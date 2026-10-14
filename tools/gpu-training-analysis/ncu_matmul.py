"""用 Nsight Compute 测 matmul kernel 在 num_stages = 1 和 3 时 warp 在等什么（RTX 5090，图 2-4b 的数据）。

A 4032 × 2048、B 2048 × 512：504 个 program，每个 SM 3 个时正好一批，不受 tail effect 影响；
另测一次 A 4096 × 2048（512 个 program，多出 2 个要单独跑一批）对照 tail effect。
--clock-control none：不锁频率。ncu 每个形状只跑一次，耗时另用 do_bench 测（清 L2，取平均）。
Run: NCU=/path/to/ncu python3 ncu_matmul.py   (needs CUDA)，结果写到 ncu_matmul.json。
"""
import csv, io, json, os, subprocess, sys
import torch

K, N = 2048, 512
RUNS = [(4032, 1), (4032, 3), (4096, 3)]                # (M, num_stages)
METRICS = ["gpu__time_duration.sum", "smsp__average_warp_latency_per_inst_issued.ratio",
           "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
           "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.pct_of_peak_sustained_elapsed",
           "lts__t_sector_hit_rate.pct", "sm__warps_active.avg.per_cycle_active",
           "smsp__inst_executed.sum", "launch__grid_size", "launch__block_size", "smsp__cycles_elapsed.avg.per_second"]
STALL = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"
REASONS = ["selected", "not_selected", "dispatch_stall", "short_scoreboard", "mio_throttle", "long_scoreboard", "lg_throttle",
           "barrier", "wait", "math_pipe_throttle", "no_instruction", "branch_resolving", "drain", "misc"]


def launcher(M, ns):
    from triton_tutorial import matmul_relu_kernel
    a = torch.randn(M, K, device="cuda"); b = torch.randn(K, N, device="cuda"); c = torch.empty(M, N, device="cuda")
    return lambda: matmul_relu_kernel[(M // 64, N // 64)](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(),
                                                           BM=64, BN=64, BK=32, num_stages=ns)


def run_kernels():
    for M, ns in RUNS:
        launcher(M, ns)()
    torch.cuda.synchronize()


def main():
    ncu = os.environ.get("NCU", "ncu")
    ms = ",".join(METRICS + [STALL.format(r) for r in REASONS])
    out = subprocess.run([ncu, "-k", "regex:matmul_relu", "--clock-control", "none", "--metrics", ms, "--csv", sys.executable, __file__, "--child"],
                         capture_output=True, text=True, check=True).stdout
    rows = list(csv.DictReader(io.StringIO(out[out.index('"ID"'):])))
    res = []
    for (M, ns), kid in zip(RUNS, sorted({r["ID"] for r in rows}, key=int)):
        v = {r["Metric Name"]: float(r["Metric Value"].replace(",", "")) for r in rows if r["ID"] == kid}
        import triton
        us = triton.testing.do_bench(launcher(M, ns), warmup=100, rep=2000) * 1e3
        res.append({"M": M, "num_stages": ns, "us": us, "us_ncu": v[METRICS[0]] / 1e3 if v[METRICS[0]] > 1e4 else v[METRICS[0]],
                    "cycles_per_inst": v[METRICS[1]], "fma_pipe_pct": v[METRICS[2]], "shared_ld_pipe_pct": v[METRICS[3]],
                    "l2_hit_pct": v[METRICS[4]], "warps_per_sm": v[METRICS[5]],
                    # 每个 warp 每轮执行的指令数：总指令数 / warp 数 / (K / BK)
                    "inst_per_round": v[METRICS[6]] / (v[METRICS[7]] * v[METRICS[8]] / 32) / (K // 32),
                    "sm_hz": v[METRICS[9]],
                    "stall": {r: v[STALL.format(r)] for r in REASONS}})
    torch.backends.cuda.matmul.allow_tf32 = False                     # cuBLAS 对照：fp32，不走 tf32
    a, b = torch.randn(4096, K, device="cuda"), torch.randn(K, N, device="cuda")
    cublas_us = triton.testing.do_bench(lambda: torch.relu(a @ b), warmup=100, rep=2000) * 1e3
    json.dump({"K": K, "N": N, "runs": res, "cublas_4096_us": cublas_us}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncu_matmul.json"), "w"), indent=1)
    print("cublas 4096", round(cublas_us, 1))
    for r in res: print(r["M"], r["num_stages"], round(r["us"], 1), round(r["cycles_per_inst"], 2), {k: round(x, 2) for k, x in r["stall"].items() if x > .03})


if __name__ == "__main__":
    run_kernels() if "--child" in sys.argv else main()
