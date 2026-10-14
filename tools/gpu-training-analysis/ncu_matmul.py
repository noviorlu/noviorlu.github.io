"""用 Nsight Compute 测 matmul kernel 在 num_stages = 1 和 3 时 warp 在等什么（RTX 5090，图 2-6 的数据）。

形状 A 4032 × 2048、B 2048 × 512：504 个 program，每个 SM 3 个时正好一批，不受 tail effect 影响。
Run: NCU=/path/to/ncu python3 ncu_matmul.py   (needs CUDA)，结果写到 ncu_matmul.json。
ncu 会把 SM 频率锁在基础频率，耗时比平时长一些。
"""
import csv, io, json, os, subprocess, sys
import torch

M, K, N, STAGES = 4032, 2048, 512, (1, 3)
METRICS = ["gpu__time_duration.sum", "smsp__average_warp_latency_per_inst_issued.ratio",
           "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
           "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.pct_of_peak_sustained_elapsed",
           "lts__t_sector_hit_rate.pct", "sm__warps_active.avg.per_cycle_active"]
STALL = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"
REASONS = ["selected", "not_selected", "dispatch_stall", "short_scoreboard", "mio_throttle", "long_scoreboard", "lg_throttle",
           "barrier", "wait", "math_pipe_throttle", "no_instruction", "branch_resolving", "drain", "misc"]


def run_kernels():
    from triton_tutorial import matmul_relu_kernel
    a = torch.randn(M, K, device="cuda"); b = torch.randn(K, N, device="cuda"); c = torch.empty(M, N, device="cuda")
    for ns in STAGES:
        matmul_relu_kernel[(M // 64, N // 64)](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(), BM=64, BN=64, BK=32, num_stages=ns)
    torch.cuda.synchronize()


def main():
    ncu = os.environ.get("NCU", "ncu")
    ms = ",".join(METRICS + [STALL.format(r) for r in REASONS])
    out = subprocess.run([ncu, "-k", "regex:matmul_relu", "--metrics", ms, "--csv", sys.executable, __file__, "--child"],
                         capture_output=True, text=True, check=True).stdout
    rows = list(csv.DictReader(io.StringIO(out[out.index('"ID"'):])))
    res = []
    for ns, kid in zip(STAGES, sorted({r["ID"] for r in rows}, key=int)):
        v = {r["Metric Name"]: float(r["Metric Value"].replace(",", "")) for r in rows if r["ID"] == kid}
        res.append({"num_stages": ns, "us": v[METRICS[0]] / 1e3 if v[METRICS[0]] > 1e4 else v[METRICS[0]],
                    "cycles_per_inst": v[METRICS[1]], "fma_pipe_pct": v[METRICS[2]], "shared_ld_pipe_pct": v[METRICS[3]],
                    "l2_hit_pct": v[METRICS[4]], "warps_per_sm": v[METRICS[5]],
                    "stall": {r: v[STALL.format(r)] for r in REASONS}})
    json.dump({"M": M, "K": K, "N": N, "runs": res}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncu_matmul.json"), "w"), indent=1)
    for r in res: print(r["num_stages"], round(r["us"], 1), round(r["cycles_per_inst"], 2), {k: round(x, 2) for k, x in r["stall"].items() if x > .03})


if __name__ == "__main__":
    run_kernels() if "--child" in sys.argv else main()
