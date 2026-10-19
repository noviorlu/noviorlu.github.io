"""用 Nsight Compute 对比 cuBLAS 和 Triton 的 fp32 matmul（RTX 5090，A 4096 × 2048、B 2048 × 512，3.4 节 notes 的数据）。

cuBLAS 关掉 tf32，调到的是 CUTLASS 的 SIMT SGEMM；Triton 测正文的 64 × 64 × 32 和 autotune 选中的 128 × 128 × 32。
--clock-control none：不锁频率，拍数 = kernel 耗时 × SM 时钟频率。
Run: NCU=/path/to/ncu python3 cublas_vs_triton.py   (needs CUDA)，结果写到 cublas_vs_triton.json。
"""
import csv, io, json, os, subprocess, sys
import torch

M, N, K = 4096, 512, 2048
CFG = [("triton_64", 64, 64, 32, 4, 3), ("triton_128", 128, 128, 32, 8, 3)]
MET = ["gpu__time_duration.sum", "sm__cycles_elapsed.avg.per_second", "launch__grid_dim_x", "launch__grid_dim_y", "launch__grid_dim_z",
       "launch__block_size", "launch__registers_per_thread", "launch__shared_mem_per_block_dynamic", "sm__warps_active.avg.per_cycle_active",
       "smsp__inst_executed.sum", "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "sm__sass_inst_executed_op_shared_ld.sum",
       "sm__sass_data_bytes_mem_shared_op_ld.sum", "smsp__average_warp_latency_per_inst_issued.ratio",
       "smsp__issue_active.avg.pct_of_peak_sustained_active", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"]
STALL = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"
REASONS = ["selected", "not_selected", "dispatch_stall", "short_scoreboard", "mio_throttle", "long_scoreboard", "lg_throttle",
           "barrier", "wait", "math_pipe_throttle", "no_instruction"]


def run_kernels():
    from triton_tutorial import matmul_relu_kernel
    torch.backends.cuda.matmul.allow_tf32 = False
    a, b = torch.randn(M, K, device="cuda"), torch.randn(K, N, device="cuda"); c = torch.empty(M, N, device="cuda")
    a @ b
    for _, BM, BN, BK, nw, ns in CFG:
        matmul_relu_kernel[(M // BM, N // BN)](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(),
                                               BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    torch.cuda.synchronize()


def main():
    ncu = os.environ.get("NCU", "ncu")
    out = subprocess.run([ncu, "--kernel-name-base", "demangled", "-k", "regex:sgemm|matmul_relu", "--clock-control", "none",
                          "--metrics", ",".join(MET + [STALL.format(r) for r in REASONS]), "--csv", sys.executable, __file__, "--child"],
                         capture_output=True, text=True, check=True).stdout
    rows = list(csv.DictReader(io.StringIO(out[out.index('"ID"'):])))
    res = {}
    for label, kid in zip(["cublas"] + [c[0] for c in CFG], sorted({r["ID"] for r in rows}, key=int)):
        rs = [r for r in rows if r["ID"] == kid]
        v = {r["Metric Name"]: float(r["Metric Value"].replace(",", "")) for r in rs}
        res[label] = {"kernel": rs[0]["Kernel Name"], "us": v[MET[0]] / 1e3, "cycles": v[MET[0]] * 1e-9 * v[MET[1]],
                      "grid": [int(v[m]) for m in MET[2:5]], "block": int(v[MET[5]]), "regs": int(v[MET[6]]), "smem": v[MET[7]],
                      "warps_per_sm": v[MET[8]], "inst": v[MET[9]], "ffma": v[MET[10]], "lds_inst": v[MET[11]], "lds_bytes": v[MET[12]],
                      "cycles_per_inst": v[MET[13]], "issue_pct": v[MET[14]], "fma_pipe_pct": v[MET[15]],
                      "stall": {r: v[STALL.format(r)] for r in REASONS}}
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cublas_vs_triton.json"), "w"), indent=1)
    for k, r in res.items():
        print(k, r["kernel"][:50], round(r["us"], 1), r["grid"], r["regs"], round(r["lds_bytes"] / 1e9, 3), round(r["cycles_per_inst"], 2))


if __name__ == "__main__":
    run_kernels() if "--child" in sys.argv else main()
