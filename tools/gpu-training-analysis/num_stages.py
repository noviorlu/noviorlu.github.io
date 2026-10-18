"""num_stages 改了什么：matmul 和 row sum 各取几档，看编译出的槽数、shared memory、每 SM 能放几个 program 和耗时（RTX 5090，第 3.1 节的数据）。

槽数读自 TTGIR 里 A、B 的 local_alloc（形如 2x64x32），cp.async 条数是 async_copy_global_to_local 的个数。
耗时用 triton.testing.do_bench，每档跑 5 次取中位数。形状和正文一样。
Run: python3 num_stages.py   (needs CUDA)，结果写到 num_stages.json。
"""
import json, os, re, statistics
import torch
import triton
from triton.testing import do_bench
from occupancy_sweep import blocks_per_sm
from triton_tutorial import row_sum_kernel, matmul_relu_kernel


def compiled(fn, ns):
    fn()
    return next(k for k in fn.kernel.device_caches[0][0].values() if k.metadata.num_stages == ns)


def info(k, f):
    s = k.asm["ttgir"]
    allocs = re.findall(r"local_alloc[^\n]*memdesc<([0-9x]+)xf32", s)
    slots = [int(a.split("x")[0]) if a.count("x") == 2 else 1 for a in allocs]
    us = statistics.median(do_bench(f, warmup=50, rep=500) for _ in range(5)) * 1e3
    return {"slots": slots[0] if slots else 0, "async_copies": len(re.findall(r"async_copy_global_to_local", s)),
            "shared": k.metadata.shared, "regs": k.n_regs, "per_sm": blocks_per_sm(k), "us": round(us, 1)}


def main():
    X = torch.randn(4096, 4096, device="cuda"); s = torch.empty(4096, device="cuda")
    res = {"matmul": {}, "matmul_4032": {}, "row_sum": {}}
    for key, M in (("matmul", 4096), ("matmul_4032", 4032)):   # 4032 行时 504 个 program，每个 SM 正好 3 个，没有 tail
        A, B = torch.randn(M, 2048, device="cuda"), torch.randn(2048, 512, device="cuda"); C = torch.empty(M, 512, device="cuda")
        for ns in (1, 2, 3, 4):
            f = lambda: matmul_relu_kernel[(M // 64, 8)](A, B, C, M, 512, 2048, *A.stride(), *B.stride(), *C.stride(),
                                                          BM=64, BN=64, BK=32, num_stages=ns)
            f.kernel = matmul_relu_kernel
            res[key][ns] = info(compiled(f, ns), f)
    for ns in (1, 2, 3):
        f = lambda: row_sum_kernel[(4096,)](X, s, 4096, TILE=1024, num_stages=ns)
        f.kernel = row_sum_kernel
        res["row_sum"][ns] = info(compiled(f, ns), f)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "num_stages.json"), "w"), indent=1)
    for n, r in res.items():
        for ns, v in r.items(): print(n, ns, v)


if __name__ == "__main__":
    main()
