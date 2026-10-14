"""第 2 节四个 kernel 每个 program 用多少 register、shared memory，一个 SM 能同时放几个（RTX 5090，表 A-2 的数据）。

每 SM 能放几个用 cuOccupancyMaxActiveBlocksPerMultiprocessor 算，形状和正文一样。
Run: python3 residency.py   (needs CUDA)，结果写到 residency.json。
"""
import json, os
import torch
import triton
from occupancy_sweep import blocks_per_sm
from triton_tutorial import gelu_kernel, softmax_kernel, row_sum_kernel, matmul_relu_kernel


def main():
    x = torch.randn(8388608, device="cuda"); y = torch.empty_like(x)
    X = torch.randn(4096, 4096, device="cuda"); Y = torch.empty_like(X); s = torch.empty(4096, device="cuda")
    A, B = torch.randn(4096, 2048, device="cuda"), torch.randn(2048, 512, device="cuda"); C = torch.empty(4096, 512, device="cuda")
    ks = {"gelu": gelu_kernel[(8192,)](x, y, x.numel(), BLOCK=1024),
          "softmax": softmax_kernel[(4096,)](X, Y, X.stride(0), 4096, BLOCK=4096),
          "row_sum": row_sum_kernel[(4096,)](X, s, 4096, TILE=1024),
          "matmul": matmul_relu_kernel[(64, 8)](A, B, C, 4096, 512, 2048, *A.stride(), *B.stride(), *C.stride(), BM=64, BN=64, BK=32)}
    res = {name: {"warps": k.metadata.num_warps, "regs": k.n_regs, "shared": k.metadata.shared, "per_sm": blocks_per_sm(k)}
           for name, k in ks.items()}
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "residency.json"), "w"), indent=1)
    for n, r in res.items(): print(n, r)


if __name__ == "__main__":
    main()
