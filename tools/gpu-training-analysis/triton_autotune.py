"""第 2 节四个 kernel 的 autotune 版：kernel 本身不变，只是套一层 triton.autotune，让它在候选配置里挑最快的（第 3.7 节）。

The parts between `# post:<name>` and `# post:end` are pasted into the post by assemble.py.
计时和选中的配置由 section2_bench.py 记录。
"""
import json, os, statistics, time
import torch
import triton
from triton_tutorial import gelu_kernel, softmax_kernel, row_sum_kernel, matmul_relu_kernel

gelu_tuned = triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=w) for b in (256, 512, 1024, 2048, 4096) for w in (1, 2, 4, 8, 16)],
    key=["n"])(gelu_kernel)
softmax_tuned = triton.autotune(                      # 一行要整个装进 BLOCK，只能调 num_warps
    configs=[triton.Config({}, num_warps=w) for w in (1, 2, 4, 8, 16, 32)], key=["N"])(softmax_kernel)
row_sum_tuned = triton.autotune(
    configs=[triton.Config({"TILE": t}, num_warps=w) for t in (512, 1024, 2048, 4096) for w in (2, 4, 8, 16)],
    key=["N"])(row_sum_kernel)


# post:matmul_autotune
configs = [triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s)
           for bm, bn in ((64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128))
           for bk in (16, 32) for w in (4, 8) for s in (2, 3)]
matmul_relu_tuned = triton.autotune(configs=configs, key=["M", "N", "K"])(matmul_relu_kernel)


def matmul_relu_auto(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))  # grid 跟着选中的 BM、BN 变
    matmul_relu_tuned[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride())
    return c
# post:end


def gelu_auto(x):
    y = torch.empty_like(x); n = x.numel()
    gelu_tuned[lambda META: (triton.cdiv(n, META["BLOCK"]),)](x, y, n)
    return y


def softmax_auto(x):
    M, N = x.shape; y = torch.empty_like(x)
    softmax_tuned[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y


def row_sum_auto(x):
    M, N = x.shape; out = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_tuned[(M,)](x, out, N)
    return out


