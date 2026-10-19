"""换 block 大小以后 num_stages 的影响（RTX 5090，fp32，A 4096 × 2048、B 2048 × 512，3.3 节 notes 的数据）。

四种 block（16 × 16 × 16 到 128 × 128 × 32），num_stages 取 1 到 4。每组记录 register、shared memory、
每个 SM 放几个 program（cuOccupancyMaxActiveBlocksPerMultiprocessor）、要跑几批，耗时是 do_bench 跑 5 次的中位数。
Run: python3 stages_blocks.py   (needs CUDA)，结果写到 stages_blocks.json。
"""
import json, math, os
import torch
import triton
from occupancy_sweep import blocks_per_sm
from triton_tutorial import matmul_relu_kernel

M, N, K, SMS = 4096, 512, 2048, 170
BLOCKS = [(16, 16, 16, 4), (32, 32, 16, 4), (64, 64, 32, 4), (128, 128, 32, 8)]   # BM, BN, BK, num_warps


def main():
    A, B = torch.randn(M, K, device="cuda"), torch.randn(K, N, device="cuda"); C = torch.empty(M, N, device="cuda")
    out = []
    for BM, BN, BK, nw in BLOCKS:
        for ns in (1, 2, 3, 4):
            grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
            f = lambda: matmul_relu_kernel[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(),
                                                 BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
            k = f(); torch.cuda.synchronize()
            n = blocks_per_sm(k)
            us = sorted(triton.testing.do_bench(f, warmup=25, rep=200) * 1e3 for _ in range(5))[2]
            progs = grid[0] * grid[1]
            r = dict(BM=BM, BN=BN, BK=BK, nw=nw, ns=ns, us=us, regs=k.n_regs, spills=k.n_spills, smem=k.metadata.shared,
                     per_sm=n, progs=progs, waves=math.ceil(progs / (SMS * n)))
            out.append(r); print(r, flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stages_blocks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
