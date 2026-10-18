"""matmul 的 BM、BN、BK、num_warps、num_stages 扫一遍（RTX 5090，fp32，input_precision="ieee"，3.5 节的数据）。

两个形状：正文的 A 4096 × 2048、B 2048 × 512，和 4096 × 4096 × 4096。每组记录时间、register、spill、shared memory、
每 SM 放几个 program（cuOccupancyMaxActiveBlocksPerMultiprocessor），以及按一个 SM 算的 warp / register / shared memory 占用。
acc 每个 thread 超过 256 个、或 shared memory 超过每个 block 的上限的组合直接跳过。
Run: python3 mm_tile_sweep.py   (needs CUDA，约 15 分钟)，结果写到 mm_tile_sweep.json。
"""
import itertools, json, os
import torch
import triton
from occupancy_sweep import blocks_per_sm
from triton_tutorial import matmul_relu_kernel

SMS = 170


def main():
    out = []
    for M, N, K in ((4096, 512, 2048), (4096, 4096, 4096)):
        A, B = torch.randn(M, K, device="cuda"), torch.randn(K, N, device="cuda"); C = torch.empty(M, N, device="cuda")
        for BM, BN, BK, nw, ns in itertools.product((32, 64, 128, 256), (32, 64, 128, 256), (16, 32, 64), (4, 8, 16), (2, 3)):
            if BM * BN > 128 * 256 or BM * BN // (32 * nw) > 256 or (BM + BN) * BK * 4 * ns > 101376:
                continue
            grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
            f = lambda: matmul_relu_kernel[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(),
                                                 BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
            try:
                k = f(); torch.cuda.synchronize()
            except Exception:
                continue
            n = blocks_per_sm(k)
            if n == 0:
                continue
            us = triton.testing.do_bench(f, warmup=10, rep=60) * 1e3
            progs = grid[0] * grid[1]
            out.append(dict(M=M, N=N, K=K, BM=BM, BN=BN, BK=BK, nw=nw, ns=ns, us=us, regs=k.n_regs, spills=k.n_spills,
                            smem=k.metadata.shared, per_sm=n, progs=progs, acc_per_thread=BM * BN // (32 * nw),
                            warps_per_sm=n * nw, reg_util=n * nw * 32 * (-(-k.n_regs // 8) * 8) / 65536,
                            smem_util=n * (k.metadata.shared + 1024) / 102400, sms_busy=min(progs, SMS * n) / n / SMS))
            print(len(out), M, N, K, BM, BN, BK, nw, ns, f"{us:.1f}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_tile_sweep.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
