"""row sum 的 TILE 从 128 扫到 16384（RTX 5090，num_warps=4 默认值），两种形状元素总数一样（340 MiB）。

记录耗时、有效带宽，以及编译出的每线程 register 数和溢出字节数。
Run: python3 tile_sweep.py   (needs CUDA)，结果写到 tile_sweep.json。
"""
import json, os
import torch
from triton_tutorial import row_sum_kernel, bench

if __name__ == "__main__":
    rows = []
    for M, N in ((170, 524288), (10880, 8192)):
        x = torch.randn(M, N, device="cuda")
        out = torch.empty(M, device="cuda")
        for p in range(7, 15):
            T = 1 << p
            f = lambda: row_sum_kernel[(M,)](x, out, N, TILE=T)
            k = f()
            assert torch.allclose(out, x.sum(1), rtol=1e-3, atol=1e-2)
            t = min(bench(f, 20) for _ in range(8))
            rows.append({"M": M, "N": N, "TILE": T, "per_thread": T // 128, "us": round(t, 1),
                         "bw": 4 * M * N / (t * 1e-6), "regs": k.n_regs, "spills": k.n_spills})
            print(rows[-1])
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tile_sweep.json"), "w"), indent=1)
