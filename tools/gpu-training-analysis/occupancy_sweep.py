"""每个 SMSP 驻留 1 到 12 个 warp 时，不同 arithmetic intensity 的逐元素 kernel 能跑多快（RTX 5090，图 1-6 的数据）。

kernel 对每个元素读 4 B、写 4 B，中间串 K 次 FMA（每次 2 FLOPs），I = 2K / 8 FLOPs/B。
grid 正好是 170 × m 个 program、每个 4 个 warp，全部同时驻留：每个 SM m 个 block，每个 SMSP m 个 warp。
program 用 grid-stride 循环走完整个数组，每个线程每轮处理 1 个数。
Run: python3 occupancy_sweep.py   (needs CUDA)，结果写到 occupancy_sweep.json。
"""
import ctypes, json, os
import torch
import triton
import triton.language as tl
from triton_tutorial import bench


@triton.jit
def fma_kernel(x_ptr, y_ptr, n, K: tl.constexpr, BLOCK: tl.constexpr):
    pid, nprog = tl.program_id(0), tl.num_programs(0)
    for start in range(pid * BLOCK, n, nprog * BLOCK):   # grid-stride：program 数固定，循环走完整个数组
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask=offs < n)
        for _ in tl.static_range(K):                      # K 次相互依赖的 FMA
            x = x * 0.999 + 0.001
        tl.store(y_ptr + offs, x, mask=offs < n)


def blocks_per_sm(k):
    cu, b = ctypes.CDLL("libcuda.so.1"), ctypes.c_int()
    cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(b), ctypes.c_void_p(k.function),
                                                   ctypes.c_int(32 * k.metadata.num_warps), ctypes.c_size_t(k.metadata.shared))
    return b.value


if __name__ == "__main__":
    sms = torch.cuda.get_device_properties(0).multi_processor_count       # 170
    x = torch.randn(64 << 20, device="cuda")                               # 256 MiB，远大于 96 MB 的 L2
    y, n, W = torch.empty_like(x), x.numel(), 4
    rows = []
    for K in (1, 8, 32, 128, 512):
        for m in range(1, 13):
            f = lambda: fma_kernel[(sms * m,)](x, y, n, K=K, BLOCK=32 * W, num_warps=W)
            k = f()
            assert blocks_per_sm(k) >= m                                   # m 个 block 都能同时驻留
            t = min(bench(f, 20) for _ in range(3))                        # µs
            rows.append({"K": K, "I": 2 * K / 8, "warps_per_smsp": m, "us": round(t, 1),
                         "flops": 2 * K * n / (t * 1e-6), "bw": 8 * n / (t * 1e-6)})
            print(f"K={K:3d} I={2 * K / 8:6.2f} m={m:2d} {t:8.1f} us {rows[-1]['flops']:.2e} FLOPS {rows[-1]['bw']:.2e} B/s")
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "occupancy_sweep.json"), "w"), indent=1)
