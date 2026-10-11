"""每个 SMSP 驻留 1 到 12 个 warp 时，一个逐元素 kernel 能跑到多少带宽（RTX 5090，图 1-6 的数据）。

grid 正好是 170 × m 个 program、每个 4 个 warp，全部同时驻留：每个 SM m 个 block，每个 SMSP m 个 warp。
program 用 grid-stride 循环把整个数组走完，每个线程每轮读写 E 个 fp32（E = 1 或 4，一条 4 B 或 16 B 的 load）。
E = 1 时每个 warp 同时在路上的读很少，带宽只能靠驻留更多 warp 来撑。
Run: python3 occupancy_sweep.py   (needs CUDA)，结果写到 occupancy_sweep.json。
"""
import ctypes, json, os
import torch
import triton
import triton.language as tl
from triton_tutorial import bench


@triton.jit
def scale_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid, nprog = tl.program_id(0), tl.num_programs(0)
    for start in range(pid * BLOCK, n, nprog * BLOCK):   # grid-stride：program 数固定，循环走完整个数组
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask=offs < n)
        tl.store(y_ptr + offs, x * 2.0, mask=offs < n)


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
    for E in (1, 4):
        BLOCK = 32 * W * E                                                 # 每个线程每轮 E 个数
        for m in range(1, 13):
            f = lambda: scale_kernel[(sms * m,)](x, y, n, BLOCK=BLOCK, num_warps=W)
            k = f()
            assert blocks_per_sm(k) >= m                                   # m 个 block 都能同时驻留
            t = min(bench(f) for _ in range(3))                            # µs
            rows.append({"E": E, "warps_per_smsp": m, "us": round(t, 1), "bw": 8 * n / (t * 1e-6)})
            print(f"E={E} 每 SMSP {m:2d} 个 warp：{t:7.1f} µs，{rows[-1]['bw']:.2e} B/s")
    assert torch.equal(y, x * 2)
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "occupancy_sweep.json"), "w"), indent=1)
