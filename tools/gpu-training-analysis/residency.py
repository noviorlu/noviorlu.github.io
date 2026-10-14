"""第 2 节四个 kernel 每个 program 用多少 register、shared memory，一个 SM 能同时放几个（RTX 5090，表 B-2 的数据）。

每 SM 能放几个用 cuOccupancyMaxActiveBlocksPerMultiprocessor 算，形状和正文一样。
另外按表 B-1 的四条上限各算一次「只看这一条能放几个」（图 B-1），四个取最小，和 cuOccupancy 的结果对一遍。
Run: python3 residency.py   (needs CUDA)，结果写到 residency.json。
"""
import ctypes, json, os
import torch
import triton
from occupancy_sweep import blocks_per_sm
from triton_tutorial import gelu_kernel, softmax_kernel, row_sum_kernel, matmul_relu_kernel


REG_UNIT, SMEM_UNIT = 256, 128   # register 按 warp 以 256 个为单位分配，shared memory 以 128 B 为单位（CUDA Occupancy Calculator）


def sm_limits():
    cu, v = ctypes.CDLL("libcuda.so.1"), ctypes.c_int()
    def attr(a):
        cu.cuDeviceGetAttribute(ctypes.byref(v), a, 0); return v.value
    # CU_DEVICE_ATTRIBUTE_*：MAX_THREADS_PER_MULTIPROCESSOR, MAX_BLOCKS_PER_MULTIPROCESSOR, MAX_REGISTERS_PER_MULTIPROCESSOR,
    # MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, RESERVED_SHARED_MEMORY_PER_BLOCK
    return {"warps": attr(39) // 32, "blocks": attr(106), "regs": attr(82), "shared": attr(81), "reserved": attr(111)}


def by_limit(r, L):
    """只看一条上限时一个 SM 能放几个 program，以及每个 program 占这条上限多少。"""
    regs_prog = -(-r["regs"] * 32 // REG_UNIT) * REG_UNIT * r["warps"]
    smem_prog = -(-r["shared"] // SMEM_UNIT) * SMEM_UNIT + L["reserved"]
    use = {"warps": r["warps"], "blocks": 1, "regs": regs_prog, "shared": smem_prog}
    return use, {k: L[k] // u for k, u in use.items()}


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
    L = sm_limits()
    for r in res.values():
        r["use"], r["max"] = by_limit(r, L)
        assert min(r["max"].values()) == r["per_sm"], r
    res["limits"] = L
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "residency.json"), "w"), indent=1)
    for n, r in res.items(): print(n, r)


if __name__ == "__main__":
    main()
