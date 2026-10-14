"""matmul + ReLU 的每个 program 在哪个 SM、什么时候开始和结束（RTX 5090，图 A-3 的数据）。

kernel 和 triton_tutorial.py 里的 matmul_relu_kernel 一样，只是开头读一次 %globaltimer 和 %smid，
写回 C 之后再读一次 %globaltimer。A 取 4032 × 2048（504 个 program，一批放下）和 4096 × 2048（512 个，
每个 SM 只放 3 个，一批 510 个，多出 2 个）。记录前先写 256 MB 清掉 L2。
Run: python3 mm_wave_trace.py   (needs CUDA)，结果写到 mm_wave_trace.json。
"""
import json, os
import torch
import triton
import triton.language as tl
from wave_trace import _asm_u64


@triton.jit
def matmul_relu_traced(a_ptr, b_ptr, c_ptr, t_ptr, M, N, K,
                       sam, sak, sbk, sbn, scm, scn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    z = tl.zeros([1], tl.int32)
    t0 = _asm_u64("mov.u64 $0, %globaltimer;", z)
    sm = tl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r,r", [z], dtype=tl.int32, is_pure=False, pack=1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    acc = tl.maximum(acc, 0.0)
    c_ptrs = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
    t1 = _asm_u64("mov.u64 $0, %globaltimer;", z)
    pid = pid_m * tl.num_programs(1) + pid_n
    o = tl.arange(0, 1)
    tl.store(t_ptr + pid * 3 + o, t0)
    tl.store(t_ptr + pid * 3 + 1 + o, t1)
    tl.store(t_ptr + pid * 3 + 2 + o, sm.to(tl.int64))


def trace(M, K=2048, N=512):
    a, b = torch.randn(M, K, device="cuda"), torch.randn(K, N, device="cuda")
    c = torch.empty(M, N, device="cuda")
    P = (M // 64) * (N // 64)
    t = torch.zeros(P * 3, dtype=torch.int64, device="cuda")
    run = lambda: matmul_relu_traced[(M // 64, N // 64)](a, b, c, t, M, N, K, *a.stride(), *b.stride(), *c.stride(),
                                                          BM=64, BN=64, BK=32)
    for _ in range(20):
        run()
    us = triton.testing.do_bench(run, warmup=50, rep=1000) * 1e3          # 打点版的平均耗时
    torch.empty(256 * 2**20, dtype=torch.int8, device="cuda").zero_()    # 清掉 L2
    run(); torch.cuda.synchronize()
    tr = t.view(P, 3).cpu()
    t0 = tr[:, 0].min()
    return {"M": M, "programs": P, "us": us, "trace": [[int(x - t0), int(y - t0), int(s)] for x, y, s in tr.tolist()]}


def main():
    res = [trace(M) for M in (4032, 4096)]
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_wave_trace.json"), "w"))
    for r in res:
        tr = r["trace"]
        print(r["M"], r["programs"], "span us", max(b for _, b, _ in tr) / 1e3, "bench us", round(r["us"], 1),
              "starts after 10us:", sorted(round(a / 1e3, 1) for a, _, _ in tr if a > 10000))


if __name__ == "__main__":
    main()
