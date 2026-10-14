"""GELU 的 8192 个 program 在 170 个 SM 上什么时候开始、什么时候结束（RTX 5090，图 B-2 的数据）。

kernel 和 triton_tutorial.py 里的 gelu_kernel 一样，只是每个 program 开头读一次 %globaltimer 和 %smid，
写回显存后再读一次 %globaltimer，三个数记到 trace[pid]。记录前先写 256 MB 把 L2 清掉，x、y 都从显存读写。
Run: python3 wave_trace.py   (needs CUDA)，结果写到 wave_trace.json。
"""
import json, os
import torch
import triton
import triton.language as tl


@triton.jit
def _asm_u64(asm: tl.constexpr, dummy):
    return tl.inline_asm_elementwise(asm, "=l,r", [dummy], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def gelu_traced(x_ptr, y_ptr, t_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    z = tl.zeros([1], tl.int32)
    t0 = _asm_u64("mov.u64 $0, %globaltimer;", z)
    sm = tl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r,r", [z], dtype=tl.int32, is_pure=False, pack=1)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    e = tl.exp(2 * a)
    t = 1 - 2 / (e + 1)                       # tanh(a)
    y = 0.5 * x * (1 + t)
    tl.store(y_ptr + offsets, y, mask=mask)
    t1 = _asm_u64("mov.u64 $0, %globaltimer;", z)
    o = tl.arange(0, 1)
    tl.store(t_ptr + pid * 3 + o, t0)
    tl.store(t_ptr + pid * 3 + 1 + o, t1)
    tl.store(t_ptr + pid * 3 + 2 + o, sm.to(tl.int64))


def main():
    n, BLOCK = 8388608, 1024
    P = n // BLOCK
    x = torch.randn(n, device="cuda")
    y = torch.empty_like(x)
    t = torch.zeros(P * 3, dtype=torch.int64, device="cuda")
    for _ in range(20):
        gelu_traced[(P,)](x, y, t, n, BLOCK=BLOCK)
    flush = torch.empty(256 * 2**20, dtype=torch.int8, device="cuda")
    flush.zero_()                                   # 写 256 MB 把 L2（96 MB）里的 x、y 挤出去
    gelu_traced[(P,)](x, y, t, n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    tr = t.view(P, 3).cpu()
    t0 = tr[:, 0].min()
    rows = [[int(a - t0), int(b - t0), int(s)] for a, b, s in tr.tolist()]
    out = {"n": n, "BLOCK": BLOCK, "programs": P, "trace": rows}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "wave_trace.json"), "w"))
    d = [b - a for a, b, _ in rows]
    print("span ns", max(r[1] for r in rows), "mean dur", sum(d) / len(d), "sms", len({r[2] for r in rows}))


if __name__ == "__main__":
    main()
