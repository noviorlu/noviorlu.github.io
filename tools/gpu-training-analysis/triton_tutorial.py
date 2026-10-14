"""Triton 入门的四个例子：GELU、softmax、row sum、matmul + ReLU。直接运行是和 PyTorch 对照的自检和计时。

The parts between `# post:<name>` and `# post:end` are pasted into the post by assemble.py.
"""
import math
import torch
import triton
import triton.language as tl


# post:gelu_kernel
@triton.jit
def gelu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)               # 第几个 program（≈ CUDA 的 blockIdx.x）
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # 这个 program 负责的 BLOCK 个下标
    mask = offsets < n                        # 最后一个 program 可能越界
    x = tl.load(x_ptr + offsets, mask=mask)   # 从显存读到 register
    # tanh 近似：0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))；tanh(a) = 1 − 2/(e^{2a} + 1)，e^{2a} 溢出成 inf 时仍得 1
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    e = tl.exp(2 * a)
    t = 1 - 2 / (e + 1)                       # tanh(a)
    y = 0.5 * x * (1 + t)
    tl.store(y_ptr + offsets, y, mask=mask)   # 写回显存
# post:end


# post:gelu_launch
def gelu(x):
    y = torch.empty_like(x)
    n, BLOCK = x.numel(), 1024                # 每个 program 处理 1024 个元素
    grid = (triton.cdiv(n, BLOCK),)           # program 个数，不是 SM 个数
    gelu_kernel[grid](x, y, n, BLOCK=BLOCK)
    return y
# post:end


# post:softmax_naive
def softmax_naive(x):                         # x: [M, N]，按行
    m = x.max(dim=1, keepdim=True).values     # 读 MN，写 M
    z = x - m                                 # 读 MN + M，写 MN
    e = torch.exp(z)                          # 读 MN，写 MN
    s = e.sum(dim=1, keepdim=True)            # 读 MN，写 M
    return e / s                              # 读 MN + M，写 MN
# post:end


# post:softmax_kernel
@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)                    # 一个 program 处理一整行
    cols = tl.arange(0, BLOCK)                # BLOCK ≥ N，一次装下整行
    x = tl.load(x_ptr + row * stride + cols, mask=cols < N, other=float("-inf"))
    x = x - tl.max(x, axis=0)                 # 以下全在 register 里
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * stride + cols, y, mask=cols < N)
# post:end


# post:softmax_launch
def softmax(x):
    M, N = x.shape
    y = torch.empty_like(x)
    softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
# post:end


# post:row_sum_kernel
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, TILE: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([TILE], dtype=tl.float32)  # TILE 路部分和，不是一个标量
    for start in range(0, N, TILE):           # 行比 TILE 长：一个 block 一个 block 串行读
        cols = start + tl.arange(0, TILE)
        acc += tl.load(x_ptr + row * N + cols, mask=cols < N, other=0.0)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))  # 最后只做一次跨 thread 归约
# post:end


# post:row_sum_launch
def row_sum(x, TILE=1024):
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, out, N, TILE=TILE)
    return out
# post:end


# post:matmul_kernel
@triton.jit
def matmul_relu_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       sam, sak, sbk, sbn, scm, scn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)   # 负责 C 的第 (pid_m, pid_n) 个 block
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak  # [BM, BK] 的指针
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn  # [BK, BN] 的指针
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):                 # 沿 K 一个 block 一个 block 乘加
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")  # fp32 默认会走 tf32，这里要求精确的 fp32
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    acc = tl.maximum(acc, 0.0)                # ReLU 在 register 里做完，不多一次读写
    c_ptrs = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
# post:end


# post:matmul_launch
def matmul_relu(a, b, BM=64, BN=64, BK=32):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_relu_kernel[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(), BM=BM, BN=BN, BK=BK)
    return c
# post:end


def bench(f, it=100):
    for _ in range(10): f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1e3  # us


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(8192 * 1024 + 7, device=dev)
    ref = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    print("gelu", (gelu(x) - ref).abs().max().item())
    X = torch.randn(4096, 4096, device=dev)
    print("softmax", (softmax(X) - torch.softmax(X, 1)).abs().max().item(),
          (softmax_naive(X) - torch.softmax(X, 1)).abs().max().item())
    Y = torch.randn(1000, 5000, device=dev)
    print("row_sum", (row_sum(Y) - Y.sum(1)).abs().max().item())
    A, B = torch.randn(1000, 700, device=dev), torch.randn(700, 900, device=dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    print("matmul_relu", (matmul_relu(A, B) - torch.relu(A @ B)).abs().max().item())

    # 计时：do_bench 每次调用前写 256 MB 清掉 L2，取平均；带宽按读写的总字节数算，FLOPS 按 2MNK 算
    us = lambda f: triton.testing.do_bench(f) * 1e3
    x = torch.randn(8388608, device=dev)
    for name, f in (("triton", lambda: gelu(x)), ("torch", lambda: torch.nn.functional.gelu(x, approximate="tanh"))):
        t = us(f); print(f"gelu n=8388608 {name}: {t:.1f} us, {2 * x.numel() * 4 / t / 1e-6:.2e} B/s")
    for name, f, rw in (("eager", lambda: softmax_naive(X), 8 * X.numel()), ("triton", lambda: softmax(X), 2 * X.numel()),
                        ("torch", lambda: torch.softmax(X, 1), 2 * X.numel())):
        t = us(f); print(f"softmax 4096x4096 {name}: {t:.1f} us, {rw * 4 / t / 1e-6:.2e} B/s")
    for name, f in (("triton", lambda: row_sum(X)), ("torch", lambda: X.sum(1))):
        t = us(f); print(f"row_sum 4096x4096 {name}: {t:.1f} us, {X.numel() * 4 / t / 1e-6:.2e} B/s")
    A, B = torch.randn(4096, 2048, device=dev), torch.randn(2048, 512, device=dev)
    for name, f in (("triton", lambda: matmul_relu(A, B)), ("torch", lambda: torch.relu(A @ B))):
        t = us(f); print(f"matmul_relu 4096x2048x512 {name}: {t:.1f} us, {2 * 4096 * 2048 * 512 / t / 1e-6:.2e} FLOPS")
