"""RMSNorm: eager vs hand-written Triton (one forward kernel, one backward kernel).

The parts between `# post:<name>` and `# post:end` are pasted into the post by assemble.py
via {{code:rmsnorm_triton.py:<name>}}. Run this file to check the Triton version against eager:
  python3 rmsnorm_triton.py                      # on a GPU
  TRITON_INTERPRET=1 python3 rmsnorm_triton.py   # on CPU, Triton's interpreter
"""
# post:eager
import torch

def rmsnorm_eager(x, weight, eps=1e-6):
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # ① pow ② mean ③ rsqrt，各一个 kernel
    x_hat = x * rms                                            # ④
    return weight * x_hat                                      # ⑤
# autograd 为反向存下 x、r、x̂、w（表 3-1），反向再由 autograd 逐个 op 执行
# post:end

# post:triton
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_fwd(X, W, Y, R, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)                       # 一个 program 算一行
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * D + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)  # ①②③ 在寄存器里做完
    tl.store(R + row, r)                           # 只存 r：每行 4 B
    tl.store(Y + row * D + cols, w * (x * r), mask=mask)  # ④⑤，x̂ 不落显存

@triton.jit
def rmsnorm_bwd(DY, X, W, R, DX, DW, M, D, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    w = tl.load(W + cols, mask=mask, other=0.0)
    dw = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(ROWS):                        # 一个 program 算 ROWS 行
        row = tl.program_id(0) * ROWS + i
        m = mask & (row < M)
        x = tl.load(X + row * D + cols, mask=m, other=0.0)
        dy = tl.load(DY + row * D + cols, mask=m, other=0.0)
        r = tl.load(R + row, mask=row < M, other=0.0)
        x_hat = x * r                            # 现场重算 x̂
        g = w * dy
        dx = r * (g - x_hat * tl.sum(g * x_hat, axis=0) / D)
        tl.store(DX + row * D + cols, dx, mask=m)
        dw += dy * x_hat
    tl.atomic_add(DW + cols, dw, mask=mask)      # 各 program 的 dw 部分和累加到一起

class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        D = x.shape[-1]
        x2 = x.reshape(-1, D)
        M = x2.shape[0]
        y = torch.empty_like(x2)
        r = torch.empty(M, device=x.device, dtype=torch.float32)
        rmsnorm_fwd[(M,)](x2, weight, y, r, D, eps, BLOCK=triton.next_power_of_2(D))
        ctx.save_for_backward(x2, weight, r)     # 存 x、w、r，不存 x̂
        return y.view_as(x)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, r = ctx.saved_tensors
        M, D = x2.shape
        dx = torch.empty_like(x2)
        dw = torch.zeros(D, device=x2.device, dtype=torch.float32)
        ROWS = 16
        rmsnorm_bwd[(triton.cdiv(M, ROWS),)](dy.reshape(M, D).contiguous(), x2, weight, r, dx, dw, M, D,
                                             ROWS=ROWS, BLOCK=triton.next_power_of_2(D))
        return dx.view(dy.shape), dw, None
# post:end

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    for shape in [(4, 512, 2560), (3, 7, 100)]:      # the post's shape, and one with D not a power of 2, M % ROWS != 0
        x = torch.randn(*shape, device=dev, requires_grad=True)
        w = torch.randn(shape[-1], device=dev, requires_grad=True)
        dy = torch.randn(*shape, device=dev)
        y_ref = rmsnorm_eager(x, w)
        dx_ref, dw_ref = torch.autograd.grad(y_ref, (x, w), dy)
        y = RMSNorm.apply(x, w)
        dx, dw = torch.autograd.grad(y, (x, w), dy)
        for name, a, b in [("y", y, y_ref), ("dx", dx, dx_ref), ("dw", dw, dw_ref)]:
            err = ((a - b).abs().max() / b.abs().max()).item()
            print(shape, name, f"{err:.1e}")
            assert err < 1e-5, (name, err)
    print("ok")
