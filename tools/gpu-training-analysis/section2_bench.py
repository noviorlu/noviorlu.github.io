"""第 2 节开头的对比：同一个 op 用 PyTorch eager 直接写 vs 用 Triton 写（RTX 5090，fp32，形状和正文一样）。

eager 是读者用张量运算直接写的版本：GELU 按 tanh 公式逐个运算，softmax 是正文的 softmax_naive，
row sum 是 x.sum(1)，matmul + ReLU 是 torch.relu(A @ B)（cuBLAS，关掉 tf32）。compile 是同一份 eager 代码包一层 torch.compile。
耗时用 triton.testing.do_bench，每个跑 5 次取中位数；每个 kernel 的名字和耗时用 torch.profiler 记，只用来核对正文的说法。
Run: python3 section2_bench.py   (needs CUDA)，结果写到 section2_bench.json。
"""
import json, math, os, statistics
import torch
import triton
from torch.profiler import profile, ProfilerActivity
from triton_tutorial import gelu, softmax, softmax_naive, row_sum, row_sum_kernel, matmul_relu


def gelu_eager(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def us(f):
    return round(statistics.median(triton.testing.do_bench(f) for _ in range(5)) * 1e3, 1)


def kernels(f):
    """跑了哪些 kernel、各自多少 µs（profiler 里 L2 是热的，只用来看各 kernel 占多少）。"""
    f(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        f(); torch.cuda.synchronize()
    return [[e.name[:60], round(e.device_time, 1)] for e in p.events() if e.device_type.name == "CUDA"]


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(8388608, device="cuda")
    X = torch.randn(4096, 4096, device="cuda")
    A, B = torch.randn(4096, 2048, device="cuda"), torch.randn(2048, 512, device="cuda")
    fns = {"gelu": (lambda x: gelu_eager(x), lambda: gelu(x), (x,)),
           "softmax": (lambda X: softmax_naive(X), lambda: softmax(X), (X,)),
           "row_sum": (lambda X: X.sum(1), lambda: row_sum(X), (X,)),
           "matmul_relu": (lambda A, B: torch.relu(A @ B), lambda: matmul_relu(A, B), (A, B))}
    shapes = {"gelu": "n = 8388608", "softmax": "4096 × 4096", "row_sum": "4096 × 4096", "matmul_relu": "4096 × 2048 × 512"}
    cases = {}
    for name, (eager, tri, args) in fns.items():
        comp = torch.compile(eager)
        cases[name] = {"shape": shapes[name], "eager": lambda e=eager, a=args: e(*a),
                       "compile": lambda c=comp, a=args: c(*a), "triton": tri}
    res = {}
    for name, c in cases.items():
        r = {"shape": c["shape"]}
        for k in ("eager", "compile", "triton"):
            if k in c:
                r[k] = {"us": us(c[k]), "kernels": kernels(c[k])}
        res[name] = r
        print(name, r)
    out = torch.empty(4096, device="cuda")                                          # row sum 换一组 TILE、num_warps
    res["row_sum"]["triton_tile2048_nw16"] = {"us": us(lambda: row_sum_kernel[(4096,)](X, out, 4096, TILE=2048, num_warps=16))}
    print(res["row_sum"]["triton_tile2048_nw16"])
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "section2_bench.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
