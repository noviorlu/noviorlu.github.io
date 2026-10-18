"""第 2 节开头的对比：同一个 op 用 PyTorch eager 直接写 vs 用 Triton 写（RTX 5090，fp32，形状和正文一样）。

eager 列是读者用张量运算直接写的版本：GELU 按 tanh 公式逐个运算，softmax 是正文的 softmax_naive，
row sum 是 x.sum(1)，matmul + ReLU 是 torch.relu(A @ B)（cuBLAS，关掉 tf32）。builtin 列是 PyTorch 自带的单个融合 kernel，作参照。
耗时用 triton.testing.do_bench，每个跑 5 次取中位数；kernel 个数用 torch.profiler 数。
Run: python3 section2_bench.py   (needs CUDA)，结果写到 section2_bench.json。
"""
import json, math, os, statistics
import torch
import triton
from torch.profiler import profile, ProfilerActivity
from triton_tutorial import gelu, softmax, softmax_naive, row_sum, matmul_relu


def gelu_eager(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def us(f):
    return round(statistics.median(triton.testing.do_bench(f) for _ in range(5)) * 1e3, 1)


def kernels(f):
    f(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        f(); torch.cuda.synchronize()
    return sum(1 for e in p.events() if e.device_type.name == "CUDA")


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(8388608, device="cuda")
    X = torch.randn(4096, 4096, device="cuda")
    A, B = torch.randn(4096, 2048, device="cuda"), torch.randn(2048, 512, device="cuda")
    cases = {
        "gelu": {"shape": "n = 8388608", "eager": lambda: gelu_eager(x), "triton": lambda: gelu(x),
                 "builtin": lambda: torch.nn.functional.gelu(x, approximate="tanh")},
        "softmax": {"shape": "4096 × 4096", "eager": lambda: softmax_naive(X), "triton": lambda: softmax(X),
                    "builtin": lambda: torch.softmax(X, 1)},
        "row_sum": {"shape": "4096 × 4096", "eager": lambda: X.sum(1), "triton": lambda: row_sum(X)},
        "matmul_relu": {"shape": "4096 × 2048 × 512", "eager": lambda: torch.relu(A @ B), "triton": lambda: matmul_relu(A, B)},
    }
    res = {}
    for name, c in cases.items():
        r = {"shape": c["shape"]}
        for k in ("eager", "triton", "builtin"):
            if k in c:
                r[k] = {"us": us(c[k]), "kernels": kernels(c[k])}
        res[name] = r
        print(name, r)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "section2_bench.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
