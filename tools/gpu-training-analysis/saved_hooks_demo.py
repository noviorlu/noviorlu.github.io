"""Print which tensors autograd saves for backward (RMSNorm and attention, eager, fp32).

The parts between `# post:<name>` and `# post:end` are pasted into the post by assemble.py.
Run: python3 saved_hooks_demo.py   (needs CUDA)
"""
import math
import torch

# post:hooks
blocks, count = {}, [0]  # data_ptr -> 块编号：编号相同就是同一块内存

def block(t):
    return blocks.setdefault(t.data_ptr(), "ABCDEFGHIJ"[len(blocks)])

def pack(t):  # 前向每存下一个张量调用一次，返回值会被保存
    count[0] += 1
    print(f"Saving  {count[0]}  {list(t.shape)}  {str(t.dtype)[6:]}  块 {block(t)}")
    return t

def unpack(t):  # 反向每取出一个张量调用一次，收到的就是 pack 的返回值
    print(f"Loading    块 {block(t)}")
    return t

def show(fn, *inputs):
    blocks.clear(); count[0] = 0
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = fn(*inputs)  # 前向：触发 pack
    out.sum().backward()   # 反向：触发 unpack
# post:end


# post:models
def rmsnorm(x, w, eps=1e-5):
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # ① ② ③
    return w * (x * r)                                       # ④ ⑤

def attention(q, k, v, mask):
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])     # ① ②
    s = s.masked_fill(mask, float("-inf"))                    # ③
    m = s.max(dim=-1, keepdim=True).values                    # ④ softmax 的 5 个 kernel
    e = torch.exp(s - m)
    p = e / e.sum(dim=-1, keepdim=True)
    return p @ v                                              # ⑤

dev = "cuda"
x = torch.randn(4, 512, 2560, device=dev, requires_grad=True)
w = torch.ones(2560, device=dev, requires_grad=True)
show(rmsnorm, x, w)

q, k, v = (torch.randn(4, 16, 1024, 64, device=dev, requires_grad=True) for _ in range(3))
mask = torch.triu(torch.ones(1024, 1024, dtype=torch.bool, device=dev), diagonal=1)
show(attention, q, k, v, mask)
# post:end
