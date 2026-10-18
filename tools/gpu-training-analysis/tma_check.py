"""5090 上能不能用 TMA：用 Triton 的 tensor descriptor 写一个复制 kernel，看编出来的 PTX 里用了哪些搬运指令（第 1.4 节）。

Run: python3 tma_check.py   (needs CUDA)，打印结果对不对、PTX 里的搬运指令和 target。
"""
import re, torch, triton, triton.language as tl

@triton.jit
def copy_desc(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    t = xd.load([pid * BM, 0])
    yd.store([pid * BM, 0], t + 1)

def alloc(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(alloc)
x = torch.randn(1024, 64, device="cuda"); y = torch.empty_like(x)
k = copy_desc[(1024 // 64,)](x, y, 1024, 64, BM=64, BN=64)
torch.cuda.synchronize()
print("ok", torch.allclose(y, x + 1))
ptx = k.asm["ptx"]
print(sorted(set(re.findall(r"(cp\.async\.bulk[\w.:]*|cp\.reduce\.async\.bulk[\w.:]*|mbarrier[\w.:]*|ld\.global[\w.]*|st\.global[\w.]*|cp\.async\.\w+)", ptx))))
print("target", re.search(r"\.target\s+(\S+)", ptx).group(1))
