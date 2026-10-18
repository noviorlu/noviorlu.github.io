---
title: "单卡 GPU 优化方法：从 divergence 到 tiling"
date: 2026-10-18
draft: false
math: true
description: "单卡上常用的七种优化方法：control divergence、低精度、fusion、recomputation、memory coalescing、bank conflict 和 tiling。每种方法省的是什么、对应哪一层硬件，配上前后几篇里的实测。GPU training optimization 系列的第二篇。"
tags: ["GPU", "Triton", "CUDA", "优化", "LLM", "AI"]
categories: ["AI笔记"]
series: ["GPU training optimization"]
series_order: 2
---

[上一篇](/blog/gpu-triton-intro/)介绍了 GPU 的硬件和 Triton 的写法。这一篇把单卡上常用的优化方法放在一起：每种方法省的是什么，对应哪一层硬件，前后几篇里哪里用到了它。七种方法按从小到大的粒度排：先看一个 warp 内部（[第 1 节](#divergence)），再看数据的位宽（[第 2 节](#precision)）、kernel 之间（[第 3、4 节](#fusion)），最后是访存的方式（[第 5、6、7 节](#coalescing)）。[下一篇](/blog/gpu-training-analysis/)在 RTX 5090 上拆解一步训练，用的就是这里的几种方法。

> 环境：RTX 5090 32 GB，torch 2.11.0+cu130，Triton 3.6.0。文中的 KB、MB、GB 都按 1024 进位。

---

## 1 Control divergence {#divergence}

一个 warp 的 32 个 thread 同一时刻执行同一条指令（[入门篇 1.3 节](/blog/gpu-triton-intro/#hw-model)）。如果遇到分支，warp 里一部分 thread 走 if、另一部分走 else，硬件只能把两边都执行一遍：执行 if 那段时，走 else 的 thread 被屏蔽掉，什么也不做，反过来也一样。这叫 control divergence，代价是两段的时间加起来。分支在不同 warp 之间不一样没关系，每个 warp 只走自己那条路（[图 1-1](#fig-o-3)）。

{{fig-o-3}}

所以分支条件最好在一个 warp 里一致，比如按 program、按行来分，而不是按单个元素。逐元素的条件可以改成计算：两边都算出来，再用 select 挑一个，没有跳转，也就没有 divergence。Triton 里 `tl.where(cond, a, b)` 就是这样，`tl.load` 的 `mask` 编出来是带谓词的指令：入门篇 GELU 的 PTX 里，`@%p1 ld.global.v4.b32` 前面的 `%p1` 就是谓词，越界的 thread 不读，但大家执行的是同一条指令。Triton 里的 `if`、`for` 只能用在整个 program 共有的标量上，所以写 Triton 时很少遇到这个问题；写 CUDA 时才要自己留意。

warp 内部的问题说完，下一个是每个数占多少位。

## 2 低精度 {#precision}

把 fp32 换成 bf16、fp8、fp4，一是每个数的字节少了，同样的带宽能搬更多的数（[图 2-1](#fig-o-4)）；二是能用上更快的 Tensor core。5090 上 fp32 走 CUDA core，峰值 1.05e14 FLOPS；Tensor core 在 fp32 累加时，bf16 是 2.1e14，fp8 是 4.19e14，fp4 是 1.68e15（dense，[RTX Blackwell 白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)）。累加一般还是用 fp32：bf16 只有 7 位尾数，长求和在低精度下会丢精度。

{{fig-o-4}}

fp8 和 fp4 能表示的范围很窄，E4M3 最大只到 448，不能直接把 bf16 的数截过去，要先除以一个 scale 再取整，算完再乘回来。scale 怎么选是量化的核心问题。整个张量只用一个 scale 时，一个很大的离群值就会把 scale 撑大，其他数都挤到很少几个格子里。所以现在的格式都按 block 给 scale：[OCP 的 MX 格式](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)每 32 个连续的数共用一个 E8M0 的 scale，E8M0 只有指数位，也就是一个 2 的幂，元素是 MXFP8（E4M3 或 E5M2）或 MXFP4（E2M1）；NVIDIA 的 [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) 每 16 个 E2M1 的数共用一个 E4M3 的 scale，外面再乘一个 fp32 的 per-tensor scale。DeepSeek-V3 的 fp8 训练也是这个思路：activation 每 1 × 128 个数一个 scale，权重每 128 × 128 一个，Tensor core 每累加 128 个数就把部分和搬到 CUDA core 上用 fp32 加，避免累加误差变大（[DeepSeek-V3](https://arxiv.org/abs/2412.19437)）。

block scaling 带来一个训练时才有的麻烦：转置（[图 2-2](#fig-o-9)）。一个 block 里的数共用一个 scale，block 必须沿矩阵乘的归约方向连续。权重按 PyTorch 的 Linear 存成 [N, K]，前向 Y = X Wᵀ 沿 K 归约，block 横着分；反向算 dX = dY W 沿 N 归约，block 要竖着分，原来那组 block 和 scale 用不上。量化过的 W 再转置重新量化会叠加误差，所以 [MXFP8 的训练方案](https://arxiv.org/abs/2506.08027)给每个张量存两份，一份沿行量化、一份沿列量化，都从高精度的数量化出来。DeepSeek-V3 的 activation 在反向时也要从 1 × 128 的 tile 转成 128 × 1，它把 scale 都取成 2 的幂，让重新量化不再引入额外误差，但这一来一回要先反量化、转置、再量化，还得经过一次显存。

{{fig-o-9}}

在 Transformer 里，fp8 已经能用在训练的矩阵乘上：MXFP8 的方案在 8B 模型、15T token 上和 bf16 训练的结果对得上，权重、activation、梯度都用 E4M3；norm、softmax、loss 和优化器状态还是留在高精度。fp4 也开始用在预训练上（[NVFP4 预训练](https://arxiv.org/abs/2509.25149)），推理时更常见。再往下是只有三个值的权重：[BitNet b1.58](https://arxiv.org/abs/2402.17764) 的权重取 −1、0、+1（log₂3 ≈ 1.58 bit），先除以整个矩阵绝对值的平均再取整截断，activation 按 token 量化成 8 bit；乘以 ±1、0 只是加、减或跳过，矩阵乘几乎没有乘法。它是从头按这种精度训练的，不是训完再量化。推理时还常把训好的权重压到 2 到 4 bit，比如 [QuIP#](https://arxiv.org/abs/2402.04396)，算之前再反量化回 bf16。

还有两点容易踩。一是 Triton 的 `tl.dot` 遇到 fp32 输入默认走 tf32 的 Tensor core，结果和 fp32 矩阵乘差 0.1 左右，要精确的 fp32 得写 `input_precision="ieee"`（[入门篇 2.4 节](/blog/gpu-triton-intro/#triton-matmul)）。二是低精度省的主要是时间，显存不一定省得多：下一篇在 5090 上用 bf16 autocast 训练，前向快了 1.9–2.3 倍，显存只省了 18–21%，因为权重、梯度、Adam 状态还是 fp32，norm、softmax 这些也还在 fp32 下算（[下一篇 4.1 节](/blog/gpu-training-analysis/#bf16)）。

位宽省的是每个数的字节，下面两节省的是读写的次数。

## 3 Fusion {#fusion}

逐元素和归约类的 op 算得很少，时间几乎都花在读写显存上。几个这样的 op 接在一起时，每个 op 都把中间结果写回显存、下一个再读回来，fusion 把它们合成一个 kernel，中间结果留在 register 里，显存只读写一次（[图 3-1](#fig-o-5)）。入门篇[表 2-1](/blog/gpu-triton-intro/#tab-2-1) 里，GELU 按公式用张量运算写，eager 要跑 8 个 kernel，194.7 µs；合成一个 Triton kernel 是 46.6 µs，快 4.2 倍。softmax 从 5 个 kernel 合成 1 个，快 2.6 倍。`torch.compile` 自动做的就是这件事，它在 GPU 上生成的也是 Triton kernel。

{{fig-o-5}}

fusion 对一个 kernel 本来就受算力限制的 op 帮助不大：matmul 后面接的 ReLU 单独跑也只要 4 µs，矩阵乘本身要 150 µs 左右。下一篇把 RMSNorm 的 5 个 op 合成一个 kernel，前向、反向都少读写一遍中间结果（[下一篇 3.1 节](/blog/gpu-training-analysis/#rmsnorm)）。

合成一个 kernel 以后，中间结果不用写回显存了，但反向要用它们的时候怎么办，就是下一节的事。

## 4 Recomputation {#recompute}

训练时反向要用前向的中间结果。存下来要占显存，读回来要花带宽；如果重新算一遍比读回来便宜，就不存，反向时重算。这是拿算力换显存和带宽。三层 sigmoid 就是一个例子（[图 4-1](#fig-o-6)）：存下中间结果时，前向 1 读 3 写，反向 3 读 1 写，一共 8 次显存读写；sigmoid 的导数只要它自己的值，反向时读 x 重算一遍，前向 1 读 1 写，反向 2 读 1 写，一共 5 次。

{{fig-o-6}}

它有两个尺度。小的在一个 op 里：融合后的 RMSNorm 不存 $\hat x$，反向时用存下来的 x 和每行一个数的 r 重算。大的跨好几层，叫 activation checkpoint：前向只存每一段的入口，反向走到这一段时把前向重跑一遍。下一篇在 5090 上测过，每层一个 checkpoint 时显存从 15.0 GiB 降到 7.8 GiB，step 慢 28–33%，正好是一步从 3 次前向的量变成 4 次（[下一篇 4.2 节](/blog/gpu-training-analysis/#checkpoint)）。FlashAttention 的反向也是重算：不存 seq × seq 的 P，反向时用存下来的每行 logsumexp 重算出来（[FlashAttention 1–4](/blog/flashattention-1-to-4/)）。

前面几节都在少搬数据，后面三节看同样的数据怎么搬更快。

## 5 Memory coalescing {#coalescing}

一个 warp 发一条 load 时，硬件把 32 个 thread 的地址合起来，按 32 B 的 sector 去显存取，128 B 是一条 cache line。相邻的 thread 读相邻的地址，32 个 thread 的请求就落在几条连续的 cache line 里，取上来的字节都用得上，这叫 coalescing。相邻的 thread 隔得很远，比如按列读一个行优先的矩阵，每个 thread 都要单独取一个 sector，大部分字节是白取的（[图 5-1](#fig-o-7)）。

{{fig-o-7}}

每个 thread 一次读几个相邻的数也有关系。入门篇的 GELU 里，n 是 16 的倍数时，Triton 知道地址按 16 B 对齐，一个 thread 用一条 `ld.global.v4` 读 4 个 fp32；n 不是 16 的倍数时，就变成 4 条 4 B 的 `ld.global`（[入门篇 2.1 节 notes](/blog/gpu-triton-intro/#triton-gelu)），条数多了 4 倍。写 Triton 时，让 `tl.arange` 沿着内存里连续的那一维，block 的这一维取得够长，编译器就会排成合并的读写。

显存这边讲完，shared memory 也有自己的规矩。

## 6 Bank conflict {#bank}

shared memory 分成 32 个 bank，每个 bank 宽 4 B，地址按 4 B 轮流分到各个 bank 上（[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)）。一个 warp 读 shared memory 时，32 个 thread 落在 32 个不同的 bank 上，一拍就能读完；如果有几个 thread 落在同一个 bank 的不同地址上，这几个请求只能排队，一拍拆成几拍，这叫 bank conflict。几个 thread 读同一个地址不算冲突，会广播。

最常见的例子是按列读一个 32 × 32 的 fp32 矩阵：一列的 32 个数隔 128 B，全落在同一个 bank 上，要排 32 拍。常见的解法有两种：每行多留一个数（padding），让一列的数错开到不同的 bank；或者按一定的规律打乱存放的位置（swizzle）。[图 6-1](#fig-o-8) 按 8 个 bank 画了 padding 的效果。

{{fig-o-8}}

写 Triton 时这件事由编译器决定，shared memory 的布局写在 TTGIR 里：入门篇 2.4 节那个 fp32 的 matmul，A、B 的 slot 用的是 `swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1}>`，也就是没有打乱；走 Tensor core 的 `tl.dot` 编译器会换成别的布局。

最后一种方法把前面几件事都用上：让一个数从显存读进来以后，在片上被用很多次。

## 7 Tiling {#tiling}

matmul $C = A B$ 里，A 是 M × K，B 是 K × N。A 的每个数要和 B 的 N 列各乘一次，B 的每个数要和 A 的 M 行各乘一次。naive 的写法每个输出自己去显存读它要的那一行和那一列，A 的每个数就被读 N 次。tiling 把 C 切成 T × T 的 block，一个 program 把 A、B 对应的 block 读进 shared memory，给这个 block 里的 T × T 个输出共用（[图 7-1](#fig-o-2)）：A 的每个数只在 C 的同一行 block 里被读进来一次，一共 N / T 次，每次读进来以后在 shared memory 里被用 T 次。显存访问因此少了 T 倍（[表 7-1](#tab-7-1)）。

| | A 的每个数从显存读 | B 的每个数从显存读 | 整个 kernel 从显存读 |
|:--|--:|--:|--:|
| naive | N 次（512 次） | M 次（4096 次） | 32 GB |
| tiling（T × T 的 block） | N / T 次（8 次），每次在 shared memory 里用 T 次 | M / T 次（64 次），每次用 T 次 | 512 MB |
{#tab-7-1 caption="**表 7-1** naive 和 tiling：A、B 的每个数从显存读几次" note="括号里是 <a href=\"/blog/gpu-triton-intro/#triton-matmul\">入门篇 2.4 节</a>的形状：A 4096 × 2048、B 2048 × 512，fp32，T = 64。整个 kernel 的读量是 A 的 M × K 个数乘读的次数，加上 B 的 K × N 个数乘读的次数；512 MB 和入门篇 2.4 节 notes 里从 PTX 数出来的一致。"}

{{fig-o-2}}

tile 越大，显存读得越少，但一个 program 要占更多 register 和 shared memory，怎么取见入门篇 [3.5 节](/blog/gpu-triton-intro/#tile)。在 Triton 里 BM、BN、BK 都得是 2 的幂，这是 `tl.arange` 的要求；`tl.dot` 还要求 BK 至少 16。tile 的大小还带来两种 quantization。一是矩阵的维度不是 tile 的整数倍时，边上的 tile 只有一部分是有用的数，叫 tile quantization。二是 tile 的个数不是全卡一批能放下的整数倍时，最后一批只有几个 tile 在跑，叫 wave quantization。入门篇 3.4 节的 matmul 就是第二种：A 从 4032 行变成 4096 行，计算量只多 1.6%，耗时多 28%，因为 512 个 program 比全卡一批能放的 510 个多了 2 个（[入门篇 3.4 节](/blog/gpu-triton-intro/#sched-wave)）。

tiling 不只在 shared memory 这一层。每个 thread 从 shared memory 读进 A、B 的一小段，再和自己 register 里的一小块累加器一起算，一个数在 register 里也被用好几次；入门篇 [3.5 节](/blog/gpu-triton-intro/#tile)扫 matmul 的 block 大小时，最快的配置每个 thread 存 64 个累加器，就是这一层的 tiling。

---

这七种方法里，fusion、recomputation 和 tiling 省的是显存读写的次数，低精度省的是每次的字节，coalescing 和 bank conflict 让同样的读写更快，divergence 让 warp 少做无用功。[下一篇](/blog/gpu-training-analysis/)在 5090 上拆一步 Transformer 训练，会看到这些方法各自能省多少，以及哪里光靠它们不够。

---

## References {#references}

1. Stanford CS336. [Lecture 5: GPUs](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)。这一篇七种方法的顺序和第 1、3、4、5、7 节示意图的画法。
2. NVIDIA. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)。SIMT 和分支、global memory 的合并访问、shared memory 的 bank。
3. NVIDIA. [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。谓词执行和 `ld.global.v4` 等访存指令。
4. NVIDIA. [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)（白皮书）。第 2 节 5090 上 bf16、fp8、fp4 Tensor core 的峰值。
5. Open Compute Project. [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。MX 格式：32 个数一个 E8M0 的 scale。
6. NVIDIA. [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)。NVFP4：16 个数一个 E4M3 的 scale，外加 per-tensor 的 fp32 scale。
7. Mishra, Stosic, Layton, Micikevicius. [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027). arXiv:2506.08027, 2025. 所有张量用 E4M3，每个张量沿行、沿列各量化一份。
8. DeepSeek-AI. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). arXiv:2412.19437, 2024. fp8 训练的 1 × 128、128 × 128 scale，累加提升到 fp32，反向时 tile 的转置。
9. NVIDIA. [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149). arXiv:2509.25149, 2025.
10. Ma et al. [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764). arXiv:2402.17764, 2024. BitNet b1.58。
11. Tseng, Chee, Sun, Kuleshov, De Sa. [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396). ICML 2024. 推理时 2 到 4 bit 的权重量化。
12. Chillee (Horace He). [Min-cut optimal(*) recomputation (i.e. activation checkpointing) with AOTAutograd](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467)。第 4 节三层 sigmoid 的例子。
13. Chen, Xu, Zhang, Guestrin. [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174). arXiv:1604.06174, 2016. activation checkpoint。
14. Dao, Fu, Ermon, Rudra, Ré. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). NeurIPS 2022. 反向时重算 P。
15. [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)。fused softmax、matmul 的写法。
