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

一个 warp 的 32 个 thread 同一时刻执行同一条指令（[入门篇 1.3 节](/blog/gpu-triton-intro/#hw-model)）。如果遇到分支，warp 里一部分 thread 走 if、另一部分走 else，硬件只能把两边都执行一遍：执行 if 那段时，走 else 的 thread 被屏蔽掉，什么也不做，反过来也一样。这叫 control divergence，代价是两段的时间加起来。分支在不同 warp 之间不一样没关系，每个 warp 只走自己那条路。

所以分支条件最好在一个 warp 里一致，比如按 program、按行来分，而不是按单个元素。逐元素的条件可以改成计算：两边都算出来，再用 select 挑一个，没有跳转，也就没有 divergence。Triton 里 `tl.where(cond, a, b)` 就是这样，`tl.load` 的 `mask` 编出来是带谓词的指令：入门篇 GELU 的 PTX 里，`@%p1 ld.global.v4.b32` 前面的 `%p1` 就是谓词，越界的 thread 不读，但大家执行的是同一条指令。Triton 里的 `if`、`for` 只能用在整个 program 共有的标量上，所以写 Triton 时很少遇到这个问题；写 CUDA 时才要自己留意。

warp 内部的问题说完，下一个是每个数占多少位。

## 2 低精度 {#precision}

把 fp32 换成 bf16、fp16、fp8，一是每个数的字节少了，同样的带宽能搬更多数；二是能用上 Tensor core。5090 上 fp32 走 CUDA core，峰值 1.05e14 FLOPS；bf16 走 Tensor core，2.1e14 FLOPS，fp8 更高（[入门篇 Appendix A](/blog/gpu-triton-intro/#specs)）。累加一般还是用 fp32：bf16 只有 7 位尾数，长求和在 bf16 下会丢精度。

有两点容易踩。一是 Triton 的 `tl.dot` 遇到 fp32 输入默认走 tf32 的 Tensor core，结果和 fp32 矩阵乘差 0.1 左右，要精确的 fp32 得写 `input_precision="ieee"`（[入门篇 2.4 节](/blog/gpu-triton-intro/#triton-matmul)）。二是低精度省的主要是时间，显存不一定省得多：下一篇在 5090 上用 bf16 autocast 训练，前向快了 1.9–2.3 倍，显存只省了 18–21%，因为权重、梯度、Adam 状态还是 fp32，norm、softmax 这些也还在 fp32 下算（[下一篇 4.1 节](/blog/gpu-training-analysis/#bf16)）。

位宽省的是每个数的字节，下面两节省的是读写的次数。

## 3 Fusion {#fusion}

逐元素和归约类的 op 算得很少，时间几乎都花在读写显存上。几个这样的 op 接在一起时，每个 op 都把中间结果写回显存、下一个再读回来，fusion 把它们合成一个 kernel，中间结果留在 register 里，显存只读写一次。入门篇[表 2-1](/blog/gpu-triton-intro/#tab-2-1) 里，GELU 按公式用张量运算写，eager 要跑 8 个 kernel，194.7 µs；合成一个 Triton kernel 是 46.6 µs，快 4.2 倍。softmax 从 5 个 kernel 合成 1 个，快 2.6 倍。`torch.compile` 自动做的就是这件事，它在 GPU 上生成的也是 Triton kernel。

fusion 对一个 kernel 本来就受算力限制的 op 帮助不大：matmul 后面接的 ReLU 单独跑也只要 4 µs，矩阵乘本身要 150 µs 左右。下一篇把 RMSNorm 的 5 个 op 合成一个 kernel，前向、反向都少读写一遍中间结果（[下一篇 3.1 节](/blog/gpu-training-analysis/#rmsnorm)）。

合成一个 kernel 以后，中间结果不用写回显存了，但反向要用它们的时候怎么办，就是下一节的事。

## 4 Recomputation {#recompute}

训练时反向要用前向的中间结果。存下来要占显存，读回来要花带宽；如果重新算一遍比读回来便宜，就不存，反向时重算。这是拿算力换显存和带宽。

它有两个尺度。小的在一个 op 里：融合后的 RMSNorm 不存 $\hat x$，反向时用存下来的 x 和每行一个数的 r 重算。大的跨好几层，叫 activation checkpoint：前向只存每一段的入口，反向走到这一段时把前向重跑一遍。下一篇在 5090 上测过，每层一个 checkpoint 时显存从 15.0 GiB 降到 7.8 GiB，step 慢 28–33%，正好是一步从 3 次前向的量变成 4 次（[下一篇 4.2 节](/blog/gpu-training-analysis/#checkpoint)）。FlashAttention 的反向也是重算：不存 seq × seq 的 P，反向时用存下来的每行 logsumexp 重算出来（[FlashAttention 1–4](/blog/flashattention-1-to-4/)）。

前面几节都在少搬数据，后面三节看同样的数据怎么搬更快。

## 5 Memory coalescing {#coalescing}

一个 warp 发一条 load 时，硬件把 32 个 thread 的地址合起来，按 32 B 的 sector 去显存取，128 B 是一条 cache line。相邻的 thread 读相邻的地址，32 个 thread 的请求就落在几条连续的 cache line 里，取上来的字节都用得上，这叫 coalescing。相邻的 thread 隔得很远，比如按列读一个行优先的矩阵，每个 thread 都要单独取一个 sector，大部分字节是白取的。

每个 thread 一次读几个相邻的数也有关系。入门篇的 GELU 里，n 是 16 的倍数时，Triton 知道地址按 16 B 对齐，一个 thread 用一条 `ld.global.v4` 读 4 个 fp32；n 不是 16 的倍数时，就变成 4 条 4 B 的 `ld.global`（[入门篇 2.1 节 notes](/blog/gpu-triton-intro/#triton-gelu)），条数多了 4 倍。写 Triton 时，让 `tl.arange` 沿着内存里连续的那一维，block 的这一维取得够长，编译器就会排成合并的读写。

显存这边讲完，shared memory 也有自己的规矩。

## 6 Bank conflict {#bank}

shared memory 分成 32 个 bank，每个 bank 宽 4 B，地址按 4 B 轮流分到各个 bank 上（[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)）。一个 warp 读 shared memory 时，32 个 thread 落在 32 个不同的 bank 上，一拍就能读完；如果有几个 thread 落在同一个 bank 的不同地址上，这几个请求只能排队，一拍拆成几拍，这叫 bank conflict。几个 thread 读同一个地址不算冲突，会广播。

最常见的例子是按列读一个 32 × 32 的 fp32 矩阵：一列的 32 个数隔 128 B，全落在同一个 bank 上，要排 32 拍。常见的解法有两种：每行多留一个数（padding），让一列的数错开到不同的 bank；或者按一定的规律打乱存放的位置（swizzle）。写 Triton 时这件事由编译器决定，shared memory 的布局写在 TTGIR 里：入门篇 2.4 节那个 fp32 的 matmul，A、B 的 slot 用的是 `swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1}>`，也就是没有打乱；走 Tensor core 的 `tl.dot` 编译器会换成别的布局。

最后一种方法把前面几件事都用上：让一个数从显存读进来以后，在片上被用很多次。

## 7 Tiling {#tiling}

matmul 的 $C = A B$ 里，算一个 $C_{ij}$ 要用 A 的一行和 B 的一列，一共 2K 个数，naive 和 tiling 的区别在这些数从哪读（[表 7-1](#tab-7-1)）：

| | 每个输出从显存读 | 其余从片上读 | K = 2048、T = 64 时从显存读 | 整个 kernel 从显存读 |
|:--|--:|--:|--:|--:|
| naive（不切 block） | 2K 个数 | 0 | 4096 个数 | 32 GB |
| tiling（切成 T × T 的 block） | 2K / T 个数 | 2K − 2K / T | 64 个数 | 512 MB |
{#tab-7-1 caption="**表 7-1** naive 和 tiling：每个输出的 2K 个数从哪读" note="形状是 <a href=\"/blog/gpu-triton-intro/#triton-matmul\">入门篇 2.4 节</a>的 A 4096 × 2048、B 2048 × 512，fp32。片上指 shared memory，以及 thread 自己 register 里留着的数。512 MB 和入门篇 2.4 节 notes 里从 PTX 数出来的一致。"}

1/T 是这样来的：一个 program 从显存读进 A 的 T × K 个数和 B 的 K × T 个数，算出 C 的 T × T 个输出，平均每个输出读 2TK / T² = 2K / T 个（[图 7-1](#fig-o-2)）。一个数从显存读进来一次，在片上被 T 个输出用到。BM、BN 不相等时是 K / BN + K / BM。

block 越大读得越少，代价是一个 program 要占更多 register 和 shared memory，怎么取见入门篇 [3.5 节](/blog/gpu-triton-intro/#tile)。BM、BN、BK 都得是 2 的幂，这是 `tl.arange` 的要求；`tl.dot` 还要求 BK 至少 16。

{{fig-o-2}}

tiling 不只在 shared memory 这一层。每个 thread 从 shared memory 读进 A、B 的一小段，再和自己 register 里的一小块累加器一起算，一个数在 register 里也被用好几次；入门篇 [3.5 节](/blog/gpu-triton-intro/#tile)扫 matmul 的 block 大小时，最快的配置每个 thread 存 64 个累加器，就是这一层的 tiling。

---

这七种方法里，fusion、recomputation 和 tiling 省的是显存读写的次数，低精度省的是每次的字节，coalescing 和 bank conflict 让同样的读写更快，divergence 让 warp 少做无用功。[下一篇](/blog/gpu-training-analysis/)在 5090 上拆一步 Transformer 训练，会看到这些方法各自能省多少，以及哪里光靠它们不够。

---

## References {#references}

1. NVIDIA. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)。SIMT 和分支、global memory 的合并访问、shared memory 的 bank。
2. NVIDIA. [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。谓词执行和 `ld.global.v4` 等访存指令。
3. [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)。fused softmax、matmul 的写法。
4. Dao, Fu, Ermon, Rudra, Ré. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). NeurIPS 2022. 第 4 节的重算。
5. Chen, Xu, Zhang, Guestrin. [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174). arXiv:1604.06174, 2016. 第 4 节的 activation checkpoint。
