---
title: "GPU 与 Triton 入门：从存储层级到第一个 kernel"
date: 2026-10-03
draft: false
math: true
description: "GPU 的存储层级、SM 结构和 CUDA 执行层级；再用 GELU、softmax、row sum、matmul 四个例子入门 Triton，Appendix 对照 TPU。GPU training optimization 系列的第一篇。"
tags: ["GPU", "Triton", "CUDA", "TPU", "LLM", "AI"]
categories: ["AI笔记"]
series: ["GPU training optimization"]
series_order: 1
---

这个系列讲怎么让 Transformer 训练在 GPU 上跑得更快。这一篇先打基础：[第 1 节](#hardware)介绍 GPU 的硬件，包括数据存在哪几层、一个 SM 里有什么、CUDA 的执行层级怎么对应到硬件；[第 2 节](#triton)用四个例子介绍 Triton，后面两篇的 kernel 都用它写。TPU 和 GPU 的对照放在 [Appendix A](#tpu)，program 怎么排到 SM 上跑（驻留上限、warp 调度、wave）放在 [Appendix B](#sched)，几代 NVIDIA GPU 的规格对比放在 [Appendix C](#specs)。

下一篇[谁偷走了 5090 的算力和显存](/blog/gpu-training-analysis/)用这些知识在 RTX 5090 上拆解一步训练的时间和显存，第三篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/) 讲 attention 的 IO-aware kernel。

> 环境：RTX 5090 32 GB，torch 2.11.0+cu130，Triton 3.6.0。文中的 KB、MB、GB 都按 1024 进位。

---

## 1 GPU：越靠近计算单元越快 {#hardware}

做性能分析时最常用的两个数是峰值算力和显存带宽，一个 kernel 受哪个限制，取决于它要搬多少数据、在哪一层读写。这一节先看数据存在哪几层（[1.1](#hw-memory)）和 SM 里有什么（[1.2](#hw-sm)），再看 CUDA 的 thread、warp、block 怎么落到这些硬件上（[1.3](#hw-model)）。

### 1.1 存储层级：越近越快，也越小 {#hw-memory}

处理器的频率早就不怎么涨了，算力的增长主要来自并行：更多的 SM、更宽的 Tensor core。显存带宽涨得慢得多：过去 20 年，峰值算力涨了约 6 万倍（每两年 3.0 倍），DRAM 带宽只涨了约 100 倍（每两年 1.6 倍），芯片之间的互联带宽约 30 倍（[图 1-1](#fig-memwall)）。带宽跟不上，越来越多的 op 受显存带宽限制（memory-bound），只能少搬：让读进来的数据留在离计算单元近的地方，多用几次。

<figure id="fig-memwall" class="fg-fig">
<img src="memory-wall.webp" alt="1997 到 2023 年峰值算力、DRAM 带宽和互联带宽的增长：算力每两年 3.0 倍，DRAM 带宽每两年 1.6 倍，互联带宽每两年 1.4 倍" loading="lazy" style="width: 100%; height: auto; border-radius: 8px; background: #fff;">
<figcaption><strong>图 1-1</strong> 峰值算力与 DRAM、互联带宽 20 年来的增长，纵轴是相对 1997 年的倍数（对数轴）。图来自 Gholami 等人的 <a href="https://arxiv.org/abs/2403.14123">AI and Memory Wall</a>（IEEE Micro，2024）。</figcaption>
</figure>

数据离计算单元越近，读写越快，容量也越小（[图 1-2](#fig-h-1)）。每个 SM 里有 register 和 L1 / shared memory，所有 SM 共享芯片上的 L2，显存在芯片外面。L1 和 L2 由硬件当作 cache 自动管理，kernel 能自己安排的只有 shared memory 和 register。所以 kernel 优化的套路都一样：把要反复用的数据读进 shared memory 或 register，在片上多算几次，再写回显存。[第 2 节](#triton)的 block 矩阵乘、[下一篇](/blog/gpu-training-analysis/)的算子融合和 [FlashAttention](/blog/flashattention-1-to-4/) 都是这个思路。

{{fig-h-1}}

5090 是消费级显卡，显存用的是 GDDR7（32 GB，512-bit，28 Gbps，带宽 1.79e12 B/s），不是数据中心卡上的 HBM，也没有 ECC 和 NVLink。

register 和 L1 / shared memory 都在 SM 里面，下面看 SM 本身的结构。

### 1.2 一个 SM 里有什么 {#hw-sm}

每个 SM 分成 4 个 SMSP（SM sub-partition），每个 SMSP 有自己的 warp 调度器、register、32 个 CUDA core（做 FP32 和 INT32 运算）和一个 Tensor core，4 个 SMSP 共享 L1 / shared memory（[图 1-3](#fig-h-2)）。L1 和 shared memory 是同一块 128 KB 的片上存储，按几种固定的比例切开用，shared memory 最多切到 100 KB，这时 L1 cache 只剩 28 KB。NVIDIA 在 [GA102（RTX 30 系列）白皮书](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)里列出了这几种切法，5090 能选的 shared memory 档位和它一样：0、8、16、32、64、100 KB。

{{fig-h-2}}

[图 1-4](#fig-h-9) 是一个 SMSP 的结构，放在这里作参考。后文要用到其中两件事。

第一件是 CUDA core 和 Tensor core 的分工。CUDA core 一次算一个数：一条 FMA 指令里 warp 的 32 个 thread 各用一个 CUDA core，各做一次乘加（2 FLOPs）。5090 一共 170 × 128 个 CUDA core，fp32 峰值 1.05e14 FLOPS。Tensor core 一次算一个小矩阵乘：一条指令由整个 warp 一起发出，算一小块 A × B 再加到累加器上。它只吃 bf16、fp16、tf32、fp8 这类低精度输入，bf16 的峰值是 2.1e14 FLOPS（fp32 累加），fp8 更高。第 2 节的 `tl.dot` 默认走 Tensor core，加减乘除在 CUDA core 上算，exp 这类函数在 SFU 上算。

第二件是 thread 和 register 的关系。每个 SMSP 的 register file 有 16384 个 32-bit register（64 KB），一个 SM 共 65536 个。每个 thread 用几个 register 在编译时就定了，最多 255 个。一个 warp 固定在一个 SMSP 上，block 放到 SM 上时，它的每个 thread 从所在 SMSP 的 register file 里分到这么多个 register，只归自己读写，一直占到 block 跑完。所以每个 thread 用的 register 越多，一个 SM 能同时驻留的 thread 就越少：每个 thread 用满 255 个时，一个 SM 只放得下 256 个 thread（8 个 warp），是上限 1536 个的六分之一；第 2 节的 softmax 每个 thread 用 55 个，一个 SM 只放得下 9 个 program（[图 B-1](#fig-t-8)）。

{{fig-h-9}}

[第 2 节](#triton) kernel 图里的「register」就是这块 register file，「计算单元」指其中的 CUDA core 和 Tensor core。几代 GPU 的规格对比见 [Appendix C](#specs)。

这些是硬件提供的资源。register 按 thread 分配，shared memory 按 block 分配，要知道一个 kernel 怎么用这些资源，得先看 CUDA 怎么把 thread 组织起来。

### 1.3 编程模型：thread、warp、block、grid {#hw-model}

CUDA 的执行模型叫 SIMT（single instruction, multiple threads）：一个 warp 的 32 个 thread 同一时刻执行同一条指令。thread 往上分四层：thread、warp（32 个 thread）、block（也叫 CTA，若干个 warp）、grid（一次 kernel 启动的全部 block），每层跑在哪个硬件上见[图 1-5](#fig-h-5)。

{{fig-h-5}}

block 放到 SM 上以后一直驻留到跑完。一个 SM 同时驻留多个 block，每个 SMSP 的调度器每拍从驻留的 warp 里挑一个就绪的发射；一个 warp 等显存数据要几百个周期，这段时间里调度器就发别的 warp，所以同时驻留的 warp 越多，访存延迟越容易藏住。一个 SM 能放几个 block、block 多于全卡的名额时怎么分批跑，见 [Appendix B](#sched)。

block 之间不能直接共享数据，只能通过显存交换，而这正是最慢的一层。所以要尽量把会重复读的数据放进同一个 block 里处理，这就是 tiling。在 CUDA 里做 tiling 要自己管 shared memory 和 thread 同步，[第 2 节](#triton)的 Triton 把这部分交给编译器。

---

## 2 Triton：按 block 写 kernel {#triton}

Triton 按 block 写 kernel，正好对应 [1.3 节](#hw-model)说的 tiling。CUDA 要写清楚每个 thread 做什么，shared memory 和同步都要自己管；Triton 只写一个 block 做什么：把它负责的数据读进来，在片上算完，再写回显存，block 内怎么分给 thread、要不要经过 shared memory 由编译器决定。Triton 把一个 block 叫作一个 program（[表 2-1](#tab-2-1)）。

| CUDA | Triton | 写法 | 能否直接控制 |
|:--|:--|:--|:--|
| thread | 不暴露 | 写不到 `threadIdx` | 不能 |
| warp | 只给数量 | 启动时传 `num_warps=N`，默认 4 | 只能调数量 |
| block（CTA） | program | `triton.jit` 函数体就是一个 program 的代码；`tl.program_id(axis)` ≈ `blockIdx`，`tl.num_programs(axis)` ≈ `gridDim` | 主要的编程层 |
| grid | grid | `kernel[grid](...)`，例如 `grid = (triton.cdiv(n, BLOCK),)` | 自己定 |
| shared memory | 由编译器分配 | 没有对应语句 | 不能 |
{#tab-2-1 caption="**表 2-1** CUDA 概念在 Triton 里的对应"}

下文 program 就是 CUDA 的 block；单说 block 时指一块数据，比如 A 的 block、C 的 block；代码里的 `BLOCK`、`TILE` 是一段数据的长度。

下面四个例子由浅入深。grid 怎么切，看一个输出依赖哪些输入；program 里要不要循环，看这些输入能不能一次装进 register，装不下就要一个 block 一个 block 地读，用一个累加器（accumulator）存中间结果：

| op | 类型 | 一个输出依赖 | grid 怎么切 | program 里 |
|:--|:--|:--|:--|:--|
| GELU | 逐元素 | 同位置的一个输入 | 展平成一维，按 BLOCK 切段 | 不循环 |
| softmax | 按行归约 | 整行（max 和 sum） | 一行一个 program | 不循环，整行一次读进来 |
| row sum | 按行归约 | 整行（sum） | 一行一个 program | 沿行循环，累加器存部分和 |
| matmul | block 乘加 | A 的一行、B 的一列 | 二维，C 的一个 block 一个 program | 沿 K 循环，累加器存 C 的一个 block |
{#tab-2-2 caption="**表 2-2** 四个例子的 op 类型"}

代码都在 RTX 5090 上和 PyTorch 对照过。每个例子配一张数据流图：左边是 grid，右边从上到下是计算单元、register、shared memory 和显存（芯片外的 GDDR7）。`tl.load` 和 `tl.store` 在显存和 register 或 shared memory 之间直接搬数据，计算单元只读写 register。中间经过的 L2 和 L1 cache 由硬件自动管理，图里不画。计算单元那一行从左往右是执行顺序；register 和 shared memory 里的一块存储只画一次，横向跨过它存在的那段时间，每一步在自己那一列读写它，块上标的是它的大小。

### 2.1 GELU：逐元素 {#triton-gelu}

GELU 对每个元素单独算，$\Phi$ 是标准正态分布的累积分布函数，实际用 tanh 近似：

<p align="center">$y_i = \mathrm{GELU}(x_i) = x_i\,\Phi(x_i) \approx \tfrac{1}{2}\,x_i\left(1 + \tanh\left(\sqrt{2/\pi}\,(x_i + 0.044715\,x_i^3)\right)\right)$</p>

{{fig-t-1}}

逐元素 op 最简单，每个元素各算各的，program 之间不需要配合。把 n 个元素切成每段 BLOCK 个，一段交给一个 program（[图 2-1](#fig-t-1)）。kernel 里先用 `tl.program_id` 算出自己负责的下标，读进来，算完写回：

```python {role="program" tag="处理第 pid 段的 BLOCK 个元素"}
{{code:triton_tutorial.py:gelu_kernel}}
```

启动时只要定 grid，也就是 program 的个数，它和 SM 的个数无关。

```python {role="grid" tag="一段一个 program"}
{{code:triton_tutorial.py:gelu_launch}}
```

program 编译后就是 CUDA 的一个 block。8192 个 program 远多于 170 个 SM，GPU 会把它们一批一批地排到 SM 上，每个 SM 同时跑 12 个（为什么是 12 个见 [图 B-1](#fig-t-8)，怎么分批见 [Appendix B.3](#sched-wave)）。

n = 8388608 时 Triton 版 45.2 µs，读写共 64 MB，有效带宽 1.49e12 B/s，是显存带宽的 83%，和 PyTorch 的 `F.gelu(x, approximate="tanh")`（45.9 µs）持平。

<details class="fold doc aside">
<summary>GELU 编译成的 PTX</summary>

Triton 把 kernel 编译成 PTX，PTX 描述的是单个 thread 的指令。n = 8388608 时的节选：

```
mov.u32          %r17, %ctaid.x;                                // program 编号，即 blockIdx.x
mov.u32          %r20, %tid.x;                                  // thread 编号
%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];     // 一条读 4 个 fp32（16 B），共 2 条
ex2.approx.f32   %r66, %r65;                                    // tl.exp 编成以 2 为底的指数
div.full.f32     %r90, %r89, %r81;                              // 2 / (e + 1)
%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r9, %r10, %r11, %r12 };  // 一条写 4 个，共 2 条
```

一个 program 处理 1024 个元素、128 个 thread，每个 thread 分到 8 个。n 是 16 的倍数，Triton 知道地址按 16 B 对齐，把 8 次 4 B 的读合并成 2 条 `ld.global.v4`，写也一样；n 不是 16 的倍数时（比如 n + 7），PTX 里就是 8 条 4 B 的 `ld.global`。

</details>

GELU 的每个元素互不依赖，怎么切都可以。softmax 的每个元素都要用到整行的最大值和总和，切分就得按行来。

### 2.2 Softmax：按行归约，一行一个 program {#triton-softmax}

softmax 对 `[M, N]` 的输入按行算，先减去每行的最大值 $m_i$ 防止 exp 溢出：

<p align="center">$m_i = \max_k x_{ik}, \qquad y_{ij} = \dfrac{e^{x_{ij} - m_i}}{\sum_{k} e^{x_{ik} - m_i}}$</p>

$y_{ij}$ 要用到第 $i$ 行全部 N 个数。eager 下 softmax 要 5 个 kernel（和[下一篇](/blog/gpu-training-analysis/#attention)里 attention 的 softmax 一样），一共读 5MN + 2M 个数，写 3MN + 2M 个：

```python
{{code:triton_tutorial.py:softmax_naive}}
```

{{fig-t-2}}

如果一个 program 处理一整行（[图 2-2](#fig-t-2)），max、exp、求和、除都可以在 register 里做完，只读 MN、写 MN，读写量是原来的 1/4：

```python {role="program" tag="处理第 row 行"}
{{code:triton_tutorial.py:softmax_kernel}}
```

启动时一行一个 program：

```python {role="grid" tag="一行一个 program"}
{{code:triton_tutorial.py:softmax_launch}}
```

4096 × 4096 的 fp32 输入，eager 版 246.9 µs，Triton 版 92.4 µs，快 2.7 倍，和 `torch.softmax`（92.3 µs）持平，有效带宽 1.45e12 B/s，是显存带宽的 81%。

[图 2-2](#fig-t-2) 里的 warp max、warp sum 是 shared memory 在这个 kernel 里唯一的用处：`tl.max`、`tl.sum` 先在每个 warp 内用 shuffle 归约，4 个 warp 的部分结果再经 shared memory 交换，一共 16 B。把一整个 block 的数据放进 shared memory 什么时候值得，[2.4 节](#triton-matmul)再讲。

这个写法要求 BLOCK ≥ N，也就是一行能一次装进一个 program。行再长，program 就只能沿行一个 block 一个 block 地读，读过的 block 要留下一个中间结果。下一节先看最简单的按行归约：求和。

### 2.3 Row sum：按行归约，按 block 累加 {#triton-rowsum}

row sum 对 `[M, N]` 的输入每行求和：

<p align="center">$s_i = \sum_{j=1}^{N} x_{ij} = \sum_{T} \ \sum_{j \in \mathcal{J}_T} x_{ij}$</p>

$\mathcal{J}_T$ 是第 $T$ 个 block 的 TILE 个列下标，代码里 TILE 取 1024。加法满足结合律，先把每个 block 加起来再相加，和一次加完是同一个数（浮点数下只差舍入误差），所以 program 可以沿行一个 block 一个 block 地读，每读一个 block 就加到累加器上（[图 2-3](#fig-t-3)）。

{{fig-t-3}}

累加器 `acc` 是一个长度为 TILE 的向量，不是一个数：循环里每个 thread 只加自己负责的几个位置，thread 之间不需要通信；循环结束后，`tl.sum` 再做一次跨 thread 归约，把 TILE 个部分和加成一个数。

```python {role="program" tag="沿第 row 行循环累加"}
{{code:triton_tutorial.py:row_sum_kernel}}
```

grid 和 softmax 一样，一行一个 program：

```python {role="grid" tag="一行一个 program"}
{{code:triton_tutorial.py:row_sum_launch}}
```

4096 × 4096 时 54.6 µs，有效带宽 1.23e12 B/s，是显存带宽的 69%，比 `x.sum(1)`（50.8 µs）慢 7%。

softmax 不能直接这样按 block 读。$e^{x_{ij} - m_i}$ 要先知道整行的最大值 $m_i$，读完整行才知道。按 block 读时，累加器里要同时存目前的最大值和部分和，最大值变大时把已有的部分和乘上 $e^{m_{\text{old}} - m_{\text{new}}}$ 修正，这就是 online softmax，[FlashAttention](/blog/flashattention-1-to-4/) 那篇会讲。

前面三个例子里每个数只读一次，切 block 只是为了把工作分给不同的 program。矩阵乘不同：A 的每个元素要和 B 的一整列相乘，切 block 是为了让读进片上的数据被多次使用。它也要沿 K 循环，累加器是 C 的一个 block。

### 2.4 Matmul：block 乘加，顺手融合 ReLU {#triton-matmul}

`[M, K] × [K, N]` 的矩阵乘再接 ReLU：

<p align="center">$C_{ij} = \max\left(0,\ \sum_{k=1}^{K} A_{ik} B_{kj}\right)$</p>

把 M、N、K 分别按 BM、BN、BK 切成 block，记 $A_{IT}$ 是 A 的第 $I$ 个 row block 和第 $T$ 个 column block 交叉处的 block（BM × BK），$B_{TJ}$、$C_{IJ}$ 同理。$C_{IJ}$ 只用到 A 的第 $I$ 个 row block 和 B 的第 $J$ 个 column block，不同的 $C_{IJ}$ 可以交给不同的 program。对 $k$ 的求和再按 BK 分段，$\mathcal{K}_T$ 是第 $T$ 段的 BK 个下标：

<p align="center">$\sum_{k=1}^{K} A_{ik} B_{kj} = \sum_{T=1}^{K/\mathrm{BK}} \ \sum_{k \in \mathcal{K}_T} A_{ik} B_{kj}, \qquad C_{IJ} = \max\left(0,\ \sum_{T} A_{IT}\, B_{TJ}\right)$</p>

和 row sum 一样，分段加完再相加等于一次加完，所以每轮只读 $A_{IT}$、$B_{TJ}$ 两个 block，乘完累加，最后就是完整的 $C_{IJ}$。ReLU 逐元素作用在累加完的结果上，放在最后做。

{{fig-t-4}}

一个 program 负责一个 $C_{IJ}$（[图 2-4](#fig-t-4)），沿 K 循环，第 $T$ 轮读 $A_{IT}$ 和 $B_{TJ}$，用 `tl.dot` 乘加到累加器上；写回前在 register 里做 ReLU，激活函数不用再单独一个 kernel 读写一遍 C：

```python {role="program" tag="算 C 的第 (pid_m, pid_n) 个 block"}
{{code:triton_tutorial.py:matmul_kernel}}
```

grid 是二维的：

```python {role="grid" tag="C 的每个 block 一个 program，二维"}
{{code:triton_tutorial.py:matmul_launch}}
```

<div class="nv-key">

**shared memory 什么时候值得用**：满足下面任一条。

- **复用**：同一份数据要被读很多次。matmul 里 A 的一个 block 要和 B 的 block 的 64 列都乘一遍；softmax 和 row sum 里每个数只被一个 thread 读一次。
- **交换**：thread 之间要交换大量数据。matmul 里 `tl.dot` 要把 A、B 的 block 重新分给不同的 thread；softmax 和 row sum 只交换 4 个 warp 的部分结果（16 B）。

所以 matmul 把 A、B 的 block 放在 shared memory 里，每个 program 32 KB；softmax 和 row sum 两条都不满足，把整个 block 放进 shared memory 只会多搬一趟。代价是 shared memory 用得越多，一个 SM 能同时放的 program 越少：每个 SM 的上限是 100 KB，matmul 每个 program 占 32 KB，再加每个 program 保留的 1 KB，只放得下 3 个（3 × 33 = 99 KB，见 [图 B-1](#fig-t-8)）。

</div>

A 4096 × 2048、B 2048 × 512 时 Triton 版 203.3 µs，4.23e13 FLOPS，是 fp32 标称峰值 1.05e14 FLOPS（170 个 SM × 128 个 CUDA core × 2 FLOPs × 2.41e9 Hz）的 40%；cuBLAS（`torch.relu(A  B)`，关掉 tf32）是 160.5 µs。差距主要来自 tail effect：每个 SM 只放 3 个 program，512 个 program 要分两批跑，第二批只有 2 个，它们跑的时候另外 168 个 SM 都空着（见 [Appendix B.3](#sched-wave)）。

<details class="fold doc aside">
<summary>matmul 的 load 藏住了吗：PTX 计数和 Nsight Compute 实测</summary>

Triton 默认 `num_stages=3`，编译器把这个循环改写成流水线：A、B 各开两个槽，算第 $T$ 轮时，第 $T+1$ 轮的 block 已经在搬了。从编译出的 TTGIR 和 PTX 里能数清楚 A、B 的 block 怎么进出 shared memory。循环前先发出第 0、1 轮的 cp.async；第 $T$ 轮先等第 $T$ 轮的数据到齐，用 ld.shared 读进 register 做乘加，读完马上往刚空出来的槽发第 $T+2$ 轮。A、B 各搬 64 次，每次 8 KB，一个 program 一共 1 MB；512 个 program 加起来 512 MB，而 A、B 本身只有 36 MB，所以 L2 命中率有 91%，大部分读不用到显存。每轮从 shared memory 读进 register 的是 192 KB，同一个数平均被读 12 次，这就是前面说的复用。写回之前 acc 还要经 shared memory 换一次排布：`tl.dot` 输出时每个 thread 拿两个 4 × 4 的小方块，编译器给 `tl.store` 选的排布是每个 thread 拿 8 段、每段一行里连续的 4 个数，同一个数换到了别的 thread 甚至别的 warp 手里，而 register 是每个 thread 私有的，只能先写进 shared memory、同步一次再按新排布读出来。两种排布写显存时都是一个 warp 一次写两整行，并不是为了合并写入；这一步是两种排布对不上留下的。代码里加了 `input_precision="ieee"`，所以 `tl.dot` 编译成 CUDA core 上的 `fma.rn.f32x2`；不加的话 fp32 输入默认走 tf32 的 Tensor core，和 fp32 矩阵乘的结果差 0.1 左右。

预取藏住了多少 load，用 Nsight Compute 测（[图 2-4a](#fig-t-6)）。ncu 给的是一个 warp 每两条指令之间平均隔几拍、这几拍里在等什么，乘上每轮的指令数（约 1220 条），就是每轮的拍数。不预取（`num_stages=1`）时每轮 5601 拍，其中 852 拍在等 L2 或显存的数据，等到了才能算，一轮接一轮。预取以后每轮 4901 拍，等数据只剩 74 拍：block $k$ 的 load 在两轮之前就发出了，约 852 拍就到，到用的时候早已在 shared memory 里。load 只占一轮的六分之一，提前一轮发其实就够。

每轮少了 13%，整个 kernel 只快 6%（168.2 µs 到 158.3 µs）：一个 warp 等数据的时候，同一个 SMSP 上另外两个 warp 本来就能发射，这段等待原本就被藏住了一部分。剩下的时间主要花在等 ld.shared 的结果和轮不到发射上：FMA 管线只忙了 66%，shared memory 读只用了 55%，两边都没满。每个 SMSP 上只有 3 个 warp，一个 warp 等 ld.shared 时，没有足够多的别的 warp 补上。

{{fig-t-6}}

</details>

第 2 节的四个例子是同一个模式：program 把一个 block 的数据读进 register，在片上做完能做的计算，再写回显存。[下一篇](/blog/gpu-training-analysis/)在 5090 上拆解一步训练的时间和显存，用的就是这个模式：把 RMSNorm 的几个 kernel 融合成一个，少读写一遍中间结果。

---

## Appendix A TPU：更大的矩阵单元，更简单的控制 {#tpu}

正文的 GPU 靠 kernel 自己切 block 来复用数据。TPU 面对同样的带宽问题，思路是把控制逻辑做轻，把矩阵乘单元做大。

[图 A-1](#fig-h-3) 是 TPU v5p 的一个 TensorCore：Scalar unit 取指令、发给 VPU 和 MXU，整个核只有这一路控制，没有 warp 和调度器之间的切换；VPU 做逐元素运算和归约；矩阵乘交给 4 个 MXU，每个是 128 × 128 的 systolic array。片上存储只有 VMEM 和 VREG 两层，VMEM 是编译器管理的 scratchpad，不是硬件 cache，数据什么时候从 HBM 搬进来、搬多少，都由编译器（XLA）提前排好。

{{fig-h-3}}

MXU 和 GPU 最大的不同在乘加单元怎么拿到数据（[图 A-2](#fig-h-4)）。GPU 上每个 CUDA core 做一次乘加，要从 register file 读两个数、写回一个；一个数被用很多次，得靠 kernel 自己切 block，先搬进 shared memory 和 register 再反复读，[2.4 节](#triton-matmul)的 matmul 就是这样写的。systolic array 把复用做进了硬件：权重先装进 128 × 128 个格子，固定不动；输入 x 从左边进入，每拍向右传一格，经过的每一格把它乘上自己的权重，加到从上面传下来的部分和上，再把部分和往下传。一个 x 从 VMEM 读一次，在一行里被 128 个格子用到；稳定以后每拍从左边进 128 个数、从底部出 128 个结果，阵列里同时做 16384 次乘加，中间结果都不经过 register file。

{{fig-h-4}}

代价是形状要规整。矩阵的维度要是 128 的倍数才能填满阵列，小矩阵和不规则的计算会浪费格子；数据流由编译器提前排好，也不能像 GPU 那样靠切换 warp 应付不确定的访存延迟。多卡之间怎么互联是另一个大区别，超出了本文的范围。两边术语的对应见[表 A-1](#tab-a-1)。

| GPU | TPU | 作用 | RTX 5090 | TPU v5p |
|:--|:--|:--|--:|--:|
| SM | TensorCore | 包含其他运算单元的核心 | 170 | 2 |
| warp 调度器 | VPU | SIMD 向量运算的调度 | 680 | 8 |
| CUDA core | VPU ALU | 普通的 SIMD 运算单元 | 21760 | — |
| L1 / shared memory | VMEM | 片上的快速存储 | 21.3 MB | 128 MB |
| register | VREG | vector register | 42.5 MB | 512 KB |
| Tensor core | MXU | 矩阵乘单元 | 680 | 8 |
| 显存 | HBM | 大容量主存 | 32 GB GDDR7 | 95 GB |
{#tab-a-1 caption="**表 A-1** GPU 与 TPU 的术语对照" note="RTX 5090 的片上存储是 170 个 SM 加起来的总量。TPU 一列参考 Google 的 <a href=\"https://jax-ml.github.io/scaling-book/gpus/\">How to Scale Your Model</a>。"}

---

## Appendix B 调度：program 怎么排到 SM 上跑 {#sched}

正文只用到一个结论：一个 SM 同时驻留多个 program，一个 warp 等数据时调度器换别的 warp 发射。这个 appendix 把它展开：一个 SM 能同时放几个 program（[B.1](#sched-limits)），驻留的 warp 怎么轮流发射（[B.2](#sched-warp)），program 多于全卡的名额时怎么分批跑（[B.3](#sched-wave)）。

### B.1 驻留上限 {#sched-limits}

program（也就是 CUDA block）放到 SM 上以后就一直驻留，直到它的 thread 全部跑完。一个 SM 能同时驻留多少，由四条硬件上限决定（[表 B-1](#tab-b-1)），哪条先用完就卡在哪条。warp 的上限分摊到每个 SMSP 的调度器；program 的上限记在整个 SM 上，因为一个 program 的 shared memory 和同步由 4 个 SMSP 共用。

| 资源 | RTX 5090 每个 SM 的上限 |
|:--|:--|
| 驻留 warp | 48 个（每个 SMSP 12 个），即 1536 个 thread |
| 驻留 program（CUDA block） | 24 个 |
| register | 65536 个 32-bit（256 KB），每个 thread 最多 255 个 |
| shared memory | 100 KB（128 KB 的 L1 / shared memory 里最多切出这么多），每个 program 另占 1 KB |
{#tab-b-1 caption="**表 B-1** RTX 5090 一个 SM 的驻留上限" note="用 <code>cuDeviceGetAttribute</code> 在 5090 上读出，和 CUDA Programming Guide 里 compute capability 12.0 的数一致。"}

第 2 节的图画的是一个 program 怎么读写数据。放到 SM 上时，一个 program 就是一个 CUDA block：4 个 warp、128 个 thread，每个 thread 带着自己的 register，整个 program 共用一块 shared memory。所以它在 SM 上要占四样东西，正好对应表 B-1 的四条上限：4 个 warp 名额，1 个 program 名额，128 × 每个 thread 的 register 数，shared memory 的用量再加 1 KB 保留。每条上限各自能放下几个 program，取最小的那个，就是一个 SM 实际放几个（[图 B-1](#fig-t-8)）。

{{fig-t-8}}

GELU 和 row sum 用的 register 少、几乎不用 shared memory，先用完的是 warp 名额：48 / 4 = 12。softmax 把一整行 4096 个数放在 register 里，每个 thread 55 个，按 56 分配，一个 program 占 56 × 128 = 7168 个，65536 / 7168 = 9.1，只放得下 9 个，比 warp 名额允许的 12 个少。matmul 的 `tl.dot` 要把 A、B 的 block 先放进 shared memory，一个 program 32 KB，加上保留的 1 KB 是 33 KB，100 KB 只放得下 3 个；按 register 能放 5 个，用不满。program 名额有 24 个，四个 kernel 都用不完。

四个 kernel 都用默认的 `num_warps=4`，一个 program 4 个 warp。这个数调大调小影响不大，只有 program 大到 48 个 warp 名额放不满时才明显变慢。

<details class="fold doc aside">
<summary>num_warps 调大调小会怎样</summary>

`num_warps` 是 Triton 启动 kernel 时的参数，决定一个 program 有几个 warp。program 个数不变时把它调大，同样的数据分给更多 thread：GELU 一个 program 1024 个数，`num_warps=1` 时 32 个 thread 每个处理 32 个数，用 80 个 register；`num_warps=32` 时 1024 个 thread 每个只处理 1 个数，用 10 个 register。

每个 SM 放的 program 也跟着变少：48 个 warp 名额只够放 48 / num_warps 个。`num_warps` 从 2 到 16，warp 名额都是满的，每个 SM 驻留的 warp 总数不变，GELU 的耗时也基本不变。变化出在两头。`num_warps=1` 时先用完的是 24 个 program 名额，每个 SM 只驻留 24 个 warp，但 GELU 没有变慢，因为每个 thread 要处理 32 个数，一次能发出好几个 load。`num_warps=32` 时一个 program 就占 32 个 warp，第二个放不下，48 个名额空着 16 个，每个 thread 又只处理 1 个数，GELU 明显变慢。

softmax 在 `num_warps=1` 时每个 SM 只驻留 12 个 warp，也没怎么变慢：每个 thread 处理这一行里的 128 个数，同时有很多个 load 在路上，几个 warp 就能让显存一直有请求在处理。藏延迟有两种办法，一是多驻留 warp，二是让每个 thread 多干活。

</details>

几代 GPU 的规格和驻留上限对比见 [Appendix C](#specs)。

驻留只说明 program 占住了名额，不说明它在跑。同一时刻真正在发射指令的，是每个 SMSP 的调度器挑中的那一个 warp。

### B.2 warp 调度：驻留不等于在跑 {#sched-warp}

一个 program 的 warp 按编号轮流分到 4 个 SMSP，每个 SMSP 有自己的 warp scheduler、register file 和 32 个 CUDA core（[图 1-4](#fig-h-9)）。调度器每拍从驻留的 warp 里挑一个就绪的，发出它的一条指令。一个 warp 等显存数据要几百个周期，这段时间里调度器改发另一个就绪的 warp。所有驻留 warp 的 register 一直留在 register file 里，换 warp 不用保存或恢复状态，所以同时驻留的 warp 越多，访存延迟越容易藏住（[图 B-2](#fig-h-6)）。

{{fig-h-6}}

名额是有限的。grid 里的 program 多于全卡的名额时，GPU 只能一批一批地往 SM 上排。

### B.3 wave 和 tail effect {#sched-wave}

GELU 每个 SM 放 12 个 program，全卡一次放得下 170 × 12 = 2040 个，8192 个要 5 批，第 5 批只有 32 个。[图 B-3](#fig-t-5) 上半部分的纵轴是某一时刻全卡已经开始、还没结束的 program 数：在横轴上任取一个时刻，曲线的高度就是那一刻 170 个 SM 上一共在跑几个 program。实测的时间线上批次之间却没有分界：一个 program 跑完，它的名额很快就换下一个进来，第 2 批在 2.6 µs 就开始了，第 1 批到 15.4 µs 才全部跑完。曲线也没有一直贴着 2040：从一个 program 发出最后一条 store，到同一个名额上的下一个 program 执行第一条指令，中间有一段空档，名额上没有 program 在执行指令，叫 bubble（图里浅红色的部分）。空档里可能是上一个 program 还在等 store 写完、名额还没回收，也可能是新 program 还没启动，打点分不出来。从第一个 program 结束到最后一个 program 开始，平均有 25% 的名额处在 bubble 里。同一批的 program 都在抢显存带宽，耗时从 4.2 µs 到 11.9 µs 不等（10% 和 90% 分位）。36.5 µs 时最后一个 program 开始，之后名额一个个空出来，在跑的 program 从 1400 个左右降到 0，这段约 10 µs，占整个 kernel 的 22%，叫 tail effect。打点的开销很小：打点版 46.9 µs，不打点 45.2 µs。

{{fig-t-5}}

program 少而每个都重时，尾巴更明显。matmul 每个 SM 只放 3 个 program，全卡一批 510 个，512 个 program 要跑 2 批，第 2 批只有 2 个（[图 B-4](#fig-t-7)）。第 1 批的 510 个 program 在开头一起开始，各跑 125 到 152 µs；全卡最先跑完的两个在 SM 68 和 SM 135 上，约 125 µs 结束，第 2 批的 2 个这时才开始。152 µs 以后全卡只剩这 2 个 program，另外 168 个 SM 空着，一直等到 200 µs。这 2 个 program 后半段独占一个 SM，只跑了 75 µs，比第 1 批的中位数 135 µs 快得多。

{{fig-t-7}}

A 换成 4032 × 2048 时，504 个 program 一批放下，同一个 kernel 只要 158.3 µs，和 cuBLAS 的 160.5 µs 相当。4096 比 4032 的计算量只多 1.6%，耗时多 28%。打点的开销不大：打点版 210.8 µs 和 162.8 µs，不打点 203.3 µs 和 158.3 µs。


---

## Appendix C 几代 NVIDIA GPU 的规格 {#specs}

正文和 Appendix B 都只用 RTX 5090 的数。[表 C-1](#tab-c-1) 把它和三代数据中心卡放在一起，硬件规格、标称峰值和驻留上限都在一张表里。

| | A100 | H100 | B200 | RTX 5090 |
|:--|--:|--:|--:|--:|
| SM 数 | 108 | 132 | 148 | 170 |
| 每 SM 的 CUDA core | 64 | 128 | 128 | 128 |
| 每 SM 的 Tensor core | 4 | 4 | 4 | 4 |
| fp32 峰值（CUDA core） | 1.95e13 FLOPS | 6.7e13 FLOPS | 7.5e13 FLOPS | 1.05e14 FLOPS |
| bf16 峰值（Tensor core，dense） | 3.12e14 FLOPS | 9.89e14 FLOPS | 2.25e15 FLOPS | 2.1e14 FLOPS |
| L2 | 40 MB | 50 MB | — | 96 MB |
| 显存 | 80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e | 32 GB GDDR7 |
| 显存带宽 | 2.0e12 B/s | 3.35e12 B/s | 8e12 B/s | 1.79e12 B/s |
| 每 SM 的 L1 + shared | 192 KB | 256 KB | 256 KB | 128 KB |
| 每 SM 的 register | 256 KB | 256 KB | 256 KB | 256 KB |
| 每 SMSP 最多驻留 warp | 16 | 16 | 16 | 12 |
| 每 SM 最多驻留 warp | 64 | 64 | 64 | 48 |
| 每 SM 最多驻留 block | 32 | 32 | 32 | 24 |
{#tab-c-1 caption="**表 C-1** 几代 NVIDIA GPU 的规格和驻留上限" note="A100 是 80GB SXM，H100 是 SXM，B200 按 HGX B200 的单卡算。峰值都是标称值，不含 structural sparsity，bf16 按 fp32 累加。规格来自 NVIDIA 各代架构白皮书和产品页，驻留上限来自 CUDA Programming Guide 的 compute capability 表（8.0、9.0、10.0、12.0）。B200 的 L2 没有找到可靠的公开数字，暂缺。"}

5090 的 SM 最多，fp32 峰值比 H100 还高；但每个 SM 能驻留的 warp 和 block 比数据中心卡少四分之一，L1 + shared 只有 H100、B200 的一半，bf16 的 Tensor core 峰值也只有 H100 的约五分之一。

---

## References {#references}

1. Gholami, Yao, Kim, Hooper, Mahoney, Keutzer. [AI and Memory Wall](https://arxiv.org/abs/2403.14123). IEEE Micro, 2024. 图 1-1。
2. NVIDIA. [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)（白皮书）。RTX 5090 的存储层级和 SM 结构，图 1-2、图 1-3。
3. NVIDIA. [NVIDIA Ampere GA102 GPU Architecture](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)（白皮书）。compute 模式下 L1 和 shared memory 的几种切法（28 KB L1 + 100 KB shared memory 等），第 1.2 节。
4. NVIDIA. [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)、[NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)、[NVIDIA Blackwell Architecture](https://resources.nvidia.com/en-us-blackwell-architecture)。表 C-1 的 A100、H100、B200 一列；峰值另参考 [A100](https://www.nvidia.com/en-us/data-center/a100/)、[H100](https://www.nvidia.com/en-us/data-center/h100/) 产品页和 [HGX](https://www.nvidia.com/en-us/data-center/hgx/) 平台页（B200 的单卡数按 8 卡的 dense 值除以 8）。
5. NVIDIA. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)。各 compute capability 每 SM 的 warp、block、register 和 shared memory 上限，表 B-1、表 C-1，CUDA 执行层级。
6. NVIDIA. [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。warp 停顿原因的定义，图 2-4a。
7. Austin et al. [How to Scale Your Model](https://jax-ml.github.io/scaling-book/gpus/), Google DeepMind, 2025. Appendix A 的 TPU v5p 规格、systolic array 的数据流和表 A-1。
8. Jouppi et al. [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760). ISCA 2017. 第一代 TPU 和它的 systolic array。
9. Kung. [Why Systolic Architectures?](https://doi.org/10.1109/MC.1982.1653825). IEEE Computer, 1982.
10. Tillet, Kung, Cox. [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://doi.org/10.1145/3315508.3329973). MAPL 2019.
11. [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)。第 2 节 softmax 和 matmul 例子的写法。
12. Hendrycks, Gimpel. [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415). arXiv:1606.08415, 2016. 第 2.1 节的 tanh 近似。
13. Milakov, Gimelshein. [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867). arXiv:1805.02867, 2018. 第 2.3 节提到的 online softmax。
