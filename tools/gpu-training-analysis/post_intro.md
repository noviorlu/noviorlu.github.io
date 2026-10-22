---
title: "GPU 与 Triton 入门：从存储层级到第一个 kernel"
date: 2026-10-03
draft: false
math: true
description: "GPU 的存储层级、SM 结构和 CUDA 执行层级；再用 GELU、softmax、row sum、matmul 四个例子入门 Triton，再看 kernel 在 SM 上怎么调度、怎么调参，Appendix 对照 TPU。GPU training optimization 系列的第一篇。"
tags: ["GPU", "Triton", "CUDA", "TPU", "LLM", "AI"]
categories: ["AI笔记"]
series: ["GPU training optimization"]
series_order: 1
---

这个系列讲怎么让 Transformer 训练在 GPU 上跑得更快。这一篇先打基础：[第 1 节](#hardware)介绍 GPU 的硬件，包括数据存在哪几层、一个 SM 里有什么、CUDA 的执行层级怎么对应到硬件、数据能在哪几层之间搬；[第 2 节](#triton)用四个例子介绍 Triton，后面两篇的 kernel 都用它写。[第 3 节](#tune)看写好的 kernel 在 SM 上怎么跑、怎么调：warp 怎么藏住访存延迟，一个 SM 放几个 program，program 怎么分批，`num_warps`、`num_stages` 和 block 的大小各改变了什么。几代 NVIDIA GPU 的规格对比放在 [Appendix A](#specs)，TPU 和 GPU 的对照放在 [Appendix B](#tpu)，TMA 和异步同步放在 [Appendix C](#tma)。

下一篇[单卡 GPU 优化方法](/blog/gpu-single-gpu-optimization/)把单卡上常用的优化方法放在一起，第三篇[谁偷走了 5090 的算力和显存](/blog/gpu-training-analysis/)用这些知识在 RTX 5090 上拆解一步训练的时间和显存，第四篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/) 讲 attention 的 IO-aware kernel。

> 环境：RTX 5090 32 GB，torch 2.11.0+cu130，Triton 3.6.0。文中的 KB、MB、GB 都按 1024 进位。

---

## 1 GPU 硬件 {#hardware}

做性能分析时最常用的两个数是峰值算力和显存带宽，一个 kernel 受哪个限制，取决于它要搬多少数据、在哪一层读写。这一节先看数据存在哪几层、能在哪几层之间搬（[1.1](#hw-memory)）和 SM 里有什么（[1.2](#hw-sm)），再看 CUDA 的 thread、warp、block 怎么落到这些硬件上（[1.3](#hw-model)）。

### 1.1 存储层级 {#hw-memory}

处理器的频率早就不怎么涨了，算力的增长主要来自并行：更多的 SM、更宽的 Tensor core。显存带宽涨得慢得多：过去 20 年，峰值算力涨了约 6 万倍（每两年 3.0 倍），DRAM 带宽只涨了约 100 倍（每两年 1.6 倍），芯片之间的互联带宽约 30 倍（[图 1-1](#fig-memwall)）。带宽跟不上，越来越多的 op 受显存带宽限制（memory-bound），只能少搬：让读进来的数据留在离计算单元近的地方，多用几次。

<figure id="fig-memwall" class="fg-fig">
<img src="memory-wall.webp" alt="1997 到 2023 年峰值算力、DRAM 带宽和互联带宽的增长：算力每两年 3.0 倍，DRAM 带宽每两年 1.6 倍，互联带宽每两年 1.4 倍" loading="lazy" style="width: 100%; height: auto; border-radius: 8px; background: #fff;">
<figcaption><strong>图 1-1</strong> 峰值算力与 DRAM、互联带宽 20 年来的增长，纵轴是相对 1997 年的倍数（对数轴）。图来自 Gholami 等人的 <a href="https://arxiv.org/abs/2403.14123">AI and Memory Wall</a>（IEEE Micro，2024）。</figcaption>
</figure>

数据离计算单元越近，读写越快，容量也越小（[图 1-2](#fig-h-1)）。每个 SM 里有 register 和 L1 / shared memory，所有 SM 共享芯片上的 L2，显存在芯片外面。L1 和 L2 由硬件当作 cache 自动管理，kernel 能自己安排的只有 shared memory 和 register。所以 kernel 优化的套路都一样：把要反复用的数据读进 shared memory 或 register，在片上多算几次，再写回显存。[第 2 节](#triton)的 block 矩阵乘、[训练分析那篇](/blog/gpu-training-analysis/)的算子融合和 [FlashAttention](/blog/flashattention-1-to-4/) 都是这个思路。

{{fig-h-1}}

#### Memory access {#hw-move}

数据能从哪一层搬到哪一层，取决于每一层归谁用：计算单元只读写 register；register 只归一个 thread，shared memory 只归一个 block（thread、block 见 [1.3 节](#hw-model)）；显存和 L2 全卡共享。L1 和 L2 是 cache，没有指令能指定读写它们，数据只是经过。[表 1-1](#tab-1-1) 是各层之间能怎么搬，表里的指令名是 PTX 里的写法。PTX（Parallel Thread Execution）是 NVIDIA 定义的 GPU 虚拟指令集，相当于 GPU 上的汇编：CUDA C++ 和 Triton 写的 kernel 都先编译成 PTX，再由 ptxas 翻译成某一代 GPU 的机器码（SASS）。PTX 按单个 thread 的视角写，一条指令由一个 warp 的 32 个 thread 一起执行。

| 从 | 到 | 怎么搬 |
|:--|:--|:--|
| 显存 | register | ld.global（Triton 的 `tl.load`），经过 L2，一般也经过 L1 |
| register | 显存 | st.global（`tl.store`），经过 L2 |
| register | shared memory | st.shared |
| shared memory | register | ld.shared |
| 显存 | shared memory | 没有一步到位的普通 load：ld.global 读进 register 再 st.shared；或者 cp.async（Ampere 起）、TMA（Hopper 起，见 [1.2 节](#hw-sm)），都经过 L2，不经过 register |
| shared memory | 显存 | ld.shared 读进 register 再 st.global；或者 TMA。cp.async 只能从显存往 shared memory 搬 |
| 一个 thread 的 register | 同一个 warp 里别的 thread | shuffle（shfl.sync）：一条指令里，每个 thread 直接读同一个 warp 里另一个 thread 的一个 register；出了 warp 就不行 |
| 一个 block 的 shared memory | 别的 block | 不行，只能写回显存再读 |
{#tab-1-1 caption="**表 1-1** 数据在各层之间能怎么搬" note="Hopper 起的 thread block cluster 里，几个 block 可以互读 shared memory，这篇用不到。"}

cp.async 是 Ampere（compute capability 8.0）开始有的 PTX 指令，把显存里的数据复制到 shared memory。数据照样先经过 L2，到了 SM 以后直接写进 shared memory，不落到 register。每个 thread 自己算地址，一条搬 4、8 或 16 B；`cp.async.cg` 只在 L2 缓存，`cp.async.ca` 还会在 L1 留一份。

普通的 ld.global 发出以后 thread 也不会马上停，但数据要先落进 register，再由 st.shared 写进 shared memory，st.shared 用到这个 register 时就得等数据到齐；想提前几轮搬，register 里就得多存几轮数据。cp.async 不占 register，发出以后 thread 接着往下执行，真正要用时再等：几条 cp.async 用 `cp.async.commit_group` 打成一组，`cp.async.wait_group N` 等到最多只剩 N 组还在路上，再用 bar.sync 让同一个 block 的所有 thread 都看到搬好的数据。

Triton 代码里只有 `tl.load` 和 `tl.store`，走[表 1-1](#tab-1-1)的哪一行由编译器决定：[第 2 节](#triton)的 GELU、softmax、row sum 都是显存和 register 之间直接读写；matmul 的 A、B 要经过 shared memory，编出来用的是 cp.async（[3.3 节](#stages)）；用 tensor descriptor 写的 load 才会用 TMA（[Appendix C](#tma)）。

register 和 L1 / shared memory 都在 SM 里面，下面看 SM 本身的结构。

### 1.2 SM 的结构 {#hw-sm}

每个 SM 分成 4 个 SMSP（SM sub-partition），每个 SMSP 有自己的 warp 调度器、register、32 个 CUDA core（做 FP32 和 INT32 运算）、一个 Tensor core 和 4 个 LD/ST 单元，4 个 SMSP 共享 L1 / shared memory（[图 1-3](#fig-h-2)）。load / store 从 LD/ST 发出，先到 L1 / shared memory，不在这里的再经片上互联去 L2 和显存。L1 和 shared memory 是同一块 128 KB 的片上存储，shared memory 最多切到 100 KB，下面的 notes 讲两者的区别和几种切法。

{{fig-h-2}}

<details class="fold doc aside">
<summary>为什么 shared memory 最多 100 KB</summary>

两者是同一块 SRAM，区别在谁来管、给谁用。L1 是 cache：从显存读进来的数据经过它，硬件按 128 B 的 cache line 自动放进去、自动换出，kernel 指定不了哪些数据留下。shared memory 是 kernel 按地址读写的一块空间，数据什么时候放进去、什么时候被覆盖都由 kernel 决定；写 Triton 时这件事由编译器替你做，比如 [2.4 节](#triton-matmul) `tl.dot` 的 A、B block 和 `num_stages` 的两个 slot。shared memory 按 program 分配，只有这个 program 的 thread 能读写，program 跑完才释放，所以它算在驻留上限里（[表 3-1](#tab-3-1)）。L1 由 SM 上所有 program 共用，不占名额，但里面的数据随时可能被别的 program 挤掉。

[GA102（RTX 30 系列）白皮书](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)列出的切法见[表 1-2](#tab-1-2)。跑 kernel 时（compute 模式）有 6 档，跑图形时固定一种，另外留 16 KB 给图形管线。

| 模式 | L1 | shared memory | 图形管线 |
|:--|--:|--:|--:|
| compute | 128 KB | 0 KB | — |
| compute | 120 KB | 8 KB | — |
| compute | 112 KB | 16 KB | — |
| compute | 96 KB | 32 KB | — |
| compute | 64 KB | 64 KB | — |
| compute | 28 KB | 100 KB | — |
| graphics | 64 KB | 48 KB | 16 KB |
{#tab-1-2 caption="**表 1-2** 每个 SM 的 128 KB 怎么切（GA102 白皮书）" note="5090 能选的 shared memory 档位和 GA102 一样。"}

一个 kernel 用哪一档由驱动在启动时决定，kernel 可以通过 carveout 属性表达偏好，驱动不一定照办。

白皮书只列了切法，没有说为什么 shared memory 最多 100 KB、至少留 28 KB 给 L1，CUDA 文档里也没有说明。能验证的是这 28 KB 确实还在当 L1 用。在 5090 上让一个 thread 在一段数据里按随机顺序跳着读，每次读的地址取决于上一次读到的值（pointer chase），这样每次读的耗时就是一次访存的延迟。shared memory 切到最大时，数据在 20 KB 以内每次读约 22 ns，都在 L1 命中；超过 24 KB 开始变慢，到 48 KB 约 140 ns，基本都要去 L2 读。切成 L1 优先时，64 KB 以内都还在 30 ns 以内。

</details>

[图 1-3](#fig-h-2) 右下角的 TMA（Tensor Memory Accelerator）是 Hopper 起 SM 里专门在显存和 shared memory 之间整块搬数据的硬件单元。它和 LD/ST 走同一条路，区别在于只要一个 thread 交给它一块数据的描述，地址由它自己算。正文的 kernel 用不到它，细节放在 [Appendix C](#tma)。

再往里看一层，一个 SMSP 里有两种计算单元（[图 1-4](#fig-h-9)）。CUDA core 一次算一个数。最常用的是 FMA（fused multiply-add）指令，算 $d = a \times b + c$，乘和加在一条指令里做完、只舍入一次：一条 FMA 指令里 warp 的 32 个 thread 各用一个 CUDA core，各做一次乘加（2 FLOPs）。5090 一共 170 × 128 个 CUDA core，fp32 峰值 1.05e14 FLOPS。Tensor core 一次算一个小矩阵乘：一条指令由整个 warp 一起发出，算一小块 A × B 再加到累加器上。它只吃 bf16、fp16、tf32、fp8 这类低精度输入，bf16 的峰值是 2.1e14 FLOPS（fp32 累加），fp8 更高。第 2 节的 `tl.dot` 默认走 Tensor core，加减乘除在 CUDA core 上算，exp 这类函数在 SFU 上算。

SMSP 里另一块要紧的是 register file。每个 SMSP 有 16384 个 32-bit register（64 KB），一个 SM 共 65536 个。每个 thread 用几个 register 在编译时就定了，最多 255 个；block 放上 SM 时，它的每个 thread 从所在 SMSP 的 register file 里分到这么多个，只归自己读写，block 跑完才还回去。所以每个 thread 用的 register 越多，一个 SM 能同时放的 thread 越少，[3.1 节](#sched-warp)会算这笔账。

{{fig-h-9}}

[第 2 节](#triton) kernel 图里的「register」就是这块 register file，「计算单元」指其中的 CUDA core 和 Tensor core。几代 GPU 的规格对比见 [Appendix A](#specs)。

这些是硬件提供的资源。register 按 thread 分配，shared memory 按 block 分配，要知道一个 kernel 怎么用这些资源，得先看 CUDA 怎么把 thread 组织起来。

### 1.3 编程模型 {#hw-model}

CUDA 的执行模型叫 SIMT（single instruction, multiple threads）：一个 warp 的 32 个 thread 同一时刻执行同一条指令。thread 往上分四层：thread、warp（32 个 thread）、block（也叫 CTA，若干个 warp）、grid（一次 kernel 启动的全部 block），每层跑在哪个硬件上见[图 1-5](#fig-h-5)。

{{fig-h-5}}

block 放到 SM 上以后一直驻留到跑完，一个 SM 可以同时驻留多个 block。一个 SM 能放几个 block、驻留的 warp 怎么轮流执行、block 多于全卡的名额时怎么分批跑，[第 3 节](#tune)再讲。

block 之间不能直接共享数据，只能通过显存交换，而这正是最慢的一层。所以要尽量把会重复读的数据放进同一个 block 里处理，这就是 tiling。在 CUDA 里做 tiling 要自己管 shared memory 和 thread 同步，[第 2 节](#triton)的 Triton 把这部分交给编译器。

---

## 2 Triton 入门 {#triton}

第 1 节说过，kernel 优化的套路是把数据读进片上，算完能算的再写回显存。Triton 就是按这个思路写 kernel 的语言：你只写一个 block 的数据怎么处理，也就是 [1.3 节](#hw-model)说的 tiling。Triton 把这个 block 叫作一个 program，block 内怎么分给 thread、要不要经过 shared memory 由编译器决定，CUDA 的各层在 Triton 里怎么写见[图 1-5](#fig-h-5)。先看这样写能换来什么，[表 2-1](#tab-2-1) 把这一节的四个例子各写三遍：用 PyTorch 的张量运算直接写（eager），同一份代码包一层 `torch.compile`，以及用 Triton 手写一个 kernel。

| | [GELU](#triton-gelu) | [softmax](#triton-softmax) | [row sum](#triton-rowsum) | [matmul + ReLU](#triton-matmul) |
|:--|--:|--:|--:|--:|
| eager | 194.7 | 244.0 | 51.3 | 158.5 |
| torch.compile | 47.3（4.1×） | 91.0（2.7×） | 48.2（1.1×） | 164.7（1.0×） |
| Triton | 46.6（4.2×） | 95.4（2.6×） | 48.9（1.0×） | 173.0（0.9×） |
{#tab-2-1 caption="**表 2-1** 同一个 op 的三种写法（µs，括号里是相对 eager 的加速）" note="RTX 5090，fp32，形状和各小节一样：GELU n = 8388608，softmax 和 row sum 4096 × 4096，matmul A 4096 × 2048、B 2048 × 512。eager 分别是按 tanh 公式逐个运算、<code>softmax_naive</code>、<code>x.sum(1)</code>、<code>torch.relu(A @ B)</code>，matmul 关掉了 tf32。Triton 一行是各小节的 kernel 套上 <code>triton.autotune</code> 以后的结果，候选和选中的配置见 3.4 节，其中 softmax 选中的配置比第 2 节的固定参数（90.8 µs）慢 5%；<code>torch.compile</code> 也会自己挑 block 大小和 <code>num_warps</code>，matmul 的另外两行直接调 cuBLAS。<code>triton.testing.do_bench</code> 跑 5 次取中位数。换一张卡数字会变，三行之间的对比关系一般不变。"}

GELU 和 softmax 的 eager 写法里，每个运算都单独跑一遍，中间结果要写回显存、下一个运算再读回来；`torch.compile` 和 Triton 都把它们合成一个 kernel，中间结果留在 register 里，所以 GELU 快了 4 倍多，softmax 快了 2.6 倍左右。两者差不多快，因为 `torch.compile` 在 GPU 上生成的也是 Triton kernel。row sum 本来只有一步求和，三种写法差不多。matmul + ReLU 的另外两行都调 cuBLAS，Triton 版慢 9%，原因在 [3.4 节](#tile)最后。

下文 program 就是 CUDA 的 block；单说 block 时指一块数据，比如 A 的 block、C 的 block；代码里的 `BLOCK`、`TILE` 是一段数据的长度。四个例子由浅入深（[图 2-1](#fig-o-1)）：grid 怎么切，看一个输出依赖哪些输入；program 里要不要循环，看这些输入能不能一次装进 register，装不下就一个 block 一个 block 地读，用累加器（accumulator）存中间结果。

{{fig-o-1}}

每个例子配一张数据流图：左边是 grid，右边从上到下是计算单元、register、shared memory 和显存。`tl.load`、`tl.store` 画成显存和 register 或 shared memory 之间的一条箭头，中间经过的 L2 不画；计算单元那一行从左往右是执行顺序，register 和 shared memory 里的一块存储只画一次，横向跨过它存在的那段时间，块上标的是它的大小。

### 2.1 GELU {#triton-gelu}

GELU 对每个元素单独算，$\Phi$ 是标准正态分布的累积分布函数，实际用 tanh 近似：

<p align="center">$y_i = \mathrm{GELU}(x_i) = x_i\,\Phi(x_i) \approx \tfrac{1}{2}\,x_i\left(1 + \tanh\left(\sqrt{2/\pi}\,(x_i + 0.044715\,x_i^3)\right)\right)$</p>

{{fig-t-1}}

逐元素 op 最简单，每个元素各算各的，program 之间不需要配合。把 n 个元素切成每段 BLOCK 个，一段交给一个 program（[图 2-2](#fig-t-1)）。kernel 里先用 `tl.program_id` 算出自己负责的下标，读进来，算完写回：

```python {role="program" tag="处理第 pid 段的 BLOCK 个元素"}
{{code:triton_tutorial.py:gelu_kernel}}
```

启动时只要定 grid，也就是 program 的个数，它和 SM 的个数无关。

```python {role="grid" tag="一段一个 program"}
{{code:triton_tutorial.py:gelu_launch}}
```

program 编译后就是 CUDA 的一个 block。8192 个 program 远多于 170 个 SM，GPU 会把它们一批一批地排到 SM 上，每个 SM 同时跑 12 个（为什么是 12 个见 [3.1 节](#sched-warp)，怎么分批见 [3.2 节](#sched-wave)）。

<details class="fold doc aside">
<summary>GELU 编译成的 PTX</summary>

Triton 把 kernel 编译成 PTX（[1.1 节](#hw-move)），PTX 描述的是单个 thread 的指令。n = 8388608 时的节选：

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

### 2.2 Softmax {#triton-softmax}

softmax 对 `[M, N]` 的输入按行算，先减去每行的最大值 $m_i$ 防止 exp 溢出：

<p align="center">$m_i = \max_k x_{ik}, \qquad y_{ij} = \dfrac{e^{x_{ij} - m_i}}{\sum_{k} e^{x_{ik} - m_i}}$</p>

$y_{ij}$ 要用到第 $i$ 行全部 N 个数。eager 下 softmax 要 5 个 kernel（和[训练分析那篇](/blog/gpu-training-analysis/#attention)里 attention 的 softmax 一样），一共读 5MN + 2M 个数，写 3MN + 2M 个：

```python
{{code:triton_tutorial.py:softmax_naive}}
```

{{fig-t-2}}

如果一个 program 处理一整行（[图 2-3](#fig-t-2)），max、exp、求和、除都可以在 register 里做完，只读 MN、写 MN，读写量是原来的 1/4：

```python {role="program" tag="处理第 row 行"}
{{code:triton_tutorial.py:softmax_kernel}}
```

启动时一行一个 program：

```python {role="grid" tag="一行一个 program"}
{{code:triton_tutorial.py:softmax_launch}}
```

[图 2-3](#fig-t-2) 里的 warp max、warp sum 是 shared memory 在这个 kernel 里唯一的用处：`tl.max`、`tl.sum` 分三步归约。每个 thread 先把自己手里的 32 个数归约成一个。再在 warp 内用 shuffle 归约，shuffle 让每个 thread 直接读同一个 warp 里另一个 thread 的 register，不经过 shared memory（[表 1-1](#tab-1-1)）。编出来的 PTX 用的是 `shfl.sync.bfly`：每一步 thread $i$ 读 thread $i \oplus d$（按位异或）的值并和自己的合并，$d$ 依次取 16、8、4、2、1，5 步以后 warp 里每个 thread 都拿到这个 warp 的最大值或和。最后 4 个 warp 的部分结果经 shared memory 交换（一共 16 B），再用 $d$ 取 2、1 的两步 shuffle 合并成整行的结果。把一整个 block 的数据放进 shared memory 什么时候值得，[2.4 节](#triton-matmul)再讲。

这个写法要求 BLOCK ≥ N，也就是一行能一次装进一个 program。行再长，program 就只能沿行一个 block 一个 block 地读，读过的 block 要留下一个中间结果。下一节先看最简单的按行归约：求和。

### 2.3 Row sum {#triton-rowsum}

row sum 对 `[M, N]` 的输入每行求和：

<p align="center">$s_i = \sum_{j=1}^{N} x_{ij} = \sum_{T} \ \sum_{j \in \mathcal{J}_T} x_{ij}$</p>

$\mathcal{J}_T$ 是第 $T$ 个 block 的 TILE 个列下标，代码里 TILE 取 1024。加法满足结合律，先把每个 block 加起来再相加，和一次加完是同一个数（浮点数下只差舍入误差），所以 program 可以沿行一个 block 一个 block 地读，每读一个 block 就加到累加器上（[图 2-4](#fig-t-3)）。

{{fig-t-3}}

累加器 `acc` 是一个长度为 TILE 的向量，不是一个数：循环里每个 thread 只加自己负责的几个位置，thread 之间不需要通信；循环结束后，`tl.sum` 再做一次跨 thread 归约，把 TILE 个部分和加成一个数。

```python {role="program" tag="沿第 row 行循环累加"}
{{code:triton_tutorial.py:row_sum_kernel}}
```

grid 和 softmax 一样，一行一个 program：

```python {role="grid" tag="一行一个 program"}
{{code:triton_tutorial.py:row_sum_launch}}
```

softmax 不能直接这样按 block 读。$e^{x_{ij} - m_i}$ 要先知道整行的最大值 $m_i$，读完整行才知道。按 block 读时，累加器里要同时存目前的最大值和部分和，最大值变大时把已有的部分和乘上 $e^{m_{\text{old}} - m_{\text{new}}}$ 修正，这就是 online softmax，[FlashAttention](/blog/flashattention-1-to-4/) 那篇会讲。

前面三个例子里每个数只读一次，切 block 只是为了把工作分给不同的 program。矩阵乘不同：A 的每个元素要和 B 的一整列相乘，切 block 是为了让读进片上的数据被多次使用。它也要沿 K 循环，累加器是 C 的一个 block。

### 2.4 Matmul + ReLU {#triton-matmul}

`[M, K] × [K, N]` 的矩阵乘再接 ReLU：

<p align="center">$C_{ij} = \max\left(0,\ \sum_{k=1}^{K} A_{ik} B_{kj}\right)$</p>

把 M、N、K 分别按 BM、BN、BK 切成 block，记 $A_{IT}$ 是 A 的第 $I$ 个 row block 和第 $T$ 个 column block 交叉处的 block（BM × BK），$B_{TJ}$、$C_{IJ}$ 同理。$C_{IJ}$ 只用到 A 的第 $I$ 个 row block 和 B 的第 $J$ 个 column block，不同的 $C_{IJ}$ 可以交给不同的 program。对 $k$ 的求和再按 BK 分段，$\mathcal{K}_T$ 是第 $T$ 段的 BK 个下标：

<p align="center">$\sum_{k=1}^{K} A_{ik} B_{kj} = \sum_{T=1}^{K/\mathrm{BK}} \ \sum_{k \in \mathcal{K}_T} A_{ik} B_{kj}, \qquad C_{IJ} = \max\left(0,\ \sum_{T} A_{IT}\, B_{TJ}\right)$</p>

和 row sum 一样，分段加完再相加等于一次加完，所以每轮只读 $A_{IT}$、$B_{TJ}$ 两个 block，乘完累加，最后就是完整的 $C_{IJ}$。ReLU 逐元素作用在累加完的结果上，放在最后做。

切 block 能把每个输出从显存读的数降到原来的 1/T，这笔账在[单卡 GPU 优化方法](/blog/gpu-single-gpu-optimization/#tiling)里算。BM、BN、BK 都得是 2 的幂，这是 `tl.arange` 的要求；`tl.dot` 还要求 BK 至少 16。

{{fig-t-4}}

一个 program 负责一个 $C_{IJ}$（[图 2-5](#fig-t-4)），沿 K 循环，第 $T$ 轮读 $A_{IT}$ 和 $B_{TJ}$，用 `tl.dot` 乘加到累加器上；写回前在 register 里做 ReLU，激活函数不用再单独一个 kernel 读写一遍 C：

```python {role="program" tag="算 C 的第 (pid_m, pid_n) 个 block"}
{{code:triton_tutorial.py:matmul_kernel}}
```

grid 是二维的：

```python {role="grid" tag="C 的每个 block 一个 program，二维"}
{{code:triton_tutorial.py:matmul_launch}}
```

<div class="nv-key">

**shared memory 什么时候值得用**：看一个数读进来以后有几个 thread 要用。register 只归一个 thread，要给别的 thread 用就得经过 shared memory。

- **只有自己用**：放在 register 里就够。GELU、softmax、row sum 的每个数都只被读它的那个 thread 用。
- **只交换几个数**：softmax、row sum 最后合并整行的最大值或和，warp 内用 shuffle 直接传，4 个 warp 之间才经过 shared memory，一共 16 B。
- **整块数据给多个 thread 用**：matmul 里 $A_{ik}$ 要送到存着 C 第 $i$ 行的 16 个 thread，$B_{kj}$ 要送到 8 个，所以 A、B 的 block 整块放进 shared memory。

代价是 shared memory 用得越多，一个 SM 放的 program 越少：matmul 每个 program 占 32 KB（两个 slot，见 [3.3 节](#stages)），加上保留的 1 KB，100 KB 只放得下 3 个（[3.1 节](#sched-warp)）。

</div>

<details class="fold doc aside">
<summary>matmul 编译出来什么样</summary>

从编译出的 TTGIR 和 PTX 里能数清楚 A、B 的 block 怎么进出 shared memory，流水线本身见 [3.3 节](#stages)。A、B 各搬 64 次，每次 8 KB，一个 program 一共 1 MB；512 个 program 加起来 512 MB，而 A、B 本身只有 36 MB，所以 L2 命中率有 91%，大部分读不用到显存。每轮从 shared memory 读进 register 的是 192 KB，同一个数平均被读 12 次，这就是前面说的复用。写回之前 acc 还要经 shared memory 换一次排布：`tl.dot` 输出时每个 thread 拿两个 4 × 4 的小方块，编译器给 `tl.store` 选的排布是每个 thread 拿 8 段、每段一行里连续的 4 个数，同一个数换到了别的 thread 甚至别的 warp 手里，而 register 是每个 thread 私有的，只能先写进 shared memory、同步一次再按新排布读出来。两种排布写显存时都是一个 warp 一次写两整行，并不是为了合并写入；这一步是两种排布对不上留下的。代码里加了 `input_precision="ieee"`，所以 `tl.dot` 编译成 CUDA core 上的 `fma.rn.f32x2`；不加的话 fp32 输入默认走 tf32 的 Tensor core，和 fp32 矩阵乘的结果差 0.1 左右。

</details>

第 2 节的四个例子是同一个模式：program 把一个 block 的数据读进 register，在片上做完能做的计算，再写回显存。代码只决定 program 做什么；它在 SM 上怎么排、能不能把 SM 用满，还取决于启动参数和 block 的大小，这是第 3 节的内容。

---

## 3 调度和调参 {#tune}

[表 2-1](#tab-2-1) 里 matmul + ReLU 的 Triton 版比 cuBLAS 慢，调参能补回一部分，剩下的差距在 kernel 内部（[3.4 节](#tile)最后）。这一节先看一个 SM 放几个 program（[3.1](#sched-warp)）、放不下时怎么分批（[3.2](#sched-wave)），再看能调的三个参数：`num_warps`、`num_stages`（[3.3](#stages)）和 block 的大小（[3.4](#tile)）。

> 注：同一个配置在不同次测量之间最多差 8%，所以下面只在同一张表或同一张图里比较。

### 3.1 驻留和 warp 调度 {#sched-warp}

program 放上 SM 后一直驻留到跑完，它的 warp 分到 4 个 SMSP（[图 3-1](#fig-h-6)）。每个 SMSP 的调度器每拍（一个时钟周期）挑一个就绪的 warp 发一条指令，一个 warp 等显存数据的几百拍里，调度器改发别的 warp。要让调度器每拍都有 warp 可发，有三种办法：

1. 多驻留 warp；
2. 让每个 thread 有更多互不依赖的 load 或乘加（FMA，[1.2 节](#hw-sm)）；
3. 提前发 load。

{{fig-h-6}}

一个 SM 能驻留几个 program，看[表 3-1](#tab-3-1) 的四条上限里哪条先用完。第 2 节的 kernel 都是 4 个 warp：GELU 和 row sum 先用完 warp 名额，每个 SM 放 12 个；softmax 先用完 register，放 9 个；matmul 先用完 shared memory，放 3 个。

| 资源 | RTX 5090 每个 SM 的上限 |
|:--|:--|
| 驻留 warp | 48 个（每个 SMSP 12 个），即 1536 个 thread |
| 驻留 program（CUDA block） | 24 个 |
| register | 65536 个 32-bit（256 KB），每个 thread 最多 255 个 |
| shared memory | 100 KB（128 KB 的 L1 / shared memory 里最多切出这么多，见 [1.2 节](#hw-sm)），每个 program 另占 1 KB |
{#tab-3-1 caption="**表 3-1** RTX 5090 一个 SM 的驻留上限" note="用 <code>cuDeviceGetAttribute</code> 在 5090 上读出，和 CUDA Programming Guide 里 compute capability 12.0 的数一致。一个 program 占 <code>num_warps</code> 个 warp 名额、1 个 program 名额、32 × <code>num_warps</code> × 每个 thread 的 register 数（按 8 取整），以及 shared memory 用量加 1 KB。几代 GPU 的对比见 Appendix A。"}

### 3.2 wave 和 tail effect {#sched-wave}

grid 比全卡一次放得下的多时，program 要分批（wave）。GELU 一批 170 × 12 = 2040 个，8192 个要 5 批，实测批次之间没有分界，跑完一个就补进一个（[图 3-2](#fig-t-5)）。

{{fig-t-5}}

matmul 一批 510 个，512 个 program 多出 2 个，最后 168 个 SM 空等这 2 个跑完，这叫 tail effect（[图 3-3](#fig-t-7)）。A 换成 4032 × 2048 时，504 个 program 一批放下，耗时从 203.3 µs 降到 158.3 µs，按每秒的 FLOP 算和 cuBLAS 一样快（4096 行 160.5 µs）。所以 Triton 版慢在多出的第 2 批，不在 kernel 本身。

{{fig-t-7}}

### 3.3 num_warps 和 num_stages {#stages}

两个参数都在启动 kernel 时传：`kernel[grid](..., num_warps=4, num_stages=3)`，这是 Triton 的默认值。

`num_warps` 是一个 program 的 warp 数。block 的大小不变时，warp 多了，每个 thread 分到的数据就少，所以它是在 [3.1 节](#sched-warp)的前两种办法之间做分配。GELU 的 `num_warps` 从 1 到 16 耗时差不多；取 32 时一个 SM 只放得下 1 个 program，每个 thread 只剩 1 个数，才明显变慢。逐元素和按行归约的 kernel 用默认值就够。

`num_stages` 是第三种办法。matmul 每轮把 A、B 的 block 搬进 shared memory 里的 slot，再读进 register 乘加。`num_stages` 取 S 时，第 $k$ 轮的 load 在第 $k-(S-1)$ 轮算完时就用 cp.async 发出（[1.1 节](#hw-move)），算第 $k$ 轮时，下一轮的数据已经在另一个 slot 里（[图 3-4](#fig-t-9)）。代价是每多一级多一个 slot 的 shared memory（[表 3-2](#tab-3-2)）。

{{fig-t-9}}

| num_stages | A、B 各几个 slot (shmem) | load 提前几整轮 | shared memory | 每个 SM 放几个 program | 耗时 |
|:--|--:|--:|--:|--:|--:|
| 1 | 1 | — | 16 KB | 5 | 181.4 µs |
| 2 | 1 | 0 | 16 KB | 5 | 175.8 µs |
| 3（默认） | 2 | 1 | 32 KB | 3 | 169.3 µs |
| 4 | 3 | 2 | 48 KB | 2 | 210.0 µs |
{#tab-3-2 caption="**表 3-2** matmul 的 num_stages 改了什么（A 4032 × 2048，B 2048 × 512，BM = BN = 64，BK = 32，4 个 warp）" note="slot 数读自编译出的 TTGIR，提前几轮读自 PTX 里 cp.async 的发出位置，shared memory 读自编译结果的 metadata。用 4032 行是为了让 504 个 program 正好一批跑完，避开 3.2 节的 tail effect。耗时是 <code>do_bench</code> 跑 5 次的中位数，四档是同一次测量。"}

默认的 3 级已经把每轮等数据的时间从 852 拍压到 74 拍，再加一级也没有多少等待可藏。`num_stages=4` 反而慢了 24%。它每一轮其实更短（[图 3-4](#fig-t-9) 最下面一栏），慢在 shared memory：每多一级多一个 16 KB 的 slot，一个 program 占 48 KB，加上保留的 1 KB 是 49 KB，100 KB 只放得下 2 个（3 级时 33 KB，放得下 3 个）。全卡一批从 510 个降到 340 个，504 个 program 要分两批，第 2 批 164 个（[3.2 节](#sched-wave)，[图 3-5](#fig-s-1)）。四个例子里只有 matmul 的循环用得上 `num_stages`。它也改不了 program 的个数：4096 行时取 1 到 4，耗时都在 209 到 214 µs。

{{fig-s-1}}

### 3.4 block 大小和 autotune {#tile}

block 的大小同时决定 program 的个数、每个 program 占的资源和每个 thread 存几个 acc。我们在正文形状和 4096 × 4096 × 4096 上各扫了 198 组 BM、BN、BK、`num_warps` 和 `num_stages`。正文形状下最快的是 BM = BN = 128、BK = 32、8 个 warp，176.9 µs，比 64 × 64 的 203.0 µs 少用 13% 的时间（[图 3-6](#fig-d-1)）。它只有 128 个 program，空着 42 个 SM，但没有哪个 SM 要多跑一个 program，每个 thread 存的 acc 也多一倍。

{{fig-d-1}}

两个形状的规律一样（[图 3-7](#fig-d-2)）：每个 thread 存 32 到 64 个 acc、每个 SM 只驻留 8 到 16 个 warp 时最快。acc 越多，每从 shared memory 读一个数能做的乘加越多（[单卡优化篇第 7 节](/blog/gpu-single-gpu-optimization/#tiling)）；驻留的 warp 一多，每个 thread 分到的 register 就放不下这么多 acc。所以 matmul 靠第二、三种办法藏延迟，不靠多驻留 warp。

{{fig-d-2}}

形状一变，最快的配置就变，Triton 用 autotune 自动挑：

```python {role="program" tag="matmul 的 kernel 不变，外面套一层 autotune"}
{{code:triton_autotune.py:matmul_autotune}}
```

`configs` 是 48 个候选，`key` 里的参数一变就重新挑，grid 跟着选中的 BM、BN 变。第一次调用时，每个候选都编译、试跑一遍（[表 3-3](#tab-3-3)）。如果 kernel 原地修改输出，要加 `reset_to_zero` 或 `restore_value`，否则试跑会重复累加。

| op | 候选数 | 第一次调用 | 选中的配置 | 第 2 节的固定参数 | autotune 后 |
|:--|--:|--:|:--|--:|--:|
| GELU | 25 | 3.6 s | BLOCK = 4096，8 个 warp | 46.5 | 46.6 |
| softmax | 6 | 1.0 s | 8 个 warp | 90.8 | 95.4 |
| row sum | 16 | 2.3 s | TILE = 2048，16 个 warp | 52.8 | 48.9 |
| matmul + ReLU | 48 | 79.5 s | BM = BN = 128，BK = 32，8 个 warp，3 级 | 197.5 | 173.0 |
{#tab-3-3 caption="**表 3-3** 四个 kernel autotune 前后（µs）" note="RTX 5090，fp32，形状同<a href=\"#tab-2-1\">表 2-1</a>，和表 2-1 是同一次测量。第一次调用从空的编译缓存开始算，包含编译所有候选和试跑。耗时是 <code>do_bench</code> 跑 5 次的中位数。softmax 的候选之间只差几个百分点，选中的反而慢了。"}

matmul 调到最好仍比 cuBLAS 慢 9%，差在 register 这一层的 tiling：cuBLAS 每个 thread 存 128 个 acc，Triton 只存得下 64 个；`num_stages` 的最优值也随 block 大小变。

<details class="fold doc aside">
<summary>num_stages 随 block 怎么变，Triton 比 cuBLAS 慢在哪</summary>

block 一变，`num_stages` 的最优值也跟着变（[表 3-4](#tab-3-4)），看的还是 [3.3 节](#stages)那两件事：多一级会不会多出一批，每个 SM 本来有多少个 warp。16 × 16 × 16 时一个 SM 有 40 到 48 个 warp，等数据时总有别的 warp 顶上，1 级最快；2 级起每个 thread 多用 2 到 4 个 register，按 8 个一组分配后一个 SM 从 12 个 program 降到 10 个。128 × 128 × 32 时一个 SM 只有 1 个 program、8 个 warp，只能靠预取，3 级比 1 级少用 19% 的时间；4 级要 96 KB 也还放得下这 1 个，和 3 级一样快。32 × 32 × 16 的 4 级每个 SM 从 10 个降到 7 个，但还是 2 批，所以不吃亏。

| block | num_stages = 1 | 2 | 3 | 4 |
|:--|--:|--:|--:|--:|
| 16 × 16 × 16，4 个 warp | **510.4（12 个 / 5 批）** | 567.1（10 个 / 5 批） | 546.9（10 个 / 5 批） | 564.1（10 个 / 5 批） |
| 32 × 32 × 16，4 个 warp | 272.5（10 个 / 2 批） | 260.0（10 个 / 2 批） | 243.2（10 个 / 2 批） | **242.9（7 个 / 2 批）** |
| 64 × 64 × 32，4 个 warp | 207.5（5 个 / 1 批） | **202.6（5 个 / 1 批）** | 206.3（3 个 / 2 批） | 213.3（2 个 / 2 批） |
| 128 × 128 × 32，8 个 warp | 220.6（1 个 / 1 批） | 200.9（1 个 / 1 批） | 179.4（1 个 / 1 批） | **179.3（1 个 / 1 批）** |
{#tab-3-4 caption="**表 3-4** 换 block 大小以后，num_stages 的影响（µs，括号里是每个 SM 放几个 program、整个 kernel 跑几批）" note="RTX 5090，fp32，A 4096 × 2048，B 2048 × 512，每行最快的加粗。耗时是 <code>do_bench</code> 跑 5 次的中位数。"}

调到最好的 128 × 128 × 32 仍比 cuBLAS 慢，慢的原因不在 program 怎么排（[表 3-5](#tab-3-5)）。cuBLAS 调到的 CUTLASS kernel 也是 128 个 block、每个 SM 8 个 warp：它的 block 算 256 × 128 的 C，只有 64 个，grid 的第三维是 2，看起来是把 K 也切成了两半。两边的乘加次数一样，指令数只差 1%，Triton 的拍数却多 16%，差在同样的指令发得更慢：每条指令平均隔 3.03 拍，cuBLAS 是 2.56 拍，多出来的主要在两项：等 shared memory，和 dispatch_stall（指令选中了却没能发到执行单元）。根源在 register 这一层的 tiling（[单卡优化篇第 7 节](/blog/gpu-single-gpu-optimization/#tiling)）：cuBLAS 每个 thread 存 128 个 acc，只用 210 个 register；Triton 存 64 个就用了 220 个，每次乘加要从 shared memory 读的数多，读的字节是 cuBLAS 的 2.2 倍。在 Triton 里让每个 thread 存 128 个 acc，扫描里这样的配置 register 都顶到 255 个，大多有 spill，最快的也要 185.0 µs。这是编译器怎么生成代码的问题，block 大小、warp 数、级数这些参数调不到。

| | cuBLAS | Triton 128 × 128 × 32 | Triton 64 × 64 × 32 |
|:--|--:|--:|--:|
| 拍数（耗时 × SM 频率） | 3.93e5 | 4.58e5 | 4.71e5 |
| block 数 × 每个 block 的 thread | 128 × 256 | 128 × 256 | 512 × 128 |
| 每个 SM 驻留的 warp | 8.0 | 8.0 | 11.0 |
| FFMA 次数 | 4.297e9 | 4.295e9 | 4.295e9 |
| 指令数（warp 级） | 1.510e8 | 1.521e8 | 1.610e8 |
| 每个 thread 存几个 acc | 128 | 64 | 32 |
| 每个 thread 的 register | 210 | 220 | 96 |
| 从 shared memory 读的字节 | 5.54e8 | 1.22e9 | 1.35e9 |
| 每条指令平均隔几拍 | 2.56 | 3.03 | 4.00 |
| 其中等 shared memory（short_scoreboard + mio_throttle） | 0.17 | 0.42 | 0.61 |
| 其中 dispatch_stall（选中了没发出去） | 0.22 | 0.60 | 0.70 |
| 其中等显存（long_scoreboard） | 0.03 | 0.03 | 0.06 |
{#tab-3-5 caption="**表 3-5** cuBLAS 和 Triton 的 fp32 matmul 在 Nsight Compute 里的对比" note="RTX 5090，A 4096 × 2048，B 2048 × 512，<code>--clock-control none</code>。cuBLAS 关掉 tf32，调到的 kernel 是 CUTLASS 的 <code>cutlass_80_simt_sgemm_256x128_8x4</code>，也走 CUDA core。每个 thread 的 acc 数按一个 block 算的 C 除以 thread 数：cuBLAS 的 block 是 256 × 128、256 个 thread。等待的各项是每发一条指令平均有几个 warp 在等这一项，和「每条指令平均隔几拍」同一个单位。"}

</details>

这一节的数都是 fp32 走 CUDA core 量的，换成 Tensor core 要重新量。

[下一篇](/blog/gpu-single-gpu-optimization/)把这些写法背后的优化方法放在一起讲；[训练分析那篇](/blog/gpu-training-analysis/)用第 2 节的模式把 RMSNorm 的几个 kernel 融合成一个。

---

## Appendix A 几代 GPU 的规格 {#specs}

正文只用 RTX 5090 的数。[表 A-1](#tab-a-1) 把它和三代数据中心卡放在一起，硬件规格、标称峰值和驻留上限都在一张表里。

| | A100 | H100 | B200 | RTX 5090 |
|:--|--:|--:|--:|--:|
| SM 数 | 108 | 132 | 148 | 170 |
| 每 SM 的 CUDA core | 64 | 128 | 128 | 128 |
| 每 SM 的 Tensor core | 4 | 4 | 4 | 4 |
| fp32 峰值（CUDA core） | 1.95e13 FLOPS | 6.7e13 FLOPS | 7.5e13 FLOPS | 1.05e14 FLOPS |
| bf16 峰值（Tensor core，dense） | 3.12e14 FLOPS | 9.89e14 FLOPS | 2.25e15 FLOPS | 2.1e14 FLOPS |
| L2 | 40 MB | 50 MB | — | 96 MB |
| 显存 | 80 GB HBM2e | 80 GB HBM3 | 180 GB HBM3e | 32 GB GDDR7 |
| 显存带宽 | 2.0e12 B/s | 3.35e12 B/s | 7.7e12 B/s | 1.79e12 B/s |
| 每 SM 的 L1 + shared | 192 KB | 256 KB | 256 KB | 128 KB |
| 每 SM 的 register | 256 KB | 256 KB | 256 KB | 256 KB |
| 每 SMSP 最多驻留 warp | 16 | 16 | 16 | 12 |
| 每 SM 最多驻留 warp | 64 | 64 | 64 | 48 |
| 每 SM 最多驻留 block | 32 | 32 | 32 | 24 |
{#tab-a-1 caption="**表 A-1** 几代 NVIDIA GPU 的规格和驻留上限" note="A100 是 80GB SXM，H100 是 SXM，B200 按 HGX B200 的单卡算。峰值都是标称值，不含 structural sparsity，bf16 按 fp32 累加。规格来自 NVIDIA 各代架构白皮书和 datasheet，驻留上限来自 CUDA Programming Guide 的 compute capability 表（8.0、9.0、10.0、12.0）。B200 的 L2 没有找到可靠的公开数字，暂缺。"}

5090 的 SM 最多，fp32 峰值比 H100 还高；但每个 SM 能驻留的 warp 和 block 比数据中心卡少四分之一，L1 + shared 只有 H100、B200 的一半，bf16 的 Tensor core 峰值也只有 H100 的约五分之一。

---

## Appendix B TPU {#tpu}

正文的 GPU 靠 kernel 自己切 block 来复用数据。TPU 面对同样的带宽问题，思路是把控制逻辑做轻，把矩阵乘单元做大。

[图 B-1](#fig-h-3) 是 TPU v5p 的一个 TensorCore：Scalar unit 取指令、发给 VPU 和 MXU，整个核只有这一路控制，没有 warp 和调度器之间的切换；VPU 做逐元素运算和归约；矩阵乘交给 4 个 MXU，每个是 128 × 128 的 systolic array。片上存储只有 VMEM 和 VREG 两层，VMEM 是编译器管理的 scratchpad，不是硬件 cache，数据什么时候从 HBM 搬进来、搬多少，都由编译器（XLA）提前排好。

{{fig-h-3}}

MXU 和 GPU 最大的不同在乘加单元怎么拿到数据（[图 B-2](#fig-h-4)）。GPU 上每个 CUDA core 做一次乘加，要从 register file 读两个数、写回一个；一个数被用很多次，得靠 kernel 自己切 block，先搬进 shared memory 和 register 再反复读，[2.4 节](#triton-matmul)的 matmul 就是这样写的。systolic array 把复用做进了硬件：权重先装进 128 × 128 个格子，固定不动；输入 x 从左边进入，每拍向右传一格，经过的每一格把它乘上自己的权重，加到从上面传下来的部分和上，再把部分和往下传。一个 x 从 VMEM 读一次，在一行里被 128 个格子用到；稳定以后每拍从左边进 128 个数、从底部出 128 个结果，阵列里同时做 16384 次乘加，中间结果都不经过 register file。

{{fig-h-4}}

代价是形状要规整。矩阵的维度要是 128 的倍数才能填满阵列，小矩阵和不规则的计算会浪费格子；数据流由编译器提前排好，也不能像 GPU 那样靠切换 warp 应付不确定的访存延迟。多卡之间怎么互联是另一个大区别，超出了本文的范围。两边术语的对应见[表 B-1](#tab-b-1)。

| GPU | TPU | 作用 | RTX 5090 | TPU v5p |
|:--|:--|:--|--:|--:|
| SM | TensorCore | 包含其他运算单元的核心 | 170 | 2 |
| warp 调度器 | VPU | SIMD 向量运算的调度 | 680 | 8 |
| CUDA core | VPU ALU | 普通的 SIMD 运算单元 | 21760 | — |
| L1 / shared memory | VMEM | 片上的快速存储 | 21.3 MB | 128 MB |
| register | VREG | vector register | 42.5 MB | 512 KB |
| Tensor core | MXU | 矩阵乘单元 | 680 | 8 |
| 显存 | HBM | 大容量主存 | 32 GB GDDR7 | 95 GB |
{#tab-b-1 caption="**表 B-1** GPU 与 TPU 的术语对照" note="RTX 5090 的片上存储是 170 个 SM 加起来的总量。TPU 一列参考 Google 的 <a href=\"https://jax-ml.github.io/scaling-book/gpus/\">How to Scale Your Model</a>。"}

---

## Appendix C TMA 和异步同步 {#tma}

[1.1 节](#hw-move)讲了 cp.async，[1.2 节](#hw-sm)提到 SM 里还有一个专门搬数据的单元 TMA。正文的 kernel 都没用到 TMA，这个 appendix 讲它和 cp.async 差在哪（[C.1](#tma-vs-cpasync)），怎么知道它搬完了（[C.2](#tma-barrier)），以及有了它以后 kernel 怎么分工（[C.3](#tma-ws)）。

### C.1 TMA 和 cp.async 的区别 {#tma-vs-cpasync}

TMA 是一个硬件单元，每个 SM 一个，4 个 SMSP 共用：[H100 白皮书](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)的 SM 结构图把它画在 4 个 SMSP 之外、紧挨着 L1 / shared memory；5090 的白皮书没有单独画它。程序通过 `cp.async.bulk.tensor` 这类指令来用它，就像通过 ld / st 指令用 LD/ST 单元、通过 mma 指令用 Tensor core。它也不是另一条通路，和 LD/ST 发出的请求一样经片上互联到 L2，区别在请求由谁发、地址由谁算。它还能反过来，把 shared memory 里的一块写回显存，这是 cp.async 做不到的。

cp.async 要每个 thread 自己算地址、各搬一小段；TMA 只要一个 thread 发一条指令，给它一个 tensor descriptor（整个张量的起点、形状、stride，以及这次要搬的一块有多大），地址由 TMA 算，越界的部分自动补零。5090 也有 TMA：用 Triton 的 `tl.make_tensor_descriptor` 写一个复制 kernel，编出来的 PTX 里读是 `cp.async.bulk.tensor.2d.shared::cluster.global`、用 mbarrier 等，写回是 `cp.async.bulk.tensor.2d.global.shared::cta`。

从显存到 shared memory 的三种搬法放在一起比（[表 C-1](#tab-c-1)）：

| | ld.global + st.shared | cp.async | TMA |
|:--|:--|:--|:--|
| 数据经不经过 register | 经过：先读进 register，再写进 shared memory | 不经过 | 不经过 |
| 发出以后 thread 要不要等 | 用到那个 register 时要等 | 不等，要用时用 `cp.async.wait_group` 等 | 不等，要用时用 mbarrier 等 |
| 谁发指令、谁算地址 | 每个 thread 自己算地址、自己发指令；一条指令每个 thread 最多搬 16 B，从 Blackwell 开始最多 32 B | 每个 thread 自己算地址、自己发指令；一条指令每个 thread 搬 4、8 或 16 B | 一个 thread 发一条指令，搬一整块，地址由 TMA 生成 |
| 越界 | 每个 thread 自己用 mask 判断 | 每个 thread 自己用 mask 判断 | TMA 自动处理，越界补零 |
| 比方：显存是仓库，shared memory 是 SM 里的货架 | 每个人自己去仓库搬箱子，先揣在身上（register），回来再放上货架 | 每个人各自下单，仓库直接送到货架，之后等自己的单都到了 | 只派一个人，把一张清单（tensor descriptor）交给叉车，叉车自己算位置、整块搬到货架，搬完按一下铃（mbarrier） |
{#tab-c-1 caption="**表 C-1** 从显存搬到 shared memory 的三种办法" note="一条 load 指令里每个 thread 最多装满 4 个 32-bit register，所以是 16 B；数据多就多发几条，条数没有上限。Blackwell 加了 256-bit 的 load / store，一条能装 8 个 register，用 ptxas 编给 sm_100a、sm_120a 得到一条 <code>LDG.E.ENL2.256</code>，编给 sm_90a 会报错。cp.async 的大小只有 4、8、16 B 三种。"}

ld.global 和 cp.async 的区别是数据进不进 register；cp.async 和 TMA 都是异步、都直接进 shared memory，区别是谁来干搬运的活。拿 [2.4 节](#triton-matmul)的 matmul 算一下：每轮 A、B 的 block 各 8 KB。用 cp.async，一条搬 16 B，8 KB 要 512 条，分给 128 个 thread，每个 thread 每轮 A、B 各发 4 条，每条之前还要自己算地址和 mask。换成 TMA，一轮只要 A、B 各一条指令，由一个 thread 发出。TMA 省下的是 thread 的指令和 register：算地址、算 mask、发几百条小请求的活都不用 thread 做，于是可以只让一个 warp 发 TMA，其余的 warp 专心算。

### C.2 怎么知道搬完了 {#tma-barrier}

[表 C-1](#tab-c-1) 里叉车按的那个铃，是 shared memory 里的一个 barrier（mbarrier）。普通的同步 bar.sync 要所有 thread 都走到这一行才一起往下走，先到的只能干等。Ampere 加的 asynchronous barrier 把它拆成两步：thread 干完自己那份先 arrive，arrive 不会停，接着做不依赖这批数据的活；真要用数据时再 wait，等所有 thread 都 arrive 过（[图 C-1](#fig-b-1) 上）。cp.async 用这种办法就够了，每个 thread 搬的是自己发出的那几条，搬完自己知道。TMA 不一样：数据是硬件搬的，没有哪个 thread 能说「我的那份到了」。所以 Hopper 加了 asynchronous transaction barrier，它除了数 thread 的 arrive，还数字节：发起 TMA 的 thread 先登记这次要等多少字节，TMA 每写进 shared memory 一段就把字节数加到 barrier 上，wait 要等 thread 都 arrive 过、字节数也到齐才放行（图 C-1 下）。图里上下搬的是同样的 16 KB，显存带宽一样，所以放行得一样早；transaction barrier 不会让数据来得更快，省下的是 thread 自己搬运的时间，这段时间可以拿去做别的活。5090 上用 TMA 编出来的 PTX 就是这一套：`mbarrier.init` 建 barrier，`mbarrier.arrive.expect_tx` 登记字节数，`cp.async.bulk.tensor ... mbarrier::complete_tx::bytes` 搬完时把字节数报给 barrier，`mbarrier.try_wait.parity` 等。

{{fig-b-1}}

### C.3 有了 TMA 以后怎么分工 {#tma-ws}

[H100 白皮书](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)说，有了 TMA，只要很少几个 thread 就能用满显存带宽，其余的 thread 可以专心计算。常见的写法叫 warp specialization：把一个 program 的 warp 分成两种，producer warp 只发 TMA，把后面几轮的 block 搬进 slot；consumer warp 只用 Tensor core 算。两边按 slot 交接：每个 slot 配一个 transaction barrier 表示「满了」，TMA 搬完时它放行，consumer 等它再开始算；再配一个只数 arrive 的 barrier 表示「空了」，consumer 算完时 arrive，producer 等它再往这个 slot 搬下一批。这和 [3.3 节](#stages)的流水线是同一件事，只是搬和算交给了不同的 warp，不再是同一个 warp 里先发 load 再算。FlashAttention-3 在 Hopper 上就是这么写的，见 [FlashAttention 1–4](/blog/flashattention-1-to-4/)。

白皮书里还有 thread block cluster：一组保证同时调度、分布在几个 SM 上的 block。transaction barrier 可以在 cluster 里跨 SM 同步，TMA 也能在 cluster 里的 block 之间直接搬数据。这篇用不到，不展开。

---

## References {#references}

1. Gholami, Yao, Kim, Hooper, Mahoney, Keutzer. [AI and Memory Wall](https://arxiv.org/abs/2403.14123). IEEE Micro, 2024. 图 1-1。
2. NVIDIA. [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)（白皮书，[备份](https://web.archive.org/web/20261008150304/https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)）。RTX 5090 的存储层级和 SM 结构，图 1-2、图 1-3。
3. NVIDIA. [NVIDIA Ampere GA102 GPU Architecture](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)（白皮书，[备份](https://web.archive.org/web/20261011221321/https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)）。compute 模式下 L1 和 shared memory 的几种切法（28 KB L1 + 100 KB shared memory 等），第 1.2 节。
4. NVIDIA. [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)（白皮书，[备份](https://web.archive.org/web/20261003180516/https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)）、[A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)（[备份](https://web.archive.org/web/20261018221712/https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)）、[NVIDIA Blackwell Architecture Technical Brief](https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf)（[备份](https://web.archive.org/web/20261003103759/https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf)）、[Blackwell datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf)（[备份](https://web.archive.org/web/20260927220200/https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf)）。表 A-1 的 A100、B200 一列：A100 是 80GB SXM，B200 按 datasheet 里 HGX B200 的单卡值（bf16 取 sparse 值的一半）；H100 一列见第 5 条。
5. NVIDIA. [NVIDIA H100 Tensor Core GPU Architecture](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)（白皮书，[备份](https://web.archive.org/web/20261018221616/https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)）。第 1.2 节和 Appendix C 的 TMA：SM 结构图（Figure 7）里每个 SM 一个，由一个 thread 发起，地址生成和数据搬运由硬件完成（Figure 18、19）；Appendix C 的两种 barrier（Figure 20）和 warp 分工；表 A-1 的 H100 一列（白皮书里的规格对比表，H100 SXM5）。
6. NVIDIA. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)。各 compute capability 每 SM 的 warp、block、register 和 shared memory 上限，表 3-1、表 A-1，CUDA 执行层级。
7. NVIDIA. [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。warp 停顿原因的定义，图 3-4。
8. Austin et al. [How to Scale Your Model](https://jax-ml.github.io/scaling-book/gpus/), Google DeepMind, 2025. Appendix B 的 TPU v5p 规格、systolic array 的数据流和表 B-1。
9. Jouppi et al. [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760). ISCA 2017. 第一代 TPU 和它的 systolic array。
10. Kung. [Why Systolic Architectures?](https://doi.org/10.1109/MC.1982.1653825). IEEE Computer, 1982.
11. Tillet, Kung, Cox. [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://doi.org/10.1145/3315508.3329973). MAPL 2019.
12. [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)。第 2 节 softmax 和 matmul 例子的写法，3.4 节提到的 `@triton.autotune`。
13. Hendrycks, Gimpel. [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415). arXiv:1606.08415, 2016. 第 2.1 节的 tanh 近似。
14. Milakov, Gimelshein. [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867). arXiv:1805.02867, 2018. 第 2.3 节提到的 online softmax。
15. [triton.autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html) 和 [triton.Config](https://triton-lang.org/main/python-api/generated/triton.Config.html)（Triton API 文档）。3.4 节的参数。
16. NVIDIA. [PTX ISA: cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) 和 [cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)。1.1 节和 Appendix C 里 cp.async、commit_group、wait_group 和 TMA、mbarrier 指令的语义。
