---
title: "谁偷走了 5090 的算力和显存：一步 Transformer 训练的 roofline 侦查"
date: 2026-10-04
draft: false
math: true
description: "在 RTX 5090 上实测一步 Transformer 训练的时间和显存，再把 RMSNorm 和 attention 逐个 op 拆开，看每一步算了多少、搬了多少、为反向存了多少。最后都落到 attention 的两个 seq × seq 矩阵上：分数矩阵 S = QKᵀ，和它过 softmax 之后的 P。FlashAttention 一文的前置。"
tags: ["GPU", "Roofline", "Transformer", "显存", "LLM", "AI"]
categories: ["AI笔记"]
series: ["AI笔记"]
series_order: 2
---

我在一张 RTX 5090 上把一步 Transformer 训练拆开，分别量了时间和显存。结果和预想的不太一样：拖慢速度的不是矩阵乘，占显存最多的也不是权重。两边查到最后，都落在 attention 里的两个 seq × seq 矩阵上，一个是分数矩阵 S = QKᵀ（每个 query 对每个 key 的打分），一个是 S 按行过 softmax 之后的注意力权重 P，最后输出是 PV。

文章分四步走：[第 1 节](#basics)准备工具 roofline；[第 2 节](#step)算一步训练的总账，找出时间和显存的大头；[第 3 节](#ops)逐个 op 拆开，看每一步算了多少、读写了多少显存、为反向存了什么，先拿最简单的 RMSNorm 走一遍，再用同样的方法拆 attention；[第 4 节](#savings)试 bf16 和 activation checkpoint 能省多少。怎么把 S、P 彻底去掉，留给下一篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/)。

模型是我自己写的 Transformer LM（RMSNorm、RoPE、SwiGLU，pre-norm），一共五档：small 0.13B、medium 0.42B、large 0.97B、xl 3.41B，以及 10B（实际 12.83B 参数）。

> 环境：RTX 5090 32 GB，torch 2.11.0+cu130。fp32 基准关掉 tf32（`allow_tf32=False`）。除注明外 batch 4、seq 512，预热 5 步、计时 10 步。「full step」指前向、反向加 optimizer 的一整步，「前向 + 反向」不含 optimizer。显存一律 GiB = 2³⁰ B（`max_memory_allocated() / 1024³`）。

---

## 1 Roofline：一把尺子 {#basics}

一个 op 跑多快，取决于它要算多少和要搬多少。

算的量用 FLOPs（浮点运算次数）数。矩阵乘 `[M, K] × [K, N]` 要 2·M·N·K 次（M·N 个输出，每个做 K 次乘加）；逐元素 op（加、乘、exp、mask）每个元素只算一到几十次，比同样大小的矩阵乘少几个数量级。GPU 每秒最多能做的浮点运算次数叫峰值算力，记作 $\pi$，单位 FLOPS。5090 的 fp32 峰值是 $\pi$ = 1.05e14 FLOPS，用 bf16 Tensor core 时翻倍到 2.1e14。

搬的量是 op 读写显存的字节数：输入要从显存读进来，结果要写回去。显存每秒最多能读写的字节数叫带宽，记作 $\beta$，5090 是 $\beta$ = 1.79e12 B/s。一个 op 的耗时不会低于算的时间和搬的时间中较大的那个：

<p align="center">$t \ge \max\left(\dfrac{\mathrm{FLOPs}}{\pi},\ \dfrac{\mathrm{bytes}}{\beta}\right)$</p>

算和搬的比值叫**算术强度** $I = \mathrm{FLOPs} / \mathrm{bytes}$，即每搬 1 字节做多少次运算。把上式改写成「最多能跑到多少 FLOPS」，就是 **roofline**：

<p align="center">$\mathrm{FLOPS}_{\max}(I) = \min(\pi,\ I \cdot \beta)$</p>

在 log-log 坐标上它像一个屋顶（[图 1-1](#fig-1-1)）：左边是斜坡，被带宽限制；右边是平顶，被算力限制；拐角 $I^* = \pi / \beta$ 叫 **ridge point**，5090 fp32 是 58 FLOPs/B。落在拐角左边的 op 是 **memory-bound**，耗时由搬数据决定，减少 FLOPs 没有用；落在右边的是 **compute-bound**，耗时由算力决定。衡量 op 跑得好不好也分两种：compute-bound 的看 MFU（实际 FLOPS / $\pi$），memory-bound 的看 MBU（实际带宽 / $\beta$）。

{{fig-1-1}}

一个 op 在拐角哪一边，用张量形状就能估。逐元素 op 在 fp32 下每个元素算 1 次、读写 8 字节，$I \approx 0.13$，远在拐角左边。矩阵乘的 $I$ 由 M、N、K 里最小的那个决定，fp32 下不超过它的一半。Linear 的三个维度都上千（medium、seq 1024 时 FFN 第一层是 `[4096, 1024] × [1024, 4096]`），$I \approx 340$，在拐角右边；attention 里的 QKᵀ 和 PV 都有一个维度是 d_head（64），$I$ 只有 28，在拐角左边。所以 QKᵀ 和 PV 虽然是矩阵乘，在 attention 里也是 memory-bound。[3.2 节](#attention)会把实测的 op 放到 fp32 的 roofline 上看（[图 3-5](#fig-3-5)）。

---

## 2 总账：时间和显存花在哪 {#step}

### 2.1 时间：矩阵乘没在偷懒 {#time}

Transformer 的 FLOPs 几乎都在 Linear 上。一个 Linear 前向只做一次矩阵乘 $X_L = X_{L-1} W_L$；反向收到误差 $\nabla X_L$ 后要做两次：算参数梯度 $\nabla W_L = X_{L-1}^{\top} \nabla X_L$，再算传给上一层的 $\nabla X_{L-1} = \nabla X_L W_L^{\top}$，每次都和前向一样大（[图 2-1](#fig-2-1)）。

{{fig-2-1}}

每个参数对每个 token 做一次乘加，也就是 2 FLOPs，所以前向每个 token 约 2N FLOPs（N 是参数量），反向 4N，一步一共 6N × token 数，这里是 batch 4 × seq 512 = 2048 个 token。attention 的 QKᵀ 和 PV 没有参数，不在 6N 里，seq 512 时只占 2–4%，可以忽略。实测也是这样，反向耗时差不多是前向的两倍（[图 2-2](#fig-2-2)），因为反向要把 weight grad 和 activation grad 各算一遍。

{{fig-2-2}}

按 6N 算，medium 和 large 的 MFU 只有 31%（实际 3.2e13 FLOPS，fp32 峰值 1.05e14），small 是 27%。矩阵乘本身不慢，单个大矩阵乘能跑到峰值的 64%；问题是所有矩阵乘加起来只占一步 GPU 时间的 60%，剩下 40% 花在几乎不做计算的逐元素 kernel 上，比如 norm、softmax、mask、激活函数。

### 2.2 显存：权重只是小头 {#memory}

fp32 + AdamW 训练时，每个参数要占 16 B：权重 4 B、梯度 4 B、Adam 的 m 和 v 各 4 B，此外还有前向为反向留下的 activation。xl 光这部分就要 50.8 GiB，5090 放不下；10B 在建模型时就 OOM 了。下面用这几个记号：

- <span class="sw" style="background: var(--fig-1)"></span>**W** 全部权重；<span class="sw" style="background: var(--fig-hi)"></span>**G** 全部梯度 `.grad`，大小等于 W；<span class="sw" style="background: var(--fig-mute)"></span>Adam 的 m、v 合计 2W，第一步之后常驻；
- <span class="sw" style="background: var(--fig-2)"></span>**A** 前向为反向存下的张量（saved tensors）；<span class="sw" style="background: var(--fig-3)"></span>**T** 当前层的临时量，算完即释放。

实测 full step 的峰值里只有权重、Adam 状态和 A，没有梯度（[图 2-3](#fig-2-3)）。

{{fig-2-3}}

梯度不在峰值里，是因为 G 和 A 此消彼长。反向走完 j 层（共 L 层）时，显存里是：

<p align="center">$M(j) = W + G \cdot \dfrac{j}{L} + A \cdot \dfrac{L-j}{L} + T$</p>

A 在前向一层层攒起来，反向再一层层释放；G 正好相反，前向时还不存在，`.grad` 要等反向算到那个参数才分配，optimizer step 结束后又被 `zero_grad(set_to_none=True)` 释放。一个涨一个降，M(j) 是一条直线，最高点只可能在两端。full step 里还有一直占着的 Adam 状态，m 和 v 各和权重一样大，共 2W，和权重加起来常驻 3W。所以 full step 的峰值是

<p align="center">$\mathrm{peak}_{\mathrm{full}} \approx 3W + \max(A,\ G)$</p>

拿 large 验算：3 × 3.61 + 16.58 = 27.4 GiB，实测 27.51 GiB，差的 0.1 GiB 就是 T。正常训练 token 多，A 比 G 大，峰值出现在前向刚结束时；token 很少时才反过来，比如 xl 在 seq 128（[图 2-4](#fig-2-4) 左）。

{{fig-2-4}}

总账算下来有两条线索：时间上，40% 花在逐元素 kernel 上；显存上，峰值主要由 <span class="sw" style="background: var(--fig-2)"></span>A 决定，而且四项里只有 A 随 seq 变大。下面把单个 op 拆开来看。

---

## 3 拆解：逐个 op 看 {#ops}

这一节对每个 op 问三件事：算了多少 FLOPs，读写了多少字节显存，为反向存了哪些张量。前两件看张量形状就能算出来；第三件由 autograd 决定，要实测。

PyTorch 的 `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` 就是用来看第三件事的。它是一个上下文管理器：在它里面跑前向时，autograd 每存下一个张量就调用一次 `pack(t)`，保存的是 `pack` 的返回值；反向每用到一个存下的张量就调用一次 `unpack`，收到的就是当初 `pack` 的返回值。下面的两个函数都原样返回张量，只是打印出来，并用 `data_ptr()` 给每块内存编号，编号相同就是同一块内存：

```python
{{code:saved_hooks_demo.py:hooks}}
```

[3.1 节](#rmsnorm)拿最简单的 RMSNorm 把这套方法走一遍，[3.2 节](#attention)再用到 attention 上。

### 3.1 RMSNorm：融合省掉 x̂ {#rmsnorm}

RMSNorm 算的是 $y = w \odot (x \cdot r)$，其中 $r = (\tfrac{1}{d}\sum_j x_j^2 + \epsilon)^{-1/2}$ 每行一个数。eager 模式下它拆成 5 个 op（fp32，`x: [4, 512, 2560]`，一份 x 是 20 MiB）：

```python
rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # ① pow ② mean ③ rsqrt
x_hat = x * rms                                            # ④
y = weight * x_hat                                         # ⑤
```

把上面三行包成函数 `rmsnorm(x, w)`，用 `show(rmsnorm, x, w)` 跑一遍，打印如下。块 A 是 x，B 是 r，C 是 $\hat{x}$，D 是 w：

```
Saving  1  [4, 512, 2560]  float32  块 A
Saving  2  [4, 512, 1]  float32  块 B
Saving  3  [4, 512, 1]  float32  块 B
Saving  4  [4, 512, 2560]  float32  块 A
Saving  5  [4, 512, 2560]  float32  块 C
Saving  6  [2560]  float32  块 D
Loading    块 C
Loading    块 D
Loading    块 B
Loading    块 A
Loading    块 B
Loading    块 A
```

打印的结果说明了 autograd 存张量的规则：**一个 op 的局部偏导里用到谁，前向就存谁；偏导是常数就什么都不存**。存的是引用，不是拷贝，所以本来就在显存里的输入 `x` 和参数 `w` 不额外占显存。6 次 Saving 只落在 4 块内存上，新占显存的只有 $r$（8 KiB）和 $\hat{x}$（20 MiB），见[表 3-1](#tab-3-1) 和[图 3-1](#fig-3-1)。

| op | 反向要的偏导 | 存下 | 新占显存 |
|:--|:--|:--|--:|
| ① $x^2$ | $\partial x^2/\partial x = 2x$ | $x$ | 0 |
| ② $v=\tfrac1d\sum x^2$ | $\partial v/\partial x^2 = \tfrac1d$，常数 | 不存 | 0 |
| ③ $r=(v+\epsilon)^{-1/2}$ | $\partial r/\partial v = -\tfrac12 r^3$ | $r$ | **8 KiB** |
| ④ $\hat{x}=x\cdot r$ | $\partial\hat{x}/\partial x = r$，$\partial\hat{x}/\partial r = x$ | $r$、$x$ | 0 |
| ⑤ $y=w\odot\hat{x}$ | $\partial y/\partial w = \hat{x}$，$\partial y/\partial\hat{x} = w$ | $\hat{x}$、$w$ | **20 MiB** |
{#tab-3-1 caption="**表 3-1** RMSNorm 五个 op 为反向存的张量（eager）" note="④ 存的 $r$、$x$ 和 ③、① 存的是同一块内存，所以不再占显存。新占显存按 fp32 算：$r$ 是 [4, 512, 1]，$\hat{x}$ 是 [4, 512, 2560]。"}

{{fig-3-1}}

时间上，RMSNorm 每个元素只算约 4 次，5 个 op 却要读写约 140 MiB 显存：①、④、⑤ 各读 20 MiB、写 20 MiB，② 读 20 MiB。$I \approx 0.14$，是 memory-bound。显存上，它为反向多存了一份 $\hat{x}$。

$\hat{x}$ 其实不用存：它就是 $x \cdot r$，反向时用 $x$ 和 $r$ 重算一次就有。eager 模式做不到，因为每个 op 单独执行，⑤ 的反向只知道自己需要 $\hat{x}$，不知道它能由 $x$ 和 $r$ 算出来。用 `torch.compile` 把 ①–⑤ 编译成一个前向 kernel、反向编成 3 个 kernel 后，只存 $x$、$w$、$r$：

```
Saving  1  [4,512,2560]  grad_fn=None  ptr=…3c80   # x
Saving  2  [2560]        grad_fn=None  ptr=…b3c0   # w
Saving  3  [4,512,1]     grad_fn=None  ptr=…f9c0   # r
Loading    1 → 2 → 3（与 Saving 同序）
```

反向要算 $\nabla x = r\,\big(g - \hat{x} \cdot \mathrm{mean}(g \odot \hat{x})\big)$，其中 $g = w \odot \nabla y$。式子里的 $\hat{x}$ 都在反向 kernel 里用 $x \cdot r$ 现算（[图 3-2](#fig-3-2)），前后对比见[表 3-2](#tab-3-2)。

{{fig-3-2}}

| | eager | 融合后 |
|:--|:--|:--|
| 前向 kernel | 6 个（③ 的 +ε 和 rsqrt 各一个） | 1 个 |
| 反向 kernel | 13 个 | 3 个（dx 1 个，dw 两段归约 2 个） |
| 为反向存的张量 | $x$、$w$、$r$、$\hat{x}$ | $x$、$w$、$r$ |
| 新占显存 | $r + \hat{x}$ ≈ 20 MiB | $r$ ≈ 8 KiB |
| 前向读写显存 | ~140 MiB | ~40 MiB（只读 $x$、写 $y$） |
| FLOPs / 元素 | 前向 ~4 | 前向 ~4，反向多 1（重算 $\hat{x} = x\cdot r$） |
| 前向算术强度 $I$ | ~4 / 28 B ≈ 0.14 | ~4 / 8 B ≈ 0.5 |
{#tab-3-2 caption="**表 3-2** RMSNorm：eager 与 `torch.compile` 融合" note="kernel 数和存的张量是实测（`torch.profiler`，不含 memset、拷贝和 `.grad` 累加），FLOPs 和读写是纸面计数。I 按每个元素算：eager 每元素读写 7 次 × 4 B = 28 B（共 ~140 MiB），融合后只读 x、写 y，8 B。两者都远低于 ridge point 58，仍是 memory-bound。"}

融合没有让 RMSNorm 变成 compute-bound。把两种写法放到 fp32 roofline 上（[图 3-3](#fig-3-3)），I 从 0.14 移到 0.5，两个点都贴着带宽斜线，MBU 都在 80% 左右；要读写的字节少了 3.5 倍，前向耗时也从 816 µs 降到 236 µs，正好快 3.5 倍。

{{fig-3-3}}

融合版也可以手写成 Triton（下面折叠的代码，eager 版就是本节开头那三行），思路和 `torch.compile` 一样：前向一个 kernel，一个 program 算一行，①–⑤ 都在寄存器里做完，只写出 y 和 r。反向比 `torch.compile` 少两个 kernel：dx 是按行归约，dw 却要把所有行加起来，`torch.compile` 为 dw 单独拆了两段归约；手写版让每个 program 读回 x、w、r，现场重算 x̂，算完自己这一行的 dx，再用 `atomic_add` 把这一行对 dw 的贡献直接累加上去，反向就只有一个 kernel。

<details class="fold">
<summary>Triton 融合：前向 1 个 kernel，反向 1 个 kernel</summary>

```python
{{code:rmsnorm_triton.py:triton}}
```

</details>

> 融合解决了两件事：几个 op 合成一个 kernel，中间结果不再进出显存；能重算的张量不存，反向时再算。[3.2 节](#attention)的 attention 和 [4.2 节](#checkpoint)的 checkpoint 都会再遇到这两件事。

### 3.2 Attention：都在搬 S 和 P {#attention}

一层 attention 的完整公式是

<p align="center">$O = \mathrm{softmax}\!\left(\dfrac{QK^{\top}}{\sqrt{d}} + M\right) V$</p>

$Q$、$K$、$V$ 的形状都是 `[b, h, seq, d]`（d 是 d_head），$M$ 是 causal mask（未来位置为 $-\infty$）。eager 模式下它拆成 5 步：① 算分数 $S = QK^{\top}$；② 除以 $\sqrt{d}$；③ 加 mask；④ 按行 softmax 得到 $P$；⑤ $O = PV$。S、P 的形状都是 `[b, h, seq, seq]`，O 和 Q 一样大。

和 RMSNorm 一样，逐步看它算了多少、读写了多少显存、为反向存了什么（[图 3-4](#fig-3-4)，medium、seq 1024：b = 4，h = 16，d = 64）。一份 S 或 P 是 4 × 16 × 1024 × 1024 × 4 B = 256 MiB，而 Q、K、V、O 各只有 16 MiB。

eager 的写法如下，softmax 按公式拆成 5 个 kernel：

```python
{{code:saved_hooks_demo.py:attention}}
```

用 `show` 跑一遍，打印如下。块 A 是 K（转置后的 view），B 是 Q，C 是 mask，D 是每行 max 的下标，E 是 e = exp(S − m)，F 是行和 Σ，G 是 V，H 是 P：

```
Saving  1  [64, 64, 1024]  float32  块 A
Saving  2  [64, 1024, 64]  float32  块 B
Saving  3  [1024, 1024]  bool  块 C
Saving  4  [4, 16, 1024, 1]  int64  块 D
Saving  5  [4, 16, 1024, 1024]  float32  块 E
Saving  6  [4, 16, 1024, 1]  float32  块 F
Saving  7  [4, 16, 1024, 1024]  float32  块 E
Saving  8  [64, 1024, 64]  float32  块 G
Saving  9  [64, 1024, 1024]  float32  块 H
Loading    块 G
Loading    块 H
Loading    块 F
Loading    块 E
Loading    块 E
Loading    块 D
Loading    块 C
Loading    块 A
Loading    块 B
```

按 RMSNorm 那条规则逐个 op 对一遍（[表 3-3](#tab-3-3)）：9 次 Saving 里，Q、K、V、mask 本来就在显存里，只是引用；softmax 的 5 个 kernel 里，max 只把梯度传给最大值所在的位置，存下每行最大值的下标；exp 的导数就是它自己的输出，除法要用分子和分母，所以存下 e 和 Σ；⑤ 要用 P 和 V。新占显存的是 e 和 P 两个 seq × seq 张量，各 256 MiB，加起来是 Q、K、V、O 总和的 8 倍。

| op | 反向要的偏导 | 存下 | 新占显存 |
|:--|:--|:--|--:|
| ① $S = QK^{\top}$ | $\partial S/\partial Q = K$，$\partial S/\partial K = Q$ | $Q$、$K$ | 0 |
| ② $\div\sqrt{d}$ | $1/\sqrt{d}$，常数 | 不存 | 0 |
| ③ $+M$ | 被 mask 的位置梯度为 0 | mask | 0 |
| ④ max | 只有最大值的位置有梯度 | 下标 | 0.5 MiB |
| ④ $e = \exp(S - m)$ | $\partial e/\partial S = e$ | $e$ | **256 MiB** |
| ④ $P = e / \Sigma$ | $\partial P/\partial e = 1/\Sigma$，$\partial P/\partial \Sigma = -e/\Sigma^2$ | $e$、$\Sigma$ | 0.25 MiB |
| ⑤ $O = PV$ | $\partial O/\partial P = V$，$\partial O/\partial V = P$ | $P$、$V$ | **256 MiB** |
{#tab-3-3 caption="**表 3-3** attention 各 op 为反向存的张量（eager，medium，seq 1024）" note="存下的张量是实测。④ 的减 max 和求和偏导是常数，不存；除法存的 e 和 exp 存的是同一块内存。"}

{{fig-3-4}}

时间上，FLOPs 集中在 ① 和 ⑤ 两个矩阵乘上，读写却集中在 ② ③ ④ 上：② 和 ③ 各把整个 256 MiB 的 S 读一遍、写一遍，④ 读写得更多。放到 roofline 上（[图 3-5](#fig-3-5)），attention 的 op 全在斜坡上，包括 QKᵀ 和 PV 这两个矩阵乘。它们的 MBU 已经有 57–86%，kernel 本身没多少优化空间，要更快只能少搬数据。

{{fig-3-5}}

搬得最多的是 ④ softmax。它对 S 的每一行算 $P_{ij} = e^{S_{ij} - m_i} / \sum_k e^{S_{ik} - m_i}$，其中 $m_i = \max_k S_{ik}$，减掉行最大值是为了防止 exp 溢出。eager 模式下这个公式拆成 5 个 kernel：求 max、减 max、exp、求和、除。每个 kernel 都要读或写和 S 一样大的张量，5 个加起来一共读写 8 次，2048 MiB（图 3-4 里 max 到 ÷Σ 这 5 个 kernel 进出显存的箭头）。

FLOPs 和耗时因此对不上：softmax 的运算量只有 PV 的 1/5，耗时却是 PV 的 6 倍（[图 3-6](#fig-3-6)）。

{{fig-3-6}}

S、P 的读写量随 seq² 增长，Linear 只随 seq 线性增长。seq 从 256 增加到 1024，attention 占前向时间的比例从 10% 涨到 46%，多出来的几乎全是 softmax、除以 √d、mask 这类只搬数据的 op（[图 3-7](#fig-3-7)）。这就是[第 2 节](#time)那 40% 里随 seq 涨得最快的部分。

{{fig-3-7}}

要少搬，就得把几步合进一个 kernel，中间结果留在片上：融合的 softmax 只读一次 S、写一次 P；FlashAttention 更进一步，S、P 根本不写回显存。

显存上，[表 3-3](#tab-3-3) 是一层 attention 单独测的：新占显存的主要是两个 seq × seq 张量。放到整层 Transformer block 上也是这样，只是 `torch.compile` 之后存下的两个 seq × seq 张量换成了 S 和 P。xl 的一层（RMSNorm 这类中间量已经被省掉）一共要为反向存 3655 MiB，其中一半以上是 S 和 P（[图 3-8](#fig-3-8)）。这组测量用的是 16 头，S、P 各 1 GiB；标准 xl 是 32 头，S、P 还要再大一倍。

{{fig-3-8}}

xl 有 32 层，按 16 头算加起来也有 114 GiB，是 5090 显存的三倍多。而且只有 S、P 随 seq² 增长：32 头时，同一个 `[b, h, s, s]` 张量在 seq 128 时是 8 MiB，seq 2048 时是 2 GiB，是残差流上一个 `[b, s, d]` 张量的 25 倍。

[图 3-9](#fig-3-9) 是 xl 一步的显存时间线。seq 2048 只跑前向时，每层 attention 都让显存冲高约 8 GiB，算完再落回去，32 层都能跑完；加上反向后，每层的 S、P 都得留下，第 1 层就多占约 4.7 GiB，到第 2 层就 OOM 了。

{{fig-3-9}}

---

## 4 优化：bf16 和 checkpoint 都差一口气 {#savings}

要减小 A 有两种办法：把每个张量存得小一点（bf16），或者少存一些、反向时重算（checkpoint）。

### 4.1 bf16：快了，省得不多 {#bf16}

autocast 只把矩阵乘的输入换成 bf16，权重、梯度和 Adam 状态还是 fp32。速度提升很明显，前向快了 1.9–2.3 倍（[图 4-1](#fig-4-1)）：矩阵乘换到 bf16 的 Tensor core 上，峰值从 1.05e14 翻倍到 2.1e14，要搬的字节也少了一半。显存只省了 18–21%：W、G 和 Adam 状态大小不变；A 也没有减半，因为 norm、softmax、残差和 loss 还在 fp32 下算（这些累加在 bf16 下不准，bf16 只有 7 位尾数，把 0.01 累加 1000 次只能得到 4.0），反向还要多存一份 bf16 的权重副本。存下的张量还是那些，只是一部分从 4 字节变成了 2 字节。

{{fig-4-1}}

### 4.2 Checkpoint：省显存，多一遍前向 {#checkpoint}

checkpoint 和 [3.1 节](#rmsnorm)里融合 RMSNorm 的做法一样，只是从一个 op 扩大到几层：前向只存每段的入口（entry，xl、seq 2048 时 80 MiB），反向走到这一段时，用入口把这段的前向重跑一遍，用完就释放。4 层 xl block 每 2 层设一个 checkpoint，峰值就从 4 × 3655 MiB = 14.6 GiB 降到「2 个 entry 加一段」的 7.5 GiB（[图 4-2](#fig-4-2)）。

{{fig-4-2}}

代价是整个网络要多跑一遍前向。xl 在 seq 2048 下光参数加梯度就要 25.4 GiB，放不下 activation，所以我在 large 上扫了每段放几层（[图 4-3](#fig-4-3)）。不管怎么切，step 都是 302–313 ms，比不用 checkpoint 的 236 ms 慢 28–33%，正好对应一步从 3F 变成 4F（F 是一次前向，反向约 2F）。显存则是切得越细越省，每层一个 checkpoint 时最低，7.8 GiB，不用时是 15.0 GiB。

{{fig-4-3}}

切得越细越省，是因为入口很小。设共 $L$ 层、每段 $e$ 层、入口大小 $a$、一层的 A 为 $A_1$，峰值约为 $\frac{L}{e}a + e\,A_1$。只要所有入口加起来比一层的 A 小（这里 36 × 5 MiB = 180 MiB < 220 MiB），$e = 1$ 就是最优。

但 checkpoint 解决不了 S、P。重算到某一层时，这一层的 S、P 照样要完整写进显存再读出来，seq 2048 时每层约 8 GiB 的峰值还在。

---

## 5 小结：问题都在 S、P {#conclusion}

时间和显存的问题最后都落在 S、P 上。时间上，它们让 attention 成了 memory-bound，seq 1024 时占前向将近一半的时间；显存上，它们占一层 saved tensors 的一半以上，xl 在 seq 2048 时第 2 层就 OOM。bf16 只能把它们存小一点，checkpoint 只能推迟它们出现，都没能让它们离开显存。

要解决，得把 RMSNorm 的做法用到 attention 上：把 QKᵀ、softmax、PV 合进一个 kernel，分块在片上算完，S、P 不写回显存，反向需要时再重算。下一篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/) 讲的就是这个。
