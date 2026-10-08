---
title: "GPU 训练分析：FLOPs、Roofline 与显存峰值（RTX 5090 实测）"
date: 2026-10-08
draft: false
math: true
description: "FLOPs 与 FLOPS、算术强度、roofline、MFU 这些基础概念怎么算；前向和反向各算了多少，什么是 memory bound，显存峰值落在哪一刻——全部在一张 RTX 5090 上实测。FlashAttention 一文的前置。"
tags: ["GPU", "Roofline", "Transformer", "显存", "LLM", "AI"]
categories: ["AI笔记"]
series: ["AI笔记"]
series_order: 2
---

训练一步 Transformer，GPU 的时间和显存花在了哪里？这篇先讲清 FLOPs、算术强度、roofline、MFU 这几个基础概念，再在一张 RTX 5090 上把一步训练拆开实测：前向和反向各算了多少，哪些 op 受算力限制、哪些受带宽限制，显存峰值落在哪一刻。几条线最后都指向同一个东西——attention 的 seq × seq 分数矩阵，这也是下一篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/) 的出发点。

> 环境：RTX 5090 32 GB，torch 2.11.0+cu130；模型是自己写的 Transformer LM（RMSNorm + RoPE + SwiGLU，pre-norm）。fp32 基准关掉 tf32（`allow_tf32=False`，否则 fp32 矩阵乘会悄悄走 Tensor core）；除注明外 batch 4、seq 512，预热 5 步、计时 10 步；显存一律 GiB = 2³⁰ B（`max_memory_allocated() / 1024³`）。

## 省流不看

1. **速度上限看 roofline：耗时 ≥ max(FLOPs / 峰值算力, bytes / 带宽)**（§1）：5090 的 fp32 峰值 1.05e14 FLOPS、带宽 1.79e12 B/s，两者之比 ≈ 60 FLOPs/B；每搬 1 字节算不到 60 次的 op 受带宽限制，快慢和 FLOPs 无关。
2. **训练一步 ≈ 6 × 参数量 × token 数 FLOPs**（§2）：前向每个参数做一次乘加（2N），反向对输入、对权重各做一次同样大的矩阵乘（4N）；实测反向 ≈ 2.0× 前向，整步只跑到峰值的 31%。
3. **attention 是 memory-bound 的**（§3）：除了 Linear，attention 里的 op 全贴着带宽斜线，连 QKᵀ、PV 两个矩阵乘也是；softmax 的 FLOPs 只有 PV 的 1/5，耗时却是它的 6×。
4. **显存峰值 = 🟦 权重 + max(🟩 activation, 🟥 梯度)**（§4）：梯度在反向才逐层长出、activation 同时逐层释放，两者不会同时满；但 3.41B 的 xl 在 seq 2048 时，attention 每层冒出一串 2 GiB 的分数矩阵，第 2 层就 OOM。
5. **bf16、checkpoint 都只能缓解**（§5）：bf16 快 ~2×、省 ~20%；checkpoint 多算一遍前向（+30%）换来显存减半，但 S、P 重算时照样要落显存——根治要改 kernel，见 [FlashAttention 1–4](/blog/flashattention-1-to-4/)。

---

## 1 基础概念：FLOPs、算术强度、roofline、MFU

**表 1-1** RTX 5090 的硬件上限，以及本文实测能跑到多少

| 量 | 规格 | 本文实测 |
|:--|--:|:--|
| 显存容量 | 32 GB | PyTorch 可用 31.3 GiB |
| 显存带宽 B | 1.79e12 B/s | memory-bound 的 op 跑到 57–86%（§3） |
| fp32 峰值算力 P（CUDA core） | 1.05e14 FLOPS | 单个大矩阵乘 64%，整步训练 31%（§2、§3） |
| bf16 峰值算力（Tensor core） | 2.1e14 FLOPS | bf16 前向比 fp32 快 1.9–2.3×（§5） |
| ridge point P / B | ≈ 60 FLOPs/B（fp32） | 低于它的 op 受带宽限制（§1.2） |

CUDA core 是通用的标量乘加单元，什么都能算；Tensor core 只做小矩阵块的乘加、只收 16 位 / tf32 / fp8 输入，吞吐高得多。纯 fp32 矩阵乘走不了 Tensor core，所以本文 fp32 基准的算力上限是 1.05e14。

### 1.1 FLOPs 与 FLOPS

- **FLOPs**（floating-point operations）是浮点运算的**次数**，**FLOPS**（per second）是**速率**。本文都用 10 的幂写：seq 1024 的一次 PV 矩阵乘是 8.6e9 FLOPs，5090 的 fp32 峰值是 1.05e14 FLOPS。
- **矩阵乘** `[M, K] × [K, N]`：M·N 个输出，每个做 K 次乘加，一次乘加记 2 FLOPs，共 **2·M·N·K**。
- **逐元素 op**（加、乘、exp、mask）：每个元素一到几十次运算（exp 按 ~20 次计），总量 = 元素数 × 每元素次数，比同尺寸的矩阵乘小几个数量级。

### 1.2 算术强度与 roofline

一个 op 在 GPU 上要做两件事：在计算单元上算（FLOPs），从显存读输入、写输出（bytes）。两件事各有上限，所以耗时有两个下限，实际取较大的那个：

<p align="center">$t \ge \max\left(\dfrac{\mathrm{FLOPs}}{P},\ \dfrac{\mathrm{bytes}}{B}\right)$</p>

FLOPs 与 bytes 之比叫**算术强度**（arithmetic intensity）$I = \mathrm{FLOPs} / \mathrm{bytes}$，即每搬 1 字节做几次运算。把上式改写成「最快能跑多少 FLOPS」，就是 **roofline 模型**：

<p align="center">$\mathrm{FLOPS}_{\max}(I) = \min(P,\ I \cdot B)$</p>

在 log-log 坐标上它是一条斜线接一条平线，像屋顶（图 1-1）。拐点 $I^* = P / B$ 叫 **ridge point**，5090 fp32 是 1.05e14 / 1.79e12 ≈ 60 FLOPs/B：

- **I < 60：memory-bound**。点落在斜线下，耗时 ≈ bytes / B，和 FLOPs 无关；想更快只能少搬字节。
- **I > 60：compute-bound**。点落在平线下，耗时 ≈ FLOPs / P；想更快只能少算，或换更快的单元（Tensor core）。

估 I 有两条捷径：

- **逐元素 op**：每个元素 1 次运算，fp32 读 4 B、写 4 B，I ≈ 0.13，永远是 memory-bound。
- **矩阵乘**：fp32 下 $I = \dfrac{2MNK}{4(MK + KN + MN)}$，输出远大于输入时 ≈ K/2——**内维 K 决定一切**。Linear 的 K 是 d_model（1024），I ≈ 340；attention 里 QKᵀ 的 K 是 d_head（64），I ≈ 28，同样是矩阵乘，却落在 ridge 左边。

<figure class="rf-fig">
<svg viewBox="0 0 640 372" width="100%" role="img" aria-label="RTX 5090 的 roofline：横轴算术强度，纵轴可达算力，带宽斜线与 fp32 平线交于约 60 FLOPs/B；attention 的 op 都贴着斜线，只有 Linear 在平线下">
  <style>
    .rf-grid { stroke: currentColor; stroke-opacity: .12; stroke-width: 1; }
    .rf-axis { stroke: currentColor; stroke-opacity: .45; stroke-width: 1; }
    .rf-tick { font-size: 11px; fill: currentColor; opacity: .7; }
    .rf-atitle { font-size: 12px; fill: currentColor; opacity: .8; }
    .rf-roof { fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    .rf-roof16 { fill: none; stroke: currentColor; stroke-opacity: .5; stroke-width: 1.5; stroke-dasharray: 6 4; }
    .rf-ridge { stroke: currentColor; stroke-opacity: .3; stroke-width: 1; }
    .rf-lab { font-size: 12px; fill: currentColor; }
    .rf-sub { font-size: 11px; fill: currentColor; opacity: .6; }
    .rf-pt { fill: rgb(var(--color-primary-500)); stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .rf-hit { fill: transparent; }
    .rf-g:hover .rf-pt, .rf-g:focus .rf-pt { stroke-width: 3; }
    @media (max-width: 640px) { .rf-fig { overflow-x: auto; } .rf-fig > svg { min-width: 520px; } }
  </style>
  <text class="rf-tick" x="70.0" y="337" text-anchor="middle">0.1</text>
  <line class="rf-grid" x1="207.5" y1="20" x2="207.5" y2="320"/>
  <text class="rf-tick" x="207.5" y="337" text-anchor="middle">1</text>
  <line class="rf-grid" x1="345.0" y1="20" x2="345.0" y2="320"/>
  <text class="rf-tick" x="345.0" y="337" text-anchor="middle">10</text>
  <line class="rf-grid" x1="482.5" y1="20" x2="482.5" y2="320"/>
  <text class="rf-tick" x="482.5" y="337" text-anchor="middle">100</text>
  <line class="rf-grid" x1="620.0" y1="20" x2="620.0" y2="320"/>
  <text class="rf-tick" x="620.0" y="337" text-anchor="middle">1000</text>
  <text class="rf-tick" x="62" y="324.0" text-anchor="end">1e11</text>
  <line class="rf-grid" x1="70" y1="236.7" x2="620" y2="236.7"/>
  <text class="rf-tick" x="62" y="240.7" text-anchor="end">1e12</text>
  <line class="rf-grid" x1="70" y1="153.3" x2="620" y2="153.3"/>
  <text class="rf-tick" x="62" y="157.3" text-anchor="end">1e13</text>
  <line class="rf-grid" x1="70" y1="70.0" x2="620" y2="70.0"/>
  <text class="rf-tick" x="62" y="74.0" text-anchor="end">1e14</text>
  <line class="rf-axis" x1="70" y1="320" x2="620" y2="320"/>
  <line class="rf-axis" x1="70" y1="20" x2="70" y2="320"/>
  <text class="rf-atitle" x="345" y="362" text-anchor="middle">算术强度 I（FLOPs/B，对数轴）</text>
  <text class="rf-atitle" transform="translate(16 170) rotate(-90)" text-anchor="middle">可达算力（FLOPS，对数轴）</text>
  <line class="rf-ridge" x1="450.6" y1="68.2" x2="450.6" y2="320"/>
  <text class="rf-sub" transform="translate(463.6 312) rotate(-90)">ridge point ≈ 60</text>
  <polyline class="rf-roof16" points="450.6,68.2 492.0,43.1 620,43.1"/>
  <polyline class="rf-roof" points="70,298.9 450.6,68.2 620,68.2"/>
  <text class="rf-sub" x="620" y="36.1" text-anchor="end">bf16 Tensor core 2.1e14 FLOPS</text>
  <text class="rf-lab" x="620" y="61.2" text-anchor="end">fp32 1.05e14 FLOPS</text>
  <text class="rf-lab" transform="translate(250 181.8) rotate(-31.2)" text-anchor="middle">带宽 1.79e12 B/s</text>
  <text class="rf-sub" x="332" y="306" text-anchor="middle">memory-bound</text>
  <text class="rf-sub" x="548" y="306" text-anchor="middle">compute-bound</text>
  <g class="rf-g" tabindex="0"><title>Linear（FFN w1）：I = 341 FLOPs/B，实测 6.87e13 FLOPS，正上方屋顶 1.05e14，MFU 64%</title><circle class="rf-hit" cx="555.8" cy="83.6" r="12"/><circle class="rf-pt" cx="555.8" cy="83.6" r="5"/></g>
  <text class="rf-lab" x="545.8" y="101.6" text-anchor="end">Linear</text>
  <g class="rf-g" tabindex="0"><title>S = QKᵀ：I = 28.4 FLOPs/B，实测 2.86e13 FLOPS，正上方屋顶 5.09e13，MBU 57%</title><circle class="rf-hit" cx="407.4" cy="115.3" r="12"/><circle class="rf-pt" cx="407.4" cy="115.3" r="5"/></g>
  <text class="rf-lab" x="417.4" y="129.3" text-anchor="start">QKᵀ</text>
  <g class="rf-g" tabindex="0"><title>O = PV：I = 30.1 FLOPs/B，实测 3.73e13 FLOPS，正上方屋顶 5.39e13，MBU 70%</title><circle class="rf-hit" cx="410.8" cy="105.6" r="12"/><circle class="rf-pt" cx="410.8" cy="105.6" r="5"/></g>
  <text class="rf-lab" x="420.8" y="103.6" text-anchor="start">PV</text>
  <g class="rf-g" tabindex="0"><title>softmax（5 个 kernel）：I = 0.844 FLOPs/B，实测 1.27e12 FLOPS，正上方屋顶 1.51e12，MBU 84%</title><circle class="rf-hit" cx="197.4" cy="228.1" r="12"/><circle class="rf-pt" cx="197.4" cy="228.1" r="5"/></g>
  <text class="rf-lab" x="207.4" y="244.1" text-anchor="start">softmax</text>
  <g class="rf-g" tabindex="0"><title>S / √d：I = 0.125 FLOPs/B，实测 1.92e11 FLOPS，正上方屋顶 2.24e11，MBU 86%</title><circle class="rf-hit" cx="83.3" cy="296.4" r="12"/><circle class="rf-pt" cx="83.3" cy="296.4" r="5"/></g>
  <text class="rf-lab" x="93.3" y="308.4" text-anchor="start">S / √d</text>
</svg>
</figure>

**图 1-1** RTX 5090 的 roofline（log-log）。实线是 fp32 的两个屋顶：斜线 = 带宽 1.79e12 B/s，平线 = CUDA core 峰值 1.05e14 FLOPS；虚线是 bf16 Tensor core 的屋顶 2.1e14。点是 §3 表 3-1 实测的 op（medium，seq 1024，一层），纵坐标 = FLOPs / 实测耗时，悬停可看数值；causal mask 的 FLOPs 为 0，画不上。

### 1.3 MFU 与 MBU

- **MFU**（model FLOPs utilization）= 实际 FLOPS / 峰值 FLOPS，衡量 compute-bound 的代码离平线多远。整步训练的 MFU = 每步模型 FLOPs（§2 的 6N × token 数）/ step 时间 / P。
- **MBU**（memory bandwidth utilization）= 实际带宽 / 峰值带宽，衡量 memory-bound 的 op 离斜线多远。

图 1-1 里每个点到它正上方屋顶的距离就是它的利用率。memory-bound 的 op MFU 必然很低（softmax 只有 ~1%），这不说明它写得差，要看 MBU（84%）。

---

## 2 前向与反向：FLOPs 怎么数，实际跑多快

**表 2-1** 五档模型规格（vocab 10000，d_head 64，10B 为 128；参数量 N 在 meta device 上数）

| Size | d_model | d_ff | 层数 L | 头数 | N |
|:--|--:|--:|--:|--:|--:|
| small  |  768 |  3072 | 12 | 12 |  0.13B |
| medium | 1024 |  4096 | 24 | 16 |  0.42B |
| large  | 1280 |  5120 | 36 | 20 |  0.97B |
| xl     | 2560 | 10240 | 32 | 32 |  3.41B |
| 10B    | 4608 | 12288 | 50 | 36 | 12.83B |

Transformer 的 FLOPs 几乎全在 Linear 的矩阵乘里，按 §1.1 的 2·M·N·K 数：

- **前向 ≈ 2N / token**：一个 token 穿过每个权重矩阵时，每个参数恰好做一次乘加（embedding 查表除外）。attention 的 QKᵀ、PV 不含参数，另加 4·L·seq·d_model，seq 512 时只占 2–4%。
- **反向 ≈ 4N / token**：每个 Linear `y = x·Wᵀ` 要算两个梯度——`dx = dy·W` 传给前一层，`dW = dyᵀ·x` 交给 optimizer——各是一次和前向同样大的矩阵乘。
- **训练一步 ≈ 6N × token 数**，这里 token 数 = batch × seq = 2048。

nsys 里能直接数出这三组矩阵乘：

**表 2-2** medium 一步训练里的 GEMM kernel（nsys；24 层 × 7 个 Linear + lm_head = 169）

| kernel | 每步次数 | 算的是 |
|:--|--:|:--|
| `sgemm_128x256_tn` + `sgemm_256x128_tn` | 73 + 96 = 169 | 前向 `y = x·Wᵀ` |
| `sgemm_256x128_nn` | 169 | 反向 `dx = dy·W` |
| `sgemm_128x128_nt` + `sgemm_128x64_nt` | 73 + 96 = 169 | 反向 `dW = dyᵀ·x` |

`sgemm` 是 fp32 矩阵乘，`128x256` 是每个 thread block 负责的输出分块，`tn` / `nn` / `nt` 是两个输入是否转置；73 / 96 是 d_model 宽和 d_ff 宽的矩阵分到了不同的分块。

**表 2-3** 实测各阶段耗时与 MFU（fp32，batch 4 seq 512；MFU = 训练 FLOPs / step ÷ full step 时间 ÷ 1.05e14）

| Size | 训练 FLOPs / step | 前向 (ms) | 反向 (ms) | 反向 / 前向 | optimizer (ms) | full step (ms) | MFU |
|:--|--:|--:|--:|--:|--:|--:|--:|
| small  | 1.6e12 |  17.2 |  34.8 | 2.02 |  3.7 |  55.7 | 27% |
| medium | 5.4e12 |  51.1 | 103.3 | 2.02 | 12.9 | 167.4 | 31% |
| large  | 1.2e13 | 118.1 | 227.8 | 1.93 | 26.9 | 372.8 | 31% |

xl 带图前向就 OOM（§4），10B 建模型就 OOM。计时前必须预热：第一步多出 ~300 ms 的一次性开销（kernel 懒加载、cuBLAS 初始化、显存池首次 cudaMalloc），不预热的话 10 步均值虚高 7–59%。

- **反向为什么是前向的 2 倍？** 表 2-2：反向的矩阵乘次数正好是前向的两倍、尺寸相同；表 2-3 实测 1.93–2.02×。optimizer（AdamW）几乎没有 FLOPs，是对每个参数读写权重、梯度和 m、v 的逐元素更新——纯 memory-bound，占一步的 7%。
- **5090 实际跑到多少？** 3.2e13 FLOPS，fp32 峰值的 31%。单个大矩阵乘能到 64%（§3），但整步里矩阵乘只占 GPU 时间的 60%，剩下几乎全是 FLOPs 很少的逐元素 kernel——它们为什么这么费时间，是 §3 的内容。
- **训一个模型要多久？** 每个模型训 20N token（Chinchilla 配比），按实测 step 时间（每步 2048 token）折算：small 18 h、medium 7.8 天、large 41 天；换 bf16（§5）是 12 h、4.5 天、21 天。单卡 5090 认真训的上限大约是 medium（0.42B）。

---

## 3 Memory bound：attention 慢在哪

把 medium@1024 一层里的 op 逐个放到 roofline 上，就是图 1-1 的那些点：

**表 3-1** 逐 op 的两个下限与实测（medium，seq 1024，一层；S、P 为 `[4, 16, 1024, 1024]` fp32 = 256 MiB；Linear 取 FFN 的 w1，`[4096, 1024] × [1024, 4096]`；实测 = nsys 里对应 kernel 的 GPU 时间，24 层平均；粗体是瓶颈）

| op | FLOPs | 读写 bytes | I (FLOPs/B) | FLOPs / P (ms) | bytes / B (ms) | 实测 (ms) | 利用率 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Linear（参照） | 3.4e10 | 96 MiB | 340 | **0.32** | 0.06 | ≈ 0.5 | MFU 64% |
| S = QKᵀ | 8.6e9 | 288 MiB | 28 | 0.08 | **0.17** | 0.30 | MBU 57% |
| S / √d | 6.7e7 | 512 MiB | 0.13 | 0.0006 | **0.30** | 0.35 | MBU 86% |
| causal mask | 0 | 512 MiB | 0 | 0 | **0.30** | 0.40 | MBU 75% |
| softmax（5 个 kernel） | 1.8e9 | 2 GiB | 0.85 | 0.02 | **1.2** | 1.43 | MBU 84% |
| O = PV | 8.6e9 | 272 MiB | 30 | 0.08 | **0.16** | 0.23 | MBU 70% |

- **哪些 op 是 compute-bound？** 只有 Linear。attention 里全是 memory-bound，连两个矩阵乘也是——内维 d_head = 64，只有 Linear 的 1/16（§1.2）。
- **还能更快吗？** 利用率已经 57–86%，贴着屋顶了。要更快只能把下限本身压下去：少搬字节。
- **softmax 为什么最贵？** eager 下它是 5 个 kernel（求 max、减 max、exp、求和、除），每个都把 S 大小的张量完整读写一遍，合计 8 遍、2 GiB；`/√d` 和 mask 各 2 遍。

**表 3-2** attention 三段占 forward GPU 时间（medium，24 层合计，按 kernel 名归因）

| | seq 256 | seq 512 | seq 1024 |
|:--|--:|--:|--:|
| scores（QKᵀ + /√d + mask） | 0.9 ms (4%) | 3.6 ms (8%) | 25.1 ms (18%) |
| softmax | 1.0 ms (4%) | 5.0 ms (11%) | 34.3 ms (24%) |
| PV | 0.4 ms (2%) | 1.6 ms (3%) | 5.5 ms (4%) |
| 其余（投影、FFN、norm、残差） | 21.1 ms (90%) | 37.2 ms (78%) | 76.8 ms (54%) |

- **FLOPs 和时间对得上吗？** 对不上。seq 1024 时 softmax 用 1.8e9 FLOPs 花了 34.3 ms，PV 用 8.6e9 FLOPs 只花 5.5 ms——FLOPs 少 5 倍，反而慢 6 倍。
- **seq 变长呢？** attention 的 bytes ∝ seq²，Linear ∝ seq，所以三段合计从 10% 涨到 46%，增量几乎全在那些不怎么算数的逐元素 kernel 上。

> **GPU 时间 = 张量被搬了几遍，不是 FLOPs。** 对 memory-bound 的 op，唯一的办法是融合：把几步放进一个 kernel，中间结果留在片上不回显存。fused softmax 把 8 遍降到 2 遍；FlashAttention 更进一步，让 S、P 根本不写回显存。

---

## 4 显存峰值：W + max(A, G)

**纸面账**：fp32 + AdamW 训练，每个参数常驻 16 B——权重 4、梯度 4、Adam 的 m 和 v 各 4。此外还有 activation（为反向存下的中间张量），正比于 batch × seq。xl（3.41B）光这 16 B/参数就是 50.8 GiB，5090 装不下；10B 建模型就 OOM。记号（颜色与图 4-1 一致）：

- 🟦 **W** 全部权重；🟥 **G** 全部梯度 `.grad`，大小等于 W；
- 🟩 **A** 前向为反向存下的张量（autograd 的 saved tensors）；🟨 **T** 正在算的那一层的临时量，算完即释放。

**表 4-1** 各模式的峰值显存（GiB，batch 4 seq 512，`max_memory_allocated`；带图 = 训练时的前向，no_grad = 推理）

| Size | 🟦 W | forward（no_grad） | forward（带图） | fwd_bwd | full | 纸面 16 B/参数 + A |
|:--|--:|--:|--:|--:|--:|--:|
| small  |  0.48 |  0.72 |  3.98 |  4.08 |  5.04 |  5.4 |
| medium |  1.58 |  1.90 | 10.49 | 10.58 | 13.74 | 15.2 |
| large  |  3.61 |  4.11 | 20.19 | 20.28 | 27.51 | 31.0 |
| xl     | 12.70 | 13.47 | OOM | OOM | OOM | 50.8 + A |

- 带图前向 − W 就是 🟩 A：small 3.5、medium 8.9、large 16.6 GiB，是权重的 4.6–7.3 倍。
- fwd_bwd 只比带图前向多 0.1 GiB；full 再多 2W，是 Adam 的 m、v（第一步之后常驻）。
- 实测 full 比纸面少一个 W（large 27.5 vs 31.0）——梯度并不常驻。

**峰值落在哪一刻。** 反向走完 j 层（共 L 层）时，活着的显存是

<p align="center">$M(j) = W + G \cdot \dfrac{j}{L} + A \cdot \dfrac{L-j}{L} + T$</p>

🟦 W 常驻；🟩 A 前向逐层堆上、反向逐层放掉；🟥 G 反向逐层长出来；🟨 T 层内即造即释。

> **梯度不是常驻的底座**：`.grad` 在反向算到那个参数时才创建，optimizer step 用完就被 `zero_grad(set_to_none=True)`（PyTorch 2.0 起的默认）释放——前向期间不存在，反向期间从 0 长到 W。

M(j) 对 j 是直线，峰值只能在两端：A > G 时在前向末尾（W + A），G > A 时在反向末尾（W + G）。full 再加常驻的 Adam 2W：

<p align="center">$\mathrm{peak}_{\mathrm{full}} \approx 3W + \max(A,\ G)$</p>

验证 large：3 × 3.61 + 16.58 = 27.4，实测 27.51。正常训练 token 多，A > G（表 4-1 三档都是）；只有 token 很少时才翻过来，比如 xl 在 seq 128（A 5.3 < G 12.7），峰值就在反向末尾 = 2W（图 4-1 左）。

![一步 fwd_bwd 的显存按 W、G、A、T 堆叠](peak_moment.png "**图 4-1** 一步 fwd_bwd 的显存。色带按 M(j) 用实测的 W、G、A、T 堆叠，× 是 hook 在每层前向 / 反向结束时读到的 memory_allocated()，都压在公式上。左：xl@128，G > A，峰值在反向末尾；右：small@512，A > G，峰值在前向末尾。虚线是 bf16（§5）。")

**A 里存的是什么。** autograd 的规则只有一条：每个 op 的局部偏导里出现什么，前向就存什么；偏导是常数就不存。比如 RMSNorm 的最后一步 $y = w \odot \hat{x}$，$\partial y / \partial w = \hat{x}$，所以 $\hat{x}$ 要存（`[4, 512, 2560]` fp32，20 MiB）；中间求均值那步偏导是常数 $1/d$，什么都不存。矩阵乘 $y = xW^\top$ 对 W 的偏导是 x——所以**每个矩阵乘的输入都得存**。用 `saved_tensors_hooks` 数 xl 的一层：

**表 4-2** xl 一层为反向存的 3655 MiB（batch 4 seq 2048，`torch.compile` 后，`saved_tensors_hooks` 实测；这组用 16 头，S、P 各 1 GiB，32 头时各 2 GiB）

| 张量 | 形状 | 大小 (MiB) | 占比 |
|:--|:--|--:|--:|
| attention 的 S = QKᵀ、P = softmax(S) | `[b, h, s, s]` | 1024 × 2 | 56% |
| FFN 的 w1(x)、w3(x)、silu·gate | `[b, s, d_ff]` | 320 × 3 | 26% |
| x、两个 norm 的输出、Q、K、V、attention 输出等 | `[b, s, d]` | 80 × 8 | 17% |
| mask、RoPE 的 cos/sin、softmax 统计量等 | — | ~7 | 0.2% |

全是矩阵乘的输入，融合也省不掉；32 层就是 114 GiB，远超 5090 的 31.3 GiB。其中一半以上是 S、P，而且只有它们 ∝ seq²，其余 ∝ seq。

seq 一长，这个 seq² 项直接决定能不能跑。用 `torch.cuda.memory._record_memory_history` 记下 xl（batch 4，32 头）一步里的每次分配和释放：

![xl 的四张显存时间线](mem_xl_timelines.png "**图 4-2** xl 的显存时间线，每个点是一次真实的分配 / 释放。左上 seq 128 纯前向：不为反向留东西，是平的。右上 seq 2048 纯前向：每层 attention 那串 [b, h, s, s] 中间量（S、/√d、mask、softmax 的三步）有 ~4 份 2 GiB 同时活着，冲出 32 根尖峰。左下 seq 128 full step：前向逐层上坡，反向继续爬（每层放掉 🟩 166 MiB、长出 🟥 410 MiB），分配 Adam 的 m、v 时 OOM。右下 seq 2048 带图前向：第 1 层留下 ~4.7 GiB，第 2 层的尖峰撞到 25.96 GiB，OOM。")

- **为什么 seq 2048 连前向都过不了？** 同一个 `[b, h, s, s]` 张量在 seq 128 时只有 8 MiB，seq 2048 时 2 GiB——是残差流上 `[b, s, d]` 张量（80 MiB）的 25 倍。纯前向每层用完就释放，所以 32 层都过得去；带图前向每层还要把 S、P 留给反向，第 2 层就没有空间给尖峰了。

---

## 5 bf16 与 checkpoint：能省多少

**bf16 autocast** 只在算子调用时把矩阵乘的输入 cast 成 bf16；权重、梯度、Adam 状态仍是 fp32。

**表 5-1** bf16 autocast 相对 fp32（fwd_bwd 模式，batch 4 seq 512）

| Size | 前向 (ms) | 反向 (ms) | 峰值显存 (GiB) |
|:--|:--|:--|:--|
| small  |  17.2 →  9.2（1.87×） |  33.9 →  20.1（1.69×） |  4.08 →  3.18（−21%） |
| medium |  50.1 → 24.4（2.05×） | 102.3 →  57.3（1.79×） | 10.58 →  8.36（−21%） |
| large  | 114.8 → 49.9（2.30×） | 220.0 → 117.6（1.87×） | 20.28 → 16.61（−18%） |

- **为什么快 ~2×？** 字节减半，矩阵乘也从 CUDA core 换到了 Tensor core（表 1-1）。模型越大、矩阵乘占比越高，加速越多。
- **为什么只省 ~20%？** 🟦🟥 不变，🟩 A 也没减半：norm、softmax、残差、loss 留在 fp32，反向还要用一份 bf16 权重副本（large：A −5.50 GiB，副本 +1.78 GiB）。
- **为什么 norm、softmax 留 fp32？** 它们都是求和类的归约，精度由累加器决定。bf16 只有 7 位尾数：0.01 累加 1000 次，bf16 累加器停在 4.0（4.0 + 0.01 舍入回 4.0），fp32 累加器得 10.0001。

**activation checkpoint** 用 FLOPs 换显存：前向只存每段的输入（entry，一个 `[b, s, d]` 张量），反向到这段时用它把前向重跑一遍，造出 saved tensors、用完即丢。

![checkpoint 段长扫描：step 时间与峰值显存](checkpoint_large_sweep.png "**图 5-1** checkpoint 段长扫描（large 36 层，batch 1 seq 1024，fwd_bwd，fp32 eager）。左：step 时间；右：峰值显存；红虚线是不 checkpoint。")

- **时间**：不管每段几层，整网都恰好多算一遍前向，所以都是 302–313 ms，比不 checkpoint 的 236 ms 多 28–33%。按 §2，一步原本是 1 份前向 + 2 份（反向），现在再加 1 份。
- **显存**：随每段层数单调上升，每层一个 checkpoint 最低（7.8 GiB，不 checkpoint 是 15.0 GiB）。

设每段 e 层、entry 大小 a、一层 saved tensors 大小 r，峰值 ≈ 全部 entry + 正在重算的那一段：

<p align="center">$M(e) = \dfrac{L}{e}\,a + e\,r$</p>

只要全部 entry 加起来不到一层（L·a < r；这里 36 × 5 MiB = 180 MiB < 220 MiB），e = 1 就最优——Transformer 每层都很胖，几乎总是这样。实践中先上 FlashAttention 或选择性重算（只丢 S、P 这类大而便宜的张量，前向 +5%），不够再按显存缺口挑 N 层、每层包一个 checkpoint。

> **S、P 是绕不开的那部分。** 时间上，它们让 attention 成为 memory-bound（§3）；显存上，它们占一层 saved tensors 的一半以上（§4）。bf16 和 checkpoint 都只是缓解：checkpoint 只能把它们的存活时间缩到重算那一层，重算时照样要完整写进显存、再读出来。要让 S、P 根本不进显存，只能把 QKᵀ → softmax → PV 融成一个 kernel，在片上分块算完，反向时再在片上重算——这就是下一篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/)。
