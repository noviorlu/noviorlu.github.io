---
title: "GPU 训练分析：FLOPs、Roofline 与显存峰值（RTX 5090 实测）"
date: 2026-10-04
draft: false
math: true
description: "FLOPs、算术强度、roofline（fp32 / bf16 / fp8 / nvfp4 的 ridge point）、MFU 怎么算，前向和反向各算多少，哪些 op 是 memory-bound，显存峰值在哪一刻、存的是什么。全部在一张 RTX 5090 上实测，是 FlashAttention 一文的前置。"
tags: ["GPU", "Roofline", "Transformer", "显存", "LLM", "AI"]
categories: ["AI笔记"]
series: ["AI笔记"]
series_order: 2
---

这篇在一张 RTX 5090 上把一步 Transformer 训练拆开，看时间和显存分别花在哪。GPU 上一个 op 的耗时由两部分决定：计算（FLOPs）和读写显存（bytes），所以先讲 FLOPs、算术强度、roofline、MFU 这几个概念，再逐项实测。几部分的结论最后都落在 attention 的 seq × seq 分数矩阵上，下一篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/) 讲的就是怎么处理它。

> 环境：RTX 5090 32 GB，torch 2.11.0+cu130；模型是自己写的 Transformer LM（RMSNorm + RoPE + SwiGLU，pre-norm）。fp32 基准关掉 tf32（`allow_tf32=False`）；除注明外 batch 4、seq 512，预热 5 步、计时 10 步；显存一律 GiB = 2³⁰ B（`max_memory_allocated() / 1024³`）。

## 省流不看 {#tldr}

1. **速度上限看 roofline：耗时 ≥ max(FLOPs / 峰值算力, bytes / 带宽)**（[§1](#basics)）：5090 带宽 1.79e12 B/s，fp32 / bf16 / fp8 / nvfp4 峰值 1.05e14 / 2.1e14 / 4.19e14 / 1.68e15 FLOPS，ridge point 58 / 117 / 234 / 935 FLOPs/B；算术强度低于 ridge 的 op 受带宽限制，快慢和 FLOPs 无关。
2. **训练一步 ≈ 6 × 参数量 × token 数 FLOPs**（[§2](#fwd-bwd)）：前向每个参数做一次乘加（2N），反向对输入、对权重各做一次同样大的矩阵乘（4N）；实测反向 ≈ 2.0× 前向，整步只跑到 fp32 峰值的 31%。
3. **attention 是 memory-bound 的**（[§3](#memory-bound)）：softmax 的 FLOPs 只有 PV 矩阵乘的 1/5，耗时却是它的 6×；eager 下 5 个 kernel 把 seq² 大小的分数矩阵进出显存 8 次。
4. **显存峰值 = 🟦 权重 + max(🟩 activation, 🟥 梯度)**（[§4](#peak-memory)）：每个 op 的局部偏导里有什么，前向就得存什么；一层 activation 的一半以上是 attention 的 S、P，3.41B 的 xl 在 seq 2048 时第 2 层就 OOM。
5. **bf16、checkpoint 都只能缓解**（[§5](#bf16-checkpoint)）：bf16 快 ~2×、省 ~20%；checkpoint 多算一遍前向（+30%），显存减半，但 S、P 重算时照样要写进显存。要让它们不进显存只能改 kernel，见 [FlashAttention 1–4](/blog/flashattention-1-to-4/)。

---

## 1 基础概念：FLOPs、算术强度、roofline、MFU {#basics}

### 1.1 FLOPs 与 FLOPS {#flops}

- **FLOPs**（floating-point operations）是浮点运算的**次数**，**FLOPS**（per second）是**速率**。本文都用 10 的幂写：seq 1024 的一次 PV 矩阵乘是 8.6e9 FLOPs，5090 的 fp32 峰值是 1.05e14 FLOPS。
- **矩阵乘** `[M, K] × [K, N]`：M·N 个输出，每个做 K 次乘加，一次乘加记 2 FLOPs，共 **2·M·N·K**。
- **逐元素 op**（加、乘、exp、mask）：每个元素一到几十次运算（exp 按 ~20 次计），总量 = 元素数 × 每元素次数，比同尺寸的矩阵乘少几个数量级。

除了计算，每个 op 还要从显存读输入、把输出写回显存，这部分用字节数（bytes）衡量。

### 1.2 算术强度与 roofline {#roofline}

计算受峰值算力 P 限制，读写受显存带宽 B 限制，所以一个 op 的耗时有两个下限，实际耗时不低于其中较大的那个：

<p align="center">$t \ge \max\left(\dfrac{\mathrm{FLOPs}}{P},\ \dfrac{\mathrm{bytes}}{B}\right)$</p>

FLOPs 与 bytes 之比叫**算术强度**（arithmetic intensity）$I = \mathrm{FLOPs} / \mathrm{bytes}$，表示每读写 1 字节做几次运算，只由 op 本身决定。把上式换成可达的 FLOPS，就是 **roofline 模型**：

<p align="center">$\mathrm{FLOPS}_{\max}(I) = \min(P,\ I \cdot B)$</p>

画在 log-log 坐标上是一条斜线（带宽）接一条水平线（算力），交点 $I^* = P / B$ 叫 **ridge point**：

- $I < I^*$：**memory-bound**。落在斜线下，耗时 ≈ bytes / B，与 FLOPs 无关，要提速只能减少读写。
- $I > I^*$：**compute-bound**。落在水平线下，耗时 ≈ FLOPs / P，要提速只能减少计算，或换更快的计算单元。

I 可以直接从张量形状估出来：

- **逐元素 op**：fp32 下每个元素 1 次运算，读 4 B、写 4 B，I ≈ 0.13。
- **矩阵乘**：fp32 下 $I = \dfrac{2MNK}{4(MK + KN + MN)}$，输出远大于输入时约为 K/2，主要看内维 K。Linear 的 K 是 d_model（1024），I ≈ 340；attention 里 QKᵀ 的 K 是 d_head（64），I 只有 ≈ 28。

ridge point 由硬件决定。5090 的带宽是 1.79e12 B/s，峰值算力则随精度不同：

| 精度 | 计算单元 | 累加 | 峰值 (FLOPS) | ridge point (FLOPs/B) |
|:--|:--|:--|--:|--:|
| **fp32** | CUDA core | fp32 | **1.05e14** | **58** |
| tf32 | Tensor core | fp32 | 1.05e14 | 58 |
| **bf16** / fp16 | Tensor core | fp32 | **2.1e14** | **117** |
| fp16 | Tensor core | fp16 | 4.19e14 | 234 |
| **fp8** | Tensor core | fp32 | **4.19e14** | **234** |
| fp8 | Tensor core | fp16 | 8.38e14 | 468 |
| **nvfp4** | Tensor core | fp32 | **1.68e15** | **935** |
{#tab-1-1 caption="**表 1-1** RTX 5090 各精度的峰值算力与 ridge point" note="dense 峰值，按 boost clock 2407 MHz，来自 NVIDIA RTX Blackwell 白皮书附录 A 表 3；ridge point = 峰值 / 1.792e12 B/s。"}

- **fp32 用的是 CUDA core**：CUDA core 是通用的标量乘加单元；Tensor core 只做小矩阵块的乘加，只接受 16 位、tf32、fp8 等输入。tf32 是把 fp32 的尾数截到 10 位后交给 Tensor core 算，本文关掉了，所以 fp32 的峰值按 1.05e14 算。
- **精度越低，峰值越高**：bf16、fp8、nvfp4 分别是 2.1e14、4.19e14、1.68e15，ridge point 从 117 升到 935。
- **GeForce 上 fp32 累加是半速**：同样的 fp16 / fp8 输入，fp32 累加的峰值只有 fp16 累加的一半。训练时矩阵乘用 fp32 累加，所以本文 bf16 的峰值取 2.1e14。
- **nvfp4**：白皮书只给「FP4 Tensor」一档；NVFP4（E2M1 元素 + 每 16 个元素一个 E4M3 scale）在 5090（sm_120a）上由 `mma.sync … kind::mxf4nvf4.block_scale` 执行，这里按 FP4 这一档计。

[图 1-1](#fig-1-1) 把这些上限和本文实测的几个 op 画在一起（点是 [§3](#memory-bound) 里 medium、seq 1024 一层 attention 的实测，悬停可看数值；causal mask 没有 FLOPs，不在图上）。

<figure id="fig-1-1" class="rf-fig">
<svg viewBox="0 0 640 392" width="100%" role="img" aria-label="RTX 5090 的 roofline：横轴算术强度，纵轴可达算力；带宽斜线与 fp32、bf16、fp8、nvfp4 四条平线分别交于 58、117、234、935 FLOPs/B；attention 的 op 都贴着斜线，只有 Linear 在 fp32 平线下">
  <style>
    .rf-grid { stroke: currentColor; stroke-opacity: .12; stroke-width: 1; }
    .rf-axis { stroke: currentColor; stroke-opacity: .45; stroke-width: 1; }
    .rf-tick { font-size: 11px; fill: currentColor; opacity: .7; }
    .rf-atitle { font-size: 12px; fill: currentColor; opacity: .8; }
    .rf-roof { fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    .rf-roof2 { fill: none; stroke: currentColor; stroke-opacity: .55; stroke-width: 1.5; stroke-dasharray: 6 4; }
    .rf-knee { fill: currentColor; }
    .rf-ridge { stroke: currentColor; stroke-opacity: .3; stroke-width: 1; }
    .rf-lab { font-size: 12px; fill: currentColor; }
    .rf-sub { font-size: 11px; fill: currentColor; opacity: .65; }
    .rf-pt { fill: rgb(var(--color-primary-500)); stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .rf-hit { fill: transparent; }
    .rf-g:hover .rf-pt, .rf-g:focus .rf-pt { stroke-width: 3; }
    @media (max-width: 640px) { .rf-fig { overflow-x: auto; } .rf-fig > svg { min-width: 520px; } }
  </style>
  <text class="rf-tick" x="70.0" y="357" text-anchor="middle">0.1</text>
  <line class="rf-grid" x1="180.0" y1="20" x2="180.0" y2="340"/>
  <text class="rf-tick" x="180.0" y="357" text-anchor="middle">1</text>
  <line class="rf-grid" x1="290.0" y1="20" x2="290.0" y2="340"/>
  <text class="rf-tick" x="290.0" y="357" text-anchor="middle">10</text>
  <line class="rf-grid" x1="400.0" y1="20" x2="400.0" y2="340"/>
  <text class="rf-tick" x="400.0" y="357" text-anchor="middle">100</text>
  <line class="rf-grid" x1="510.0" y1="20" x2="510.0" y2="340"/>
  <text class="rf-tick" x="510.0" y="357" text-anchor="middle">1000</text>
  <line class="rf-grid" x1="620.0" y1="20" x2="620.0" y2="340"/>
  <text class="rf-tick" x="620.0" y="357" text-anchor="middle">10000</text>
  <text class="rf-tick" x="62" y="344.0" text-anchor="end">1e11</text>
  <line class="rf-grid" x1="70" y1="268.9" x2="620" y2="268.9"/>
  <text class="rf-tick" x="62" y="272.9" text-anchor="end">1e12</text>
  <line class="rf-grid" x1="70" y1="197.8" x2="620" y2="197.8"/>
  <text class="rf-tick" x="62" y="201.8" text-anchor="end">1e13</text>
  <line class="rf-grid" x1="70" y1="126.7" x2="620" y2="126.7"/>
  <text class="rf-tick" x="62" y="130.7" text-anchor="end">1e14</text>
  <line class="rf-grid" x1="70" y1="55.6" x2="620" y2="55.6"/>
  <text class="rf-tick" x="62" y="59.6" text-anchor="end">1e15</text>
  <line class="rf-axis" x1="70" y1="340" x2="620" y2="340"/>
  <line class="rf-axis" x1="70" y1="20" x2="70" y2="340"/>
  <text class="rf-atitle" x="345" y="382" text-anchor="middle">算术强度 I（FLOPs/B，对数轴）</text>
  <text class="rf-atitle" transform="translate(16 180) rotate(-90)" text-anchor="middle">可达算力（FLOPS，对数轴）</text>
  <line class="rf-ridge" x1="374.4" y1="125.2" x2="374.4" y2="340"/>
  <line class="rf-roof2" x1="374.4" y1="125.2" x2="506.8" y2="39.6"/>
  <line class="rf-roof" x1="374.4" y1="125.2" x2="620" y2="125.2"/>
  <circle class="rf-knee" cx="374.4" cy="125.2" r="2.5"><title>fp32：ridge point = 1.05e14 / 1.792e12 = 58 FLOPs/B</title></circle>
  <text class="rf-sub" x="367.4" y="123.2" text-anchor="end">58</text>
  <text class="rf-lab" x="618" y="139.2" text-anchor="end">fp32 1.05e14</text>
  <line class="rf-roof2" x1="407.5" y1="103.8" x2="620" y2="103.8"/>
  <circle class="rf-knee" cx="407.5" cy="103.8" r="2.5"><title>bf16：ridge point = 2.1e14 / 1.792e12 = 117 FLOPs/B</title></circle>
  <text class="rf-sub" x="400.5" y="101.8" text-anchor="end">117</text>
  <text class="rf-sub" x="618" y="117.8" text-anchor="end">bf16 2.1e14</text>
  <line class="rf-roof2" x1="440.6" y1="82.4" x2="620" y2="82.4"/>
  <circle class="rf-knee" cx="440.6" cy="82.4" r="2.5"><title>fp8：ridge point = 4.19e14 / 1.792e12 = 234 FLOPs/B</title></circle>
  <text class="rf-sub" x="433.6" y="80.4" text-anchor="end">234</text>
  <text class="rf-sub" x="618" y="96.4" text-anchor="end">fp8 4.19e14</text>
  <line class="rf-roof2" x1="506.8" y1="39.6" x2="620" y2="39.6"/>
  <circle class="rf-knee" cx="506.8" cy="39.6" r="2.5"><title>nvfp4：ridge point = 1.68e15 / 1.792e12 = 935 FLOPs/B</title></circle>
  <text class="rf-sub" x="499.8" y="37.6" text-anchor="end">935</text>
  <text class="rf-sub" x="618" y="53.6" text-anchor="end">nvfp4 1.68e15</text>
  <polyline class="rf-roof" points="70,322.0 374.4,125.2"/>
  <text class="rf-sub" x="80" y="36">拐点旁的数字 = ridge point（FLOPs/B）</text>
  <text class="rf-lab" transform="translate(230 210.6) rotate(-32.9)" text-anchor="middle">带宽 1.79e12 B/s</text>
  <text class="rf-sub" x="279" y="326" text-anchor="middle">memory-bound</text>
  <text class="rf-sub" x="529" y="326" text-anchor="middle">compute-bound</text>
  <g class="rf-g" tabindex="0"><title>Linear（FFN w1）：I = 341 FLOPs/B，实测 6.87e13 FLOPS，正上方 fp32 屋顶 1.05e14，MFU 64%</title><circle class="rf-hit" cx="458.6" cy="138.3" r="12"/><circle class="rf-pt" cx="458.6" cy="138.3" r="5"/></g>
  <text class="rf-lab" x="448.6" y="156.3" text-anchor="end">Linear</text>
  <g class="rf-g" tabindex="0"><title>S = QKᵀ：I = 28.4 FLOPs/B，实测 2.86e13 FLOPS，正上方 fp32 屋顶 5.1e13，MBU 57%</title><circle class="rf-hit" cx="339.9" cy="165.3" r="12"/><circle class="rf-pt" cx="339.9" cy="165.3" r="5"/></g>
  <text class="rf-lab" x="349.9" y="179.3" text-anchor="start">QKᵀ</text>
  <g class="rf-g" tabindex="0"><title>O = PV：I = 30.1 FLOPs/B，实测 3.73e13 FLOPS，正上方 fp32 屋顶 5.4e13，MBU 70%</title><circle class="rf-hit" cx="342.7" cy="157.1" r="12"/><circle class="rf-pt" cx="342.7" cy="157.1" r="5"/></g>
  <text class="rf-lab" x="352.7" y="155.1" text-anchor="start">PV</text>
  <g class="rf-g" tabindex="0"><title>softmax（5 个 kernel）：I = 0.844 FLOPs/B，实测 1.27e12 FLOPS，正上方 fp32 屋顶 1.51e12，MBU 84%</title><circle class="rf-hit" cx="171.9" cy="261.6" r="12"/><circle class="rf-pt" cx="171.9" cy="261.6" r="5"/></g>
  <text class="rf-lab" x="181.9" y="277.6" text-anchor="start">softmax</text>
  <g class="rf-g" tabindex="0"><title>S / √d：I = 0.125 FLOPs/B，实测 1.92e11 FLOPS，正上方 fp32 屋顶 2.24e11，MBU 86%</title><circle class="rf-hit" cx="80.7" cy="319.9" r="12"/><circle class="rf-pt" cx="80.7" cy="319.9" r="5"/></g>
  <text class="rf-lab" x="90.7" y="331.9" text-anchor="start">S / √d</text>
</svg>
<figcaption><strong>图 1-1</strong> RTX 5090 的 roofline：斜线是带宽，平线是各精度的峰值（<a href="#tab-1-1">表 1-1</a>），拐点旁是 ridge point；点是 fp32 下实测的 op（<a href="#memory-bound">§3</a>），悬停可看数值。</figcaption>
</figure>

attention 的几个 op 都在斜线附近，属于 memory-bound；只有 Linear 落在水平线下。换成低精度后峰值变高、ridge point 右移，但同一个 op 的字节数也按比例减少，点同样右移，所以 Linear 仍在 ridge 右侧，逐元素 op 仍在左侧，两类 op 都会变快（bf16 实测快 ~2×，见 [§5.1](#bf16)）。

### 1.3 MFU 与 MBU {#mfu}

实测点与它正上方那条线的比值就是利用率，按所在区域分两种：

- **MFU**（model FLOPs utilization）= 实际 FLOPS / 峰值 FLOPS，衡量 compute-bound 的代码离水平线多远。整步训练的 MFU = 每步模型 FLOPs / step 时间 / P。
- **MBU**（memory bandwidth utilization）= 实际带宽 / 峰值带宽，衡量 memory-bound 的 op 离斜线多远。

memory-bound 的 op MFU 一定很低，比如[图 1-1](#fig-1-1) 里的 softmax 只有 ~1%，但它的 MBU 有 84%，已经接近带宽上限。

---

## 2 前向与反向：FLOPs 怎么数，实际跑多快 {#fwd-bwd}

实验用了五档模型，参数量 N 分别是 small 0.13B、medium 0.42B、large 0.97B、xl 3.41B、10B 12.83B。

Transformer 的 FLOPs 基本都来自 Linear 层的矩阵乘。以一个 Linear 为例（[图 2-1](#fig-2-1)），前向做一次矩阵乘，反向做两次：

<figure id="fig-2-1" class="gx-fig">
<svg viewBox="0 0 640 262" width="100%" role="img" aria-label="一个 Linear 的前向与反向：前向一次矩阵乘，反向算 dx、dW 两次同样大的矩阵乘；算 dW 要用前向的输入 x，所以 x 要留到反向">
  <style>
    .gx-band { fill: currentColor; fill-opacity: .045; }
    .gx-row { font-size: 12px; fill: currentColor; opacity: .8; }
    .gx-rowsub { font-size: 10.5px; fill: currentColor; opacity: .55; }
    .gx-op { fill: rgba(128,128,128,.12); stroke: currentColor; stroke-opacity: .5; stroke-width: 1.2; }
    .gx-t { font-size: 12.5px; fill: currentColor; }
    .gx-tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .gx-s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .gx-note { font-size: 11px; fill: currentColor; opacity: .7; }
    @media (max-width: 640px) { .gx-fig { overflow-x: auto; } .gx-fig > svg { min-width: 540px; } }
  </style>
  <defs>
    <marker id="fig-2-1-m0" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker>
    <marker id="fig-2-1-m1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#43a047"/></marker>
  </defs>
  <rect class="gx-band" x="74" y="14" width="560" height="66" rx="8"/>
  <text class="gx-row" x="6" y="45">前向</text>
  <text class="gx-rowsub" x="6" y="60">1 次矩阵乘</text>
  <rect class="gx-band" x="74" y="104" width="560" height="148" rx="8"/>
  <text class="gx-row" x="6" y="176">反向</text>
  <text class="gx-rowsub" x="6" y="191">2 次矩阵乘</text>
  <rect x="104.0" y="21.0" width="112" height="24" rx="6" fill="rgba(67,160,71,0.16)" stroke="#43a047" stroke-width="1.4"/>
  <text class="gx-t" x="160" y="33.0" text-anchor="middle" dominant-baseline="central">x  [tokens, in]</text>
  <line x1="216.0" y1="33.0" x2="271.0" y2="43.5" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-2-1-m0)"/>
  <rect x="104.0" y="49.0" width="112" height="24" rx="6" fill="rgba(30,136,229,0.16)" stroke="#1e88e5" stroke-width="1.4"/>
  <text class="gx-t" x="160" y="61.0" text-anchor="middle" dominant-baseline="central">W  [out, in]</text>
  <line x1="216.0" y1="61.0" x2="271.0" y2="50.5" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-2-1-m0)"/>
  <rect x="274.0" y="25.0" width="196" height="44" rx="6" class="gx-op"/>
  <text class="gx-tb" x="372" y="39.5" text-anchor="middle" dominant-baseline="central">y = x · Wᵀ</text>
  <text class="gx-s" x="372" y="54.5" text-anchor="middle" dominant-baseline="central">2 · tokens · in · out FLOPs</text>
  <rect x="496.0" y="33.0" width="128" height="28" rx="6" class="gx-op"/>
  <text class="gx-t" x="560" y="47.0" text-anchor="middle" dominant-baseline="central">y → 后一层</text>
  <line x1="470.0" y1="47.0" x2="493.0" y2="47.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-2-1-m0)"/>
  <rect x="104.0" y="114.0" width="112" height="24" rx="6" fill="rgba(249,168,37,0.16)" stroke="#f9a825" stroke-width="1.4"/>
  <text class="gx-t" x="160" y="126.0" text-anchor="middle" dominant-baseline="central">dy ← 后一层</text>
  <line x1="216.0" y1="126.0" x2="271.0" y2="136.5" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-2-1-m0)"/>
  <rect x="104.0" y="142.0" width="112" height="24" rx="6" fill="rgba(30,136,229,0.16)" stroke="#1e88e5" stroke-width="1.4"/>
  <text class="gx-t" x="160" y="154.0" text-anchor="middle" dominant-baseline="central">W</text>
  <line x1="216.0" y1="154.0" x2="271.0" y2="143.5" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-2-1-m0)"/>
  <rect x="274.0" y="118.0" width="196" height="44" rx="6" class="gx-op"/>
  <text class="gx-tb" x="372" y="132.5" text-anchor="middle" dominant-baseline="central">dx = dy · W</text>
  <text class="gx-s" x="372" y="147.5" text-anchor="middle" dominant-baseline="central">同样多 FLOPs</text>
  <rect x="496.0" y="126.0" width="128" height="28" rx="6" fill="rgba(249,168,37,0.16)" stroke="#f9a825" stroke-width="1.4"/>
  <text class="gx-t" x="560" y="140.0" text-anchor="middle" dominant-baseline="central">dx → 前一层</text>
  <line x1="470.0" y1="140.0" x2="493.0" y2="140.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-2-1-m0)"/>
  <rect x="104.0" y="188.0" width="112" height="24" rx="6" fill="rgba(249,168,37,0.16)" stroke="#f9a825" stroke-width="1.4"/>
  <text class="gx-t" x="160" y="200.0" text-anchor="middle" dominant-baseline="central">dy</text>
  <line x1="216.0" y1="200.0" x2="271.0" y2="210.5" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-2-1-m0)"/>
  <rect x="104.0" y="216.0" width="112" height="24" rx="6" fill="rgba(67,160,71,0.16)" stroke="#43a047" stroke-width="1.4" stroke-dasharray="4 3"/>
  <text class="gx-t" x="160" y="228.0" text-anchor="middle" dominant-baseline="central">x（前向留下的）</text>
  <line x1="216.0" y1="228.0" x2="271.0" y2="217.5" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-2-1-m0)"/>
  <rect x="274.0" y="192.0" width="196" height="44" rx="6" class="gx-op"/>
  <text class="gx-tb" x="372" y="206.5" text-anchor="middle" dominant-baseline="central">dW = dyᵀ · x</text>
  <text class="gx-s" x="372" y="221.5" text-anchor="middle" dominant-baseline="central">同样多 FLOPs</text>
  <rect x="496.0" y="200.0" width="128" height="28" rx="6" fill="rgba(229,57,53,0.16)" stroke="#e53935" stroke-width="1.4"/>
  <text class="gx-t" x="560" y="214.0" text-anchor="middle" dominant-baseline="central">dW → optimizer</text>
  <line x1="470.0" y1="214.0" x2="493.0" y2="214.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-2-1-m0)"/>
  <path d="M104,33 C84,33 84,228 104,228" fill="none" stroke="#43a047" stroke-width="1.6" stroke-dasharray="5 3" marker-end="url(#fig-2-1-m1)"/>
  <text class="gx-note" x="88.0" y="132.0" text-anchor="end">x 留到反向</text>
</svg>
<figcaption><strong>图 2-1</strong> 一个 Linear 的前向与反向：灰框是矩阵乘（花 FLOPs），彩框是张量，颜色同 <a href="#peak-memory">§4</a> 的 🟩 A / 🟦 W / 🟥 G / 🟨 T。</figcaption>
</figure>

- **前向 ≈ 2N / token**：每个 token 经过一个权重矩阵时，矩阵里每个参数做一次乘加（embedding 是查表，不算）。attention 的 QKᵀ、PV 没有参数，另算 4·L·seq·d_model，seq 512 时只占 2–4%。
- **反向 ≈ 4N / token**：要算 dx 和 dW 两个梯度，各是一次和前向同样大的矩阵乘。其中 dW = dyᵀ·x 要用前向的输入 x，所以 x 得一直保存到反向，这部分就是 [§4](#peak-memory) 的 🟩 A。
- **训练一步 ≈ 6N × token 数**，这里 token 数 = batch × seq = 2048。

用 nsys 看 medium 的一步，GEMM kernel 正好分成三组，每组 169 个（24 层 × 7 个 Linear + lm_head），分别对应前向的 y = x·Wᵀ、反向的 dx = dy·W 和 dW = dyᵀ·x。实测一步里各阶段的耗时占比见[图 2-2](#fig-2-2)：

<figure id="fig-2-2" class="cv-fig">
<svg class="cv" viewBox="0 0 640 172" width="100%" role="img" aria-label="三档模型一步训练的耗时构成：前向约 31%，反向约 62%，optimizer 约 7%；右侧是每步总耗时和 MFU">
  <style>
    .cv { --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#e34948; --cg:#a19f9a; }
    html.dark .cv { --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#e66767; --cg:#6f6e6a; }
    .cv .grid { stroke: currentColor; stroke-opacity: .1; }
    .cv .axis { stroke: currentColor; stroke-opacity: .35; }
    .cv .ref { stroke: currentColor; stroke-opacity: .55; stroke-dasharray: 5 4; }
    .cv .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .lab { font-size: 12px; fill: currentColor; }
    .cv .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .cv .val { font-size: 11px; fill: currentColor; }
    .cv .in { font-size: 11px; fill: #111; }
    .cv .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .cv g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .cv-fig { overflow-x: auto; } .cv-fig > svg { min-width: 540px; } }
  </style>
  <rect x="112" y="11" width="10" height="10" rx="2" style="fill: var(--c1)"/>
  <text class="lab" x="127" y="20">前向</text>
  <rect x="173" y="11" width="10" height="10" rx="2" style="fill: var(--c2)"/>
  <text class="lab" x="188" y="20">反向</text>
  <rect x="234" y="11" width="10" height="10" rx="2" style="fill: var(--c3)"/>
  <text class="lab" x="249" y="20">optimizer</text>
  <text class="lab" x="102" y="53" text-anchor="end">small</text>
  <text class="lab2" x="102" y="67" text-anchor="end">0.13B</text>
  <g class="m"><title>small 前向：17.2 ms（31%）</title><rect x="112.0" y="44.0" width="106.7" height="26.0" rx="3" style="fill: var(--c1)"/></g>
  <text class="in" x="165.3" y="61" text-anchor="middle">31%</text>
  <g class="m"><title>small 反向：34.8 ms（62%）</title><rect x="220.7" y="44.0" width="217.9" height="26.0" rx="3" style="fill: var(--c2)"/></g>
  <text class="in" x="329.7" y="61" text-anchor="middle">62%</text>
  <g class="m"><title>small optimizer：3.7 ms（7%）</title><rect x="440.6" y="44.0" width="21.4" height="26.0" rx="3" style="fill: var(--c3)"/></g>
  <text class="val" x="474" y="56">55.7 ms / 步</text>
  <text class="lab2" x="474" y="70">MFU 27%</text>
  <text class="lab" x="102" y="95" text-anchor="end">medium</text>
  <text class="lab2" x="102" y="109" text-anchor="end">0.42B</text>
  <g class="m"><title>medium 前向：51.1 ms（31%）</title><rect x="112.0" y="86.0" width="105.5" height="26.0" rx="3" style="fill: var(--c1)"/></g>
  <text class="in" x="164.8" y="103" text-anchor="middle">31%</text>
  <g class="m"><title>medium 反向：103.3 ms（62%）</title><rect x="219.5" y="86.0" width="215.3" height="26.0" rx="3" style="fill: var(--c2)"/></g>
  <text class="in" x="327.2" y="103" text-anchor="middle">62%</text>
  <g class="m"><title>medium optimizer：12.9 ms（8%）</title><rect x="436.9" y="86.0" width="25.1" height="26.0" rx="3" style="fill: var(--c3)"/></g>
  <text class="val" x="474" y="98">167.4 ms / 步</text>
  <text class="lab2" x="474" y="112">MFU 31%</text>
  <text class="lab" x="102" y="137" text-anchor="end">large</text>
  <text class="lab2" x="102" y="151" text-anchor="end">0.97B</text>
  <g class="m"><title>large 前向：118.1 ms（32%）</title><rect x="112.0" y="128.0" width="109.5" height="26.0" rx="3" style="fill: var(--c1)"/></g>
  <text class="in" x="166.8" y="145" text-anchor="middle">32%</text>
  <g class="m"><title>large 反向：227.8 ms（61%）</title><rect x="223.5" y="128.0" width="213.1" height="26.0" rx="3" style="fill: var(--c2)"/></g>
  <text class="in" x="330.1" y="145" text-anchor="middle">61%</text>
  <g class="m"><title>large optimizer：26.9 ms（7%）</title><rect x="438.6" y="128.0" width="23.4" height="26.0" rx="3" style="fill: var(--c3)"/></g>
  <text class="val" x="474" y="140">372.8 ms / 步</text>
  <text class="lab2" x="474" y="154">MFU 31%</text>
</svg>
<figcaption><strong>图 2-2</strong> 一步训练里前向、反向、optimizer 的耗时占比（fp32，batch 4，seq 512），右侧是每步总耗时和 MFU。</figcaption>
</figure>

xl 在带梯度的前向就 OOM（[§4.3](#one-layer)），10B 在建模型时就 OOM。计时前要先预热：第一步有 ~300 ms 的一次性开销（kernel 懒加载、cuBLAS 初始化、显存池第一次 cudaMalloc），不预热会让 10 步均值偏高 7–59%。

- **反向为什么是前向的 2 倍？** 反向的矩阵乘个数是前向的两倍、尺寸相同，所以三档模型的反向都占一步的 ~62%，是前向（~31%）的 2 倍。optimizer（AdamW）基本没有 FLOPs，只是对每个参数读写权重、梯度和 m、v，属于 memory-bound，占一步的 7%。
- **实际算力多少？** medium 和 large 是 3.2e13 FLOPS，fp32 峰值的 31%（small 27%）。单个大矩阵乘能到 64%（[图 1-1](#fig-1-1) 里的 Linear），但矩阵乘只占整步 GPU 时间的 60%，其余 40% 基本是 FLOPs 很少的逐元素 kernel。
- **训一个模型要多久？** 按 Chinchilla 配比训 20N token，用实测的 step 时间（每步 2048 token）折算：small 18 h、medium 7.8 天、large 41 天，bf16（[§5.1](#bf16)）下是 12 h、4.5 天、21 天。单卡 5090 实际能训的大概到 medium（0.42B）。

这 40% 的时间花在 FLOPs 很少的 kernel 上，按 [§1.2](#roofline) 的判据它们都是 memory-bound，下一节以 attention 为例逐个看。

---

## 3 Memory bound：attention 慢在哪 {#memory-bound}

[图 1-1](#fig-1-1) 里的点就是 medium、seq 1024 时一层 attention 里的 op：

- **哪些 op 是 compute-bound？** 只有 Linear（I = 340，大于 fp32 的 ridge point 58）。attention 里的 op 都是 memory-bound，包括 QKᵀ 和 PV 两个矩阵乘：内维 d_head = 64，I 只有 ~30（[§1.2](#roofline)）。
- **还能再快吗？** 这些 op 已经跑到带宽上限的 57–86%，只能减少读写的字节数。
- **softmax 为什么最慢？** eager 实现是 5 个 kernel，每个都要完整读或写一遍和 S 一样大的张量（medium@1024 一层 256 MiB），一共 8 次、2 GiB（[图 3-1](#fig-3-1)）；`/√d` 和 mask 各 2 次。

<figure id="fig-3-1" class="sp-fig">
<svg viewBox="0 0 640 262" width="100%" role="img" aria-label="eager softmax 的 5 个 kernel 共读写显存里 S 大小的张量 8 次；融合成一个 kernel 后只读 S、写 P 两次">
  <style>
    .sp-band { fill: currentColor; fill-opacity: .045; }
    .sp-row { font-size: 12px; fill: currentColor; opacity: .75; }
    .sp-rowsub { font-size: 10.5px; fill: currentColor; opacity: .55; }
    .sp-t { fill: rgba(var(--color-primary-500), .14); stroke: rgb(var(--color-primary-500)); stroke-width: 1.5; }
    .sp-k { fill: rgba(128,128,128,.12); stroke: currentColor; stroke-opacity: .5; stroke-width: 1.2; }
    .sp-txt { font-size: 12.5px; fill: currentColor; text-anchor: middle; dominant-baseline: central; }
    .sp-a { stroke: rgb(var(--color-primary-500)); stroke-width: 1.8; fill: none; marker-end: url(#sp-arr); }
    .sp-s { stroke: currentColor; stroke-opacity: .5; stroke-width: 1.2; fill: none; marker-end: url(#sp-arr2); }
    .sp-lab { font-size: 11.5px; fill: currentColor; }
    .sp-slab { font-size: 11px; fill: currentColor; opacity: .6; text-anchor: middle; }
    @media (max-width: 640px) { .sp-fig { overflow-x: auto; } .sp-fig > svg { min-width: 520px; } }
  </style>
  <defs>
    <marker id="sp-arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: rgb(var(--color-primary-500))"/></marker>
    <marker id="sp-arr2" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="currentColor" fill-opacity=".5"/></marker>
  </defs>
  <rect class="sp-band" x="70" y="24" width="562" height="56" rx="8"/>
  <rect class="sp-band" x="70" y="122" width="562" height="56" rx="8"/>
  <text class="sp-row" x="8" y="50">显存</text><text class="sp-rowsub" x="8" y="66">(HBM)</text>
  <text class="sp-row" x="8" y="148">kernel</text><text class="sp-rowsub" x="8" y="164">(片上算)</text>
  <rect class="sp-t" x="133" y="37" width="64" height="30" rx="6"/>
  <text class="sp-txt" x="165" y="52">S</text>
  <rect class="sp-t" x="237" y="37" width="76" height="30" rx="6"/>
  <text class="sp-txt" x="275" y="52">S − m</text>
  <rect class="sp-t" x="388" y="37" width="104" height="30" rx="6"/>
  <text class="sp-txt" x="440" y="52">exp(S − m)</text>
  <rect class="sp-t" x="562" y="37" width="56" height="30" rx="6"/>
  <text class="sp-txt" x="590" y="52">P</text>
  <rect class="sp-k" x="81" y="135" width="62" height="30" rx="6"/>
  <text class="sp-txt" x="112" y="150">max</text>
  <rect class="sp-k" x="183" y="135" width="70" height="30" rx="6"/>
  <text class="sp-txt" x="218" y="150">减 max</text>
  <rect class="sp-k" x="301" y="135" width="58" height="30" rx="6"/>
  <text class="sp-txt" x="330" y="150">exp</text>
  <rect class="sp-k" x="409" y="135" width="62" height="30" rx="6"/>
  <text class="sp-txt" x="440" y="150">求和</text>
  <rect class="sp-k" x="519" y="135" width="58" height="30" rx="6"/>
  <text class="sp-txt" x="548" y="150">除</text>
  <line class="sp-a" x1="152" y1="67" x2="122" y2="133"/>
  <text class="sp-lab" x="99" y="104">① 读</text>
  <line class="sp-a" x1="178" y1="67" x2="208" y2="133"/>
  <text class="sp-lab" x="199" y="104">② 读</text>
  <line class="sp-a" x1="232" y1="135" x2="262" y2="69"/>
  <text class="sp-lab" x="255" y="108">③ 写</text>
  <line class="sp-a" x1="288" y1="67" x2="320" y2="133"/>
  <text class="sp-lab" x="311" y="104">④ 读</text>
  <line class="sp-a" x1="342" y1="135" x2="412" y2="69"/>
  <text class="sp-lab" x="383" y="112">⑤ 写</text>
  <line class="sp-a" x1="440" y1="67" x2="440" y2="133"/>
  <text class="sp-lab" x="446" y="104">⑥ 读</text>
  <line class="sp-a" x1="468" y1="67" x2="536" y2="133"/>
  <text class="sp-lab" x="510" y="104">⑦ 读</text>
  <line class="sp-a" x1="560" y1="135" x2="584" y2="69"/>
  <text class="sp-lab" x="578" y="108">⑧ 写</text>
  <line class="sp-s" x1="143" y1="150" x2="181" y2="150"/>
  <text class="sp-slab" x="162" y="143">m</text>
  <line class="sp-s" x1="471" y1="150" x2="517" y2="150"/>
  <text class="sp-slab" x="494" y="143">Σ</text>
  <text class="sp-rowsub" x="351" y="192" text-anchor="middle">eager：5 个 kernel，S 大小的张量进出显存 8 次；m、Σ 每行一个数，可忽略</text>
  <text class="sp-row" x="8" y="240">融合后</text>
  <rect class="sp-t" x="133" y="221" width="64" height="30" rx="6"/>
  <text class="sp-txt" x="165" y="236">S</text>
  <rect class="sp-k" x="255" y="221" width="150" height="30" rx="6"/>
  <text class="sp-txt" x="330" y="236">fused softmax</text>
  <rect class="sp-t" x="467" y="221" width="56" height="30" rx="6"/>
  <text class="sp-txt" x="495" y="236">P</text>
  <line class="sp-a" x1="197" y1="236" x2="253" y2="236"/>
  <text class="sp-lab" x="209" y="228">① 读</text>
  <line class="sp-a" x1="405" y1="236" x2="465" y2="236"/>
  <text class="sp-lab" x="419" y="228">② 写</text>
</svg>
<figcaption><strong>图 3-1</strong> eager softmax 的显存读写：每条编号箭头是一次完整的读或写，共 8 次；融合后只剩 2 次。</figcaption>
</figure>

把这 5 步融合成一个 kernel，读写降到 2 次；FlashAttention 更进一步，让 S、P 不写回显存。同一层里各 op 的 FLOPs 和实测时间对比见[图 3-2](#fig-3-2)：

<figure id="fig-3-2" class="cv-fig">
<svg class="cv" viewBox="0 0 640 186" width="100%" role="img" aria-label="medium、seq 1024 一层 attention 里各 op 的 FLOPs 与实测 GPU 时间：softmax 和 /√d、mask 的 FLOPs 很少，时间却最多">
  <style>
    .cv { --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#e34948; --cg:#a19f9a; }
    html.dark .cv { --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#e66767; --cg:#6f6e6a; }
    .cv .grid { stroke: currentColor; stroke-opacity: .1; }
    .cv .axis { stroke: currentColor; stroke-opacity: .35; }
    .cv .ref { stroke: currentColor; stroke-opacity: .55; stroke-dasharray: 5 4; }
    .cv .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .lab { font-size: 12px; fill: currentColor; }
    .cv .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .cv .val { font-size: 11px; fill: currentColor; }
    .cv .in { font-size: 11px; fill: #111; }
    .cv .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .cv g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .cv-fig { overflow-x: auto; } .cv-fig > svg { min-width: 540px; } }
  </style>
  <rect x="112" y="11" width="10" height="10" rx="2" style="fill: var(--c1)"/>
  <text class="lab" x="127" y="20">矩阵乘</text>
  <rect x="185" y="11" width="10" height="10" rx="2" style="fill: var(--c2)"/>
  <text class="lab" x="200" y="20">逐元素</text>
  <text class="ttl" x="112" y="48">FLOPs</text>
  <text class="ttl" x="392" y="48">GPU 时间（ms）</text>
  <text class="lab" x="102" y="75" text-anchor="end">QKᵀ</text>
  <g class="m"><title>QKᵀ：8.6e9 FLOPs（一层）</title><rect x="112.0" y="62.0" width="191.1" height="18.0" rx="3" style="fill: var(--c1)"/></g>
  <text class="val" x="309.1" y="75">8.6e9</text>
  <g class="m"><title>QKᵀ：0.3 ms（一层）</title><rect x="392.0" y="62.0" width="40.0" height="18.0" rx="3" style="fill: var(--c1)"/></g>
  <text class="val" x="438.0" y="75">0.30</text>
  <text class="lab" x="102" y="105" text-anchor="end">/√d + mask</text>
  <g class="m"><title>/√d + mask：6.7e7 FLOPs（一层）</title><rect x="112.0" y="92.0" width="1.5" height="18.0" rx="3" style="fill: var(--c2)"/></g>
  <text class="val" x="119.5" y="105">6.7e7</text>
  <g class="m"><title>/√d + mask：0.75 ms（一层）</title><rect x="392.0" y="92.0" width="100.0" height="18.0" rx="3" style="fill: var(--c2)"/></g>
  <text class="val" x="498.0" y="105">0.75</text>
  <text class="lab" x="102" y="135" text-anchor="end">softmax</text>
  <g class="m"><title>softmax：1.8e9 FLOPs（一层）</title><rect x="112.0" y="122.0" width="40.0" height="18.0" rx="3" style="fill: var(--c2)"/></g>
  <text class="val" x="158.0" y="135">1.8e9</text>
  <g class="m"><title>softmax：1.43 ms（一层）</title><rect x="392.0" y="122.0" width="190.7" height="18.0" rx="3" style="fill: var(--c2)"/></g>
  <text class="val" x="588.7" y="135">1.43</text>
  <text class="lab" x="102" y="165" text-anchor="end">PV</text>
  <g class="m"><title>PV：8.6e9 FLOPs（一层）</title><rect x="112.0" y="152.0" width="191.1" height="18.0" rx="3" style="fill: var(--c1)"/></g>
  <text class="val" x="309.1" y="165">8.6e9</text>
  <g class="m"><title>PV：0.23 ms（一层）</title><rect x="392.0" y="152.0" width="30.7" height="18.0" rx="3" style="fill: var(--c1)"/></g>
  <text class="val" x="428.7" y="165">0.23</text>
  <line class="axis" x1="112" y1="58" x2="112" y2="174"/>
  <line class="axis" x1="392" y1="58" x2="392" y2="174"/>
</svg>
<figcaption><strong>图 3-2</strong> 一层 attention 里各 op 的 FLOPs 与实测 GPU 时间（medium，seq 1024）：逐元素 op 的 FLOPs 少得几乎看不见，时间却占大头。</figcaption>
</figure>

- **FLOPs 和时间对得上吗？** 对不上。softmax 的 FLOPs 是 PV 的 1/5，时间是它的 6 倍；`/√d` 和 mask 几乎没有 FLOPs，时间却是 QKᵀ 的 2.5 倍。

这部分开销随 seq 增长得很快（[图 3-3](#fig-3-3)）：

<figure id="fig-3-3" class="cv-fig">
<svg class="cv" viewBox="0 0 640 262" width="100%" role="img" aria-label="attention 三段占 forward 时间随 seq 的变化：softmax 从 4% 涨到 24%，scores 从 4% 涨到 18%，PV 只从 2% 到 4%，合计从 10% 到 46%">
  <style>
    .cv { --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#e34948; --cg:#a19f9a; }
    html.dark .cv { --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#e66767; --cg:#6f6e6a; }
    .cv .grid { stroke: currentColor; stroke-opacity: .1; }
    .cv .axis { stroke: currentColor; stroke-opacity: .35; }
    .cv .ref { stroke: currentColor; stroke-opacity: .55; stroke-dasharray: 5 4; }
    .cv .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .lab { font-size: 12px; fill: currentColor; }
    .cv .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .cv .val { font-size: 11px; fill: currentColor; }
    .cv .in { font-size: 11px; fill: #111; }
    .cv .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .cv g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .cv-fig { overflow-x: auto; } .cv-fig > svg { min-width: 540px; } }
  </style>
  <rect x="70" y="11" width="10" height="10" rx="2" style="fill: var(--c2)"/>
  <text class="lab" x="85" y="20">softmax</text>
  <rect x="153.2" y="11" width="10" height="10" rx="2" style="fill: var(--c1)"/>
  <text class="lab" x="168.2" y="20">scores（QKᵀ + /√d + mask）</text>
  <rect x="359.4" y="11" width="10" height="10" rx="2" style="fill: var(--c3)"/>
  <text class="lab" x="374.4" y="20">PV</text>
  <rect x="409.59999999999997" y="11" width="10" height="10" rx="2" style="fill: var(--cg)"/>
  <text class="lab" x="424.59999999999997" y="20">attention 合计</text>
  <line class="grid" x1="70" y1="236.0" x2="520" y2="236.0"/>
  <text class="tick" x="62" y="240.0" text-anchor="end">0%</text>
  <line class="grid" x1="70" y1="200.0" x2="520" y2="200.0"/>
  <text class="tick" x="62" y="204.0" text-anchor="end">10%</text>
  <line class="grid" x1="70" y1="164.0" x2="520" y2="164.0"/>
  <text class="tick" x="62" y="168.0" text-anchor="end">20%</text>
  <line class="grid" x1="70" y1="128.0" x2="520" y2="128.0"/>
  <text class="tick" x="62" y="132.0" text-anchor="end">30%</text>
  <line class="grid" x1="70" y1="92.0" x2="520" y2="92.0"/>
  <text class="tick" x="62" y="96.0" text-anchor="end">40%</text>
  <line class="grid" x1="70" y1="56.0" x2="520" y2="56.0"/>
  <text class="tick" x="62" y="60.0" text-anchor="end">50%</text>
  <text class="tick" x="110" y="254" text-anchor="middle">seq 256</text>
  <text class="tick" x="290" y="254" text-anchor="middle">seq 512</text>
  <text class="tick" x="470" y="254" text-anchor="middle">seq 1024</text>
  <text class="lab2" x="70" y="46">占 forward GPU 时间（medium，24 层）</text>
  <polyline points="110,200.0 290,156.8 470,70.4" fill="none" style="stroke: var(--cg)" stroke-width="2" stroke-linejoin="round" stroke-dasharray="6 4"/>
  <g class="m"><title>seq 256：合计 2.3 ms，占 10%</title><circle cx="110" cy="200.0" r="4.5" class="ring" style="fill: var(--cg)"/></g>
  <g class="m"><title>seq 512：合计 10.2 ms，占 22%</title><circle cx="290" cy="156.8" r="4.5" class="ring" style="fill: var(--cg)"/></g>
  <g class="m"><title>seq 1024：合计 64.9 ms，占 46%</title><circle cx="470" cy="70.4" r="4.5" class="ring" style="fill: var(--cg)"/></g>
  <text class="val" x="482" y="74.4">合计 46%</text>
  <polyline points="110,221.6 290,196.4 470,149.6" fill="none" style="stroke: var(--c2)" stroke-width="2" stroke-linejoin="round"/>
  <g class="m"><title>seq 256：softmax 1.0 ms，占 4%</title><circle cx="110" cy="221.6" r="4.5" class="ring" style="fill: var(--c2)"/></g>
  <g class="m"><title>seq 512：softmax 5.0 ms，占 11%</title><circle cx="290" cy="196.4" r="4.5" class="ring" style="fill: var(--c2)"/></g>
  <g class="m"><title>seq 1024：softmax 34.3 ms，占 24%</title><circle cx="470" cy="149.6" r="4.5" class="ring" style="fill: var(--c2)"/></g>
  <text class="val" x="482" y="153.6">softmax 24%</text>
  <polyline points="110,221.6 290,207.2 470,171.2" fill="none" style="stroke: var(--c1)" stroke-width="2" stroke-linejoin="round"/>
  <g class="m"><title>seq 256：scores 0.9 ms，占 4%</title><circle cx="110" cy="221.6" r="4.5" class="ring" style="fill: var(--c1)"/></g>
  <g class="m"><title>seq 512：scores 3.6 ms，占 8%</title><circle cx="290" cy="207.2" r="4.5" class="ring" style="fill: var(--c1)"/></g>
  <g class="m"><title>seq 1024：scores 25.1 ms，占 18%</title><circle cx="470" cy="171.2" r="4.5" class="ring" style="fill: var(--c1)"/></g>
  <text class="val" x="482" y="175.2">scores 18%</text>
  <polyline points="110,228.8 290,225.2 470,221.6" fill="none" style="stroke: var(--c3)" stroke-width="2" stroke-linejoin="round"/>
  <g class="m"><title>seq 256：PV 0.4 ms，占 2%</title><circle cx="110" cy="228.8" r="4.5" class="ring" style="fill: var(--c3)"/></g>
  <g class="m"><title>seq 512：PV 1.6 ms，占 3%</title><circle cx="290" cy="225.2" r="4.5" class="ring" style="fill: var(--c3)"/></g>
  <g class="m"><title>seq 1024：PV 5.5 ms，占 4%</title><circle cx="470" cy="221.6" r="4.5" class="ring" style="fill: var(--c3)"/></g>
  <text class="val" x="482" y="225.6">PV 4%</text>
</svg>
<figcaption><strong>图 3-3</strong> attention 三段占 forward GPU 时间的比例随 seq 变化（medium）。</figcaption>
</figure>

- **seq 变长后？** S、P 的形状是 `[b, h, seq, seq]`，读写量 ∝ seq²，Linear 的 ∝ seq。attention 三段的占比从 10% 涨到 46%，增加的部分主要在 softmax 和 scores 上（scores 里大半是 `/√d`、mask 这类逐元素 op），PV 只从 2% 到 4%。

> **结论**：GPU 时间看张量在显存里读写了几遍，而不是 FLOPs。memory-bound 的 op 只能靠融合提速：几步放进一个 kernel，中间结果留在片上，不写回显存。

S、P 除了读写次数多，还是 softmax 和 PV 的输入，反向要用，所以会一直留在显存里（和[图 2-1](#fig-2-1) 里的 x 一样），这部分在下一节算。

---

## 4 显存峰值：W + max(A, G) {#peak-memory}

**纸面估算**：fp32 + AdamW 训练，每个参数要 16 B：权重 4、梯度 4、Adam 的 m 和 v 各 4。另外还有 activation（为反向存下的中间张量），与 batch × seq 成正比。xl（3.41B）只算这 16 B/参数就要 50.8 GiB，超过 5090 的显存；10B 在建模型时就 OOM。下文的记号：

- 🟦 **W** 全部权重；🟥 **G** 全部梯度 `.grad`，大小等于 W；Adam 的 m、v 合计 2W，第一步之后常驻；
- 🟩 **A** 前向为反向存下的张量（autograd 的 saved tensors）；🟨 **T** 正在算的那一层的临时量，算完即释放。

[图 4-1](#fig-4-1) 把纸面估算和实测的 full step 峰值按组成画在一起：

<figure id="fig-4-1" class="cv-fig">
<svg class="cv" viewBox="0 0 640 304" width="100%" role="img" aria-label="三档模型 full step 的峰值显存，纸面估算与实测对比：实测少的正好是梯度 G 这一块">
  <style>
    .cv { --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#e34948; --cg:#a19f9a; }
    html.dark .cv { --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#e66767; --cg:#6f6e6a; }
    .cv .grid { stroke: currentColor; stroke-opacity: .1; }
    .cv .axis { stroke: currentColor; stroke-opacity: .35; }
    .cv .ref { stroke: currentColor; stroke-opacity: .55; stroke-dasharray: 5 4; }
    .cv .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .lab { font-size: 12px; fill: currentColor; }
    .cv .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .cv .val { font-size: 11px; fill: currentColor; }
    .cv .in { font-size: 11px; fill: #111; }
    .cv .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .cv g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .cv-fig { overflow-x: auto; } .cv-fig > svg { min-width: 540px; } }
  </style>
  <rect x="70" y="11" width="10" height="10" rx="2" style="fill: var(--c1)"/>
  <text class="lab" x="85" y="20">W 权重</text>
  <rect x="144.2" y="11" width="10" height="10" rx="2" style="fill: var(--c2)"/>
  <text class="lab" x="159.2" y="20">Adam m、v</text>
  <rect x="239.39999999999998" y="11" width="10" height="10" rx="2" style="fill: var(--c3)"/>
  <text class="lab" x="254.39999999999998" y="20">A activation</text>
  <rect x="355.59999999999997" y="11" width="10" height="10" rx="2" style="fill: var(--c4)"/>
  <text class="lab" x="370.59999999999997" y="20">G 梯度</text>
  <line class="grid" x1="70" y1="262.0" x2="630" y2="262.0"/>
  <text class="tick" x="62" y="266.0" text-anchor="end">0</text>
  <line class="grid" x1="70" y1="200.0" x2="630" y2="200.0"/>
  <text class="tick" x="62" y="204.0" text-anchor="end">10</text>
  <line class="grid" x1="70" y1="138.0" x2="630" y2="138.0"/>
  <text class="tick" x="62" y="142.0" text-anchor="end">20</text>
  <line class="grid" x1="70" y1="76.0" x2="630" y2="76.0"/>
  <text class="tick" x="62" y="80.0" text-anchor="end">30</text>
  <text class="lab2" x="70" y="40">GiB</text>
  <line class="ref" x1="70" y1="67.9" x2="630" y2="67.9"/>
  <text class="lab2" x="630" y="61.9" text-anchor="end">5090 可用 31.3 GiB</text>
  <g class="m"><title>small 纸面：W 0.48 GiB</title><rect x="112.0" y="260.0" width="44.0" height="3.0" rx="2" style="fill: var(--c1)"/></g>
  <g class="m"><title>small 纸面：Adam 0.96 GiB</title><rect x="112.0" y="254.1" width="44.0" height="4.0" rx="2" style="fill: var(--c2)"/></g>
  <g class="m"><title>small 纸面：A 3.50 GiB</title><rect x="112.0" y="232.4" width="44.0" height="19.7" rx="2" style="fill: var(--c3)"/></g>
  <text class="in" x="134.0" y="246.2" text-anchor="middle">A</text>
  <g class="m"><title>small 纸面：G 0.48 GiB</title><rect x="112.0" y="229.4" width="44.0" height="3.0" rx="2" style="fill: var(--c4)"/></g>
  <text class="val" x="134.0" y="222.4" text-anchor="middle">5.4</text>
  <text class="lab2" x="134.0" y="277" text-anchor="middle">纸面</text>
  <g class="m"><title>small 实测：W 0.48 GiB</title><rect x="164.0" y="260.0" width="44.0" height="3.0" rx="2" style="fill: var(--c1)"/></g>
  <g class="m"><title>small 实测：Adam 0.96 GiB</title><rect x="164.0" y="254.1" width="44.0" height="4.0" rx="2" style="fill: var(--c2)"/></g>
  <g class="m"><title>small 实测：A 3.50 GiB</title><rect x="164.0" y="232.4" width="44.0" height="19.7" rx="2" style="fill: var(--c3)"/></g>
  <text class="in" x="186.0" y="246.2" text-anchor="middle">A</text>
  <g class="m"><title>small 实测：其他 0.10 GiB</title><rect x="164.0" y="231.8" width="44.0" height="0.6" rx="2" style="fill: var(--cg)"/></g>
  <text class="val" x="186.0" y="224.8" text-anchor="middle">5.0</text>
  <text class="lab2" x="186.0" y="277" text-anchor="middle">实测</text>
  <text class="lab" x="160" y="294" text-anchor="middle">small 0.13B</text>
  <g class="m"><title>medium 纸面：W 1.58 GiB</title><rect x="262.0" y="253.2" width="44.0" height="7.8" rx="2" style="fill: var(--c1)"/></g>
  <g class="m"><title>medium 纸面：Adam 3.16 GiB</title><rect x="262.0" y="233.6" width="44.0" height="17.6" rx="2" style="fill: var(--c2)"/></g>
  <text class="in" x="284.0" y="246.4" text-anchor="middle">Adam</text>
  <g class="m"><title>medium 纸面：A 8.91 GiB</title><rect x="262.0" y="178.4" width="44.0" height="53.2" rx="2" style="fill: var(--c3)"/></g>
  <text class="in" x="284.0" y="209.0" text-anchor="middle">A</text>
  <g class="m"><title>medium 纸面：G 1.58 GiB</title><rect x="262.0" y="168.6" width="44.0" height="7.8" rx="2" style="fill: var(--c4)"/></g>
  <text class="val" x="284.0" y="161.6" text-anchor="middle">15.2</text>
  <text class="lab2" x="284.0" y="277" text-anchor="middle">纸面</text>
  <g class="m"><title>medium 实测：W 1.58 GiB</title><rect x="314.0" y="253.2" width="44.0" height="7.8" rx="2" style="fill: var(--c1)"/></g>
  <g class="m"><title>medium 实测：Adam 3.16 GiB</title><rect x="314.0" y="233.6" width="44.0" height="17.6" rx="2" style="fill: var(--c2)"/></g>
  <text class="in" x="336.0" y="246.4" text-anchor="middle">Adam</text>
  <g class="m"><title>medium 实测：A 8.91 GiB</title><rect x="314.0" y="178.4" width="44.0" height="53.2" rx="2" style="fill: var(--c3)"/></g>
  <text class="in" x="336.0" y="209.0" text-anchor="middle">A</text>
  <g class="m"><title>medium 实测：其他 0.09 GiB</title><rect x="314.0" y="177.8" width="44.0" height="0.6" rx="2" style="fill: var(--cg)"/></g>
  <text class="val" x="336.0" y="170.8" text-anchor="middle">13.7</text>
  <text class="lab2" x="336.0" y="277" text-anchor="middle">实测</text>
  <text class="lab" x="310" y="294" text-anchor="middle">medium 0.42B</text>
  <g class="m"><title>large 纸面：W 3.61 GiB</title><rect x="412.0" y="240.6" width="44.0" height="20.4" rx="2" style="fill: var(--c1)"/></g>
  <text class="in" x="434.0" y="254.8" text-anchor="middle">W</text>
  <g class="m"><title>large 纸面：Adam 7.22 GiB</title><rect x="412.0" y="195.9" width="44.0" height="42.8" rx="2" style="fill: var(--c2)"/></g>
  <text class="in" x="434.0" y="221.2" text-anchor="middle">Adam</text>
  <g class="m"><title>large 纸面：A 16.58 GiB</title><rect x="412.0" y="93.1" width="44.0" height="100.8" rx="2" style="fill: var(--c3)"/></g>
  <text class="in" x="434.0" y="147.5" text-anchor="middle">A</text>
  <g class="m"><title>large 纸面：G 3.61 GiB</title><rect x="412.0" y="70.7" width="44.0" height="20.4" rx="2" style="fill: var(--c4)"/></g>
  <text class="in" x="434.0" y="84.9" text-anchor="middle">G</text>
  <text class="val" x="434.0" y="63.7" text-anchor="middle">31.0</text>
  <text class="lab2" x="434.0" y="277" text-anchor="middle">纸面</text>
  <g class="m"><title>large 实测：W 3.61 GiB</title><rect x="464.0" y="240.6" width="44.0" height="20.4" rx="2" style="fill: var(--c1)"/></g>
  <text class="in" x="486.0" y="254.8" text-anchor="middle">W</text>
  <g class="m"><title>large 实测：Adam 7.22 GiB</title><rect x="464.0" y="195.9" width="44.0" height="42.8" rx="2" style="fill: var(--c2)"/></g>
  <text class="in" x="486.0" y="221.2" text-anchor="middle">Adam</text>
  <g class="m"><title>large 实测：A 16.58 GiB</title><rect x="464.0" y="93.1" width="44.0" height="100.8" rx="2" style="fill: var(--c3)"/></g>
  <text class="in" x="486.0" y="147.5" text-anchor="middle">A</text>
  <g class="m"><title>large 实测：其他 0.10 GiB</title><rect x="464.0" y="92.4" width="44.0" height="0.6" rx="2" style="fill: var(--cg)"/></g>
  <text class="val" x="486.0" y="85.4" text-anchor="middle">27.5</text>
  <text class="lab2" x="486.0" y="277" text-anchor="middle">实测</text>
  <text class="lab" x="460" y="294" text-anchor="middle">large 0.97B</text>
  <line class="axis" x1="70" y1="262" x2="630" y2="262"/>
</svg>
<figcaption><strong>图 4-1</strong> full step 的峰值显存：纸面 16 B/参数 + A 与实测对比（batch 4，seq 512）。实测里没有 G，「其他」是 ~0.1 GiB 的临时量。</figcaption>
</figure>

- 实测比纸面少的正好是 🟥 G 这一块（large 27.5 vs 31.0 GiB）。
- A 由带梯度的前向峰值减去 W 得到：small 3.5、medium 8.9、large 16.6 GiB，是权重的 4.6–7.3 倍。fwd_bwd 的峰值只比带梯度的前向多 0.1 GiB；full 再多 2W，就是 Adam 的 m、v。

少的这个 W 是梯度：纸面估算假设梯度一直占着显存，实际并不是，要看显存随时间怎么变化。

### 4.1 峰值落在哪一刻 {#peak-moment}

反向走完 j 层（共 L 层）时，活着的显存是

<p align="center">$M(j) = W + G \cdot \dfrac{j}{L} + A \cdot \dfrac{L-j}{L} + T$</p>

🟦 W 常驻；🟩 A 前向逐层堆上、反向逐层放掉；🟥 G 反向逐层长出来；🟨 T 层内即造即释。

> **梯度不是一直都在**：`.grad` 在反向算到对应参数时才分配，optimizer step 之后被 `zero_grad(set_to_none=True)`（PyTorch 2.0 起的默认）释放。所以前向期间没有梯度，反向期间从 0 增长到 W。

M(j) 是 j 的线性函数，最大值只能在两端：A > G 时在前向结束时（W + A），G > A 时在反向结束时（W + G）。full 模式还要加上常驻的 Adam 状态 2W：

<p align="center">$\mathrm{peak}_{\mathrm{full}} \approx 3W + \max(A,\ G)$</p>

用 large 验证：3 × 3.61 + 16.58 = 27.4，实测 27.51。token 数正常时 A > G（[图 4-1](#fig-4-1) 三档都是）；token 很少时反过来，比如 xl 在 seq 128 时 A 5.3 < G 12.7，峰值出现在反向结束时，等于 2W（[图 4-2](#fig-4-2) 左）。

<div id="fig-4-2"></div>

![一步 fwd_bwd 的显存按 W、G、A、T 堆叠](peak_moment.png "**图 4-2** 一步 fwd_bwd 的显存：色带按 M(j) 用实测的 W、G、A、T 堆叠，× 是逐层实测值。左 xl@128（G > A），右 small@512（A > G）；虚线是 bf16。")

峰值里最大、而且随 seq 变化的一项是 🟩 A，下面看它由哪些张量组成。

### 4.2 A 里存的是什么：拆开 RMSNorm {#rmsnorm}

`torch.autograd.graph.saved_tensors_hooks` 可以在 pack / unpack 时打印每个张量，精确到每个 op。先用最简单的 RMSNorm（fp32，`x: [4, 512, 2560]`）看规则，再看 `torch.compile` 融合之后的变化。

$\mathrm{RMSNorm}(x)_i = w_i \cdot \dfrac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}}$，拆成 5 个 op：

<p align="center">$\underbrace{r = \big(\underbrace{\tfrac{1}{d}\textstyle\sum_j \underbrace{x_j^2}_{\text{①}}}_{\text{②}} + \epsilon\big)^{-1/2}}_{\text{③}}$，$\underbrace{\hat{x} = x \cdot r}_{\text{④}}$，$\underbrace{y = w \odot \hat{x}}_{\text{⑤}}$</p>

```python
rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # ①②③
x_hat = x * rms                                            # ④
y = weight * x_hat                                         # ⑤
```

pack / unpack 打印：

```
Saving  1  [4,512,2560]  grad_fn=None            ptr=…9040
Saving  2  [4,512,1]     grad_fn=RsqrtBackward0  ptr=…6b00
Saving  3  [4,512,1]     grad_fn=RsqrtBackward0  ptr=…6b00
Saving  4  [4,512,2560]  grad_fn=None            ptr=…9040
Saving  5  [4,512,2560]  grad_fn=MulBackward0    ptr=…3c00
Saving  6  [2560]        grad_fn=None            ptr=…1000
Loading    5 → 6 → 2 → 4 → 3 → 1
```

规则是：**一个 op 的局部偏导里用到哪个变量，前向就要把它存下来；偏导是常数就不用存。**这里的「存」只是反向节点持有一个引用，不会拷贝，所以输入 `x` 和参数 `w` 本来就在显存里，不额外占空间。

| op | 前向 | FLOPs / 元素 | 反向要的偏导 | 存 | 额外显存 | print |
|:--|:--|--:|:--|:--|--:|:--|
| ① | $x^2$ | 1 | $\partial x^2/\partial x = 2x$ | $x$ | 0 | 1 |
| ② | $v=\tfrac1d\sum x^2$ | 1 | $\partial v/\partial x^2 = \tfrac1d$ | 常数，不存 | 0 | — |
| ③ | $r=(v+\epsilon)^{-1/2}$ | 每行 2 | $\partial r/\partial v = -\tfrac12 r^3$ | $r$ | **8 KiB** | 3 |
| ④ | $\hat{x}=x\cdot r$ | 1 | $\partial\hat{x}/\partial x = r$，$\partial\hat{x}/\partial r = x$ | $r$、$x$ | 0 | 2、4 |
| ⑤ | $y=w\odot\hat{x}$ | 1 | $\partial y/\partial w = \hat{x}$，$\partial y/\partial\hat{x} = w$ | $\hat{x}$、$w$ | **20 MiB** | 5、6 |
{#tab-4-1 caption="**表 4-1** RMSNorm 五个 op 的 FLOPs 与为反向存的张量（eager）" note="FLOPs 是纸面计数；print 列是上面打印里的第几条 Saving。"}

额外显存为 0 的几项：$x$ 是输入，$w$ 是参数，本来就在显存里；④ 存的 $r$ 和 ③ 是同一块。6 次 Saving 实际只对应 4 块内存（ptr 显示第 1、4 条同址，第 2、3 条同址），新增的只有 $r$（`[4,512,1]`）和 $\hat{x}$（`[4,512,2560]`），约等于一份输入的大小。[图 4-3](#fig-4-3) 把计算和存储分开画：

<figure id="fig-4-3" class="gx-fig">
<svg viewBox="0 0 640 332" width="100%" role="img" aria-label="RMSNorm eager 的前向、为反向留下的张量和反向：前向 5 个算子每元素约 1 次运算；显存里多留 r 8 KiB 和 x̂ 20 MiB；反向各步读取这些张量">
  <style>
    .gx-band { fill: currentColor; fill-opacity: .045; }
    .gx-row { font-size: 12px; fill: currentColor; opacity: .8; }
    .gx-rowsub { font-size: 10.5px; fill: currentColor; opacity: .55; }
    .gx-op { fill: rgba(128,128,128,.12); stroke: currentColor; stroke-opacity: .5; stroke-width: 1.2; }
    .gx-t { font-size: 12.5px; fill: currentColor; }
    .gx-tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .gx-s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .gx-note { font-size: 11px; fill: currentColor; opacity: .7; }
    @media (max-width: 640px) { .gx-fig { overflow-x: auto; } .gx-fig > svg { min-width: 540px; } }
  </style>
  <defs>
    <marker id="fig-4-3-m0" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker>
    <marker id="fig-4-3-m1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#fb8c00"/></marker>
    <marker id="fig-4-3-m2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#1e88e5"/></marker>
    <marker id="fig-4-3-m3" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#ab47bc"/></marker>
    <marker id="fig-4-3-m4" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#43a047"/></marker>
  </defs>
  <rect class="gx-band" x="74" y="18" width="560" height="74" rx="8"/>
  <text class="gx-row" x="6" y="53">前向</text>
  <text class="gx-rowsub" x="6" y="68">花 FLOPs</text>
  <rect class="gx-band" x="74" y="124" width="560" height="72" rx="8"/>
  <text class="gx-row" x="6" y="158">显存</text>
  <text class="gx-rowsub" x="6" y="173">为反向留下</text>
  <rect class="gx-band" x="74" y="228" width="560" height="74" rx="8"/>
  <text class="gx-row" x="6" y="263">反向</text>
  <text class="gx-rowsub" x="6" y="278">再花 FLOPs</text>
  <rect x="108.0" y="32.0" width="76" height="46" rx="6" class="gx-op"/>
  <text class="gx-tb" x="146" y="47.5" text-anchor="middle" dominant-baseline="central">① x²</text>
  <text class="gx-s" x="146" y="62.5" text-anchor="middle" dominant-baseline="central">1 FLOP/元素</text>
  <rect x="194.0" y="32.0" width="76" height="46" rx="6" class="gx-op"/>
  <text class="gx-tb" x="232" y="47.5" text-anchor="middle" dominant-baseline="central">② mean</text>
  <text class="gx-s" x="232" y="62.5" text-anchor="middle" dominant-baseline="central">1 FLOP/元素</text>
  <rect x="280.0" y="32.0" width="76" height="46" rx="6" class="gx-op"/>
  <text class="gx-tb" x="318" y="47.5" text-anchor="middle" dominant-baseline="central">③ rsqrt</text>
  <text class="gx-s" x="318" y="62.5" text-anchor="middle" dominant-baseline="central">每行 2 FLOPs</text>
  <rect x="366.0" y="32.0" width="76" height="46" rx="6" class="gx-op"/>
  <text class="gx-tb" x="404" y="47.5" text-anchor="middle" dominant-baseline="central">④ x · r</text>
  <text class="gx-s" x="404" y="62.5" text-anchor="middle" dominant-baseline="central">1 FLOP/元素</text>
  <rect x="452.0" y="32.0" width="76" height="46" rx="6" class="gx-op"/>
  <text class="gx-tb" x="490" y="47.5" text-anchor="middle" dominant-baseline="central">⑤ w ⊙ x̂</text>
  <text class="gx-s" x="490" y="62.5" text-anchor="middle" dominant-baseline="central">1 FLOP/元素</text>
  <line x1="184.0" y1="55.0" x2="191.0" y2="55.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <line x1="270.0" y1="55.0" x2="277.0" y2="55.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <line x1="356.0" y1="55.0" x2="363.0" y2="55.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <line x1="442.0" y1="55.0" x2="449.0" y2="55.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <text class="gx-t" x="84.0" y="59.0" text-anchor="start">x</text>
  <line x1="95.0" y1="55.0" x2="105.0" y2="55.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <line x1="528.0" y1="55.0" x2="580.0" y2="55.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <text class="gx-t" x="587.0" y="59.0" text-anchor="start">y</text>
  <rect x="129.0" y="138.0" width="120" height="44" rx="6" fill="rgba(67,160,71,0.16)" stroke="#43a047" stroke-width="1.4" stroke-dasharray="4 3"/>
  <text class="gx-t" x="189" y="152.5" text-anchor="middle" dominant-baseline="central">x  …9040</text>
  <text class="gx-s" x="189" y="167.5" text-anchor="middle" dominant-baseline="central">0：输入本来就在</text>
  <rect x="313.0" y="138.0" width="96" height="44" rx="6" fill="rgba(171,71,188,0.16)" stroke="#ab47bc" stroke-width="1.4"/>
  <text class="gx-t" x="361" y="152.5" text-anchor="middle" dominant-baseline="central">r  …6b00</text>
  <text class="gx-tb" x="361" y="167.5" text-anchor="middle" dominant-baseline="central">+8 KiB</text>
  <rect x="446.0" y="138.0" width="88" height="44" rx="6" fill="rgba(251,140,0,0.16)" stroke="#fb8c00" stroke-width="2"/>
  <text class="gx-t" x="490" y="152.5" text-anchor="middle" dominant-baseline="central">x̂  …3c00</text>
  <text class="gx-tb" x="490" y="167.5" text-anchor="middle" dominant-baseline="central">+20 MiB</text>
  <rect x="544.0" y="138.0" width="88" height="44" rx="6" fill="rgba(30,136,229,0.16)" stroke="#1e88e5" stroke-width="1.4" stroke-dasharray="4 3"/>
  <text class="gx-t" x="588" y="152.5" text-anchor="middle" dominant-baseline="central">w  …1000</text>
  <text class="gx-s" x="588" y="167.5" text-anchor="middle" dominant-baseline="central">0：参数</text>
  <line x1="324.0" y1="78.0" x2="341.0" y2="138.0" stroke="#ab47bc" stroke-width="1.2" fill="none"/>
  <line x1="398.0" y1="78.0" x2="381.0" y2="138.0" stroke="#ab47bc" stroke-width="1.2" fill="none"/>
  <line x1="490.0" y1="78.0" x2="490.0" y2="138.0" stroke="#fb8c00" stroke-width="1.2" fill="none"/>
  <rect x="108.0" y="248.0" width="76" height="34" rx="6" class="gx-op"/>
  <text class="gx-t" x="146" y="265.0" text-anchor="middle" dominant-baseline="central">① 反向</text>
  <rect x="194.0" y="248.0" width="76" height="34" rx="6" class="gx-op"/>
  <text class="gx-t" x="232" y="265.0" text-anchor="middle" dominant-baseline="central">② 反向</text>
  <rect x="280.0" y="248.0" width="76" height="34" rx="6" class="gx-op"/>
  <text class="gx-t" x="318" y="265.0" text-anchor="middle" dominant-baseline="central">③ 反向</text>
  <rect x="366.0" y="248.0" width="76" height="34" rx="6" class="gx-op"/>
  <text class="gx-t" x="404" y="265.0" text-anchor="middle" dominant-baseline="central">④ 反向</text>
  <rect x="452.0" y="248.0" width="76" height="34" rx="6" class="gx-op"/>
  <text class="gx-t" x="490" y="265.0" text-anchor="middle" dominant-baseline="central">⑤ 反向</text>
  <line x1="194.0" y1="265.0" x2="187.0" y2="265.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <line x1="280.0" y1="265.0" x2="273.0" y2="265.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <line x1="366.0" y1="265.0" x2="359.0" y2="265.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <line x1="452.0" y1="265.0" x2="445.0" y2="265.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <text class="gx-t" x="600.0" y="269.0" text-anchor="start">dy</text>
  <line x1="596.0" y1="265.0" x2="531.0" y2="265.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <line x1="108.0" y1="265.0" x2="102.0" y2="265.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-3-m0)"/>
  <text class="gx-t" x="82.0" y="269.0" text-anchor="start">dx</text>
  <line x1="490.0" y1="182.0" x2="490.0" y2="246.0" stroke="#fb8c00" stroke-width="1.6" fill="none" stroke-dasharray="5 3" marker-end="url(#fig-4-3-m1)"/>
  <line x1="568.0" y1="182.0" x2="508.0" y2="246.0" stroke="#1e88e5" stroke-width="1.6" fill="none" stroke-dasharray="5 3" marker-end="url(#fig-4-3-m2)"/>
  <line x1="375.0" y1="182.0" x2="400.0" y2="246.0" stroke="#ab47bc" stroke-width="1.6" fill="none" stroke-dasharray="5 3" marker-end="url(#fig-4-3-m3)"/>
  <line x1="347.0" y1="182.0" x2="322.0" y2="246.0" stroke="#ab47bc" stroke-width="1.6" fill="none" stroke-dasharray="5 3" marker-end="url(#fig-4-3-m3)"/>
  <line x1="229.0" y1="182.0" x2="382.0" y2="246.0" stroke="#43a047" stroke-width="1.6" fill="none" stroke-dasharray="5 3" marker-end="url(#fig-4-3-m4)"/>
  <line x1="149.0" y1="182.0" x2="146.0" y2="246.0" stroke="#43a047" stroke-width="1.6" fill="none" stroke-dasharray="5 3" marker-end="url(#fig-4-3-m4)"/>
  <text class="gx-note" x="232.0" y="295.0" text-anchor="middle">偏导是常数 1/d，什么都不读</text>
</svg>
<figcaption><strong>图 4-3</strong> RMSNorm（eager）：上排算子花 FLOPs，中排张量占显存（实线框新占，虚线框本来就在），下排是反向，虚线箭头是它读回的张量。</figcaption>
</figure>

反向从 ⑤ 到 ①：⑤ 读 $\hat{x}$ 和 $w$，④ 读 $r$ 和 $x$，③ 读 $r$，① 读 $x$，和打印的 Loading 顺序一致。$x$ 被 ① 和 ④ 各读一次，所以 $x$ 的梯度有两路，最后在叶子节点上累加。

计算量上，每个元素只有 ~4 次运算，而 5 个 kernel 每个都要读写一遍 20 MiB 的张量（纸面合计 ~140 MiB），I ≈ 0.14，和 softmax 一样是 memory-bound。显存上则多存了一份输入大小的 $\hat{x}$。这两项都可以通过融合减少。

**`torch.compile` 融合后存什么？** [表 4-1](#tab-4-1) 里 ⑤ 要的 $\hat{x}$ 只是 $x \cdot r$ 的一次逐元素乘，反向时有 $x$ 和 $r$ 就能重新算出来，不需要存这 20 MiB。逐算子执行时做不到，因为 ⑤ 的 MulBackward 只知道需要 $\hat{x}$，不知道它来自 $x \cdot r$。`torch.compile(RMSNorm(...))` 会把 ①–⑤ 追踪成一张图，由 AOTAutograd 生成一个前向 kernel 和一个反向 kernel，并在存和重算之间选开销小的：

```
Saving  1  [4,512,2560]  grad_fn=None  ptr=…3c80   # x
Saving  2  [2560]        grad_fn=None  ptr=…b3c0   # w
Saving  3  [4,512,1]     grad_fn=None  ptr=…f9c0   # r
Loading    1 → 2 → 3（与 Saving 同序）
```

| | eager | 融合后 |
|:--|:--|:--|
| kernel 数 | 前向 5 个 | 前向 1 个 + 反向 1 个 |
| 为反向存的张量 | $x$、$w$、$r$、$\hat{x}$（6 次 Saving，4 块内存） | $x$、$w$、$r$（3 次 Saving） |
| 额外显存 | $r + \hat{x}$ ≈ 20 MiB | $r$ ≈ 8 KiB |
| 前向读写显存 | ~140 MiB | ~40 MiB（只读 $x$、写 $y$） |
| FLOPs / 元素 | 前向 ~4 | 前向 ~4，反向多 1（重算 $\hat{x} = x\cdot r$） |
{#tab-4-2 caption="**表 4-2** RMSNorm：eager 与 `torch.compile` 融合" note="存的张量是实测，FLOPs 和读写是纸面计数。"}

反向 kernel 里的公式是 $\partial y/\partial w = \hat{x}$ 和 $\partial y/\partial x = r\,w\odot(I-\tfrac1d\hat{x}\hat{x}^{\!\top})$，其中的 $\hat{x}$ 都用 $x\cdot r$ 现算（[图 4-4](#fig-4-4)）。

<figure id="fig-4-4" class="gx-fig">
<svg viewBox="0 0 640 300" width="100%" role="img" aria-label="torch.compile 融合后的 RMSNorm：前向一个 kernel；显存只多留 r 8 KiB，x̂ 不存；反向一个 kernel，用 x 和 r 现场重算 x̂">
  <style>
    .gx-band { fill: currentColor; fill-opacity: .045; }
    .gx-row { font-size: 12px; fill: currentColor; opacity: .8; }
    .gx-rowsub { font-size: 10.5px; fill: currentColor; opacity: .55; }
    .gx-op { fill: rgba(128,128,128,.12); stroke: currentColor; stroke-opacity: .5; stroke-width: 1.2; }
    .gx-t { font-size: 12.5px; fill: currentColor; }
    .gx-tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .gx-s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .gx-note { font-size: 11px; fill: currentColor; opacity: .7; }
    @media (max-width: 640px) { .gx-fig { overflow-x: auto; } .gx-fig > svg { min-width: 540px; } }
  </style>
  <defs>
    <marker id="fig-4-4-m0" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker>
    <marker id="fig-4-4-m1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#43a047"/></marker>
    <marker id="fig-4-4-m2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#ab47bc"/></marker>
    <marker id="fig-4-4-m3" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#1e88e5"/></marker>
  </defs>
  <rect class="gx-band" x="74" y="18" width="560" height="68" rx="8"/>
  <text class="gx-row" x="6" y="50">前向</text>
  <text class="gx-rowsub" x="6" y="65">花 FLOPs</text>
  <rect class="gx-band" x="74" y="114" width="560" height="72" rx="8"/>
  <text class="gx-row" x="6" y="148">显存</text>
  <text class="gx-rowsub" x="6" y="163">为反向留下</text>
  <rect class="gx-band" x="74" y="214" width="560" height="72" rx="8"/>
  <text class="gx-row" x="6" y="248">反向</text>
  <text class="gx-rowsub" x="6" y="263">再花 FLOPs</text>
  <rect x="180.0" y="30.0" width="300" height="44" rx="6" class="gx-op"/>
  <text class="gx-tb" x="330" y="44.5" text-anchor="middle" dominant-baseline="central">fused forward：①–⑤ 一个 kernel</text>
  <text class="gx-s" x="330" y="59.5" text-anchor="middle" dominant-baseline="central">~4 FLOPs/元素</text>
  <text class="gx-t" x="80.0" y="56.0" text-anchor="start">x</text>
  <line x1="90.0" y1="52.0" x2="177.0" y2="52.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-4-m0)"/>
  <line x1="480.0" y1="52.0" x2="585.0" y2="52.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-4-m0)"/>
  <text class="gx-t" x="592.0" y="56.0" text-anchor="start">y</text>
  <rect x="116.0" y="128.0" width="120" height="44" rx="6" fill="rgba(67,160,71,0.16)" stroke="#43a047" stroke-width="1.4" stroke-dasharray="4 3"/>
  <text class="gx-t" x="176" y="142.5" text-anchor="middle" dominant-baseline="central">x  …3c80</text>
  <text class="gx-s" x="176" y="157.5" text-anchor="middle" dominant-baseline="central">0：输入本来就在</text>
  <rect x="304.0" y="128.0" width="96" height="44" rx="6" fill="rgba(171,71,188,0.16)" stroke="#ab47bc" stroke-width="1.4"/>
  <text class="gx-t" x="352" y="142.5" text-anchor="middle" dominant-baseline="central">r  …f9c0</text>
  <text class="gx-tb" x="352" y="157.5" text-anchor="middle" dominant-baseline="central">+8 KiB</text>
  <rect x="438.0" y="128.0" width="92" height="44" rx="6" fill="rgba(251,140,0,0.16)" stroke="#fb8c00" stroke-width="1.4" stroke-dasharray="4 3"/>
  <text class="gx-t" x="484" y="142.5" text-anchor="middle" dominant-baseline="central">x̂ 不存</text>
  <text class="gx-s" x="484" y="157.5" text-anchor="middle" dominant-baseline="central">省下 20 MiB</text>
  <line x1="444.0" y1="166.0" x2="524.0" y2="134.0" stroke="#fb8c00" stroke-width="1.4" stroke-opacity=".7"/>
  <rect x="540.0" y="128.0" width="92" height="44" rx="6" fill="rgba(30,136,229,0.16)" stroke="#1e88e5" stroke-width="1.4" stroke-dasharray="4 3"/>
  <text class="gx-t" x="586" y="142.5" text-anchor="middle" dominant-baseline="central">w  …b3c0</text>
  <text class="gx-s" x="586" y="157.5" text-anchor="middle" dominant-baseline="central">0：参数</text>
  <line x1="352.0" y1="75.0" x2="352.0" y2="128.0" stroke="#ab47bc" stroke-width="1.2" fill="none"/>
  <rect x="140.0" y="228.0" width="380" height="44" rx="6" class="gx-op"/>
  <text class="gx-tb" x="330" y="242.5" text-anchor="middle" dominant-baseline="central">fused backward：∂y/∂w、∂y/∂x 一个 kernel</text>
  <text class="gx-s" x="330" y="257.5" text-anchor="middle" dominant-baseline="central">x̂ = x · r 现场重算，每元素多 1 FLOP</text>
  <line x1="186.0" y1="172.0" x2="216.0" y2="226.0" stroke="#43a047" stroke-width="1.6" fill="none" stroke-dasharray="5 3" marker-end="url(#fig-4-4-m1)"/>
  <line x1="352.0" y1="172.0" x2="352.0" y2="226.0" stroke="#ab47bc" stroke-width="1.6" fill="none" stroke-dasharray="5 3" marker-end="url(#fig-4-4-m2)"/>
  <line x1="566.0" y1="172.0" x2="500.0" y2="226.0" stroke="#1e88e5" stroke-width="1.6" fill="none" stroke-dasharray="5 3" marker-end="url(#fig-4-4-m3)"/>
  <text class="gx-t" x="600.0" y="254.0" text-anchor="start">dy</text>
  <line x1="596.0" y1="250.0" x2="523.0" y2="250.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-4-m0)"/>
  <line x1="140.0" y1="250.0" x2="102.0" y2="250.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-4-4-m0)"/>
  <text class="gx-t" x="82.0" y="254.0" text-anchor="start">dx</text>
</svg>
<figcaption><strong>图 4-4</strong> 融合后的 RMSNorm（泳道同<a href="#fig-4-3">图 4-3</a>）：只多留 $r$（8 KiB），$\hat{x}$ 在反向用 $x$、$r$ 现场重算。</figcaption>
</figure>

> **融合用少量重算换显存和带宽**：额外显存从 $r + \hat{x}$ ≈ 20 MiB 降到 $r$ ≈ 8 KiB，前向读写从 ~140 MiB 降到 ~40 MiB（只读 $x$、写 $y$），代价是反向每个元素多一次乘法。[§5.2](#checkpoint) 的 checkpoint 在整层的粒度上做同样的事。

整层也按同样的规则存：每个矩阵乘的输入都要保存（[图 2-1](#fig-2-1) 的 x），attention 的 S、P 也在其中。

### 4.3 一层与整网：S、P 占一半以上 {#one-layer}

用 `saved_tensors_hooks` 统计 xl 的一层，`torch.compile` 已经按 [§4.2](#rmsnorm) 的方式去掉了 RMSNorm 这类逐元素中间量（[图 4-5](#fig-4-5)）：

<figure id="fig-4-5" class="cv-fig">
<svg class="cv" viewBox="0 0 640 246" width="100%" role="img" aria-label="xl 一层为反向存的 3655 MiB：S、P 占 56%，FFN 中间量 26%，[b, s, d] 级张量 17.5%，其他 0.2%">
  <style>
    .cv { --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#e34948; --cg:#a19f9a; }
    html.dark .cv { --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#e66767; --cg:#6f6e6a; }
    .cv .grid { stroke: currentColor; stroke-opacity: .1; }
    .cv .axis { stroke: currentColor; stroke-opacity: .35; }
    .cv .ref { stroke: currentColor; stroke-opacity: .55; stroke-dasharray: 5 4; }
    .cv .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .lab { font-size: 12px; fill: currentColor; }
    .cv .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .cv .val { font-size: 11px; fill: currentColor; }
    .cv .in { font-size: 11px; fill: #111; }
    .cv .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .cv g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .cv-fig { overflow-x: auto; } .cv-fig > svg { min-width: 540px; } }
  </style>
  <g class="m"><title>S、P（[b, h, s, s]）：2048 MiB，56.0%</title><path d="M150.00,30.00 A92,92 0 1 1 115.96,207.47 L128.54,175.88 A58,58 0 1 0 150.00,64.00 Z" class="ring" style="fill: var(--c2)"/></g>
  <g class="m"><title>FFN 中间量（[b, s, d_ff]）：960 MiB，26.3%</title><path d="M115.96,207.47 A92,92 0 0 1 67.50,81.28 L97.99,96.33 A58,58 0 0 0 128.54,175.88 Z" class="ring" style="fill: var(--c1)"/></g>
  <g class="m"><title>[b, s, d] 级张量（x、norm 输出、Q、K、V 等）：640 MiB，17.5%</title><path d="M67.50,81.28 A92,92 0 0 1 148.89,30.01 L149.30,64.00 A58,58 0 0 0 97.99,96.33 Z" class="ring" style="fill: var(--c3)"/></g>
  <g class="m"><title>其他（mask、RoPE、softmax 统计量）：7 MiB，0.2%</title><path d="M148.89,30.01 A92,92 0 0 1 150.00,30.00 L150.00,64.00 A58,58 0 0 0 149.30,64.00 Z" class="ring" style="fill: var(--cg)"/></g>
  <text class="ttl" x="150" y="120" text-anchor="middle">3655 MiB</text>
  <text class="lab2" x="150" y="137" text-anchor="middle">xl 一层</text>
  <rect x="290" y="52" width="12" height="12" rx="2" style="fill: var(--c2)"/>
  <text class="lab" x="310" y="62">S、P</text>
  <text class="lab2" x="310" y="78">[b, h, s, s]</text>
  <text class="val" x="630" y="62" text-anchor="end">2048 MiB · 56.0%</text>
  <rect x="290" y="92" width="12" height="12" rx="2" style="fill: var(--c1)"/>
  <text class="lab" x="310" y="102">FFN 中间量</text>
  <text class="lab2" x="310" y="118">[b, s, d_ff]</text>
  <text class="val" x="630" y="102" text-anchor="end">960 MiB · 26.3%</text>
  <rect x="290" y="132" width="12" height="12" rx="2" style="fill: var(--c3)"/>
  <text class="lab" x="310" y="142">[b, s, d] 级张量</text>
  <text class="lab2" x="310" y="158">x、norm 输出、Q、K、V 等</text>
  <text class="val" x="630" y="142" text-anchor="end">640 MiB · 17.5%</text>
  <rect x="290" y="172" width="12" height="12" rx="2" style="fill: var(--cg)"/>
  <text class="lab" x="310" y="182">其他</text>
  <text class="lab2" x="310" y="198">mask、RoPE、softmax 统计量</text>
  <text class="val" x="630" y="182" text-anchor="end">7 MiB · 0.2%</text>
</svg>
<figcaption><strong>图 4-5</strong> xl 一层为反向存的张量（batch 4，seq 2048，<code>torch.compile</code> 后用 <code>saved_tensors_hooks</code> 实测；这组用 16 头，S、P 各 1 GiB）。</figcaption>
</figure>

剩下的都是矩阵乘的输入，融合去不掉。xl 共 32 层，合计 114 GiB，远超 5090 的 31.3 GiB；其中一半以上是 S、P，也只有它们随 seq² 增长，其余随 seq 线性增长。

seq 变大后，这一项决定了能不能跑起来。用 `torch.cuda.memory._record_memory_history` 记录 xl（batch 4，32 头）一步里的每次分配和释放（[图 4-6](#fig-4-6)）：

<div id="fig-4-6"></div>

![xl 的四张显存时间线](mem_xl_timelines.png "**图 4-6** xl 的显存时间线，每个点是一次分配或释放：上排纯前向、下排带反向；左 seq 128，右 seq 2048。")

- **左上**（seq 128 纯前向）是平的，因为不为反向保存任何东西。
- **右上**（seq 2048 纯前向）有 32 个尖峰：每层 attention 的 `[b, h, s, s]` 中间量（S、`/√d`、mask、softmax 的三步）有 ~4 份 2 GiB 同时存在。
- **左下**（seq 128 full step）：前向逐层上升，反向继续上升（每层释放 🟩 166 MiB、新增 🟥 410 MiB），到 optimizer 分配 Adam 的 m、v 时 OOM。
- **右下**（seq 2048 带梯度的前向）：第 1 层留下 ~4.7 GiB，第 2 层的尖峰到 25.96 GiB 时 OOM。
- **为什么 seq 2048 时前向就 OOM？** 同样的 `[b, h, s, s]` 张量在 seq 128 时只有 8 MiB，seq 2048 时是 2 GiB，是残差流 `[b, s, d]` 张量（80 MiB）的 25 倍。纯前向时每层用完就释放，32 层都能跑完；带梯度的前向每层还要为反向留下 S、P，到第 2 层就放不下尖峰了。

要降低 A，可以让每个张量变小（bf16），也可以少存一些、反向时重算（checkpoint）。

---

## 5 bf16 与 checkpoint：能省多少 {#bf16-checkpoint}

### 5.1 bf16 autocast {#bf16}

autocast 只在算子调用时把矩阵乘的输入 cast 成 bf16；权重、梯度、Adam 状态仍是 fp32。实测结果见[图 5-1](#fig-5-1)：

<figure id="fig-5-1" class="cv-fig">
<svg class="cv" viewBox="0 0 640 252" width="100%" role="img" aria-label="bf16 autocast 相对 fp32：前向快 1.87 到 2.30 倍，反向快 1.69 到 1.87 倍；fwd_bwd 峰值显存少 18% 到 21%">
  <style>
    .cv { --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#e34948; --cg:#a19f9a; }
    html.dark .cv { --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#e66767; --cg:#6f6e6a; }
    .cv .grid { stroke: currentColor; stroke-opacity: .1; }
    .cv .axis { stroke: currentColor; stroke-opacity: .35; }
    .cv .ref { stroke: currentColor; stroke-opacity: .55; stroke-dasharray: 5 4; }
    .cv .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .lab { font-size: 12px; fill: currentColor; }
    .cv .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .cv .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .cv .val { font-size: 11px; fill: currentColor; }
    .cv .in { font-size: 11px; fill: #111; }
    .cv .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .cv g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .cv-fig { overflow-x: auto; } .cv-fig > svg { min-width: 540px; } }
  </style>
  <rect x="64" y="11" width="10" height="10" rx="2" style="fill: var(--c1)"/>
  <text class="lab" x="79" y="20">前向</text>
  <rect x="125" y="11" width="10" height="10" rx="2" style="fill: var(--c2)"/>
  <text class="lab" x="140" y="20">反向</text>
  <text class="ttl" x="64" y="46">加速比（bf16 相对 fp32）</text>
  <line class="grid" x1="64" y1="226" x2="300" y2="226"/>
  <text class="tick" x="56" y="230" text-anchor="end">0×</text>
  <line class="ref" x1="64" y1="162" x2="300" y2="162"/>
  <text class="tick" x="56" y="166" text-anchor="end">1×</text>
  <line class="grid" x1="64" y1="98" x2="300" y2="98"/>
  <text class="tick" x="56" y="102" text-anchor="end">2×</text>
  <g class="m"><title>small 前向：1.87×</title><rect x="78.0" y="106.3" width="24.0" height="119.7" rx="3" style="fill: var(--c1)"/></g>
  <text class="val" x="90" y="101.3" text-anchor="middle">1.87</text>
  <g class="m"><title>small 反向：1.69×</title><rect x="106.0" y="117.8" width="24.0" height="108.2" rx="3" style="fill: var(--c2)"/></g>
  <text class="val" x="118" y="112.8" text-anchor="middle">1.69</text>
  <text class="lab" x="104" y="243" text-anchor="middle">small</text>
  <g class="m"><title>medium 前向：2.05×</title><rect x="156.0" y="94.8" width="24.0" height="131.2" rx="3" style="fill: var(--c1)"/></g>
  <text class="val" x="168" y="89.8" text-anchor="middle">2.05</text>
  <g class="m"><title>medium 反向：1.79×</title><rect x="184.0" y="111.4" width="24.0" height="114.6" rx="3" style="fill: var(--c2)"/></g>
  <text class="val" x="196" y="106.4" text-anchor="middle">1.79</text>
  <text class="lab" x="182" y="243" text-anchor="middle">medium</text>
  <g class="m"><title>large 前向：2.3×</title><rect x="234.0" y="78.8" width="24.0" height="147.2" rx="3" style="fill: var(--c1)"/></g>
  <text class="val" x="246" y="73.8" text-anchor="middle">2.30</text>
  <g class="m"><title>large 反向：1.87×</title><rect x="262.0" y="106.3" width="24.0" height="119.7" rx="3" style="fill: var(--c2)"/></g>
  <text class="val" x="274" y="101.3" text-anchor="middle">1.87</text>
  <text class="lab" x="260" y="243" text-anchor="middle">large</text>
  <rect x="384" y="11" width="10" height="10" rx="2" style="fill: var(--cg)"/>
  <text class="lab" x="399" y="20">fp32</text>
  <rect x="447.4" y="11" width="10" height="10" rx="2" style="fill: var(--c3)"/>
  <text class="lab" x="462.4" y="20">bf16</text>
  <text class="ttl" x="384" y="46">fwd_bwd 峰值显存（GiB）</text>
  <line class="grid" x1="384" y1="226" x2="630" y2="226"/>
  <text class="tick" x="376" y="230" text-anchor="end">0</text>
  <line class="grid" x1="384" y1="156" x2="630" y2="156"/>
  <text class="tick" x="376" y="160" text-anchor="end">10</text>
  <line class="grid" x1="384" y1="86" x2="630" y2="86"/>
  <text class="tick" x="376" y="90" text-anchor="end">20</text>
  <g class="m"><title>small fp32：4.08 GiB</title><rect x="400.0" y="197.4" width="24.0" height="28.6" rx="3" style="fill: var(--cg)"/></g>
  <g class="m"><title>small bf16：3.18 GiB</title><rect x="428.0" y="203.7" width="24.0" height="22.3" rx="3" style="fill: var(--c3)"/></g>
  <text class="val" x="440" y="198.7" text-anchor="middle">−21%</text>
  <text class="lab" x="426" y="243" text-anchor="middle">small</text>
  <g class="m"><title>medium fp32：10.58 GiB</title><rect x="480.0" y="151.9" width="24.0" height="74.1" rx="3" style="fill: var(--cg)"/></g>
  <g class="m"><title>medium bf16：8.36 GiB</title><rect x="508.0" y="167.5" width="24.0" height="58.5" rx="3" style="fill: var(--c3)"/></g>
  <text class="val" x="520" y="162.5" text-anchor="middle">−21%</text>
  <text class="lab" x="506" y="243" text-anchor="middle">medium</text>
  <g class="m"><title>large fp32：20.28 GiB</title><rect x="560.0" y="84.0" width="24.0" height="142.0" rx="3" style="fill: var(--cg)"/></g>
  <g class="m"><title>large bf16：16.61 GiB</title><rect x="588.0" y="109.7" width="24.0" height="116.3" rx="3" style="fill: var(--c3)"/></g>
  <text class="val" x="600" y="104.7" text-anchor="middle">−18%</text>
  <text class="lab" x="586" y="243" text-anchor="middle">large</text>
</svg>
<figcaption><strong>图 5-1</strong> bf16 autocast 相对 fp32（fwd_bwd，batch 4，seq 512）：左边是加速比，右边是峰值显存。</figcaption>
</figure>

- **为什么快 ~2×？** 矩阵乘从 fp32 的 CUDA core（峰值 1.05e14）换到 bf16 的 Tensor core（2.1e14，[表 1-1](#tab-1-1)），相关张量的字节数也减半。模型越大，矩阵乘占比越高，加速越明显。
- **为什么只省 ~20%？** 🟦🟥 不变，🟩 A 也没减半：norm、softmax、残差、loss 留在 fp32，反向还要用一份 bf16 权重副本（large：A −5.50 GiB，副本 +1.78 GiB）。
- **为什么 norm、softmax 留 fp32？** 它们都是求和类的归约，精度由累加器决定。bf16 只有 7 位尾数：0.01 累加 1000 次，bf16 累加器停在 4.0（4.0 + 0.01 舍入回 4.0），fp32 累加器得 10.0001。

bf16 只是让张量变小，需要存的张量一个也没少。checkpoint 是前向不存、反向重算，和 [§4.2](#rmsnorm) 里 RMSNorm 融合的思路相同，只是粒度是整层。

### 5.2 activation checkpoint {#checkpoint}

`checkpoint(fn, x)` 不压缩任何东西，只是推迟：前向只保留 `fn` 的输入（entry，一个 `[b, s, d]` 张量，xl@2048 是 80 MiB），反向到这一段时用它重跑前向，生成 saved tensors，用完释放。下面以 4 层 xl block 为例（一层 saved tensors 3655 MiB，见[图 4-5](#fig-4-5)）：

<figure id="fig-5-2" class="gx-fig">
<svg viewBox="0 0 640 334" width="100%" role="img" aria-label="4 层 xl block：不 checkpoint 时每层留 3655 MiB 一起活到反向，峰值 14.6 GiB；每 2 层一个 checkpoint 时只留 entry x0、x2，反向时逐段重算，峰值 7.5 GiB">
  <style>
    .gx-band { fill: currentColor; fill-opacity: .045; }
    .gx-row { font-size: 12px; fill: currentColor; opacity: .8; }
    .gx-rowsub { font-size: 10.5px; fill: currentColor; opacity: .55; }
    .gx-op { fill: rgba(128,128,128,.12); stroke: currentColor; stroke-opacity: .5; stroke-width: 1.2; }
    .gx-t { font-size: 12.5px; fill: currentColor; }
    .gx-tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .gx-s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .gx-note { font-size: 11px; fill: currentColor; opacity: .7; }
    @media (max-width: 640px) { .gx-fig { overflow-x: auto; } .gx-fig > svg { min-width: 540px; } }
  </style>
  <defs>
    <marker id="fig-5-2-m0" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker>
  </defs>
  <rect class="gx-band" x="74" y="14" width="560" height="134" rx="8"/>
  <text class="gx-row" x="6" y="79">全部存下</text>
  <text class="gx-rowsub" x="6" y="94">峰值 14.6 GiB</text>
  <text class="gx-t" x="92.0" y="52.0" text-anchor="end">x0</text>
  <rect x="118.0" y="33.0" width="64" height="30" rx="6" class="gx-op"/>
  <text class="gx-tb" x="150" y="48.0" text-anchor="middle" dominant-baseline="central">L1</text>
  <rect x="102.0" y="92.0" width="96" height="40" rx="6" fill="rgba(67,160,71,0.16)" stroke="#43a047" stroke-width="1.4"/>
  <text class="gx-t" x="150" y="104.5" text-anchor="middle" dominant-baseline="central">3655 MiB</text>
  <text class="gx-s" x="150" y="119.5" text-anchor="middle" dominant-baseline="central">含 x0</text>
  <line x1="150.0" y1="63.0" x2="150.0" y2="92.0" stroke="#43a047" stroke-width="1.2" fill="none"/>
  <rect x="228.0" y="33.0" width="64" height="30" rx="6" class="gx-op"/>
  <text class="gx-tb" x="260" y="48.0" text-anchor="middle" dominant-baseline="central">L2</text>
  <line x1="182.0" y1="48.0" x2="225.0" y2="48.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-5-2-m0)"/>
  <rect x="212.0" y="92.0" width="96" height="40" rx="6" fill="rgba(67,160,71,0.16)" stroke="#43a047" stroke-width="1.4"/>
  <text class="gx-t" x="260" y="104.5" text-anchor="middle" dominant-baseline="central">3655 MiB</text>
  <text class="gx-s" x="260" y="119.5" text-anchor="middle" dominant-baseline="central">含 x1</text>
  <line x1="260.0" y1="63.0" x2="260.0" y2="92.0" stroke="#43a047" stroke-width="1.2" fill="none"/>
  <rect x="338.0" y="33.0" width="64" height="30" rx="6" class="gx-op"/>
  <text class="gx-tb" x="370" y="48.0" text-anchor="middle" dominant-baseline="central">L3</text>
  <line x1="292.0" y1="48.0" x2="335.0" y2="48.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-5-2-m0)"/>
  <rect x="322.0" y="92.0" width="96" height="40" rx="6" fill="rgba(67,160,71,0.16)" stroke="#43a047" stroke-width="1.4"/>
  <text class="gx-t" x="370" y="104.5" text-anchor="middle" dominant-baseline="central">3655 MiB</text>
  <text class="gx-s" x="370" y="119.5" text-anchor="middle" dominant-baseline="central">含 x2</text>
  <line x1="370.0" y1="63.0" x2="370.0" y2="92.0" stroke="#43a047" stroke-width="1.2" fill="none"/>
  <rect x="448.0" y="33.0" width="64" height="30" rx="6" class="gx-op"/>
  <text class="gx-tb" x="480" y="48.0" text-anchor="middle" dominant-baseline="central">L4</text>
  <line x1="402.0" y1="48.0" x2="445.0" y2="48.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-5-2-m0)"/>
  <rect x="432.0" y="92.0" width="96" height="40" rx="6" fill="rgba(67,160,71,0.16)" stroke="#43a047" stroke-width="1.4"/>
  <text class="gx-t" x="480" y="104.5" text-anchor="middle" dominant-baseline="central">3655 MiB</text>
  <text class="gx-s" x="480" y="119.5" text-anchor="middle" dominant-baseline="central">含 x3</text>
  <line x1="480.0" y1="63.0" x2="480.0" y2="92.0" stroke="#43a047" stroke-width="1.2" fill="none"/>
  <line x1="96.0" y1="48.0" x2="115.0" y2="48.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-5-2-m0)"/>
  <line x1="512.0" y1="48.0" x2="560.0" y2="48.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-5-2-m0)"/>
  <text class="gx-t" x="566.0" y="52.0" text-anchor="start">y</text>
  <text class="gx-note" x="626.0" y="143.0" text-anchor="end">4 份同时活到反向</text>
  <rect class="gx-band" x="74" y="168" width="560" height="156" rx="8"/>
  <text class="gx-row" x="6" y="244">每 2 层一段</text>
  <text class="gx-rowsub" x="6" y="259">峰值 7.5 GiB</text>
  <rect x="75.0" y="191.0" width="58" height="30" rx="6" fill="rgba(67,160,71,0.16)" stroke="#43a047" stroke-width="1.4"/>
  <text class="gx-t" x="104" y="206.0" text-anchor="middle" dominant-baseline="central">x0</text>
  <rect x="316.0" y="191.0" width="58" height="30" rx="6" fill="rgba(67,160,71,0.16)" stroke="#43a047" stroke-width="1.4"/>
  <text class="gx-t" x="345" y="206.0" text-anchor="middle" dominant-baseline="central">x2</text>
  <text class="gx-note" x="104.0" y="236.0" text-anchor="middle">entry 80 MiB</text>
  <text class="gx-note" x="345.0" y="236.0" text-anchor="middle">entry 80 MiB</text>
  <rect x="150.0" y="189.0" width="150" height="34" rx="6" class="gx-op"/>
  <text class="gx-tb" x="225" y="206.0" text-anchor="middle" dominant-baseline="central">checkpoint［L1 L2］</text>
  <rect x="137.0" y="256.0" width="176" height="40" rx="6" fill="rgba(249,168,37,0.16)" stroke="#f9a825" stroke-width="1.4" stroke-dasharray="4 3"/>
  <text class="gx-t" x="225" y="268.5" text-anchor="middle" dominant-baseline="central">2 × 3655 MiB</text>
  <text class="gx-s" x="225" y="283.5" text-anchor="middle" dominant-baseline="central">反向时用 x0 重算，用完即丢</text>
  <line x1="225.0" y1="223.0" x2="225.0" y2="256.0" stroke="#f9a825" stroke-width="1.2" fill="none"/>
  <rect x="390.0" y="189.0" width="150" height="34" rx="6" class="gx-op"/>
  <text class="gx-tb" x="465" y="206.0" text-anchor="middle" dominant-baseline="central">checkpoint［L3 L4］</text>
  <rect x="377.0" y="256.0" width="176" height="40" rx="6" fill="rgba(249,168,37,0.16)" stroke="#f9a825" stroke-width="1.4" stroke-dasharray="4 3"/>
  <text class="gx-t" x="465" y="268.5" text-anchor="middle" dominant-baseline="central">2 × 3655 MiB</text>
  <text class="gx-s" x="465" y="283.5" text-anchor="middle" dominant-baseline="central">反向时用 x2 重算，用完即丢</text>
  <line x1="465.0" y1="223.0" x2="465.0" y2="256.0" stroke="#f9a825" stroke-width="1.2" fill="none"/>
  <line x1="133.0" y1="206.0" x2="147.0" y2="206.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-5-2-m0)"/>
  <line x1="300.0" y1="206.0" x2="313.0" y2="206.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-5-2-m0)"/>
  <line x1="374.0" y1="206.0" x2="387.0" y2="206.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-5-2-m0)"/>
  <line x1="540.0" y1="206.0" x2="575.0" y2="206.0" stroke="currentColor" stroke-opacity=".65" stroke-width="1.6" fill="none" marker-end="url(#fig-5-2-m0)"/>
  <text class="gx-t" x="581.0" y="210.0" text-anchor="start">y</text>
  <text class="gx-note" x="626.0" y="312.0" text-anchor="end">反向时一次只物化一段</text>
</svg>
<figcaption><strong>图 5-2</strong> 4 层 xl block 有无 checkpoint：绿框一直占到反向，黄色虚线框在反向时用 entry 重算、用完即丢。</figcaption>
</figure>

不 checkpoint 时，4 层的 saved tensors 一起留到反向，峰值 4 × 3655 MiB = 14.6 GiB。每 2 层一个 checkpoint 时，前向只留 2 个 entry（160 MiB），反向一次只重算一段（2 × 3655 MiB），峰值约 7.5 GiB。

两段不能并行：L2 的反向需要 dx2，而 dx2 是 L3 反向的输出。前半段的重算只依赖 x0，理论上可以提前做，但这样两段的黄框会同时存在，峰值又回到 14.6 GiB。

每段放几层要靠实验看。xl@2048 batch 4 光参数加梯度就有 25.4 GiB，任何段长都放不下，所以改在 large（36 层）、batch 1、seq 1024 上扫描（[图 5-3](#fig-5-3)）：

<div id="fig-5-3"></div>

![checkpoint 段长扫描：step 时间与峰值显存](checkpoint_large_sweep.png "**图 5-3** checkpoint 段长扫描（large，batch 1 seq 1024，fwd_bwd）：左 step 时间，右峰值显存，红虚线是不 checkpoint。")

- **时间**：无论每段几层，整网都多算一遍前向，所以都在 302–313 ms，比不 checkpoint 的 236 ms 多 28–33%。记一次前向为 F，反向约 2F（[§2](#fwd-bwd)），一步从 3F 变成 4F。
- **显存**：随每段层数单调增加，每层一个 checkpoint 时最低（7.8 GiB，不 checkpoint 时 15.0 GiB）。整网作为一段（36 层）最不划算：多算了一遍，显存却和不 checkpoint 一样。

显存曲线没有中间的最小值。设每段 e 层、entry 大小 a、一层 saved tensors 大小 r，峰值约等于所有 entry 加上正在重算的那一段：

<p align="center">$M(e) = \dfrac{L}{e}\,a + e\,r$</p>

只要所有 entry 加起来比一层还小（L·a < r，这里 36 × 5 MiB = 180 MiB < 220 MiB），e = 1 就是最优。Transformer 每层的 saved tensors 都很大，这个条件基本总成立。实际使用时，先用 FlashAttention 或选择性重算（只重算 S、P 这类大但便宜的张量，前向多 5%），显存还不够再按缺口选 N 层，每层包一个 checkpoint。

> **小结**：S、P 让 attention 成为 memory-bound（[§3](#memory-bound)），又占了一层 saved tensors 的一半以上（[§4.3](#one-layer)）。bf16 只能让它们变小；checkpoint 只能缩短它们留在显存里的时间，重算时仍要完整写进显存再读出来。要去掉这部分读写，需要像 [§4.2](#rmsnorm) 的 RMSNorm 那样，把 QKᵀ、softmax、PV 融合成一个 kernel，在片上分块计算、反向时重算，这就是下一篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/) 的内容。
