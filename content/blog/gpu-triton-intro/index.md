---
title: "GPU 与 Triton 入门：从存储层级到第一个 kernel"
date: 2026-10-03
draft: false
math: true
description: "GPU 的存储层级、SM 结构和 CUDA 执行层级；再用 GELU、softmax、row sum、matmul 四个例子入门 Triton，附录对照 TPU。GPU training optimization 系列的第一篇。"
tags: ["GPU", "Triton", "CUDA", "TPU", "LLM", "AI"]
categories: ["AI笔记"]
series: ["GPU training optimization"]
series_order: 1
---

这个系列讲怎么让 Transformer 训练在 GPU 上跑得更快。这一篇先打基础：[第 1 节](#hardware)介绍 GPU 的硬件，包括数据存在哪几层、一个 SM 里有什么、CUDA 的执行层级怎么对应到硬件；[第 2 节](#triton)用四个例子介绍 Triton，后面两篇的 kernel 都用它写。TPU 和 GPU 的对照放在[附录 A](#tpu)。

下一篇[谁偷走了 5090 的算力和显存](/blog/gpu-training-analysis/)用这些知识在 RTX 5090 上拆解一步训练的时间和显存，第三篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/) 讲 attention 的 IO-aware kernel。

> 环境：RTX 5090 32 GB，torch 2.11.0+cu130，Triton 3.6.0。

---

## 1 GPU：越靠近计算单元越快 {#hardware}

做性能分析时最常用的两个数是峰值算力和显存带宽。一个 kernel 受哪个限制，取决于它要搬多少数据、数据在哪一层被读写。这一节按这个顺序看 5090 的硬件：先看数据存在哪几层（[1.1](#hw-memory)），再看负责计算的 SM 里有什么（[1.2](#hw-sm)）。这两节只讲硬件。最后看软件：CUDA 的 thread、warp、block 怎么落到这些硬件上，一个 SM 上同时驻留多少 warp，怎么在等数据时不让计算单元闲着（[1.3](#hw-model)）。

### 1.1 存储层级：越近越快，也越小 {#hw-memory}

处理器的频率早就不怎么涨了，算力的增长主要来自并行：更多的 SM、更宽的 Tensor core。显存带宽的增长比算力慢得多：过去 20 年，峰值算力涨了约 6 万倍（每两年 3.0 倍），DRAM 带宽只涨了约 100 倍（每两年 1.6 倍），芯片之间的互联带宽约 30 倍（[图 1-1](#fig-memwall)）。所以越来越多的 op 落在 roofline 的斜坡上，这是 memory-bound 越来越常见的原因。带宽跟不上，就只能少搬：让读进来的数据留在离计算单元近的地方，多用几次。

<figure id="fig-memwall" class="fg-fig">
<img src="memory-wall.webp" alt="1997 到 2023 年峰值算力、DRAM 带宽和互联带宽的增长：算力每两年 3.0 倍，DRAM 带宽每两年 1.6 倍，互联带宽每两年 1.4 倍" loading="lazy" style="width: 100%; height: auto; border-radius: 8px; background: #fff;">
<figcaption><strong>图 1-1</strong> 峰值算力与 DRAM、互联带宽 20 年来的增长，纵轴是相对 1997 年的倍数（对数轴）。图来自 Gholami 等人的 <a href="https://arxiv.org/abs/2403.14123">AI and Memory Wall</a>（IEEE Micro，2024）。</figcaption>
</figure>

数据离计算单元越近，读写越快，容量也越小（[图 1-2](#fig-h-1)）。每个 SM 里有 register 和 L1 / shared memory，所有 SM 共享芯片上的 L2，显存在芯片外面。写 kernel 时，L1 和 L2 由硬件当作 cache 自动管理；能自己安排的只有 shared memory（以及 register）。所以 kernel 优化的套路都是一样的：把一块数据从显存读进 shared memory 或 register，在片上尽量多算几次，再写回去。[第 2 节](#triton)的分块矩阵乘、[下一篇](/blog/gpu-training-analysis/)里的算子融合和之后的 [FlashAttention](/blog/flashattention-1-to-4/) 都是这个思路。

<figure id="fig-h-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 300" width="100%" role="img" aria-label="RTX 5090 的存储层级：每个 SM 里有 256 KB register 和 128 KB L1/shared memory，170 个 SM 共享芯片上的 96 MB L2，芯片外是 32 GB GDDR7 显存，带宽 1.79e12 B/s">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-h-1-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-mute)"/></marker></defs>
  <rect x="10" y="22" width="452" height="250" rx="12" style="fill: none; stroke: currentColor" stroke-opacity=".35" stroke-dasharray="6 4"/>
  <text class="lab2" x="22" y="40">GPU 芯片（die），共 170 个 SM</text>
  <rect x="20" y="52" width="92" height="108" rx="8" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="66" y="70" text-anchor="middle">SM</text>
  <rect x="28" y="80" width="76" height="32" rx="5" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="66" y="94" text-anchor="middle">register</text><text class="s" x="66" y="107" text-anchor="middle">256 KB</text>
  <rect x="28" y="118" width="76" height="34" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="66" y="132" text-anchor="middle">L1 / shared</text><text class="s" x="66" y="146" text-anchor="middle">128 KB</text>
  <line x1="66.0" y1="160.0" x2="66.0" y2="194.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2"/>
  <rect x="120" y="52" width="92" height="108" rx="8" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="166" y="70" text-anchor="middle">SM</text>
  <rect x="128" y="80" width="76" height="32" rx="5" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="166" y="94" text-anchor="middle">register</text><text class="s" x="166" y="107" text-anchor="middle">256 KB</text>
  <rect x="128" y="118" width="76" height="34" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="166" y="132" text-anchor="middle">L1 / shared</text><text class="s" x="166" y="146" text-anchor="middle">128 KB</text>
  <line x1="166.0" y1="160.0" x2="166.0" y2="194.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2"/>
  <rect x="220" y="52" width="92" height="108" rx="8" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="266" y="70" text-anchor="middle">SM</text>
  <rect x="228" y="80" width="76" height="32" rx="5" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="266" y="94" text-anchor="middle">register</text><text class="s" x="266" y="107" text-anchor="middle">256 KB</text>
  <rect x="228" y="118" width="76" height="34" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="266" y="132" text-anchor="middle">L1 / shared</text><text class="s" x="266" y="146" text-anchor="middle">128 KB</text>
  <line x1="266.0" y1="160.0" x2="266.0" y2="194.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2"/>
  <text class="t" x="339" y="110" text-anchor="middle">⋯</text>
  <rect x="360" y="52" width="92" height="108" rx="8" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="406" y="70" text-anchor="middle">SM</text>
  <rect x="368" y="80" width="76" height="32" rx="5" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="406" y="94" text-anchor="middle">register</text><text class="s" x="406" y="107" text-anchor="middle">256 KB</text>
  <rect x="368" y="118" width="76" height="34" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="406" y="132" text-anchor="middle">L1 / shared</text><text class="s" x="406" y="146" text-anchor="middle">128 KB</text>
  <line x1="406.0" y1="160.0" x2="406.0" y2="194.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2"/>
  <rect x="24" y="194" width="426" height="40" rx="8" style="fill: color-mix(in srgb, var(--fig-3) 45%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="t" x="237" y="219" text-anchor="middle">L2 cache 96 MB，所有 SM 共享</text>
  <text class="lab2" x="237" y="258" text-anchor="middle">片上</text>
  <rect x="504" y="70" width="124" height="150" rx="10" style="fill: color-mix(in srgb, var(--fig-mute) 14%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="tb" x="566" y="124" text-anchor="middle">GDDR7 显存</text><text class="t" x="566" y="144" text-anchor="middle">32 GB</text><text class="s" x="566" y="168" text-anchor="middle">带宽 β</text><text class="s" x="566" y="182" text-anchor="middle">1.79e12 B/s</text>
  <text class="lab2" x="566" y="238" text-anchor="middle">片外</text>
  <line x1="452.0" y1="214.0" x2="500.0" y2="214.0" style="stroke: var(--fig-mute)" stroke-width="1.8" marker-end="url(#fig-h-1-m0)"/>
  <line x1="500.0" y1="200.0" x2="452.0" y2="200.0" style="stroke: var(--fig-mute)" stroke-width="1.8" marker-end="url(#fig-h-1-m0)"/>
  <text class="lab2" x="10" y="292">越靠近 SM 越快、越小：register > L1 / shared > L2 > 显存</text>
</svg>
<figcaption><strong>图 1-2</strong> RTX 5090 的存储层级示意。规格来自 <a href="https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf">NVIDIA RTX Blackwell 白皮书</a>，官方的整芯片和 SM 结构图也在白皮书里。</figcaption>
</figure>

5090 是消费级显卡，显存用的是 GDDR7（32 GB，512-bit，28 Gbps，带宽 1.79e12 B/s），不是数据中心卡上的 HBM，也没有 ECC 和 NVLink。

register 和 L1 / shared memory 都在 SM 里面，下面看 SM 本身的结构。

### 1.2 一个 SM 里有什么 {#hw-sm}

每个 SM 分成 4 个 SMSP（SM sub-partition），每个 SMSP 有自己的 warp 调度器、register、32 个 CUDA core（FP32 运算单元）和一个 Tensor core，4 个 SMSP 共享一块 L1 / shared memory（[图 1-3](#fig-h-2)）。几代 GPU 的规格对比见[表 1-1](#tab-1-1)。

<figure id="fig-h-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 236" width="100%" role="img" aria-label="RTX 5090 的一个 SM：4 个 SMSP，每个有 1 个 warp 调度器、64 KB register、32 个 CUDA core 和 1 个 Tensor core；4 个 SMSP 共享 128 KB 的 L1/shared memory">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <rect x="10" y="10" width="620" height="216" rx="12" style="fill: color-mix(in srgb, var(--fig-1) 6%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="24" y="32">SM（每颗 5090 有 170 个）</text>
  <rect x="24" y="44" width="141" height="124" rx="8" class="op"/>
  <text class="tb" x="94" y="62" text-anchor="middle">SMSP 0</text>
  <rect x="34" y="70" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="94" y="84" text-anchor="middle">warp 调度器</text>
  <rect x="34" y="94" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="94" y="108" text-anchor="middle">register 64 KB</text>
  <rect x="34" y="118" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="94" y="132" text-anchor="middle">CUDA core ×32</text>
  <rect x="34" y="142" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="94" y="156" text-anchor="middle">Tensor core ×1</text>
  <rect x="175" y="44" width="141" height="124" rx="8" class="op"/>
  <text class="tb" x="245" y="62" text-anchor="middle">SMSP 1</text>
  <rect x="185" y="70" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="245" y="84" text-anchor="middle">warp 调度器</text>
  <rect x="185" y="94" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="245" y="108" text-anchor="middle">register 64 KB</text>
  <rect x="185" y="118" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="245" y="132" text-anchor="middle">CUDA core ×32</text>
  <rect x="185" y="142" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="245" y="156" text-anchor="middle">Tensor core ×1</text>
  <rect x="326" y="44" width="141" height="124" rx="8" class="op"/>
  <text class="tb" x="396" y="62" text-anchor="middle">SMSP 2</text>
  <rect x="336" y="70" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="396" y="84" text-anchor="middle">warp 调度器</text>
  <rect x="336" y="94" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="396" y="108" text-anchor="middle">register 64 KB</text>
  <rect x="336" y="118" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="396" y="132" text-anchor="middle">CUDA core ×32</text>
  <rect x="336" y="142" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="396" y="156" text-anchor="middle">Tensor core ×1</text>
  <rect x="477" y="44" width="141" height="124" rx="8" class="op"/>
  <text class="tb" x="547" y="62" text-anchor="middle">SMSP 3</text>
  <rect x="487" y="70" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="547" y="84" text-anchor="middle">warp 调度器</text>
  <rect x="487" y="94" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="547" y="108" text-anchor="middle">register 64 KB</text>
  <rect x="487" y="118" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="547" y="132" text-anchor="middle">CUDA core ×32</text>
  <rect x="487" y="142" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="547" y="156" text-anchor="middle">Tensor core ×1</text>
  <rect x="24" y="178" width="594" height="38" rx="8" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="t" x="321" y="202" text-anchor="middle">L1 cache / shared memory 128 KB（4 个 SMSP 共享）</text>
</svg>
<figcaption><strong>图 1-3</strong> RTX 5090 一个 SM 的结构示意（SMSP 即 SM sub-partition）。每个 SMSP 有自己的 warp 调度器、register、32 个 CUDA core 和 1 个 Tensor core，4 个 SMSP 共享 L1 / shared memory。</figcaption>
</figure>

一个 SMSP 里面再细分（[图 1-4](#fig-h-9)）：warp scheduler 每拍挑一个 warp，dispatch 把它的一条指令发给 32 个 thread；FP32 / INT32 核做普通运算，Tensor core 做矩阵乘，SFU 算 exp、rsqrt 这类函数，LD/ST 负责 load 和 store。[第 2 节](#triton)kernel 图里的「计算单元」就是这些核，「register」就是这块 64 KB 的 register file。

<figure id="fig-h-9" class="fg-fig">
<svg class="fg" viewBox="0 0 640 322" width="100%" role="img" aria-label="RTX 5090 一个 SMSP 的结构：L0 指令 cache、warp scheduler 和 dispatch，64 KB register file，32 个 FP32/INT32 核和 1 个 Tensor core，4 个 LD/ST 和 SFU；下面接 SM 共享的 128 KB L1 / shared memory">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <rect x="12" y="12" width="356" height="248" rx="12" style="fill: color-mix(in srgb, var(--fig-1) 6%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="30" y="33">一个 SMSP（每个 SM 4 个）</text>
  <rect x="30" y="42" width="96" height="24" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="78.0" y="58.0" text-anchor="middle">L0 i-cache</text>
  <rect x="132" y="42" width="112" height="24" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 18%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="188.0" y="58.0" text-anchor="middle">① warp scheduler</text>
  <rect x="250" y="42" width="104" height="24" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="302.0" y="58.0" text-anchor="middle">dispatch</text>
  <rect x="30" y="72" width="324" height="24" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 14%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="192.0" y="88.0" text-anchor="middle">② register file：16,384 × 32-bit = 64 KB</text>
  <rect x="30.0" y="102" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="55.5" y="102" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="81.0" y="102" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="106.5" y="102" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="132.0" y="102" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="157.5" y="102" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="183.0" y="102" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="208.5" y="102" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="30.0" y="128" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="55.5" y="128" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="81.0" y="128" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="106.5" y="128" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="132.0" y="128" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="157.5" y="128" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="183.0" y="128" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="208.5" y="128" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="30.0" y="154" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="55.5" y="154" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="81.0" y="154" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="106.5" y="154" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="132.0" y="154" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="157.5" y="154" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="183.0" y="154" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="208.5" y="154" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="30.0" y="180" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="55.5" y="180" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="81.0" y="180" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="106.5" y="180" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="132.0" y="180" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="157.5" y="180" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="183.0" y="180" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="208.5" y="180" width="22.5" height="23" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="66" y="137" width="132" height="34" rx="5" style="fill: var(--fig-bg); stroke: var(--fig-2)"/>
  <text class="tb" x="132" y="152" text-anchor="middle">③ FP32 / INT32</text><text class="s" x="132" y="166" text-anchor="middle">× 32</text>
  <rect x="238" y="102" width="116" height="101" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 24%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="296" y="148" text-anchor="middle">④ Tensor core</text><text class="s" x="296" y="165" text-anchor="middle">第 5 代，× 1</text>
  <text class="s" x="16" y="225" text-anchor="start" style="fill: var(--fig-hi)">⑤</text>
  <rect x="30.0" y="210" width="47" height="22" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 22%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="s" x="53.5" y="225.0" text-anchor="middle">LD/ST</text>
  <rect x="82.5" y="210" width="47" height="22" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 22%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="s" x="106.0" y="225.0" text-anchor="middle">LD/ST</text>
  <rect x="135.0" y="210" width="47" height="22" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 22%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="s" x="158.5" y="225.0" text-anchor="middle">LD/ST</text>
  <rect x="187.5" y="210" width="47" height="22" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 22%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="s" x="211.0" y="225.0" text-anchor="middle">LD/ST</text>
  <rect x="250" y="210" width="104" height="22" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 45%, transparent); stroke: var(--fig-3)" stroke-width="1.4"/>
  <text class="s" x="302.0" y="225.0" text-anchor="middle">⑥ SFU</text>
  <text class="lab2" x="190" y="251" text-anchor="middle">↓ 4 个 SMSP 共享 128 KB L1 / shared memory（图 1-3）</text>
  <text class="s" x="384" y="34" text-anchor="start" style="fill: var(--fig-hi)">①</text>
  <text class="lab2" x="400" y="34">warp scheduler：每拍从驻留的</text>
  <text class="lab2" x="400" y="49">warp 里挑一个就绪的；dispatch</text>
  <text class="lab2" x="400" y="64">把它的一条指令发给 32 个 thread</text>
  <text class="s" x="384" y="87" text-anchor="start" style="fill: var(--fig-hi)">②</text>
  <text class="lab2" x="400" y="87">register：这个 SMSP 上所有驻留的</text>
  <text class="lab2" x="400" y="102">thread 分着用，每个最多 255 个</text>
  <text class="s" x="384" y="125" text-anchor="start" style="fill: var(--fig-hi)">③</text>
  <text class="lab2" x="400" y="125">FP32 / INT32 核：一条加法、乘法</text>
  <text class="lab2" x="400" y="140">或 FMA，32 个 thread 各用一个核；</text>
  <text class="lab2" x="400" y="155">同一拍只能做 FP32 或 INT32</text>
  <text class="s" x="384" y="178" text-anchor="start" style="fill: var(--fig-hi)">④</text>
  <text class="lab2" x="400" y="178">Tensor core：一条指令算一个</text>
  <text class="lab2" x="400" y="193">小矩阵乘，tl.dot 默认走它</text>
  <text class="s" x="384" y="216" text-anchor="start" style="fill: var(--fig-hi)">⑤</text>
  <text class="lab2" x="400" y="216">LD/ST：load / store，在 register</text>
  <text class="lab2" x="400" y="231">和 L1 / shared memory、显存之间</text>
  <text class="lab2" x="400" y="246">搬数据</text>
  <text class="s" x="384" y="269" text-anchor="start" style="fill: var(--fig-hi)">⑥</text>
  <text class="lab2" x="400" y="269">SFU：exp、rsqrt、sin 这类特殊函数</text>
</svg>
<figcaption><strong>图 1-4</strong> RTX 5090 一个 SMSP 的结构示意，按 <a href="https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf">NVIDIA RTX Blackwell 白皮书</a>的图 5 画。Blackwell 把 FP32 和 INT32 合成同一批核，同一拍只能做其中一种。</figcaption>
</figure>

| | A100 | H100 | B200 | RTX 5090 |
|:--|--:|--:|--:|--:|
| SM 数 | 108 | 132 | 148 | 170 |
| L2 | 40 MB | 50 MB | — | 96 MB |
| 显存 | 80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e | 32 GB GDDR7 |
| 显存带宽 | 2.0e12 B/s | 3.35e12 B/s | 8e12 B/s | 1.79e12 B/s |
| 每 SM 的 CUDA core（FP32） | 64 | 128 | 128 | 128 |
| 每 SM 的 Tensor core | 4 | 4 | 4 | 4 |
| 每 SM 的 L1 + shared | 192 KB | 256 KB | 256 KB | 128 KB |
| 每 SM 的 register | 256 KB | 256 KB | 256 KB | 256 KB |
| 每 SM 的 SMSP（warp 调度器） | 4 | 4 | 4 | 4 |
{#tab-1-1 caption="**表 1-1** 几代 NVIDIA GPU 的规格" note="来自 NVIDIA 各代架构白皮书和产品规格（A100 80GB SXM，H100 SXM）。B200 的 L2 没有找到可靠的公开数字，暂缺。"}

这些是硬件提供的资源。register 按 thread 分配，shared memory 按 block 分配，要知道一个 kernel 怎么用这些资源，得先看 CUDA 怎么把线程组织起来。

### 1.3 编程模型：thread、warp、block、grid {#hw-model}

CUDA 的执行模型叫 SIMT（single instruction, multiple threads）：同一个 warp 里的 32 个线程在同一时刻执行同一条指令。线程分四层：thread；warp，32 个 thread；block（也叫 CTA），若干个 warp；grid，一次 kernel 启动的全部 block。[图 1-5](#fig-h-5) 把四层画在一起，左边逐层放大，右边是每层跑在哪个硬件上。

<figure id="fig-h-5" class="fg-fig">
<svg class="fg" viewBox="0 0 640 352" width="100%" role="img" aria-label="CUDA 的执行层级：grid 由 block 组成，block 由 warp 组成，warp 由 32 个 thread 组成；分别跑在整张 GPU、一个 SM、一个 SMSP 和一条 lane 上">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-h-5-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <text class="ttl" x="20" y="18">软件：一次 kernel 启动</text><text class="ttl" x="432" y="18">硬件：跑在哪里（RTX 5090）</text>
  <rect x="20" y="30" width="350" height="66" rx="8" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="432" y="30" width="196" height="66" rx="8" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="tb" x="530" y="60.0" text-anchor="middle">整张 GPU</text><text class="s" x="530" y="75.0" text-anchor="middle">block 由硬件分到 170 个 SM</text>
  <line x1="374.0" y1="63.0" x2="428.0" y2="63.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-h-5-m0)"/>
  <rect x="20" y="126" width="350" height="66" rx="8" style="fill: color-mix(in srgb, var(--fig-1) 12%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <rect x="432" y="126" width="196" height="66" rx="8" style="fill: color-mix(in srgb, var(--fig-1) 12%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="530" y="156.0" text-anchor="middle">一个 SM</text><text class="s" x="530" y="171.0" text-anchor="middle">block 整个驻留，不跨 SM</text>
  <line x1="374.0" y1="159.0" x2="428.0" y2="159.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-h-5-m0)"/>
  <rect x="20" y="222" width="350" height="50" rx="8" style="fill: color-mix(in srgb, var(--fig-2) 16%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <rect x="432" y="222" width="196" height="50" rx="8" style="fill: color-mix(in srgb, var(--fig-2) 16%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="tb" x="530" y="244.0" text-anchor="middle">一个 SMSP</text><text class="s" x="530" y="259.0" text-anchor="middle">调度器每拍发一条指令</text>
  <line x1="374.0" y1="247.0" x2="428.0" y2="247.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-h-5-m0)"/>
  <rect x="20" y="302" width="350" height="42" rx="8" style="fill: color-mix(in srgb, var(--fig-hi) 14%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="432" y="302" width="196" height="42" rx="8" style="fill: color-mix(in srgb, var(--fig-hi) 14%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="tb" x="530" y="320.0" text-anchor="middle">一条 lane</text><text class="s" x="530" y="335.0" text-anchor="middle">CUDA core，自己的 register</text>
  <line x1="374.0" y1="323.0" x2="428.0" y2="323.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-h-5-m0)"/>
  <text class="lab" x="32" y="47">grid：所有 block，互相独立</text>
  <rect x="32" y="58" width="34" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="49" y="76" text-anchor="middle">B0</text>
  <rect x="74" y="58" width="34" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="91" y="76" text-anchor="middle">B1</text>
  <rect x="116" y="58" width="34" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="133" y="76" text-anchor="middle">B2</text>
  <rect x="158" y="58" width="34" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="175" y="76" text-anchor="middle">B3</text>
  <rect x="200" y="58" width="34" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="217" y="76" text-anchor="middle">B4</text>
  <rect x="242" y="58" width="34" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="259" y="76" text-anchor="middle">B5</text>
  <rect x="284" y="58" width="34" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="301" y="76" text-anchor="middle">B6</text>
  <text class="t" x="340" y="77">⋯</text>
  <path d="M74 86 L20 126 M108 86 L370 126" stroke="currentColor" stroke-opacity=".3" stroke-dasharray="3 3" fill="none"/>
  <text class="lab" x="32" y="143">block（Triton 的 program）</text>
  <rect x="32" y="150" width="326" height="16" rx="3" style="fill: color-mix(in srgb, var(--fig-3) 45%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="195" y="162" text-anchor="middle">shared memory：块内共享，可以同步</text>
  <rect x="32" y="172" width="76" height="16" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 45%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="70" y="184" text-anchor="middle">warp 0</text>
  <rect x="115" y="172" width="76" height="16" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="153" y="184" text-anchor="middle">warp 1</text>
  <rect x="198" y="172" width="76" height="16" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="236" y="184" text-anchor="middle">warp 2</text>
  <rect x="281" y="172" width="76" height="16" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="319" y="184" text-anchor="middle">warp 3</text>
  <path d="M32 188 L20 222 M108 188 L370 222" stroke="currentColor" stroke-opacity=".3" stroke-dasharray="3 3" fill="none"/>
  <text class="lab" x="32" y="239">warp：32 个 thread，同一时刻执行同一条指令</text>
  <rect x="32" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 60%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="42" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="52" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="62" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="72" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="82" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="92" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="102" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="112" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="122" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="132" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="142" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="152" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="162" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="172" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="182" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="192" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="202" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="212" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="222" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="232" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="242" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="252" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="262" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="272" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="282" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="292" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="302" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="312" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="322" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="332" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="342" y="248" width="8" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 20%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <path d="M32 264 L20 302 M40 264 L370 302" stroke="currentColor" stroke-opacity=".3" stroke-dasharray="3 3" fill="none"/>
  <text class="lab" x="32" y="328">thread：自己的 register 和下标</text>
</svg>
<figcaption><strong>图 1-5</strong> CUDA 的执行层级。左边从上到下逐层放大：grid 里的一个 block、block 里的一个 warp、warp 里的一个 thread；右边是每一层跑在哪个硬件上，对应<a href="#fig-h-2">图 1-3</a>。</figcaption>
</figure>

block 放到 SM 上以后就一直驻留，直到它的线程全部跑完。一个 SM 能同时驻留多少，有几条硬件上限（[表 1-2](#tab-1-2)）。warp 的上限分摊到每个 SMSP 的调度器；block 的上限记在整个 SM 上，因为一个 block 的 shared memory 和同步由 4 个 SMSP 共用。

| | A100 | H100 | B200 | RTX 5090 |
|:--|--:|--:|--:|--:|
| 每 SMSP 最多驻留 warp | 16 | 16 | 16 | 12 |
| 每 SM 最多驻留 warp | 64 | 64 | 64 | 48 |
| 每 SM 最多驻留 block | 32 | 32 | 32 | 24 |
| 每 thread 最多 register | 255 | 255 | 255 | 255 |
{#tab-1-2 caption="**表 1-2** 一个 SM 的驻留上限" note="来自 CUDA Programming Guide 的 compute capability 表（8.0、9.0、10.0、12.0）。"}

一个 block 的 warp 按编号轮流分到 4 个 SMSP：warp 0 到 SMSP 0，warp 1 到 SMSP 1，依此类推。驻留不等于在跑。一个 warp 等显存数据要几百个周期，这段时间里调度器改发另一个就绪的 warp；所有驻留 warp 的 register 一直留在 register file 里，换 warp 不用保存或恢复状态。所以一个 SM 上同时驻留的 warp 越多，访存延迟越容易藏住。[图 1-6](#fig-h-6) 画的是驻留 3 个 block、每个 4 个 warp 时的某一拍：每个 SMSP 有 3 个 warp，分别来自 3 个 block，调度器每个周期只发射其中一个；SMSP 2 的 3 个都在等数据，这一拍就空转。

<figure id="fig-h-6" class="fg-fig">
<svg class="fg" viewBox="0 0 640 300" width="100%" role="img" aria-label="一个 SM 上驻留 3 个 block、每个 4 个 warp 时的某一拍：每个 SMSP 有 3 个 warp，分别来自 3 个 block；每个调度器这一拍最多发射一个，SMSP 2 的 3 个都在等数据，调度器空转">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <rect x="10" y="10" width="620" height="246" rx="12" style="fill: color-mix(in srgb, var(--fig-1) 6%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="24" y="32">一个 SM 的驻留情况</text>
  <text class="lab2" x="616" y="32" text-anchor="end">整个 SM 最多驻留 24 个 block、48 个 warp</text>
  <rect x="24" y="44" width="141" height="164" rx="8" class="op"/>
  <text class="tb" x="94" y="62" text-anchor="middle">SMSP 0</text>
  <rect x="34" y="70" width="121" height="22" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="94" y="85" text-anchor="middle">warp 调度器</text>
  <text class="s" x="94" y="110" text-anchor="middle">最多驻留 12 个 warp</text>
  <rect x="31" y="118" width="28" height="22" rx="3" style="fill: var(--fig-hi)"/>
  <text class="s" x="45" y="133" text-anchor="middle">B0</text>
  <rect x="64" y="118" width="28" height="22" rx="3" style="fill: var(--fig-2)"/>
  <text class="s" x="78" y="133" text-anchor="middle">B1</text>
  <rect x="97" y="118" width="28" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="111" y="133" text-anchor="middle">B2</text>
  <rect x="130" y="118" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="31" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="64" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="97" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="130" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="31" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="64" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="97" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="130" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="175" y="44" width="141" height="164" rx="8" class="op"/>
  <text class="tb" x="245" y="62" text-anchor="middle">SMSP 1</text>
  <rect x="185" y="70" width="121" height="22" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="245" y="85" text-anchor="middle">warp 调度器</text>
  <text class="s" x="245" y="110" text-anchor="middle">最多驻留 12 个 warp</text>
  <rect x="182" y="118" width="28" height="22" rx="3" style="fill: var(--fig-hi)"/>
  <text class="s" x="196" y="133" text-anchor="middle">B0</text>
  <rect x="215" y="118" width="28" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="229" y="133" text-anchor="middle">B1</text>
  <rect x="248" y="118" width="28" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="262" y="133" text-anchor="middle">B2</text>
  <rect x="281" y="118" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="182" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="215" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="248" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="281" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="182" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="215" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="248" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="281" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="326" y="44" width="141" height="164" rx="8" class="op"/>
  <text class="tb" x="384" y="62" text-anchor="middle">SMSP 2</text>
  <text class="s" x="430" y="62" style="fill: var(--fig-hi)">空转</text>
  <rect x="336" y="70" width="121" height="22" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="396" y="85" text-anchor="middle">warp 调度器</text>
  <text class="s" x="396" y="110" text-anchor="middle">最多驻留 12 个 warp</text>
  <rect x="333" y="118" width="28" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="347" y="133" text-anchor="middle">B0</text>
  <rect x="366" y="118" width="28" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="380" y="133" text-anchor="middle">B1</text>
  <rect x="399" y="118" width="28" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="413" y="133" text-anchor="middle">B2</text>
  <rect x="432" y="118" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="333" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="366" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="399" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="432" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="333" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="366" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="399" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="432" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="477" y="44" width="141" height="164" rx="8" class="op"/>
  <text class="tb" x="547" y="62" text-anchor="middle">SMSP 3</text>
  <rect x="487" y="70" width="121" height="22" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="547" y="85" text-anchor="middle">warp 调度器</text>
  <text class="s" x="547" y="110" text-anchor="middle">最多驻留 12 个 warp</text>
  <rect x="484" y="118" width="28" height="22" rx="3" style="fill: var(--fig-hi)"/>
  <text class="s" x="498" y="133" text-anchor="middle">B0</text>
  <rect x="517" y="118" width="28" height="22" rx="3" style="fill: var(--fig-2)"/>
  <text class="s" x="531" y="133" text-anchor="middle">B1</text>
  <rect x="550" y="118" width="28" height="22" rx="3" style="fill: var(--fig-2)"/>
  <text class="s" x="564" y="133" text-anchor="middle">B2</text>
  <rect x="583" y="118" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="484" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="517" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="550" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="583" y="146" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="484" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="517" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="550" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="583" y="174" width="28" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/>
  <rect x="24" y="218" width="594" height="28" rx="6" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="321" y="236" text-anchor="middle">B0、B1、B2 各有 4 个 warp：warp 0 到 3 依次分到 SMSP 0 到 3</text>
  <rect x="96" y="272" width="17" height="10" rx="2" style="fill: var(--fig-hi)"/><text class="lab2" x="120" y="281">正在发射</text>
  <rect x="200" y="272" width="17" height="10" rx="2" style="fill: var(--fig-2)"/><text class="lab2" x="224" y="281">就绪，没被选中</text>
  <rect x="343" y="272" width="17" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity=".55"/><text class="lab2" x="367" y="281">等数据</text>
  <rect x="434" y="272" width="17" height="10" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/><text class="lab2" x="458" y="281">空槽</text>
</svg>
<figcaption><strong>图 1-6</strong> 一个 SM 上驻留 3 个 block（B0、B1、B2）时的某一拍。格子里写的是这个 warp 属于哪个 block，每个 SMSP 有 12 个格子，整个 SM 共 48 个。每个调度器每拍最多发射一个 warp；SMSP 2 的 3 个 warp 都在等数据，这一拍空转。</figcaption>
</figure>

block 之间不能直接共享数据，只能通过显存交换，而这正是最慢的一层。所以要尽量把会重复读的数据放进同一个 block 里处理，这就是分块（tiling）。在 CUDA 里分块要自己管 shared memory 和线程同步，[第 2 节](#triton)的 Triton 把这部分交给编译器。

---

## 2 Triton：按 block 写 kernel {#triton}

Triton 按 block 写 kernel，正好对应 [1.3 节](#hw-model)说的分块。CUDA 要写清楚每个线程做什么，控制最细，但 shared memory、线程同步这些都要自己管。Triton 只要写清楚每个线程块做什么：把一块数据读进来，在片上算完，再写回显存，块内怎么分给线程、要不要经过 shared memory 由编译器决定。Triton 把一个线程块叫作一个 program，概念对应见[表 2-1](#tab-2-1)。

| CUDA | Triton | 写法 | 能否直接控制 |
|:--|:--|:--|:--|
| thread | 不暴露 | 写不到 `threadIdx` | 不能 |
| warp | 只给数量 | 启动时传 `num_warps=N`，默认 4 | 只能调数量 |
| block（CTA） | program | `@triton.jit` 函数体就是一个 program 的代码；`tl.program_id(axis)` ≈ `blockIdx`，`tl.num_programs(axis)` ≈ `gridDim` | 主要的编程层 |
| grid | grid | `kernel[grid](...)`，例如 `grid = (triton.cdiv(n, BLOCK),)` | 自己定 |
| shared memory | 由编译器分配 | 没有对应语句 | 不能 |
{#tab-2-1 caption="**表 2-1** CUDA 概念在 Triton 里的对应"}

下面四个例子由浅入深。grid 怎么切，看一个输出依赖哪些输入；program 里要不要循环，看这些输入能不能一次装进 register，装不下就要一块一块读，用一个累加器（accumulator）存中间结果：

| op | 类型 | 一个输出依赖 | grid 怎么切 | program 里 |
|:--|:--|:--|:--|:--|
| GELU | 逐元素 | 同位置的一个输入 | 展平成一维，按 BLOCK 切段 | 不循环 |
| softmax | 按行归约 | 整行（max 和 sum） | 一行一个 program | 不循环，整行一次读进来 |
| row sum | 按行归约 | 整行（sum） | 一行一个 program | 沿行循环，累加器存部分和 |
| matmul | 分块乘加 | A 的一行、B 的一列 | 二维，C 的一块一个 program | 沿 K 循环，累加器存 C 的一块 |
{#tab-2-2 caption="**表 2-2** 四个例子的 op 类型"}

代码都在 RTX 5090 上和 PyTorch 对照过。每个例子配一张图：左边是 grid，右边从上到下是计算单元、register、shared memory 和显存，箭头是数据流。`tl.load` 和 `tl.store` 在显存和 register（或 shared memory）之间直接搬数据，不经过计算单元；计算单元只从 register 读、写回 register。图里的「显存」指芯片外的 GDDR7，中间经过的 L2 和 L1 cache 由硬件自动管理，kernel 不直接控制，图里不画。[下一篇](/blog/gpu-training-analysis/#rmsnorm)手写的融合 RMSNorm 和之后的 [FlashAttention](/blog/flashattention-1-to-4/) 也都用 Triton 写。

### 2.1 GELU：逐元素 {#triton-gelu}

GELU 对每个元素单独算，$\Phi$ 是标准正态分布的累积分布函数，实际用 tanh 近似：

<p align="center">$y_i = \mathrm{GELU}(x_i) = x_i\,\Phi(x_i) \approx \tfrac{1}{2}\,x_i\left(1 + \tanh\left(\sqrt{2/\pi}\,(x_i + 0.044715\,x_i^3)\right)\right)$</p>

<figure id="fig-t-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 356" width="100%" role="img" aria-label="GELU kernel：数组展平成一维按 BLOCK 切段，一段一个 program；tl.load 把一段从显存直接读进 register，计算单元算 GELU，tl.store 从 register 直接写回显存；每个 SM 同时跑 12 个 program，1536 个 thread 全满">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-t-1-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-t-1-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-2)"/></marker><marker id="fig-t-1-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-1)"/></marker></defs>
  <text class="ttl" x="10" y="16">grid</text>
  <text class="ttl" x="162" y="16">一个 program（pid = 1）</text>
  <text class="tb" x="630" y="16" text-anchor="end" style="fill: var(--fig-hi)">逐元素：y[i] 只依赖 x[i]</text>
  <text class="lab2" x="10" y="48">x 展平成一维</text>
  <rect x="10.0" y="60" width="27.5" height="26" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="s" x="23.8" y="77" text-anchor="middle">0</text>
  <rect x="39.5" y="60" width="27.5" height="26" rx="3" style="fill: color-mix(in srgb, var(--fig-1) 30%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="53.2" y="77" text-anchor="middle">1</text>
  <rect x="69.0" y="60" width="27.5" height="26" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="s" x="82.8" y="77" text-anchor="middle">2</text>
  <rect x="98.5" y="60" width="27.5" height="26" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="s" x="112.2" y="77" text-anchor="middle">⋯</text>
  <text class="s" x="53.2" y="100" text-anchor="middle">pid=1</text>
  <text class="lab2" x="10" y="168">grid = (cdiv(n, BLOCK),)</text>
  <text class="lab2" x="10" y="183">n = 8388608</text>
  <text class="lab2" x="10" y="198">8192 个 program</text>
  <text class="ttl" x="10" y="219">一个 SM</text>
  <rect x="12" y="227" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <line x1="17.0" y1="231" x2="17.0" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="21.7" y1="231" x2="21.7" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="26.4" y1="231" x2="26.4" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="31.1" y1="231" x2="31.1" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="40" y="227" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="45.0" y1="231" x2="45.0" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="49.7" y1="231" x2="49.7" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="54.4" y1="231" x2="54.4" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="59.1" y1="231" x2="59.1" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="68" y="227" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="73.0" y1="231" x2="73.0" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="77.7" y1="231" x2="77.7" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="82.4" y1="231" x2="82.4" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="87.1" y1="231" x2="87.1" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="96" y="227" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="101.0" y1="231" x2="101.0" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="105.7" y1="231" x2="105.7" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="110.4" y1="231" x2="110.4" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="115.1" y1="231" x2="115.1" y2="241" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="12" y="249" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="17.0" y1="253" x2="17.0" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="21.7" y1="253" x2="21.7" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="26.4" y1="253" x2="26.4" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="31.1" y1="253" x2="31.1" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="40" y="249" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="45.0" y1="253" x2="45.0" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="49.7" y1="253" x2="49.7" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="54.4" y1="253" x2="54.4" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="59.1" y1="253" x2="59.1" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="68" y="249" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="73.0" y1="253" x2="73.0" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="77.7" y1="253" x2="77.7" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="82.4" y1="253" x2="82.4" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="87.1" y1="253" x2="87.1" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="96" y="249" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="101.0" y1="253" x2="101.0" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="105.7" y1="253" x2="105.7" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="110.4" y1="253" x2="110.4" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="115.1" y1="253" x2="115.1" y2="263" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="12" y="271" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="17.0" y1="275" x2="17.0" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="21.7" y1="275" x2="21.7" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="26.4" y1="275" x2="26.4" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="31.1" y1="275" x2="31.1" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="40" y="271" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="45.0" y1="275" x2="45.0" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="49.7" y1="275" x2="49.7" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="54.4" y1="275" x2="54.4" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="59.1" y1="275" x2="59.1" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="68" y="271" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="73.0" y1="275" x2="73.0" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="77.7" y1="275" x2="77.7" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="82.4" y1="275" x2="82.4" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="87.1" y1="275" x2="87.1" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="96" y="271" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="101.0" y1="275" x2="101.0" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="105.7" y1="275" x2="105.7" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="110.4" y1="275" x2="110.4" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="115.1" y1="275" x2="115.1" y2="285" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <text class="s" x="10" y="301">每格一个 program（4 warp）</text><text class="s" x="10" y="315">12 × 128 = 1536 thread</text>
  <text class="s" x="10" y="329">线程数先满</text>
  <rect class="band" x="146" y="26" width="490" height="66" rx="8"/>
  <text class="lab2" x="152" y="56.5">计算</text>
  <text class="lab2" x="152" y="69.5">单元</text>
  <rect class="band" x="146" y="100" width="490" height="76" rx="8"/>
  <text class="lab2" x="152" y="142.0">register</text>
  <rect class="band" x="146" y="184" width="490" height="32" rx="8"/>
  <text class="lab2" x="152" y="197.5">shared</text>
  <text class="lab2" x="152" y="210.5">memory</text>
  <rect class="band" x="146" y="224" width="490" height="84" rx="8"/>
  <text class="lab2" x="152" y="270.0">显存</text>
  <text class="s" x="152" y="284.0">GDDR7</text>
  <line x1="130.0" y1="73.0" x2="144.0" y2="73.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-t-1-m0)"/>
  <rect class="op" x="281.0" y="46" width="84.0" height="38" rx="6"/>
  <text class="tb" x="323.0" y="62" text-anchor="middle">offsets</text><text class="s" x="323.0" y="77" text-anchor="middle">tl.arange</text>
  <rect class="op" x="425.0" y="46" width="84.0" height="38" rx="6"/>
  <text class="tb" x="467.0" y="62" text-anchor="middle">gelu</text><text class="s" x="467.0" y="77" text-anchor="middle">tl.exp</text>
  <text class="lab2" x="152" y="324">每个 program 1024 个元素，每个 thread 8 个，都在 register 里，不用 shared memory；</text>
  <text class="lab2" x="152" y="339">offsets 是 load、store 的地址。显存只读 x 一次、写 y 一次。</text>
  <rect x="298.0" y="110" width="50" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="323.0" y="123" text-anchor="middle">offsets</text>
  <line x1="323.0" y1="86.0" x2="323.0" y2="108.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-1-m1)"/>
  <rect x="373.0" y="110" width="44" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="395.0" y="123" text-anchor="middle">x 段</text>
  <line x1="395.0" y1="250.0" x2="395.0" y2="130.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-1-m2)"/>
  <text class="s" x="399.0" y="194.0" text-anchor="start" style="fill: var(--fig-1)">tl.load</text>
  <line x1="409.0" y1="108.0" x2="451.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-1-m1)"/>
  <rect x="517.0" y="110" width="44" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="539.0" y="123" text-anchor="middle">y 段</text>
  <line x1="483.0" y1="86.0" x2="525.0" y2="108.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-1-m1)"/>
  <line x1="539.0" y1="130.0" x2="539.0" y2="250.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-1-m2)"/>
  <text class="s" x="543.0" y="194.0" text-anchor="start" style="fill: var(--fig-1)">tl.store</text>
  <rect x="355.0" y="252" width="80.0" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <rect x="385.0" y="252.0" width="20.0" height="28.0" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="395.0" y="293" text-anchor="middle">x，32 MiB</text>
  <rect x="499.0" y="252" width="80.0" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <rect x="529.0" y="252.0" width="20.0" height="28.0" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="539.0" y="293" text-anchor="middle">y，32 MiB</text>
  <text class="s" x="214" y="204.0">不用（0 B）</text>
</svg>
<figcaption><strong>图 2-1</strong> GELU kernel（n = 8388608，fp32）。左边是 grid：数组展平成一维、按 BLOCK = 1024 切段，一段一个 program；下面是一个 SM 同时跑几个 program，12 格是上限（1536 个 thread）。右边是 program 1：最上面是计算单元，下面依次是 register、shared memory 和显存（芯片外的 GDDR7）。箭头是数据流：tl.load 从显存直接读进 register，tl.store 从 register 直接写回显存，计算单元只从 register 读、写回 register。读写显存都会经过 L2 和 L1 cache，由硬件自动管理，图里不画。</figcaption>
</figure>

逐元素 op 最简单，每个元素各算各的，program 之间不需要配合。把 n 个元素切成每段 BLOCK 个，一段交给一个 program（[图 2-1](#fig-t-1)）。kernel 里先用 `tl.program_id` 算出自己负责的下标，读进来，算完写回：

```python {role="program" tag="处理第 pid 段的 BLOCK 个元素"}
@triton.jit
def gelu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)               # 第几个 program（≈ CUDA 的 blockIdx.x）
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # 这个 program 负责的 BLOCK 个下标
    mask = offsets < n                        # 最后一个 program 可能越界
    x = tl.load(x_ptr + offsets, mask=mask)   # 从显存读到 register
    # tanh 近似：0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))，tanh(a) = (e^{2a} − 1)/(e^{2a} + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    e = tl.exp(2 * a)
    y = 0.5 * x * (1 + (e - 1) / (e + 1))
    tl.store(y_ptr + offsets, y, mask=mask)   # 写回显存
```

启动时只要定 grid，也就是 program 的个数。program 个数和 SM 个数无关：5090 有 170 个 SM，program 多于 SM 能同时放下的数量时，GPU 会一批一批地调度，下面会看到实测的样子。

```python {role="grid" tag="一段一个 program"}
def gelu(x):
    y = torch.empty_like(x)
    n, BLOCK = x.numel(), 1024                # 每个 program 处理 1024 个元素
    grid = (triton.cdiv(n, BLOCK),)           # program 个数，不是 SM 个数
    gelu_kernel[grid](x, y, n, BLOCK=BLOCK)
    return y
```

program 编译后就是 CUDA 的一个 block，一个 SM 能同时驻留几个，看[表 1-2](#tab-1-2) 里哪条上限先用完。GELU 一个 program 4 个 warp，SM 最多驻留 48 个 warp，所以一个 SM 同时放 12 个 program，block 上限 24 用不满；register 和 shared memory 也都够。全卡一次放得下 170 × 12 = 2040 个，8192 个 program 要 8192 / 2040 = 4.02 批，也就是 5 批，第 5 批只有 32 个。

<figure id="fig-t-5" class="fg-fig">
<svg class="fg" viewBox="0 0 640 404" width="100%" role="img" aria-label="GELU 的 8192 个 program 在 170 个 SM 上的实测时间线：上面是全卡同时在跑的 program 数，按批次堆叠，最多 2040 个；下面是一个 SM 的 12 个名额，前一个 program 结束就换下一个进来，批次之间没有清楚的分界，最后一批只有 32 个">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <line class="grid" x1="74" y1="200.0" x2="620" y2="200.0"/><text class="tick" x="66" y="204.0" text-anchor="end">0</text>
  <line class="grid" x1="74" y1="163.7" x2="620" y2="163.7"/><text class="tick" x="66" y="167.7" text-anchor="end">500</text>
  <line class="grid" x1="74" y1="127.5" x2="620" y2="127.5"/><text class="tick" x="66" y="131.5" text-anchor="end">1000</text>
  <line class="grid" x1="74" y1="91.2" x2="620" y2="91.2"/><text class="tick" x="66" y="95.2" text-anchor="end">1500</text>
  <line class="grid" x1="74" y1="52.0" x2="620" y2="52.0"/><text class="tick" x="66" y="56.0" text-anchor="end">2040</text>
  <text class="lab2" x="74" y="28">全卡同时在跑的 program 数，按批次堆叠；虚线是 170 个 SM × 12 = 2040 个名额</text>
  <line class="ref" x1="74" y1="52.0" x2="620" y2="52.0"/>
  <g class="m"><title>第 1 批：program 0 到 2039，共 2040 个</title><polygon points="74.0,191.2 76.4,88.4 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.3 95.4,55.4 97.7,58.5 100.1,62.5 102.5,67.2 104.9,71.8 107.2,77.0 109.6,81.5 112.0,85.1 114.4,89.8 116.7,94.2 119.1,98.4 121.5,101.8 123.9,105.1 126.2,108.0 128.6,109.9 131.0,112.1 133.3,115.8 135.7,120.3 138.1,126.2 140.5,132.8 142.8,138.2 145.2,140.8 147.6,143.4 150.0,144.6 152.3,145.4 154.7,145.5 157.1,146.4 159.5,147.0 161.8,148.4 164.2,150.2 166.6,151.2 169.0,155.5 171.3,158.4 173.7,160.0 176.1,162.1 178.5,165.0 180.8,167.6 183.2,170.7 185.6,175.6 187.9,179.3 190.3,181.1 192.7,185.9 195.1,190.8 197.4,192.7 199.8,195.5 202.2,197.2 204.6,197.8 206.9,198.8 209.3,199.5 211.7,199.9 214.1,200.0 216.4,200.0 218.8,200.0 221.2,200.0 223.6,200.0 225.9,200.0 228.3,200.0 230.7,200.0 233.1,200.0 235.4,200.0 237.8,200.0 240.2,200.0 242.5,200.0 244.9,200.0 247.3,200.0 249.7,200.0 252.0,200.0 254.4,200.0 256.8,200.0 259.2,200.0 261.5,200.0 263.9,200.0 266.3,200.0 268.7,200.0 271.0,200.0 273.4,200.0 275.8,200.0 278.2,200.0 280.5,200.0 282.9,200.0 285.3,200.0 287.7,200.0 290.0,200.0 292.4,200.0 294.8,200.0 297.1,200.0 299.5,200.0 301.9,200.0 304.3,200.0 306.6,200.0 309.0,200.0 311.4,200.0 313.8,200.0 316.1,200.0 318.5,200.0 320.9,200.0 323.3,200.0 325.6,200.0 328.0,200.0 330.4,200.0 332.8,200.0 335.1,200.0 337.5,200.0 339.9,200.0 342.3,200.0 344.6,200.0 347.0,200.0 349.4,200.0 351.7,200.0 354.1,200.0 356.5,200.0 358.9,200.0 361.2,200.0 363.6,200.0 366.0,200.0 368.4,200.0 370.7,200.0 373.1,200.0 375.5,200.0 377.9,200.0 380.2,200.0 382.6,200.0 385.0,200.0 387.4,200.0 389.7,200.0 392.1,200.0 394.5,200.0 396.9,200.0 399.2,200.0 401.6,200.0 404.0,200.0 406.3,200.0 408.7,200.0 411.1,200.0 413.5,200.0 415.8,200.0 418.2,200.0 420.6,200.0 423.0,200.0 425.3,200.0 427.7,200.0 430.1,200.0 432.5,200.0 434.8,200.0 437.2,200.0 439.6,200.0 442.0,200.0 444.3,200.0 446.7,200.0 449.1,200.0 451.5,200.0 453.8,200.0 456.2,200.0 458.6,200.0 460.9,200.0 463.3,200.0 465.7,200.0 468.1,200.0 470.4,200.0 472.8,200.0 475.2,200.0 477.6,200.0 479.9,200.0 482.3,200.0 484.7,200.0 487.1,200.0 489.4,200.0 491.8,200.0 494.2,200.0 496.6,200.0 498.9,200.0 501.3,200.0 503.7,200.0 506.1,200.0 508.4,200.0 510.8,200.0 513.2,200.0 515.5,200.0 517.9,200.0 520.3,200.0 522.7,200.0 525.0,200.0 527.4,200.0 529.8,200.0 532.2,200.0 534.5,200.0 536.9,200.0 539.3,200.0 541.7,200.0 544.0,200.0 546.4,200.0 548.8,200.0 551.2,200.0 553.5,200.0 555.9,200.0 558.3,200.0 560.7,200.0 563.0,200.0 565.4,200.0 567.8,200.0 570.1,200.0 572.5,200.0 574.9,200.0 577.3,200.0 579.6,200.0 582.0,200.0 584.4,200.0 586.8,200.0 589.1,200.0 591.5,200.0 593.9,200.0 596.3,200.0 598.6,200.0 601.0,200.0 603.4,200.0 605.8,200.0 608.1,200.0 610.5,200.0 612.9,200.0 615.3,200.0 617.6,200.0 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,200.0 563.0,200.0 560.7,200.0 558.3,200.0 555.9,200.0 553.5,200.0 551.2,200.0 548.8,200.0 546.4,200.0 544.0,200.0 541.7,200.0 539.3,200.0 536.9,200.0 534.5,200.0 532.2,200.0 529.8,200.0 527.4,200.0 525.0,200.0 522.7,200.0 520.3,200.0 517.9,200.0 515.5,200.0 513.2,200.0 510.8,200.0 508.4,200.0 506.1,200.0 503.7,200.0 501.3,200.0 498.9,200.0 496.6,200.0 494.2,200.0 491.8,200.0 489.4,200.0 487.1,200.0 484.7,200.0 482.3,200.0 479.9,200.0 477.6,200.0 475.2,200.0 472.8,200.0 470.4,200.0 468.1,200.0 465.7,200.0 463.3,200.0 460.9,200.0 458.6,200.0 456.2,200.0 453.8,200.0 451.5,200.0 449.1,200.0 446.7,200.0 444.3,200.0 442.0,200.0 439.6,200.0 437.2,200.0 434.8,200.0 432.5,200.0 430.1,200.0 427.7,200.0 425.3,200.0 423.0,200.0 420.6,200.0 418.2,200.0 415.8,200.0 413.5,200.0 411.1,200.0 408.7,200.0 406.3,200.0 404.0,200.0 401.6,200.0 399.2,200.0 396.9,200.0 394.5,200.0 392.1,200.0 389.7,200.0 387.4,200.0 385.0,200.0 382.6,200.0 380.2,200.0 377.9,200.0 375.5,200.0 373.1,200.0 370.7,200.0 368.4,200.0 366.0,200.0 363.6,200.0 361.2,200.0 358.9,200.0 356.5,200.0 354.1,200.0 351.7,200.0 349.4,200.0 347.0,200.0 344.6,200.0 342.3,200.0 339.9,200.0 337.5,200.0 335.1,200.0 332.8,200.0 330.4,200.0 328.0,200.0 325.6,200.0 323.3,200.0 320.9,200.0 318.5,200.0 316.1,200.0 313.8,200.0 311.4,200.0 309.0,200.0 306.6,200.0 304.3,200.0 301.9,200.0 299.5,200.0 297.1,200.0 294.8,200.0 292.4,200.0 290.0,200.0 287.7,200.0 285.3,200.0 282.9,200.0 280.5,200.0 278.2,200.0 275.8,200.0 273.4,200.0 271.0,200.0 268.7,200.0 266.3,200.0 263.9,200.0 261.5,200.0 259.2,200.0 256.8,200.0 254.4,200.0 252.0,200.0 249.7,200.0 247.3,200.0 244.9,200.0 242.5,200.0 240.2,200.0 237.8,200.0 235.4,200.0 233.1,200.0 230.7,200.0 228.3,200.0 225.9,200.0 223.6,200.0 221.2,200.0 218.8,200.0 216.4,200.0 214.1,200.0 211.7,200.0 209.3,200.0 206.9,200.0 204.6,200.0 202.2,200.0 199.8,200.0 197.4,200.0 195.1,200.0 192.7,200.0 190.3,200.0 187.9,200.0 185.6,200.0 183.2,200.0 180.8,200.0 178.5,200.0 176.1,200.0 173.7,200.0 171.3,200.0 169.0,200.0 166.6,200.0 164.2,200.0 161.8,200.0 159.5,200.0 157.1,200.0 154.7,200.0 152.3,200.0 150.0,200.0 147.6,200.0 145.2,200.0 142.8,200.0 140.5,200.0 138.1,200.0 135.7,200.0 133.3,200.0 131.0,200.0 128.6,200.0 126.2,200.0 123.9,200.0 121.5,200.0 119.1,200.0 116.7,200.0 114.4,200.0 112.0,200.0 109.6,200.0 107.2,200.0 104.9,200.0 102.5,200.0 100.1,200.0 97.7,200.0 95.4,200.0 93.0,200.0 90.6,200.0 88.2,200.0 85.9,200.0 83.5,200.0 81.1,200.0 78.7,200.0 76.4,200.0 74.0,200.0" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>第 2 批：program 2040 到 4079，共 2040 个</title><polygon points="74.0,191.2 76.4,88.4 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.3 95.4,55.4 97.7,57.7 100.1,60.4 102.5,63.9 104.9,66.4 107.2,68.5 109.6,68.4 112.0,67.7 114.4,69.3 116.7,71.0 119.1,72.5 121.5,73.7 123.9,73.8 126.2,74.8 128.6,74.8 131.0,74.3 133.3,75.6 135.7,78.0 138.1,79.3 140.5,81.2 142.8,82.3 145.2,82.2 147.6,83.3 150.0,82.3 152.3,83.0 154.7,82.6 157.1,83.6 159.5,83.3 161.8,84.1 164.2,84.6 166.6,85.1 169.0,87.2 171.3,87.7 173.7,87.6 176.1,88.6 178.5,91.2 180.8,92.8 183.2,94.7 185.6,96.7 187.9,97.9 190.3,97.0 192.7,100.5 195.1,103.0 197.4,100.5 199.8,102.1 202.2,100.7 204.6,97.2 206.9,96.2 209.3,91.5 211.7,89.6 214.1,93.1 216.4,96.7 218.8,99.5 221.2,101.5 223.6,102.6 225.9,104.3 228.3,105.0 230.7,105.8 233.1,107.4 235.4,108.7 237.8,109.6 240.2,110.1 242.5,110.7 244.9,112.1 247.3,112.7 249.7,113.5 252.0,114.2 254.4,115.1 256.8,115.9 259.2,116.9 261.5,117.1 263.9,117.3 266.3,117.7 268.7,118.2 271.0,118.7 273.4,119.2 275.8,119.9 278.2,120.2 280.5,120.8 282.9,122.4 285.3,122.9 287.7,123.5 290.0,124.1 292.4,125.5 294.8,126.7 297.1,127.4 299.5,129.0 301.9,130.7 304.3,134.4 306.6,137.2 309.0,139.3 311.4,141.2 313.8,143.3 316.1,145.7 318.5,147.5 320.9,149.1 323.3,150.8 325.6,151.5 328.0,153.2 330.4,154.1 332.8,154.9 335.1,155.7 337.5,157.6 339.9,159.2 342.3,160.8 344.6,163.1 347.0,165.5 349.4,167.4 351.7,169.7 354.1,170.3 356.5,171.3 358.9,172.1 361.2,172.8 363.6,175.9 366.0,179.1 368.4,181.5 370.7,182.6 373.1,184.7 375.5,187.2 377.9,188.5 380.2,189.8 382.6,190.4 385.0,190.8 387.4,191.5 389.7,192.0 392.1,192.7 394.5,193.2 396.9,193.6 399.2,193.8 401.6,193.8 404.0,194.0 406.3,194.3 408.7,195.1 411.1,195.7 413.5,196.4 415.8,197.5 418.2,198.3 420.6,199.1 423.0,199.5 425.3,199.5 427.7,199.6 430.1,199.6 432.5,199.7 434.8,199.7 437.2,199.8 439.6,199.9 442.0,199.9 444.3,199.9 446.7,199.9 449.1,199.9 451.5,200.0 453.8,200.0 456.2,200.0 458.6,200.0 460.9,200.0 463.3,200.0 465.7,200.0 468.1,200.0 470.4,200.0 472.8,200.0 475.2,200.0 477.6,200.0 479.9,200.0 482.3,200.0 484.7,200.0 487.1,200.0 489.4,200.0 491.8,200.0 494.2,200.0 496.6,200.0 498.9,200.0 501.3,200.0 503.7,200.0 506.1,200.0 508.4,200.0 510.8,200.0 513.2,200.0 515.5,200.0 517.9,200.0 520.3,200.0 522.7,200.0 525.0,200.0 527.4,200.0 529.8,200.0 532.2,200.0 534.5,200.0 536.9,200.0 539.3,200.0 541.7,200.0 544.0,200.0 546.4,200.0 548.8,200.0 551.2,200.0 553.5,200.0 555.9,200.0 558.3,200.0 560.7,200.0 563.0,200.0 565.4,200.0 567.8,200.0 570.1,200.0 572.5,200.0 574.9,200.0 577.3,200.0 579.6,200.0 582.0,200.0 584.4,200.0 586.8,200.0 589.1,200.0 591.5,200.0 593.9,200.0 596.3,200.0 598.6,200.0 601.0,200.0 603.4,200.0 605.8,200.0 608.1,200.0 610.5,200.0 612.9,200.0 615.3,200.0 617.6,200.0 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,200.0 563.0,200.0 560.7,200.0 558.3,200.0 555.9,200.0 553.5,200.0 551.2,200.0 548.8,200.0 546.4,200.0 544.0,200.0 541.7,200.0 539.3,200.0 536.9,200.0 534.5,200.0 532.2,200.0 529.8,200.0 527.4,200.0 525.0,200.0 522.7,200.0 520.3,200.0 517.9,200.0 515.5,200.0 513.2,200.0 510.8,200.0 508.4,200.0 506.1,200.0 503.7,200.0 501.3,200.0 498.9,200.0 496.6,200.0 494.2,200.0 491.8,200.0 489.4,200.0 487.1,200.0 484.7,200.0 482.3,200.0 479.9,200.0 477.6,200.0 475.2,200.0 472.8,200.0 470.4,200.0 468.1,200.0 465.7,200.0 463.3,200.0 460.9,200.0 458.6,200.0 456.2,200.0 453.8,200.0 451.5,200.0 449.1,200.0 446.7,200.0 444.3,200.0 442.0,200.0 439.6,200.0 437.2,200.0 434.8,200.0 432.5,200.0 430.1,200.0 427.7,200.0 425.3,200.0 423.0,200.0 420.6,200.0 418.2,200.0 415.8,200.0 413.5,200.0 411.1,200.0 408.7,200.0 406.3,200.0 404.0,200.0 401.6,200.0 399.2,200.0 396.9,200.0 394.5,200.0 392.1,200.0 389.7,200.0 387.4,200.0 385.0,200.0 382.6,200.0 380.2,200.0 377.9,200.0 375.5,200.0 373.1,200.0 370.7,200.0 368.4,200.0 366.0,200.0 363.6,200.0 361.2,200.0 358.9,200.0 356.5,200.0 354.1,200.0 351.7,200.0 349.4,200.0 347.0,200.0 344.6,200.0 342.3,200.0 339.9,200.0 337.5,200.0 335.1,200.0 332.8,200.0 330.4,200.0 328.0,200.0 325.6,200.0 323.3,200.0 320.9,200.0 318.5,200.0 316.1,200.0 313.8,200.0 311.4,200.0 309.0,200.0 306.6,200.0 304.3,200.0 301.9,200.0 299.5,200.0 297.1,200.0 294.8,200.0 292.4,200.0 290.0,200.0 287.7,200.0 285.3,200.0 282.9,200.0 280.5,200.0 278.2,200.0 275.8,200.0 273.4,200.0 271.0,200.0 268.7,200.0 266.3,200.0 263.9,200.0 261.5,200.0 259.2,200.0 256.8,200.0 254.4,200.0 252.0,200.0 249.7,200.0 247.3,200.0 244.9,200.0 242.5,200.0 240.2,200.0 237.8,200.0 235.4,200.0 233.1,200.0 230.7,200.0 228.3,200.0 225.9,200.0 223.6,200.0 221.2,200.0 218.8,200.0 216.4,200.0 214.1,200.0 211.7,199.9 209.3,199.5 206.9,198.8 204.6,197.8 202.2,197.2 199.8,195.5 197.4,192.7 195.1,190.8 192.7,185.9 190.3,181.1 187.9,179.3 185.6,175.6 183.2,170.7 180.8,167.6 178.5,165.0 176.1,162.1 173.7,160.0 171.3,158.4 169.0,155.5 166.6,151.2 164.2,150.2 161.8,148.4 159.5,147.0 157.1,146.4 154.7,145.5 152.3,145.4 150.0,144.6 147.6,143.4 145.2,140.8 142.8,138.2 140.5,132.8 138.1,126.2 135.7,120.3 133.3,115.8 131.0,112.1 128.6,109.9 126.2,108.0 123.9,105.1 121.5,101.8 119.1,98.4 116.7,94.2 114.4,89.8 112.0,85.1 109.6,81.5 107.2,77.0 104.9,71.8 102.5,67.2 100.1,62.5 97.7,58.5 95.4,55.4 93.0,52.3 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,88.4 74.0,191.2" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>第 3 批：program 4080 到 6119，共 2040 个</title><polygon points="74.0,191.2 76.4,88.4 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.3 95.4,55.4 97.7,57.7 100.1,60.4 102.5,63.9 104.9,66.4 107.2,68.5 109.6,68.4 112.0,67.7 114.4,69.3 116.7,71.0 119.1,72.5 121.5,73.7 123.9,73.8 126.2,74.8 128.6,74.8 131.0,74.3 133.3,75.6 135.7,78.0 138.1,79.3 140.5,81.2 142.8,82.3 145.2,82.2 147.6,83.3 150.0,82.3 152.3,83.0 154.7,82.6 157.1,83.6 159.5,83.3 161.8,84.1 164.2,84.6 166.6,85.1 169.0,87.2 171.3,87.7 173.7,87.6 176.1,88.6 178.5,91.2 180.8,92.8 183.2,94.7 185.6,96.7 187.9,97.9 190.3,97.0 192.7,100.5 195.1,103.0 197.4,100.5 199.8,102.1 202.2,100.7 204.6,97.2 206.9,96.2 209.3,91.5 211.7,87.5 214.1,82.8 216.4,81.1 218.8,78.3 221.2,76.1 223.6,74.1 225.9,74.1 228.3,73.0 230.7,72.2 233.1,72.5 235.4,73.0 237.8,71.9 240.2,72.0 242.5,71.7 244.9,71.9 247.3,72.0 249.7,72.3 252.0,72.5 254.4,72.4 256.8,71.8 259.2,71.9 261.5,72.1 263.9,72.2 266.3,72.6 268.7,73.0 271.0,72.9 273.4,73.1 275.8,73.3 278.2,73.2 280.5,73.3 282.9,74.0 285.3,73.6 287.7,73.8 290.0,74.2 292.4,74.5 294.8,75.0 297.1,75.5 299.5,77.0 301.9,77.0 304.3,79.3 306.6,80.4 309.0,81.7 311.4,83.2 313.8,84.6 316.1,85.4 318.5,86.1 320.9,87.3 323.3,87.2 325.6,87.5 328.0,89.0 330.4,89.1 332.8,89.4 335.1,90.3 337.5,91.8 339.9,92.1 342.3,93.4 344.6,95.5 347.0,96.6 349.4,98.1 351.7,99.7 354.1,99.7 356.5,101.4 358.9,100.5 361.2,101.1 363.6,104.3 366.0,106.9 368.4,107.6 370.7,107.7 373.1,109.0 375.5,108.0 377.9,106.9 380.2,105.7 382.6,104.4 385.0,103.0 387.4,102.3 389.7,102.3 392.1,101.4 394.5,99.3 396.9,96.9 399.2,96.8 401.6,97.1 404.0,97.3 406.3,98.6 408.7,98.9 411.1,99.7 413.5,101.8 415.8,106.6 418.2,112.7 420.6,120.1 423.0,125.7 425.3,129.7 427.7,131.8 430.1,133.8 432.5,136.7 434.8,140.5 437.2,143.4 439.6,146.2 442.0,149.1 444.3,151.8 446.7,154.2 449.1,157.6 451.5,159.7 453.8,162.8 456.2,165.0 458.6,167.2 460.9,169.4 463.3,172.9 465.7,174.2 468.1,176.1 470.4,177.7 472.8,179.1 475.2,180.5 477.6,182.8 479.9,184.3 482.3,185.5 484.7,187.2 487.1,188.4 489.4,189.2 491.8,189.3 494.2,189.9 496.6,190.9 498.9,192.0 501.3,192.6 503.7,193.1 506.1,195.1 508.4,195.5 510.8,196.1 513.2,196.4 515.5,197.0 517.9,197.6 520.3,198.4 522.7,198.8 525.0,198.8 527.4,199.0 529.8,199.2 532.2,199.4 534.5,199.4 536.9,199.7 539.3,199.8 541.7,199.8 544.0,199.8 546.4,199.9 548.8,199.9 551.2,200.0 553.5,200.0 555.9,200.0 558.3,200.0 560.7,200.0 563.0,200.0 565.4,200.0 567.8,200.0 570.1,200.0 572.5,200.0 574.9,200.0 577.3,200.0 579.6,200.0 582.0,200.0 584.4,200.0 586.8,200.0 589.1,200.0 591.5,200.0 593.9,200.0 596.3,200.0 598.6,200.0 601.0,200.0 603.4,200.0 605.8,200.0 608.1,200.0 610.5,200.0 612.9,200.0 615.3,200.0 617.6,200.0 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,200.0 563.0,200.0 560.7,200.0 558.3,200.0 555.9,200.0 553.5,200.0 551.2,200.0 548.8,200.0 546.4,200.0 544.0,200.0 541.7,200.0 539.3,200.0 536.9,200.0 534.5,200.0 532.2,200.0 529.8,200.0 527.4,200.0 525.0,200.0 522.7,200.0 520.3,200.0 517.9,200.0 515.5,200.0 513.2,200.0 510.8,200.0 508.4,200.0 506.1,200.0 503.7,200.0 501.3,200.0 498.9,200.0 496.6,200.0 494.2,200.0 491.8,200.0 489.4,200.0 487.1,200.0 484.7,200.0 482.3,200.0 479.9,200.0 477.6,200.0 475.2,200.0 472.8,200.0 470.4,200.0 468.1,200.0 465.7,200.0 463.3,200.0 460.9,200.0 458.6,200.0 456.2,200.0 453.8,200.0 451.5,200.0 449.1,199.9 446.7,199.9 444.3,199.9 442.0,199.9 439.6,199.9 437.2,199.8 434.8,199.7 432.5,199.7 430.1,199.6 427.7,199.6 425.3,199.5 423.0,199.5 420.6,199.1 418.2,198.3 415.8,197.5 413.5,196.4 411.1,195.7 408.7,195.1 406.3,194.3 404.0,194.0 401.6,193.8 399.2,193.8 396.9,193.6 394.5,193.2 392.1,192.7 389.7,192.0 387.4,191.5 385.0,190.8 382.6,190.4 380.2,189.8 377.9,188.5 375.5,187.2 373.1,184.7 370.7,182.6 368.4,181.5 366.0,179.1 363.6,175.9 361.2,172.8 358.9,172.1 356.5,171.3 354.1,170.3 351.7,169.7 349.4,167.4 347.0,165.5 344.6,163.1 342.3,160.8 339.9,159.2 337.5,157.6 335.1,155.7 332.8,154.9 330.4,154.1 328.0,153.2 325.6,151.5 323.3,150.8 320.9,149.1 318.5,147.5 316.1,145.7 313.8,143.3 311.4,141.2 309.0,139.3 306.6,137.2 304.3,134.4 301.9,130.7 299.5,129.0 297.1,127.4 294.8,126.7 292.4,125.5 290.0,124.1 287.7,123.5 285.3,122.9 282.9,122.4 280.5,120.8 278.2,120.2 275.8,119.9 273.4,119.2 271.0,118.7 268.7,118.2 266.3,117.7 263.9,117.3 261.5,117.1 259.2,116.9 256.8,115.9 254.4,115.1 252.0,114.2 249.7,113.5 247.3,112.7 244.9,112.1 242.5,110.7 240.2,110.1 237.8,109.6 235.4,108.7 233.1,107.4 230.7,105.8 228.3,105.0 225.9,104.3 223.6,102.6 221.2,101.5 218.8,99.5 216.4,96.7 214.1,93.1 211.7,89.6 209.3,91.5 206.9,96.2 204.6,97.2 202.2,100.7 199.8,102.1 197.4,100.5 195.1,103.0 192.7,100.5 190.3,97.0 187.9,97.9 185.6,96.7 183.2,94.7 180.8,92.8 178.5,91.2 176.1,88.6 173.7,87.6 171.3,87.7 169.0,87.2 166.6,85.1 164.2,84.6 161.8,84.1 159.5,83.3 157.1,83.6 154.7,82.6 152.3,83.0 150.0,82.3 147.6,83.3 145.2,82.2 142.8,82.3 140.5,81.2 138.1,79.3 135.7,78.0 133.3,75.6 131.0,74.3 128.6,74.8 126.2,74.8 123.9,73.8 121.5,73.7 119.1,72.5 116.7,71.0 114.4,69.3 112.0,67.7 109.6,68.4 107.2,68.5 104.9,66.4 102.5,63.9 100.1,60.4 97.7,57.7 95.4,55.4 93.0,52.3 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,88.4 74.0,191.2" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>第 4 批：program 6120 到 8159，共 2040 个</title><polygon points="74.0,191.2 76.4,88.4 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.3 95.4,55.4 97.7,57.7 100.1,60.4 102.5,63.9 104.9,66.4 107.2,68.5 109.6,68.4 112.0,67.7 114.4,69.3 116.7,71.0 119.1,72.5 121.5,73.7 123.9,73.8 126.2,74.8 128.6,74.8 131.0,74.3 133.3,75.6 135.7,78.0 138.1,79.3 140.5,81.2 142.8,82.3 145.2,82.2 147.6,83.3 150.0,82.3 152.3,83.0 154.7,82.6 157.1,83.6 159.5,83.3 161.8,84.1 164.2,84.6 166.6,85.1 169.0,87.2 171.3,87.7 173.7,87.6 176.1,88.6 178.5,91.2 180.8,92.8 183.2,94.7 185.6,96.7 187.9,97.9 190.3,97.0 192.7,100.5 195.1,103.0 197.4,100.5 199.8,102.1 202.2,100.7 204.6,97.2 206.9,96.2 209.3,91.5 211.7,87.5 214.1,82.8 216.4,81.1 218.8,78.3 221.2,76.1 223.6,74.1 225.9,74.1 228.3,73.0 230.7,72.2 233.1,72.5 235.4,73.0 237.8,71.9 240.2,72.0 242.5,71.7 244.9,71.9 247.3,72.0 249.7,72.3 252.0,72.5 254.4,72.4 256.8,71.8 259.2,71.9 261.5,72.1 263.9,72.2 266.3,72.6 268.7,73.0 271.0,72.9 273.4,73.1 275.8,73.3 278.2,73.2 280.5,73.3 282.9,74.0 285.3,73.6 287.7,73.8 290.0,74.2 292.4,74.5 294.8,75.0 297.1,75.5 299.5,77.0 301.9,77.0 304.3,79.3 306.6,80.4 309.0,81.7 311.4,83.2 313.8,84.6 316.1,85.4 318.5,86.1 320.9,87.3 323.3,87.2 325.6,87.5 328.0,89.0 330.4,89.1 332.8,89.4 335.1,90.3 337.5,91.8 339.9,92.1 342.3,93.4 344.6,95.5 347.0,96.6 349.4,98.1 351.7,99.7 354.1,99.7 356.5,101.4 358.9,100.5 361.2,101.1 363.6,104.3 366.0,106.9 368.4,107.6 370.7,107.7 373.1,109.0 375.5,108.0 377.9,106.9 380.2,105.7 382.6,104.4 385.0,103.0 387.4,102.3 389.7,102.3 392.1,101.4 394.5,99.3 396.9,96.9 399.2,96.8 401.6,97.1 404.0,97.3 406.3,98.6 408.7,98.9 411.1,99.1 413.5,98.9 415.8,99.8 418.2,98.3 420.6,97.3 423.0,94.7 425.3,91.1 427.7,87.8 430.1,87.3 432.5,86.6 434.8,86.8 437.2,87.7 439.6,88.0 442.0,87.5 444.3,86.5 446.7,83.1 449.1,83.2 451.5,82.7 453.8,83.6 456.2,84.3 458.6,84.5 460.9,83.6 463.3,84.6 465.7,84.6 468.1,85.2 470.4,86.0 472.8,86.4 475.2,88.1 477.6,89.1 479.9,89.7 482.3,90.2 484.7,91.5 487.1,91.2 489.4,92.1 491.8,91.6 494.2,93.1 496.6,95.0 498.9,97.3 501.3,96.8 503.7,96.5 506.1,97.5 508.4,96.5 510.8,97.6 513.2,96.8 515.5,98.1 517.9,97.6 520.3,103.2 522.7,108.2 525.0,111.8 527.4,115.7 529.8,121.6 532.2,125.7 534.5,129.2 536.9,135.1 539.3,138.8 541.7,141.6 544.0,144.6 546.4,147.6 548.8,151.5 551.2,154.5 553.5,156.7 555.9,159.2 558.3,161.8 560.7,165.1 563.0,168.2 565.4,171.1 567.8,173.8 570.1,176.1 572.5,178.2 574.9,179.5 577.3,182.5 579.6,184.6 582.0,187.6 584.4,189.3 586.8,191.0 589.1,191.9 591.5,193.5 593.9,194.3 596.3,195.4 598.6,196.3 601.0,197.9 603.4,198.1 605.8,198.4 608.1,198.5 610.5,199.0 612.9,199.1 615.3,199.5 617.6,199.6 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,200.0 563.0,200.0 560.7,200.0 558.3,200.0 555.9,200.0 553.5,200.0 551.2,200.0 548.8,199.9 546.4,199.9 544.0,199.8 541.7,199.8 539.3,199.8 536.9,199.7 534.5,199.4 532.2,199.4 529.8,199.2 527.4,199.0 525.0,198.8 522.7,198.8 520.3,198.4 517.9,197.6 515.5,197.0 513.2,196.4 510.8,196.1 508.4,195.5 506.1,195.1 503.7,193.1 501.3,192.6 498.9,192.0 496.6,190.9 494.2,189.9 491.8,189.3 489.4,189.2 487.1,188.4 484.7,187.2 482.3,185.5 479.9,184.3 477.6,182.8 475.2,180.5 472.8,179.1 470.4,177.7 468.1,176.1 465.7,174.2 463.3,172.9 460.9,169.4 458.6,167.2 456.2,165.0 453.8,162.8 451.5,159.7 449.1,157.6 446.7,154.2 444.3,151.8 442.0,149.1 439.6,146.2 437.2,143.4 434.8,140.5 432.5,136.7 430.1,133.8 427.7,131.8 425.3,129.7 423.0,125.7 420.6,120.1 418.2,112.7 415.8,106.6 413.5,101.8 411.1,99.7 408.7,98.9 406.3,98.6 404.0,97.3 401.6,97.1 399.2,96.8 396.9,96.9 394.5,99.3 392.1,101.4 389.7,102.3 387.4,102.3 385.0,103.0 382.6,104.4 380.2,105.7 377.9,106.9 375.5,108.0 373.1,109.0 370.7,107.7 368.4,107.6 366.0,106.9 363.6,104.3 361.2,101.1 358.9,100.5 356.5,101.4 354.1,99.7 351.7,99.7 349.4,98.1 347.0,96.6 344.6,95.5 342.3,93.4 339.9,92.1 337.5,91.8 335.1,90.3 332.8,89.4 330.4,89.1 328.0,89.0 325.6,87.5 323.3,87.2 320.9,87.3 318.5,86.1 316.1,85.4 313.8,84.6 311.4,83.2 309.0,81.7 306.6,80.4 304.3,79.3 301.9,77.0 299.5,77.0 297.1,75.5 294.8,75.0 292.4,74.5 290.0,74.2 287.7,73.8 285.3,73.6 282.9,74.0 280.5,73.3 278.2,73.2 275.8,73.3 273.4,73.1 271.0,72.9 268.7,73.0 266.3,72.6 263.9,72.2 261.5,72.1 259.2,71.9 256.8,71.8 254.4,72.4 252.0,72.5 249.7,72.3 247.3,72.0 244.9,71.9 242.5,71.7 240.2,72.0 237.8,71.9 235.4,73.0 233.1,72.5 230.7,72.2 228.3,73.0 225.9,74.1 223.6,74.1 221.2,76.1 218.8,78.3 216.4,81.1 214.1,82.8 211.7,87.5 209.3,91.5 206.9,96.2 204.6,97.2 202.2,100.7 199.8,102.1 197.4,100.5 195.1,103.0 192.7,100.5 190.3,97.0 187.9,97.9 185.6,96.7 183.2,94.7 180.8,92.8 178.5,91.2 176.1,88.6 173.7,87.6 171.3,87.7 169.0,87.2 166.6,85.1 164.2,84.6 161.8,84.1 159.5,83.3 157.1,83.6 154.7,82.6 152.3,83.0 150.0,82.3 147.6,83.3 145.2,82.2 142.8,82.3 140.5,81.2 138.1,79.3 135.7,78.0 133.3,75.6 131.0,74.3 128.6,74.8 126.2,74.8 123.9,73.8 121.5,73.7 119.1,72.5 116.7,71.0 114.4,69.3 112.0,67.7 109.6,68.4 107.2,68.5 104.9,66.4 102.5,63.9 100.1,60.4 97.7,57.7 95.4,55.4 93.0,52.3 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,88.4 74.0,191.2" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>第 5 批：program 8160 到 8191，共 32 个</title><polygon points="74.0,191.2 76.4,88.4 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.3 95.4,55.4 97.7,57.7 100.1,60.4 102.5,63.9 104.9,66.4 107.2,68.5 109.6,68.4 112.0,67.7 114.4,69.3 116.7,71.0 119.1,72.5 121.5,73.7 123.9,73.8 126.2,74.8 128.6,74.8 131.0,74.3 133.3,75.6 135.7,78.0 138.1,79.3 140.5,81.2 142.8,82.3 145.2,82.2 147.6,83.3 150.0,82.3 152.3,83.0 154.7,82.6 157.1,83.6 159.5,83.3 161.8,84.1 164.2,84.6 166.6,85.1 169.0,87.2 171.3,87.7 173.7,87.6 176.1,88.6 178.5,91.2 180.8,92.8 183.2,94.7 185.6,96.7 187.9,97.9 190.3,97.0 192.7,100.5 195.1,103.0 197.4,100.5 199.8,102.1 202.2,100.7 204.6,97.2 206.9,96.2 209.3,91.5 211.7,87.5 214.1,82.8 216.4,81.1 218.8,78.3 221.2,76.1 223.6,74.1 225.9,74.1 228.3,73.0 230.7,72.2 233.1,72.5 235.4,73.0 237.8,71.9 240.2,72.0 242.5,71.7 244.9,71.9 247.3,72.0 249.7,72.3 252.0,72.5 254.4,72.4 256.8,71.8 259.2,71.9 261.5,72.1 263.9,72.2 266.3,72.6 268.7,73.0 271.0,72.9 273.4,73.1 275.8,73.3 278.2,73.2 280.5,73.3 282.9,74.0 285.3,73.6 287.7,73.8 290.0,74.2 292.4,74.5 294.8,75.0 297.1,75.5 299.5,77.0 301.9,77.0 304.3,79.3 306.6,80.4 309.0,81.7 311.4,83.2 313.8,84.6 316.1,85.4 318.5,86.1 320.9,87.3 323.3,87.2 325.6,87.5 328.0,89.0 330.4,89.1 332.8,89.4 335.1,90.3 337.5,91.8 339.9,92.1 342.3,93.4 344.6,95.5 347.0,96.6 349.4,98.1 351.7,99.7 354.1,99.7 356.5,101.4 358.9,100.5 361.2,101.1 363.6,104.3 366.0,106.9 368.4,107.6 370.7,107.7 373.1,109.0 375.5,108.0 377.9,106.9 380.2,105.7 382.6,104.4 385.0,103.0 387.4,102.3 389.7,102.3 392.1,101.4 394.5,99.3 396.9,96.9 399.2,96.8 401.6,97.1 404.0,97.3 406.3,98.6 408.7,98.9 411.1,99.1 413.5,98.9 415.8,99.8 418.2,98.3 420.6,97.3 423.0,94.7 425.3,91.1 427.7,87.8 430.1,87.3 432.5,86.6 434.8,86.8 437.2,87.7 439.6,88.0 442.0,87.5 444.3,86.5 446.7,83.1 449.1,83.2 451.5,82.7 453.8,83.6 456.2,84.3 458.6,84.5 460.9,83.6 463.3,84.6 465.7,84.6 468.1,85.2 470.4,86.0 472.8,86.4 475.2,88.1 477.6,89.1 479.9,89.7 482.3,90.2 484.7,91.5 487.1,91.2 489.4,92.1 491.8,91.6 494.2,93.1 496.6,95.0 498.9,97.3 501.3,96.8 503.7,96.5 506.1,97.5 508.4,96.5 510.8,97.6 513.2,96.8 515.5,98.1 517.9,97.6 520.3,100.9 522.7,105.9 525.0,109.5 527.4,113.4 529.8,119.3 532.2,123.4 534.5,126.9 536.9,132.8 539.3,136.5 541.7,139.3 544.0,142.3 546.4,145.3 548.8,149.1 551.2,152.2 553.5,154.4 555.9,156.9 558.3,159.7 560.7,163.0 563.0,166.1 565.4,169.0 567.8,171.9 570.1,174.4 572.5,176.6 574.9,177.9 577.3,181.1 579.6,183.3 582.0,186.4 584.4,188.1 586.8,190.1 589.1,191.1 591.5,193.0 593.9,193.9 596.3,195.1 598.6,195.9 601.0,197.6 603.4,197.8 605.8,198.2 608.1,198.3 610.5,198.8 612.9,199.0 615.3,199.4 617.6,199.6 620.0,200.0 620.0,200.0 617.6,199.6 615.3,199.5 612.9,199.1 610.5,199.0 608.1,198.5 605.8,198.4 603.4,198.1 601.0,197.9 598.6,196.3 596.3,195.4 593.9,194.3 591.5,193.5 589.1,191.9 586.8,191.0 584.4,189.3 582.0,187.6 579.6,184.6 577.3,182.5 574.9,179.5 572.5,178.2 570.1,176.1 567.8,173.8 565.4,171.1 563.0,168.2 560.7,165.1 558.3,161.8 555.9,159.2 553.5,156.7 551.2,154.5 548.8,151.5 546.4,147.6 544.0,144.6 541.7,141.6 539.3,138.8 536.9,135.1 534.5,129.2 532.2,125.7 529.8,121.6 527.4,115.7 525.0,111.8 522.7,108.2 520.3,103.2 517.9,97.6 515.5,98.1 513.2,96.8 510.8,97.6 508.4,96.5 506.1,97.5 503.7,96.5 501.3,96.8 498.9,97.3 496.6,95.0 494.2,93.1 491.8,91.6 489.4,92.1 487.1,91.2 484.7,91.5 482.3,90.2 479.9,89.7 477.6,89.1 475.2,88.1 472.8,86.4 470.4,86.0 468.1,85.2 465.7,84.6 463.3,84.6 460.9,83.6 458.6,84.5 456.2,84.3 453.8,83.6 451.5,82.7 449.1,83.2 446.7,83.1 444.3,86.5 442.0,87.5 439.6,88.0 437.2,87.7 434.8,86.8 432.5,86.6 430.1,87.3 427.7,87.8 425.3,91.1 423.0,94.7 420.6,97.3 418.2,98.3 415.8,99.8 413.5,98.9 411.1,99.1 408.7,98.9 406.3,98.6 404.0,97.3 401.6,97.1 399.2,96.8 396.9,96.9 394.5,99.3 392.1,101.4 389.7,102.3 387.4,102.3 385.0,103.0 382.6,104.4 380.2,105.7 377.9,106.9 375.5,108.0 373.1,109.0 370.7,107.7 368.4,107.6 366.0,106.9 363.6,104.3 361.2,101.1 358.9,100.5 356.5,101.4 354.1,99.7 351.7,99.7 349.4,98.1 347.0,96.6 344.6,95.5 342.3,93.4 339.9,92.1 337.5,91.8 335.1,90.3 332.8,89.4 330.4,89.1 328.0,89.0 325.6,87.5 323.3,87.2 320.9,87.3 318.5,86.1 316.1,85.4 313.8,84.6 311.4,83.2 309.0,81.7 306.6,80.4 304.3,79.3 301.9,77.0 299.5,77.0 297.1,75.5 294.8,75.0 292.4,74.5 290.0,74.2 287.7,73.8 285.3,73.6 282.9,74.0 280.5,73.3 278.2,73.2 275.8,73.3 273.4,73.1 271.0,72.9 268.7,73.0 266.3,72.6 263.9,72.2 261.5,72.1 259.2,71.9 256.8,71.8 254.4,72.4 252.0,72.5 249.7,72.3 247.3,72.0 244.9,71.9 242.5,71.7 240.2,72.0 237.8,71.9 235.4,73.0 233.1,72.5 230.7,72.2 228.3,73.0 225.9,74.1 223.6,74.1 221.2,76.1 218.8,78.3 216.4,81.1 214.1,82.8 211.7,87.5 209.3,91.5 206.9,96.2 204.6,97.2 202.2,100.7 199.8,102.1 197.4,100.5 195.1,103.0 192.7,100.5 190.3,97.0 187.9,97.9 185.6,96.7 183.2,94.7 180.8,92.8 178.5,91.2 176.1,88.6 173.7,87.6 171.3,87.7 169.0,87.2 166.6,85.1 164.2,84.6 161.8,84.1 159.5,83.3 157.1,83.6 154.7,82.6 152.3,83.0 150.0,82.3 147.6,83.3 145.2,82.2 142.8,82.3 140.5,81.2 138.1,79.3 135.7,78.0 133.3,75.6 131.0,74.3 128.6,74.8 126.2,74.8 123.9,73.8 121.5,73.7 119.1,72.5 116.7,71.0 114.4,69.3 112.0,67.7 109.6,68.4 107.2,68.5 104.9,66.4 102.5,63.9 100.1,60.4 97.7,57.7 95.4,55.4 93.0,52.3 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,88.4 74.0,191.2" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <text class="lab2" x="74" y="234">SM 62 的 12 个名额，一条是一个 program 从第一条指令到最后一条 store 发出；这个 SM 一共跑了 47 个</text>
  <text class="tick" x="66" y="254" text-anchor="end">0</text>
  <g class="m"><title>program 142（第 1 批）：0.00 到 2.62 µs</title><rect x="74.0" y="247" width="31.0" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2255（第 2 批）：3.07 到 10.11 µs</title><rect x="111.0" y="247" width="84.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4245（第 3 批）：11.68 到 27.55 µs</title><rect x="214.5" y="247" width="190.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 6565（第 4 批）：29.02 到 36.83 µs</title><rect x="423.2" y="247" width="93.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 312（第 1 批）：0.00 到 4.29 µs</title><rect x="74.0" y="256" width="51.0" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2786（第 2 批）：5.63 到 19.33 µs</title><rect x="141.8" y="256" width="164.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 5557（第 3 批）：24.99 到 30.78 µs</title><rect x="374.7" y="256" width="69.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7455（第 4 批）：32.77 到 41.70 µs</title><rect x="468.3" y="256" width="106.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 822（第 1 批）：0.03 到 3.74 µs</title><rect x="74.4" y="265" width="44.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2661（第 2 批）：5.28 到 13.41 µs</title><rect x="137.5" y="265" width="97.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4890（第 3 批）：19.36 到 26.75 µs</title><rect x="307.0" y="265" width="88.3" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 6407（第 4 批）：28.77 到 35.23 µs</title><rect x="420.2" y="265" width="77.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <text class="tick" x="66" y="281" text-anchor="end">3</text>
  <g class="m"><title>program 482（第 1 批）：0.03 到 3.81 µs</title><rect x="74.4" y="274" width="44.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2668（第 2 批）：5.28 到 19.65 µs</title><rect x="137.5" y="274" width="172.3" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 5545（第 3 批）：24.99 到 32.06 µs</title><rect x="374.7" y="274" width="84.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 8119（第 4 批）：36.74 到 43.65 µs</title><rect x="516.0" y="274" width="82.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 652（第 1 批）：0.03 到 5.34 µs</title><rect x="74.4" y="283" width="63.3" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3052（第 2 批）：8.29 到 24.29 µs</title><rect x="173.7" y="283" width="191.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 5980（第 3 批）：26.85 到 36.93 µs</title><rect x="397.1" y="283" width="120.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 992（第 1 批）：0.16 到 4.13 µs</title><rect x="75.9" y="292" width="47.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2689（第 2 批）：5.31 到 24.83 µs</title><rect x="137.9" y="292" width="234.3" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 6066（第 3 批）：27.74 到 33.31 µs</title><rect x="407.8" y="292" width="66.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 8161（第 5 批）：36.93 到 40.83 µs</title><rect x="518.3" y="292" width="46.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <text class="tick" x="66" y="308" text-anchor="end">6</text>
  <g class="m"><title>program 1162（第 1 批）：0.19 到 5.34 µs</title><rect x="76.3" y="301" width="61.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3507（第 2 批）：10.30 到 20.32 µs</title><rect x="198.0" y="301" width="119.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 5977（第 3 批）：26.82 到 30.78 µs</title><rect x="396.7" y="301" width="47.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7665（第 4 批）：34.08 到 38.37 µs</title><rect x="484.1" y="301" width="51.0" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 1332（第 1 批）：0.19 到 8.00 µs</title><rect x="76.3" y="310" width="93.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3616（第 2 批）：10.59 到 25.38 µs</title><rect x="201.5" y="310" width="177.3" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 6078（第 3 批）：27.81 到 31.68 µs</title><rect x="408.6" y="310" width="46.0" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 8024（第 4 批）：36.22 到 43.23 µs</title><rect x="509.9" y="310" width="83.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 1502（第 1 批）：0.19 到 9.41 µs</title><rect x="76.3" y="319" width="110.3" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3921（第 2 批）：11.14 到 26.59 µs</title><rect x="208.0" y="319" width="185.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 6208（第 4 批）：28.38 到 33.41 µs</title><rect x="415.5" y="319" width="59.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 8160（第 5 批）：36.93 到 42.50 µs</title><rect x="518.3" y="319" width="66.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <text class="tick" x="66" y="335" text-anchor="end">9</text>
  <g class="m"><title>program 1672（第 1 批）：0.22 到 8.19 µs</title><rect x="76.7" y="328" width="95.3" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3760（第 2 批）：10.88 到 18.11 µs</title><rect x="204.9" y="328" width="86.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 5275（第 3 批）：23.65 到 29.86 µs</title><rect x="358.6" y="328" width="74.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7397（第 4 批）：32.35 到 38.56 µs</title><rect x="463.3" y="328" width="74.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 1842（第 1 批）：0.22 到 10.08 µs</title><rect x="76.7" y="337" width="118.0" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 4098（第 3 批）：11.42 到 19.14 µs</title><rect x="211.5" y="337" width="92.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5548（第 3 批）：24.99 到 28.58 µs</title><rect x="374.7" y="337" width="42.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7394（第 4 批）：32.35 到 36.96 µs</title><rect x="463.3" y="337" width="54.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 2012（第 1 批）：0.22 到 10.94 µs</title><rect x="76.7" y="346" width="128.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 4373（第 3 批）：12.03 到 23.94 µs</title><rect x="218.8" y="346" width="142.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5982（第 3 批）：26.85 到 30.66 µs</title><rect x="397.1" y="346" width="45.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7428（第 4 批）：32.58 到 38.56 µs</title><rect x="466.0" y="346" width="71.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <text class="tick" x="74.0" y="369" text-anchor="middle">0</text>
  <line class="grid" x1="74.0" y1="52" x2="74.0" y2="200"/>
  <text class="tick" x="134.2" y="369" text-anchor="middle">5</text>
  <line class="grid" x1="134.2" y1="52" x2="134.2" y2="200"/>
  <text class="tick" x="194.3" y="369" text-anchor="middle">10</text>
  <line class="grid" x1="194.3" y1="52" x2="194.3" y2="200"/>
  <text class="tick" x="254.5" y="369" text-anchor="middle">15</text>
  <line class="grid" x1="254.5" y1="52" x2="254.5" y2="200"/>
  <text class="tick" x="314.7" y="369" text-anchor="middle">20</text>
  <line class="grid" x1="314.7" y1="52" x2="314.7" y2="200"/>
  <text class="tick" x="374.8" y="369" text-anchor="middle">25</text>
  <line class="grid" x1="374.8" y1="52" x2="374.8" y2="200"/>
  <text class="tick" x="435.0" y="369" text-anchor="middle">30</text>
  <line class="grid" x1="435.0" y1="52" x2="435.0" y2="200"/>
  <text class="tick" x="495.1" y="369" text-anchor="middle">35</text>
  <line class="grid" x1="495.1" y1="52" x2="495.1" y2="200"/>
  <text class="tick" x="555.3" y="369" text-anchor="middle">40</text>
  <line class="grid" x1="555.3" y1="52" x2="555.3" y2="200"/>
  <text class="tick" x="615.5" y="369" text-anchor="middle">45</text>
  <line class="grid" x1="615.5" y1="52" x2="615.5" y2="200"/>
  <text class="lab2" x="66" y="369" text-anchor="end">µs</text>
  <text class="lab2" x="30" y="304" text-anchor="middle" transform="rotate(-90 30 300)">名额</text>
  <text class="lab2" x="74" y="394">批次</text>
  <rect x="110" y="385" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.14"/><text class="lab" x="127" y="394">0–2039</text>
  <rect x="218" y="385" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.32"/><text class="lab" x="235" y="394">2040–4079</text>
  <rect x="326" y="385" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.5"/><text class="lab" x="343" y="394">4080–6119</text>
  <rect x="434" y="385" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.68"/><text class="lab" x="451" y="394">6120–8159</text>
  <rect x="542" y="385" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.9"/><text class="lab" x="559" y="394">8160–8191</text>
</svg>
<figcaption><strong>图 2-2</strong> GELU 的 8192 个 program 实际怎么分批跑（RTX 5090，n = 8388608，fp32，每个 program 4 个 warp）。每个 program 开头读一次 <code>%globaltimer</code> 和 <code>%smid</code>，store 发出后再读一次时间，记录前先清空 L2。颜色由淡到浓是 program 编号所在的批次，一批 2040 个。悬停可看 program 编号和时间。</figcaption>
</figure>

实测的时间线（[图 2-2](#fig-t-5)）上，批次之间没有分界。一个 program 跑完，它的名额马上换下一个 program 进来，所以第 2 批在 1.8 µs 就开始了，第 1 批到 11.6 µs 才全部跑完。同一批里的 program 都在抢显存带宽，耗时从 3.5 µs 到 12.8 µs 不等（10% 和 90% 分位）。37 µs 时最后一个 program 开始，之后没有新的 program 补进来，名额一个个空出来，在跑的 program 从 1400 个左右降到 0，这段约 8 µs，占整个 kernel 的 18%，叫 tail effect。program 多、每个都很短时，这段尾巴就短。

Triton 会把 kernel 编译成 PTX，PTX 描述的是单个线程的指令。下面是 GELU 的 PTX 节选：

```
mov.u32        %r17, %ctaid.x;                // program 编号，即 blockIdx.x
mov.u32        %r20, %tid.x;                  // 线程编号
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];     // 读显存，%p1 是 mask 算出的谓词
...                                           // 一共 8 条 ld.global
ex2.approx.f32 %r72, %r71;                    // tl.exp 编成以 2 为底的指数
@%p1 st.global.b32 [ %rd9 + 0 ], { %r9 };     // 写回显存
```

一个 program 处理 1024 个元素，默认 4 个 warp 共 128 个线程，所以每个线程分到 8 个元素，PTX 里正好是 8 条 `ld.global` 和 8 条 `st.global`。

GELU 的每个元素互不依赖，怎么切都可以。softmax 的每个元素都要用到整行的最大值和总和，切分就得按行来。

### 2.2 Softmax：按行归约，一行一个 program {#triton-softmax}

softmax 对 `[M, N]` 的输入按行算，先减去每行的最大值 $m_i$ 防止 exp 溢出：

<p align="center">$m_i = \max_k x_{ik}, \qquad y_{ij} = \dfrac{e^{x_{ij} - m_i}}{\sum_{k} e^{x_{ik} - m_i}}$</p>

$y_{ij}$ 要用到第 $i$ 行全部 N 个数。

softmax 在 eager 下要 5 个 kernel，和[下一篇](/blog/gpu-training-analysis/#attention)里 attention 的 softmax 一样。对 `[M, N]` 的输入，一共读 5MN + 2M 个数，写 3MN + 2M 个：

```python
def softmax_naive(x):                         # x: [M, N]，按行
    m = x.max(dim=1, keepdim=True).values     # 读 MN，写 M
    z = x - m                                 # 读 MN + M，写 MN
    e = torch.exp(z)                          # 读 MN，写 MN
    s = e.sum(dim=1, keepdim=True)            # 读 MN，写 M
    return e / s                              # 读 MN + M，写 MN
```

<figure id="fig-t-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 356" width="100%" role="img" aria-label="softmax kernel：一行一个 program；tl.load 把一整行从显存直接读进 register，计算单元减最大值、exp、求和、归一化，中间结果都在 register 里，跨 warp 的归约经 shared memory，最后 tl.store 写回；每个 SM 同时跑 9 个 program">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-t-2-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-t-2-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-1)"/></marker><marker id="fig-t-2-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-2)"/></marker></defs>
  <text class="ttl" x="10" y="16">grid</text>
  <text class="ttl" x="162" y="16">一个 program（pid = 1）</text>
  <text class="tb" x="630" y="16" text-anchor="end" style="fill: var(--fig-hi)">按行归约：y[i, j] 依赖整行 x[i, :]</text>
  <rect x="10" y="28" width="118" height="26" rx="5" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="t" x="69" y="46" text-anchor="middle">行 0</text>
  <rect x="10" y="60" width="118" height="26" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 30%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="t" x="69" y="78" text-anchor="middle">行 1 ← pid=1</text>
  <rect x="10" y="92" width="118" height="26" rx="5" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="t" x="69" y="110" text-anchor="middle">行 2</text>
  <rect x="10" y="124" width="118" height="26" rx="5" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="t" x="69" y="142" text-anchor="middle">⋯</text>
  <text class="lab2" x="10" y="168">grid = (M,)</text>
  <text class="lab2" x="10" y="183">M = 4096 个 program</text>
  <text class="ttl" x="10" y="204">一个 SM</text>
  <rect x="12" y="212" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <line x1="17.0" y1="216" x2="17.0" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="21.7" y1="216" x2="21.7" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="26.4" y1="216" x2="26.4" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="31.1" y1="216" x2="31.1" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="40" y="212" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="45.0" y1="216" x2="45.0" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="49.7" y1="216" x2="49.7" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="54.4" y1="216" x2="54.4" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="59.1" y1="216" x2="59.1" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="68" y="212" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="73.0" y1="216" x2="73.0" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="77.7" y1="216" x2="77.7" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="82.4" y1="216" x2="82.4" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="87.1" y1="216" x2="87.1" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="96" y="212" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="101.0" y1="216" x2="101.0" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="105.7" y1="216" x2="105.7" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="110.4" y1="216" x2="110.4" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="115.1" y1="216" x2="115.1" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="12" y="234" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="17.0" y1="238" x2="17.0" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="21.7" y1="238" x2="21.7" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="26.4" y1="238" x2="26.4" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="31.1" y1="238" x2="31.1" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="40" y="234" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="45.0" y1="238" x2="45.0" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="49.7" y1="238" x2="49.7" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="54.4" y1="238" x2="54.4" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="59.1" y1="238" x2="59.1" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="68" y="234" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="73.0" y1="238" x2="73.0" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="77.7" y1="238" x2="77.7" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="82.4" y1="238" x2="82.4" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="87.1" y1="238" x2="87.1" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="96" y="234" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="101.0" y1="238" x2="101.0" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="105.7" y1="238" x2="105.7" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="110.4" y1="238" x2="110.4" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="115.1" y1="238" x2="115.1" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="12" y="256" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="17.0" y1="260" x2="17.0" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="21.7" y1="260" x2="21.7" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="26.4" y1="260" x2="26.4" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="31.1" y1="260" x2="31.1" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="40" y="256" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="68" y="256" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="96" y="256" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="10" y="286">每格一个 program（4 warp）</text><text class="s" x="10" y="300">9 × 128 = 1152 thread</text>
  <text class="s" x="10" y="314">register 先满</text>
  <rect class="band" x="146" y="26" width="490" height="66" rx="8"/>
  <text class="lab2" x="152" y="56.5">计算</text>
  <text class="lab2" x="152" y="69.5">单元</text>
  <rect class="band" x="146" y="100" width="490" height="76" rx="8"/>
  <text class="lab2" x="152" y="142.0">register</text>
  <rect class="band" x="146" y="184" width="490" height="32" rx="8"/>
  <text class="lab2" x="152" y="197.5">shared</text>
  <text class="lab2" x="152" y="210.5">memory</text>
  <rect class="band" x="146" y="224" width="490" height="84" rx="8"/>
  <text class="lab2" x="152" y="270.0">显存</text>
  <text class="s" x="152" y="284.0">GDDR7</text>
  <line x1="130.0" y1="73.0" x2="144.0" y2="73.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-t-2-m0)"/>
  <rect class="op" x="284.0" y="46" width="84.0" height="38" rx="6"/>
  <text class="tb" x="326.0" y="62" text-anchor="middle">sub max</text><text class="s" x="326.0" y="77" text-anchor="middle">tl.max</text>
  <rect class="op" x="378.0" y="46" width="84.0" height="38" rx="6"/>
  <text class="tb" x="420.0" y="62" text-anchor="middle">exp</text><text class="s" x="420.0" y="77" text-anchor="middle">tl.exp</text>
  <rect class="op" x="472.0" y="46" width="84.0" height="38" rx="6"/>
  <text class="tb" x="514.0" y="62" text-anchor="middle">div sum</text><text class="s" x="514.0" y="77" text-anchor="middle">tl.sum</text>
  <text class="lab2" x="152" y="324">中间结果都在 register 里，每个 thread 存这一行的 32 个数；tl.max、tl.sum 先在每个 warp 里归约，</text>
  <text class="lab2" x="152" y="339">4 个 warp 的部分结果再经 shared memory 交换（16 B）。显存只读 x 一次、写 y 一次。</text>
  <rect x="232.0" y="110" width="44" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="254.0" y="123" text-anchor="middle">x 行</text>
  <line x1="254.0" y1="250.0" x2="254.0" y2="130.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-2-m1)"/>
  <text class="s" x="258.0" y="194.0" text-anchor="start" style="fill: var(--fig-1)">tl.load</text>
  <line x1="268.0" y1="108.0" x2="310.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <rect x="349.0" y="110" width="48" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="373.0" y="123" text-anchor="middle">x − m</text>
  <line x1="342.0" y1="86.0" x2="367.0" y2="108.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <line x1="379.0" y1="108.0" x2="404.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <rect x="443.0" y="110" width="48" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="467.0" y="123" text-anchor="middle">e</text>
  <line x1="436.0" y1="86.0" x2="461.0" y2="108.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <line x1="473.0" y1="108.0" x2="498.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <rect x="564.0" y="110" width="44" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="586.0" y="123" text-anchor="middle">y 行</text>
  <line x1="530.0" y1="86.0" x2="572.0" y2="108.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <line x1="586.0" y1="130.0" x2="586.0" y2="250.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-2-m1)"/>
  <text class="s" x="590.0" y="194.0" text-anchor="start" style="fill: var(--fig-1)">tl.store</text>
  <rect x="218.0" y="252" width="72.0" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <rect x="218.0" y="260.4" width="72.0" height="7.0" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="254.0" y="293" text-anchor="middle">x，64 MiB</text>
  <rect x="550.0" y="252" width="72.0" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <rect x="550.0" y="260.4" width="72.0" height="7.0" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="586.0" y="293" text-anchor="middle">y，64 MiB</text>
  <rect x="292.0" y="191" width="68" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 40%, transparent); stroke: var(--fig-3)" stroke-width="1.4"/><text class="s" x="326.0" y="204" text-anchor="middle">warp max</text>
  <line x1="321.0" y1="86.0" x2="321.0" y2="189.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-2-m2)"/>
  <line x1="331.0" y1="189.0" x2="331.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-2-m2)"/>
  <rect x="480.0" y="191" width="68" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 40%, transparent); stroke: var(--fig-3)" stroke-width="1.4"/><text class="s" x="514.0" y="204" text-anchor="middle">warp sum</text>
  <line x1="509.0" y1="86.0" x2="509.0" y2="189.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-2-m2)"/>
  <line x1="519.0" y1="189.0" x2="519.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-2-m2)"/>
</svg>
<figcaption><strong>图 2-3</strong> softmax kernel（4096 × 4096，fp32），画法同<a href="#fig-t-1">图 2-1</a>。一行一个 program，中间结果都留在 register 里，显存只读写各一次；每个 program 用的 register 多，一个 SM 只放得下 9 个。</figcaption>
</figure>

如果一个 program 处理一整行（[图 2-3](#fig-t-2)），max、exp、求和、除都可以在 register 里做完，只读 MN、写 MN，读写量是原来的 1/4：

```python {role="program" tag="处理第 row 行"}
@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)                    # 一个 program 处理一整行
    cols = tl.arange(0, BLOCK)                # BLOCK ≥ N，一次装下整行
    x = tl.load(x_ptr + row * stride + cols, mask=cols < N, other=float("-inf"))
    x = x - tl.max(x, axis=0)                 # 以下全在 register 里
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * stride + cols, y, mask=cols < N)
```

启动时 grid 是 M 个 program，一行一个：

```python {role="grid" tag="一行一个 program"}
def softmax(x):
    M, N = x.shape
    y = torch.empty_like(x)
    softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
```

在 5090 上对 4096 × 4096 的 fp32 输入实测，eager 版 249 µs，Triton 版 80 µs，快了 3.1 倍，和 `torch.softmax`（79 µs）相当。

[图 2-3](#fig-t-2) 里的 warp max、warp sum 是 shared memory 在这个 kernel 里唯一的用处：`tl.max`、`tl.sum` 先在每个 warp 里用 shuffle 交换 register 归约，4 个 warp 的部分结果再经 shared memory 交换，一共 16 B。什么时候值得把数据整块放进 shared memory，[2.4 节](#triton-matmul)再讲。

这个写法要求 BLOCK ≥ N，也就是一行能一次装进一个 program。行再长，program 就只能沿行一块一块地读，读过的块要留下一个中间结果。下一节先看最简单的按行归约：求和。

### 2.3 Row sum：按行归约，分块累加 {#triton-rowsum}

row sum 对 `[M, N]` 的输入每行求和：

<p align="center">$s_i = \sum_{j=1}^{N} x_{ij} = \sum_{T} \ \sum_{j \in \mathcal{J}_T} x_{ij}$</p>

$\mathcal{J}_T$ 是第 $T$ 块的 TILE 个列下标，代码里 TILE 取 1024。加法满足结合律，先把每块加起来再相加，和一次加完是同一个数（浮点数下只差舍入误差），所以 program 可以沿行一块一块地读，每读一块就加到累加器上（[图 2-4](#fig-t-3)）。

<figure id="fig-t-3" class="fg-fig">
<svg class="fg" viewBox="0 0 640 380" width="100%" role="img" aria-label="row sum kernel：一行一个 program；tl.load 一轮把一块从显存直接读进 register，计算单元把它加到 acc 上，颜色深浅是 for 循环的顺序；acc 一直留在 register 里，最后归约成一个数，tl.store 写回">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-t-3-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-t-3-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-1)"/></marker><marker id="fig-t-3-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-2)"/></marker><marker id="fig-t-3-m3" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker></defs>
  <text class="ttl" x="10" y="16">grid</text>
  <text class="ttl" x="162" y="16">一个 program（pid = 1）</text>
  <text class="tb" x="630" y="16" text-anchor="end" style="fill: var(--fig-hi)">按行归约，分块累加</text>
  <rect x="10" y="28" width="118" height="26" rx="5" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="t" x="69" y="46" text-anchor="middle">行 0</text>
  <rect x="10" y="60" width="118" height="26" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 30%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="t" x="69" y="78" text-anchor="middle">行 1 ← pid=1</text>
  <rect x="10" y="92" width="118" height="26" rx="5" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="t" x="69" y="110" text-anchor="middle">行 2</text>
  <rect x="10" y="124" width="118" height="26" rx="5" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="t" x="69" y="142" text-anchor="middle">⋯</text>
  <text class="lab2" x="10" y="168">grid = (M,)</text>
  <text class="lab2" x="10" y="183">M = 4096 个 program</text>
  <text class="ttl" x="10" y="204">一个 SM</text>
  <rect x="12" y="212" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <line x1="17.0" y1="216" x2="17.0" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="21.7" y1="216" x2="21.7" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="26.4" y1="216" x2="26.4" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="31.1" y1="216" x2="31.1" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="40" y="212" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="45.0" y1="216" x2="45.0" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="49.7" y1="216" x2="49.7" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="54.4" y1="216" x2="54.4" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="59.1" y1="216" x2="59.1" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="68" y="212" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="73.0" y1="216" x2="73.0" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="77.7" y1="216" x2="77.7" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="82.4" y1="216" x2="82.4" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="87.1" y1="216" x2="87.1" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="96" y="212" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="101.0" y1="216" x2="101.0" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="105.7" y1="216" x2="105.7" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="110.4" y1="216" x2="110.4" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="115.1" y1="216" x2="115.1" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="12" y="234" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="17.0" y1="238" x2="17.0" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="21.7" y1="238" x2="21.7" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="26.4" y1="238" x2="26.4" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="31.1" y1="238" x2="31.1" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="40" y="234" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="45.0" y1="238" x2="45.0" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="49.7" y1="238" x2="49.7" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="54.4" y1="238" x2="54.4" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="59.1" y1="238" x2="59.1" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="68" y="234" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="73.0" y1="238" x2="73.0" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="77.7" y1="238" x2="77.7" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="82.4" y1="238" x2="82.4" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="87.1" y1="238" x2="87.1" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="96" y="234" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="101.0" y1="238" x2="101.0" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="105.7" y1="238" x2="105.7" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="110.4" y1="238" x2="110.4" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="115.1" y1="238" x2="115.1" y2="248" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="12" y="256" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="17.0" y1="260" x2="17.0" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="21.7" y1="260" x2="21.7" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="26.4" y1="260" x2="26.4" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="31.1" y1="260" x2="31.1" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="40" y="256" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="45.0" y1="260" x2="45.0" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="49.7" y1="260" x2="49.7" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="54.4" y1="260" x2="54.4" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="59.1" y1="260" x2="59.1" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="68" y="256" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="73.0" y1="260" x2="73.0" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="77.7" y1="260" x2="77.7" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="82.4" y1="260" x2="82.4" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="87.1" y1="260" x2="87.1" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="96" y="256" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="101.0" y1="260" x2="101.0" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="105.7" y1="260" x2="105.7" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="110.4" y1="260" x2="110.4" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="115.1" y1="260" x2="115.1" y2="270" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <text class="s" x="10" y="286">每格一个 program（4 warp）</text><text class="s" x="10" y="300">12 × 128 = 1536 thread</text>
  <text class="s" x="10" y="314">线程数先满</text>
  <rect class="band" x="146" y="26" width="490" height="66" rx="8"/>
  <text class="lab2" x="152" y="56.5">计算</text>
  <text class="lab2" x="152" y="69.5">单元</text>
  <rect class="band" x="146" y="100" width="490" height="76" rx="8"/>
  <text class="lab2" x="152" y="142.0">register</text>
  <rect class="band" x="146" y="184" width="490" height="32" rx="8"/>
  <text class="lab2" x="152" y="197.5">shared</text>
  <text class="lab2" x="152" y="210.5">memory</text>
  <rect class="band" x="146" y="224" width="490" height="108" rx="8"/>
  <text class="lab2" x="152" y="282.0">显存</text>
  <text class="s" x="152" y="296.0">GDDR7</text>
  <line x1="130.0" y1="73.0" x2="144.0" y2="73.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-t-3-m0)"/>
  <rect class="op" x="262.0" y="46" width="71.3" height="38" rx="6"/>
  <text class="tb" x="297.7" y="62" text-anchor="middle">add</text><text class="s" x="297.7" y="77" text-anchor="middle">acc += x</text>
  <text class="tb" x="354.3" y="69.0" text-anchor="middle">⋯</text>
  <rect class="op" x="425.3" y="46" width="71.3" height="38" rx="6"/>
  <text class="tb" x="461.0" y="62" text-anchor="middle">add</text><text class="s" x="461.0" y="77" text-anchor="middle">acc += x</text>
  <rect class="op" x="506.7" y="46" width="71.3" height="38" rx="6"/>
  <text class="tb" x="542.3" y="62" text-anchor="middle">reduce</text><text class="s" x="542.3" y="77" text-anchor="middle">tl.sum</text>
  <text class="lab2" x="152" y="348">图里 TILE = 4（代码里 1024），一行 16 个数分 4 块，颜色由淡到浓是 for 循环读的顺序；</text>
  <text class="lab2" x="152" y="363">acc 逐位累加，一直在 register 里；最后的 tl.sum 跨 4 个 warp，经 shared memory 交换（16 B）。</text>
  <defs><linearGradient id="fig-t-3-acc" x1="0" x2="1" y1="0" y2="0"><stop offset="0.00" style="stop-color: var(--fig-1)" stop-opacity="0.12"/><stop offset="0.33" style="stop-color: var(--fig-1)" stop-opacity="0.34"/><stop offset="0.67" style="stop-color: var(--fig-1)" stop-opacity="0.56"/><stop offset="1.00" style="stop-color: var(--fig-1)" stop-opacity="0.78"/></linearGradient></defs>
  <rect x="210.0" y="112" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="210.0" y="112" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="215.5" y="123.5" text-anchor="middle" style="opacity: .9">1</text>
  <rect x="221.0" y="112" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="221.0" y="112" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="226.5" y="123.5" text-anchor="middle" style="opacity: .9">2</text>
  <rect x="232.0" y="112" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="232.0" y="112" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="237.5" y="123.5" text-anchor="middle" style="opacity: .9">0</text>
  <rect x="243.0" y="112" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="243.0" y="112" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="248.5" y="123.5" text-anchor="middle" style="opacity: .9">3</text>
  <text class="s" x="210.0" y="107">块 0</text>
  <line x1="232.0" y1="254.0" x2="232.0" y2="130.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-3-m1)"/>
  <text class="s" x="236.0" y="196.0" text-anchor="start" style="fill: var(--fig-1)">tl.load</text>
  <rect x="373.3" y="112" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="373.3" y="112" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="378.8" y="123.5" text-anchor="middle" style="fill: var(--fig-on-1)">0</text>
  <rect x="384.3" y="112" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="384.3" y="112" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="389.8" y="123.5" text-anchor="middle" style="fill: var(--fig-on-1)">3</text>
  <rect x="395.3" y="112" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="395.3" y="112" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="400.8" y="123.5" text-anchor="middle" style="fill: var(--fig-on-1)">1</text>
  <rect x="406.3" y="112" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="406.3" y="112" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="411.8" y="123.5" text-anchor="middle" style="fill: var(--fig-on-1)">2</text>
  <text class="s" x="373.3" y="107">块 3</text>
  <line x1="395.3" y1="254.0" x2="395.3" y2="130.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-3-m1)"/>
  <text class="s" x="399.3" y="196.0" text-anchor="start" style="fill: var(--fig-1)">tl.load</text>
  <line x1="246.0" y1="108.0" x2="281.7" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <line x1="409.3" y1="108.0" x2="445.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <rect x="305.7" y="144" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="305.7" y="144" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="311.2" y="155.5" text-anchor="middle" style="opacity: .9">1</text>
  <rect x="316.7" y="144" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="316.7" y="144" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="322.2" y="155.5" text-anchor="middle" style="opacity: .9">2</text>
  <rect x="327.7" y="144" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="327.7" y="144" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="333.2" y="155.5" text-anchor="middle" style="opacity: .9">0</text>
  <rect x="338.7" y="144" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="338.7" y="144" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="344.2" y="155.5" text-anchor="middle" style="opacity: .9">3</text>
  <text class="s" x="327.7" y="173" text-anchor="middle">acc</text>
  <rect x="479.7" y="144" width="44" height="16" fill="url(#fig-t-3-acc)"/>
  <rect x="479.7" y="144" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="485.2" y="155.5" text-anchor="middle" style="opacity: .9">7</text>
  <rect x="490.7" y="144" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="496.2" y="155.5" text-anchor="middle" style="opacity: .9">8</text>
  <rect x="501.7" y="144" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="507.2" y="155.5" text-anchor="middle" style="fill: var(--fig-on-1)">4</text>
  <rect x="512.7" y="144" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="518.2" y="155.5" text-anchor="middle" style="fill: var(--fig-on-1)">6</text>
  <text class="s" x="501.7" y="173" text-anchor="middle">acc</text>
  <line x1="305.7" y1="86.0" x2="317.7" y2="140.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <line x1="469.0" y1="86.0" x2="491.7" y2="140.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <line x1="511.7" y1="140.0" x2="526.3" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <path d="M297.7,44 L297.7,37 L461.0,37 L461.0,43" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.5" marker-end="url(#fig-t-3-m3)"/>
  <text class="s" x="379.3" y="33" text-anchor="middle" style="fill: var(--fig-hi)">acc 带到下一轮</text>
  <rect x="596.0" y="112" width="24" height="16" style="fill: var(--fig-1)" fill-opacity=".78"/><text class="s" x="608.0" y="123.5" text-anchor="middle" style="fill: var(--fig-on-1)">25</text>
  <line x1="558.3" y1="86.0" x2="598.0" y2="108.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <rect x="509.3" y="191" width="66" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 40%, transparent); stroke: var(--fig-3)" stroke-width="1.4"/><text class="s" x="542.3" y="204" text-anchor="middle">warp sum</text>
  <line x1="537.3" y1="86.0" x2="537.3" y2="189.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-3-m2)"/>
  <line x1="547.3" y1="189.0" x2="547.3" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-3-m2)"/>
  <rect x="204.8" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="218.4" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="232.0" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="245.6" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="259.2" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="272.8" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="286.4" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="300.1" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="313.7" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="327.3" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="340.9" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="354.5" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="368.1" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="381.7" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="395.3" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="408.9" y="238" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="204.8" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <text class="s" x="211.6" y="265.5" text-anchor="middle" style="opacity: .9">1</text>
  <rect x="204.8" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="218.4" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <text class="s" x="225.2" y="265.5" text-anchor="middle" style="opacity: .9">2</text>
  <rect x="218.4" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="232.0" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <text class="s" x="238.8" y="265.5" text-anchor="middle" style="opacity: .9">0</text>
  <rect x="232.0" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="245.6" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <text class="s" x="252.4" y="265.5" text-anchor="middle" style="opacity: .9">3</text>
  <rect x="245.6" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="259.2" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.34"/>
  <text class="s" x="266.0" y="265.5" text-anchor="middle" style="opacity: .9">4</text>
  <rect x="259.2" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="272.8" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.34"/>
  <text class="s" x="279.6" y="265.5" text-anchor="middle" style="opacity: .9">1</text>
  <rect x="272.8" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="286.4" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.34"/>
  <text class="s" x="293.2" y="265.5" text-anchor="middle" style="opacity: .9">2</text>
  <rect x="286.4" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="300.1" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.34"/>
  <text class="s" x="306.9" y="265.5" text-anchor="middle" style="opacity: .9">0</text>
  <rect x="300.1" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="313.7" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.56"/>
  <text class="s" x="320.5" y="265.5" text-anchor="middle" style="fill: var(--fig-on-1)">2</text>
  <rect x="313.7" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="327.3" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.56"/>
  <text class="s" x="334.1" y="265.5" text-anchor="middle" style="fill: var(--fig-on-1)">2</text>
  <rect x="327.3" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="340.9" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.56"/>
  <text class="s" x="347.7" y="265.5" text-anchor="middle" style="fill: var(--fig-on-1)">1</text>
  <rect x="340.9" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="354.5" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.56"/>
  <text class="s" x="361.3" y="265.5" text-anchor="middle" style="fill: var(--fig-on-1)">1</text>
  <rect x="354.5" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="368.1" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <text class="s" x="374.9" y="265.5" text-anchor="middle" style="fill: var(--fig-on-1)">0</text>
  <rect x="368.1" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="381.7" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <text class="s" x="388.5" y="265.5" text-anchor="middle" style="fill: var(--fig-on-1)">3</text>
  <rect x="381.7" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="395.3" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <text class="s" x="402.1" y="265.5" text-anchor="middle" style="fill: var(--fig-on-1)">1</text>
  <rect x="395.3" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="408.9" y="254" width="13.6" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <text class="s" x="415.7" y="265.5" text-anchor="middle" style="fill: var(--fig-on-1)">2</text>
  <rect x="408.9" y="254" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.45" stroke-width=".6"/>
  <rect x="204.8" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="218.4" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="232.0" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="245.6" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="259.2" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="272.8" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="286.4" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="300.1" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="313.7" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="327.3" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="340.9" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="354.5" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="368.1" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="381.7" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="395.3" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="408.9" y="270" width="13.6" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.18" stroke-width=".6"/>
  <rect x="204.8" y="238" width="217.8" height="48" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>
  <line x1="259.2" y1="238" x2="259.2" y2="286" style="stroke: var(--fig-1)" stroke-width="1.2"/>
  <line x1="313.7" y1="238" x2="313.7" y2="286" style="stroke: var(--fig-1)" stroke-width="1.2"/>
  <line x1="368.1" y1="238" x2="368.1" y2="286" style="stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="426.6" y="265.5">行 1</text>
  <text class="s" x="232.0" y="298" text-anchor="middle">块 0</text>
  <text class="s" x="286.4" y="298" text-anchor="middle">块 1</text>
  <text class="s" x="340.9" y="298" text-anchor="middle">块 2</text>
  <text class="s" x="395.3" y="298" text-anchor="middle">块 3</text>
  <text class="s" x="313.7" y="312" text-anchor="middle">x [4096, 4096]，64 MiB</text>
  <rect x="600.0" y="238" width="16" height="16" fill="none"/><rect x="600.0" y="238" width="16" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <rect x="600.0" y="254" width="16" height="16" style="fill: var(--fig-1)" fill-opacity=".78"/><rect x="600.0" y="254" width="16" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <rect x="600.0" y="270" width="16" height="16" fill="none"/><rect x="600.0" y="270" width="16" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="608.0" y="265.5" text-anchor="middle" style="fill: var(--fig-on-1)">25</text>
  <rect x="600.0" y="238" width="16" height="48" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="608.0" y="299" text-anchor="middle">out</text>
  <line x1="608.0" y1="130.0" x2="608.0" y2="253.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-3-m1)"/>
  <text class="s" x="604.0" y="152.0" text-anchor="end" style="fill: var(--fig-1)">tl.store</text>
</svg>
<figcaption><strong>图 2-4</strong> row sum kernel（4096 × 4096，fp32），画法同<a href="#fig-t-1">图 2-1</a>。图里 TILE = 4，一行 16 个数分 4 块，代码里 TILE = 1024。颜色由淡到浓是 for 循环读块的顺序：第 0 轮后 acc 只有第 0 块的数，最后的 acc 是 4 块逐位相加 [7, 8, 4, 6]，铺满整条渐变；中间两轮用 ⋯ 省略，最后 <code>tl.sum</code> 加成 25 写回。</figcaption>
</figure>

累加器 `acc` 是一个长度为 TILE 的向量，不是一个数：循环里每个线程只加自己负责的几个位置，线程之间不需要通信；循环结束后，`tl.sum` 再做一次跨线程归约，把 TILE 个部分和加成一个数。

```python {role="program" tag="沿第 row 行循环累加"}
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, TILE: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([TILE], dtype=tl.float32)  # TILE 路部分和，不是一个标量
    for start in range(0, N, TILE):           # 行比 TILE 长：一块一块串行读
        cols = start + tl.arange(0, TILE)
        acc += tl.load(x_ptr + row * N + cols, mask=cols < N, other=0.0)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))  # 最后只做一次跨线程归约
```

grid 和 softmax 一样，一行一个 program：

```python {role="grid" tag="一行一个 program"}
def row_sum(x, TILE=1024):
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, out, N, TILE=TILE)
    return out
```

softmax 不能直接这样分块。$e^{x_{ij} - m_i}$ 要先知道整行的最大值 $m_i$，读完整行才知道。分块时累加器里要同时存目前的最大值和部分和，最大值变大时把已有的部分和乘上 $e^{m_{\text{old}} - m_{\text{new}}}$ 修正，这就是 online softmax，[FlashAttention](/blog/flashattention-1-to-4/) 那篇会讲。

前面三个例子里每个数只读一次，分块只是为了把工作分给不同的 program。矩阵乘不同：A 的每个元素要和 B 的一整列相乘，分块是为了让读进片上的数据被多次使用。它也要沿 K 循环，累加器是 C 的一块。

### 2.4 Matmul：分块乘加，顺手融合 ReLU {#triton-matmul}

`[M, K] × [K, N]` 的矩阵乘再接 ReLU：

<p align="center">$C_{ij} = \max\left(0,\ \sum_{k=1}^{K} A_{ik} B_{kj}\right)$</p>

$C_{ij}$ 只用到 A 的第 $i$ 行和 B 的第 $j$ 列。把 M、N、K 分别按 BM、BN、BK 切块，记 $A_{IT}$ 是 A 的第 $I$ 个行块、第 $T$ 个列块（BM × BK），$B_{TJ}$、$C_{IJ}$ 同理。C 的一块 $C_{IJ}$ 里的每个元素都只用到 A 的第 $I$ 个行块和 B 的第 $J$ 个列块，所以不同的 $C_{IJ}$ 可以交给不同的 program 各算各的。再把对 $k$ 的求和按 BK 一段一段地拆开，$\mathcal{K}_T$ 是第 $T$ 段的 BK 个下标：

<p align="center">$\sum_{k=1}^{K} A_{ik} B_{kj} = \sum_{T=1}^{K/\mathrm{BK}} \ \sum_{k \in \mathcal{K}_T} A_{ik} B_{kj}, \qquad C_{IJ} = \max\left(0,\ \sum_{T} A_{IT}\, B_{TJ}\right)$</p>

加法满足结合律，分段加完再相加和一次加完是同一个数（浮点数下只差舍入误差），所以每次只读 $A_{IT}$ 和 $B_{TJ}$ 两个小块、算一个小矩阵乘 $A_{IT} B_{TJ}$ 累加起来，最后得到的就是完整的 $C_{IJ}$。ReLU 逐元素作用在累加完的结果上，放在最后做不影响答案。

<figure id="fig-t-4" class="fg-fig">
<svg class="fg" viewBox="0 0 640 465" width="100%" role="img" aria-label="matmul kernel（A 4096 × 2048，B 2048 × 512）：C 的每一块一个 program；program (1, 1) 沿 K 一轮用 cp.async 把 A、B 各一块从显存直接搬进 shared memory，tl.dot 从 shared memory 读一小片进 register 做乘加，acc 留在 register 里，中间几轮用 ⋯ 省略，最后 ReLU 后 tl.store 写回 C 的 (1, 1) 块">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-t-4-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-t-4-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-2)"/></marker><marker id="fig-t-4-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker><marker id="fig-t-4-m3" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-1)"/></marker></defs>
  <text class="ttl" x="10" y="16">grid</text>
  <text class="ttl" x="162" y="16">一个 program（pid_m = 1，pid_n = 1）</text>
  <text class="tb" x="630" y="16" text-anchor="end" style="fill: var(--fig-hi)">分块乘加</text>
  <rect x="10" y="28" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="50" y="28" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="10" y="55" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="50" y="55" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 30%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="68" y="70" text-anchor="middle">(1,1)</text>
  <rect x="10" y="82" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="50" y="82" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="10" y="109" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="50" y="109" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="lab2" x="10" y="148">C 的每一块一个 program</text>
  <text class="lab2" x="10" y="168">grid = (M / BM, N / BN)</text>
  <text class="lab2" x="10" y="183">64 × 8 = 512 个 program</text>
  <text class="ttl" x="10" y="204">一个 SM</text>
  <rect x="12" y="212" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <line x1="17.0" y1="216" x2="17.0" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="21.7" y1="216" x2="21.7" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="26.4" y1="216" x2="26.4" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="31.1" y1="216" x2="31.1" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="40" y="212" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="45.0" y1="216" x2="45.0" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="49.7" y1="216" x2="49.7" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="54.4" y1="216" x2="54.4" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="59.1" y1="216" x2="59.1" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="68" y="212" width="24" height="18" rx="3" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <line x1="73.0" y1="216" x2="73.0" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="77.7" y1="216" x2="77.7" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="82.4" y1="216" x2="82.4" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <line x1="87.1" y1="216" x2="87.1" y2="226" stroke="currentColor" stroke-opacity=".55" stroke-width="1.6"/>
  <rect x="96" y="212" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="12" y="234" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="40" y="234" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="68" y="234" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="96" y="234" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="12" y="256" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="40" y="256" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="68" y="256" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="96" y="256" width="24" height="18" rx="3" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="10" y="286">每格一个 program（4 warp）</text><text class="s" x="10" y="300">3 × 128 = 384 thread</text>
  <text class="s" x="10" y="314">shared memory 先满</text>
  <rect class="band" x="146" y="26" width="490" height="66" rx="8"/>
  <text class="lab2" x="152" y="56.5">计算</text>
  <text class="lab2" x="152" y="69.5">单元</text>
  <rect class="band" x="146" y="100" width="490" height="40" rx="8"/>
  <text class="lab2" x="152" y="124.0">register</text>
  <rect class="band" x="146" y="148" width="490" height="76" rx="8"/>
  <text class="lab2" x="152" y="183.5">shared</text>
  <text class="lab2" x="152" y="196.5">memory</text>
  <rect class="band" x="146" y="232" width="490" height="140" rx="8"/>
  <text class="lab2" x="152" y="306.0">显存</text>
  <text class="s" x="152" y="320.0">GDDR7</text>
  <line x1="90.0" y1="66.0" x2="144.0" y2="66.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-t-4-m0)"/>
  <rect class="op" x="262.0" y="46" width="71.3" height="38" rx="6"/>
  <text class="tb" x="297.7" y="62" text-anchor="middle">dot</text><text class="s" x="297.7" y="77" text-anchor="middle">tl.dot</text>
  <text class="tb" x="354.3" y="69.0" text-anchor="middle">⋯</text>
  <rect class="op" x="425.3" y="46" width="71.3" height="38" rx="6"/>
  <text class="tb" x="461.0" y="62" text-anchor="middle">dot</text><text class="s" x="461.0" y="77" text-anchor="middle">tl.dot</text>
  <rect class="op" x="506.7" y="46" width="71.3" height="38" rx="6"/>
  <text class="tb" x="542.3" y="62" text-anchor="middle">relu</text><text class="s" x="542.3" y="77" text-anchor="middle">tl.maximum</text>
  <text class="lab2" x="152" y="388">cp.async：两次 tl.load 把 A 块、B 块从显存直接搬进 shared memory，不经过 register；</text>
  <text class="lab2" x="152" y="403">乘加：tl.dot 从 shared memory 读一小片 a、b 到 register，计算单元做 FMA，加到 acc 上；</text>
  <text class="lab2" x="152" y="418">num_stages = 3：A、B 各两个槽（共 32 KiB），算第 T 轮时第 T+1 轮在搬，读完马上搬第 T+2 轮；</text>
  <text class="lab2" x="152" y="433">K = 2048 按 BK = 32 切成 64 轮，A、B 各搬 64 次，图里只画第 0 轮，每一轮见图 2-6；</text>
  <text class="lab2" x="152" y="448">写回前 acc 先经 shared memory 换一次排布（16 KiB），再从 register 写回显存。</text>
  <defs><linearGradient id="fig-t-4-acc" x1="0" x2="1" y1="0" y2="0"><stop offset="0.00" style="stop-color: var(--fig-1)" stop-opacity="0.12"/><stop offset="0.14" style="stop-color: var(--fig-1)" stop-opacity="0.21"/><stop offset="0.29" style="stop-color: var(--fig-1)" stop-opacity="0.31"/><stop offset="0.43" style="stop-color: var(--fig-1)" stop-opacity="0.40"/><stop offset="0.57" style="stop-color: var(--fig-1)" stop-opacity="0.50"/><stop offset="0.71" style="stop-color: var(--fig-1)" stop-opacity="0.59"/><stop offset="0.86" style="stop-color: var(--fig-1)" stop-opacity="0.69"/><stop offset="1.00" style="stop-color: var(--fig-1)" stop-opacity="0.78"/></linearGradient></defs>
  <rect x="203.0" y="158" width="30" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.12"/><rect x="203.0" y="158" width="30" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="218.0" y="171" text-anchor="middle" >A 块</text>
  <rect x="245.0" y="190" width="34" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.12"/><rect x="245.0" y="190" width="34" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="262.0" y="203" text-anchor="middle" >B 块</text>
  <rect x="249.0" y="111" width="34" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.12"/><rect x="249.0" y="111" width="34" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="266.0" y="124" text-anchor="middle" >a、b</text>
  <line x1="224.0" y1="156.0" x2="258.0" y2="131.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="268.0" y1="188.0" x2="272.0" y2="131.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="280.0" y1="109.0" x2="281.7" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <rect x="366.3" y="158" width="30" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="366.3" y="158" width="30" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="381.3" y="171" text-anchor="middle" style="fill: var(--fig-on-1)">A 块</text>
  <rect x="408.3" y="190" width="34" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="408.3" y="190" width="34" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="425.3" y="203" text-anchor="middle" style="fill: var(--fig-on-1)">B 块</text>
  <rect x="412.3" y="111" width="34" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="412.3" y="111" width="34" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="429.3" y="124" text-anchor="middle" style="fill: var(--fig-on-1)">a、b</text>
  <line x1="387.3" y1="156.0" x2="421.3" y2="131.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="431.3" y1="188.0" x2="435.3" y2="131.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="443.3" y1="109.0" x2="445.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <rect x="310.7" y="111" width="34" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.12"/><rect x="310.7" y="111" width="34" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="327.7" y="124" text-anchor="middle" >acc</text>
  <rect x="484.7" y="111" width="34" height="18" rx="4" fill="url(#fig-t-4-acc)"/><rect x="484.7" y="111" width="34" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="501.7" y="124" text-anchor="middle" >acc</text>
  <line x1="305.7" y1="86.0" x2="319.7" y2="109.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="469.0" y1="86.0" x2="493.7" y2="109.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="509.7" y1="109.0" x2="526.3" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <path d="M297.7,44 L297.7,37 L461.0,37 L461.0,43" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.5" marker-end="url(#fig-t-4-m2)"/>
  <text class="s" x="379.3" y="33" text-anchor="middle" style="fill: var(--fig-hi)">acc 带到下一轮</text>
  <rect x="590.0" y="111" width="36" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="590.0" y="111" width="36" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="608.0" y="124" text-anchor="middle" style="fill: var(--fig-on-1)">C 块</text>
  <line x1="558.3" y1="86.0" x2="594.0" y2="109.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <rect x="579.0" y="176" width="30" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="579.0" y="176" width="30" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="594.0" y="189" text-anchor="middle" style="fill: var(--fig-on-1)">C 块</text>
  <line x1="591.0" y1="131.0" x2="591.0" y2="174.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="600.0" y1="174.0" x2="600.0" y2="131.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <rect x="285.7" y="172" width="74" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-dasharray="3 2" stroke-opacity=".7"/>
  <text class="s" x="322.7" y="185" text-anchor="middle">第 1 轮 A、B</text><text class="s" x="322.7" y="203" text-anchor="middle">正在搬</text>
  <rect x="202.0" y="244" width="6" height="6" fill="none"/>
  <rect x="202.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="244" width="6" height="6" fill="none"/>
  <rect x="208.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="244" width="6" height="6" fill="none"/>
  <rect x="214.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="244" width="6" height="6" fill="none"/>
  <rect x="220.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="244" width="6" height="6" fill="none"/>
  <rect x="226.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="244" width="6" height="6" fill="none"/>
  <rect x="232.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="244" width="6" height="6" fill="none"/>
  <rect x="238.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="244" width="6" height="6" fill="none"/>
  <rect x="244.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="250" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="202.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="250" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.2142857142857143"/>
  <rect x="208.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="250" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.3085714285714286"/>
  <rect x="214.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="250" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.40285714285714286"/>
  <rect x="220.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="250" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.49714285714285716"/>
  <rect x="226.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="250" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.5914285714285714"/>
  <rect x="232.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="250" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.6857142857142857"/>
  <rect x="238.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="250" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="244.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="256" width="6" height="6" fill="none"/>
  <rect x="202.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="256" width="6" height="6" fill="none"/>
  <rect x="208.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="256" width="6" height="6" fill="none"/>
  <rect x="214.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="256" width="6" height="6" fill="none"/>
  <rect x="220.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="256" width="6" height="6" fill="none"/>
  <rect x="226.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="256" width="6" height="6" fill="none"/>
  <rect x="232.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="256" width="6" height="6" fill="none"/>
  <rect x="238.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="256" width="6" height="6" fill="none"/>
  <rect x="244.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="262" width="6" height="6" fill="none"/>
  <rect x="202.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="262" width="6" height="6" fill="none"/>
  <rect x="208.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="262" width="6" height="6" fill="none"/>
  <rect x="214.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="262" width="6" height="6" fill="none"/>
  <rect x="220.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="262" width="6" height="6" fill="none"/>
  <rect x="226.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="262" width="6" height="6" fill="none"/>
  <rect x="232.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="262" width="6" height="6" fill="none"/>
  <rect x="238.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="262" width="6" height="6" fill="none"/>
  <rect x="244.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="268" width="6" height="6" fill="none"/>
  <rect x="202.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="268" width="6" height="6" fill="none"/>
  <rect x="208.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="268" width="6" height="6" fill="none"/>
  <rect x="214.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="268" width="6" height="6" fill="none"/>
  <rect x="220.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="268" width="6" height="6" fill="none"/>
  <rect x="226.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="268" width="6" height="6" fill="none"/>
  <rect x="232.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="268" width="6" height="6" fill="none"/>
  <rect x="238.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="268" width="6" height="6" fill="none"/>
  <rect x="244.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="274" width="6" height="6" fill="none"/>
  <rect x="202.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="274" width="6" height="6" fill="none"/>
  <rect x="208.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="274" width="6" height="6" fill="none"/>
  <rect x="214.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="274" width="6" height="6" fill="none"/>
  <rect x="220.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="274" width="6" height="6" fill="none"/>
  <rect x="226.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="274" width="6" height="6" fill="none"/>
  <rect x="232.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="274" width="6" height="6" fill="none"/>
  <rect x="238.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="274" width="6" height="6" fill="none"/>
  <rect x="244.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="280" width="6" height="6" fill="none"/>
  <rect x="202.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="280" width="6" height="6" fill="none"/>
  <rect x="208.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="280" width="6" height="6" fill="none"/>
  <rect x="214.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="280" width="6" height="6" fill="none"/>
  <rect x="220.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="280" width="6" height="6" fill="none"/>
  <rect x="226.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="280" width="6" height="6" fill="none"/>
  <rect x="232.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="280" width="6" height="6" fill="none"/>
  <rect x="238.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="280" width="6" height="6" fill="none"/>
  <rect x="244.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="286" width="6" height="6" fill="none"/>
  <rect x="202.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="286" width="6" height="6" fill="none"/>
  <rect x="208.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="286" width="6" height="6" fill="none"/>
  <rect x="214.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="286" width="6" height="6" fill="none"/>
  <rect x="220.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="286" width="6" height="6" fill="none"/>
  <rect x="226.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="286" width="6" height="6" fill="none"/>
  <rect x="232.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="286" width="6" height="6" fill="none"/>
  <rect x="238.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="286" width="6" height="6" fill="none"/>
  <rect x="244.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="292" width="6" height="6" fill="none"/>
  <rect x="202.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="292" width="6" height="6" fill="none"/>
  <rect x="208.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="292" width="6" height="6" fill="none"/>
  <rect x="214.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="292" width="6" height="6" fill="none"/>
  <rect x="220.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="292" width="6" height="6" fill="none"/>
  <rect x="226.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="292" width="6" height="6" fill="none"/>
  <rect x="232.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="292" width="6" height="6" fill="none"/>
  <rect x="238.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="292" width="6" height="6" fill="none"/>
  <rect x="244.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="298" width="6" height="6" fill="none"/>
  <rect x="202.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="298" width="6" height="6" fill="none"/>
  <rect x="208.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="298" width="6" height="6" fill="none"/>
  <rect x="214.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="298" width="6" height="6" fill="none"/>
  <rect x="220.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="298" width="6" height="6" fill="none"/>
  <rect x="226.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="298" width="6" height="6" fill="none"/>
  <rect x="232.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="298" width="6" height="6" fill="none"/>
  <rect x="238.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="298" width="6" height="6" fill="none"/>
  <rect x="244.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="304" width="6" height="6" fill="none"/>
  <rect x="202.0" y="304" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="304" width="6" height="6" fill="none"/>
  <rect x="208.0" y="304" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="304" width="6" height="6" fill="none"/>
  <rect x="214.0" y="304" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="304" width="6" height="6" fill="none"/>
  <rect x="220.0" y="304" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="304" width="6" height="6" fill="none"/>
  <rect x="226.0" y="304" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="304" width="6" height="6" fill="none"/>
  <rect x="232.0" y="304" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="304" width="6" height="6" fill="none"/>
  <rect x="238.0" y="304" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="304" width="6" height="6" fill="none"/>
  <rect x="244.0" y="304" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="310" width="6" height="6" fill="none"/>
  <rect x="202.0" y="310" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="310" width="6" height="6" fill="none"/>
  <rect x="208.0" y="310" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="310" width="6" height="6" fill="none"/>
  <rect x="214.0" y="310" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="310" width="6" height="6" fill="none"/>
  <rect x="220.0" y="310" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="310" width="6" height="6" fill="none"/>
  <rect x="226.0" y="310" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="310" width="6" height="6" fill="none"/>
  <rect x="232.0" y="310" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="310" width="6" height="6" fill="none"/>
  <rect x="238.0" y="310" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="310" width="6" height="6" fill="none"/>
  <rect x="244.0" y="310" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="316" width="6" height="6" fill="none"/>
  <rect x="202.0" y="316" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="316" width="6" height="6" fill="none"/>
  <rect x="208.0" y="316" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="316" width="6" height="6" fill="none"/>
  <rect x="214.0" y="316" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="316" width="6" height="6" fill="none"/>
  <rect x="220.0" y="316" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="316" width="6" height="6" fill="none"/>
  <rect x="226.0" y="316" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="316" width="6" height="6" fill="none"/>
  <rect x="232.0" y="316" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="316" width="6" height="6" fill="none"/>
  <rect x="238.0" y="316" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="316" width="6" height="6" fill="none"/>
  <rect x="244.0" y="316" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="322" width="6" height="6" fill="none"/>
  <rect x="202.0" y="322" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="322" width="6" height="6" fill="none"/>
  <rect x="208.0" y="322" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="322" width="6" height="6" fill="none"/>
  <rect x="214.0" y="322" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="322" width="6" height="6" fill="none"/>
  <rect x="220.0" y="322" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="322" width="6" height="6" fill="none"/>
  <rect x="226.0" y="322" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="322" width="6" height="6" fill="none"/>
  <rect x="232.0" y="322" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="322" width="6" height="6" fill="none"/>
  <rect x="238.0" y="322" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="322" width="6" height="6" fill="none"/>
  <rect x="244.0" y="322" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="328" width="6" height="6" fill="none"/>
  <rect x="202.0" y="328" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="328" width="6" height="6" fill="none"/>
  <rect x="208.0" y="328" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="328" width="6" height="6" fill="none"/>
  <rect x="214.0" y="328" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="328" width="6" height="6" fill="none"/>
  <rect x="220.0" y="328" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="328" width="6" height="6" fill="none"/>
  <rect x="226.0" y="328" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="328" width="6" height="6" fill="none"/>
  <rect x="232.0" y="328" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="328" width="6" height="6" fill="none"/>
  <rect x="238.0" y="328" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="328" width="6" height="6" fill="none"/>
  <rect x="244.0" y="328" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="334" width="6" height="6" fill="none"/>
  <rect x="202.0" y="334" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="334" width="6" height="6" fill="none"/>
  <rect x="208.0" y="334" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="334" width="6" height="6" fill="none"/>
  <rect x="214.0" y="334" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="334" width="6" height="6" fill="none"/>
  <rect x="220.0" y="334" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="334" width="6" height="6" fill="none"/>
  <rect x="226.0" y="334" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="334" width="6" height="6" fill="none"/>
  <rect x="232.0" y="334" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="334" width="6" height="6" fill="none"/>
  <rect x="238.0" y="334" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="334" width="6" height="6" fill="none"/>
  <rect x="244.0" y="334" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="244" width="48" height="96" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="226.0" y="353.0" text-anchor="middle">A [4096, 2048]</text>
  <text class="s" x="226.0" y="366.0" text-anchor="middle">32 MiB</text>
  <rect x="258.0" y="244" width="6" height="6" fill="none"/>
  <rect x="258.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="244" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="264.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="250" width="6" height="6" fill="none"/>
  <rect x="258.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="250" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.2142857142857143"/>
  <rect x="264.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="256" width="6" height="6" fill="none"/>
  <rect x="258.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="256" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.3085714285714286"/>
  <rect x="264.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="262" width="6" height="6" fill="none"/>
  <rect x="258.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="262" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.40285714285714286"/>
  <rect x="264.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="268" width="6" height="6" fill="none"/>
  <rect x="258.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="268" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.49714285714285716"/>
  <rect x="264.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="274" width="6" height="6" fill="none"/>
  <rect x="258.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="274" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.5914285714285714"/>
  <rect x="264.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="280" width="6" height="6" fill="none"/>
  <rect x="258.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="280" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.6857142857142857"/>
  <rect x="264.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="286" width="6" height="6" fill="none"/>
  <rect x="258.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="286" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="264.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="244" width="12" height="48" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="276.0" y="258.0" text-anchor="start">B [2048, 512]</text>
  <text class="s" x="276.0" y="271.0" text-anchor="start">4 MiB</text>
  <line x1="205.0" y1="250.0" x2="212.0" y2="178.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-4-m3)"/>
  <line x1="267.0" y1="244.0" x2="265.0" y2="210.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-4-m3)"/>
  <text class="s" x="271.0" y="228.0" text-anchor="start" style="fill: var(--fig-1)">cp.async</text>
  <rect x="599.0" y="244" width="6" height="6" fill="none"/>
  <rect x="599.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="244" width="6" height="6" fill="none"/>
  <rect x="605.0" y="244" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="250" width="6" height="6" fill="none"/>
  <rect x="599.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="250" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="605.0" y="250" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="256" width="6" height="6" fill="none"/>
  <rect x="599.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="256" width="6" height="6" fill="none"/>
  <rect x="605.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="262" width="6" height="6" fill="none"/>
  <rect x="599.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="262" width="6" height="6" fill="none"/>
  <rect x="605.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="268" width="6" height="6" fill="none"/>
  <rect x="599.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="268" width="6" height="6" fill="none"/>
  <rect x="605.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="274" width="6" height="6" fill="none"/>
  <rect x="599.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="274" width="6" height="6" fill="none"/>
  <rect x="605.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="280" width="6" height="6" fill="none"/>
  <rect x="599.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="280" width="6" height="6" fill="none"/>
  <rect x="605.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="286" width="6" height="6" fill="none"/>
  <rect x="599.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="286" width="6" height="6" fill="none"/>
  <rect x="605.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="292" width="6" height="6" fill="none"/>
  <rect x="599.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="292" width="6" height="6" fill="none"/>
  <rect x="605.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="298" width="6" height="6" fill="none"/>
  <rect x="599.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="298" width="6" height="6" fill="none"/>
  <rect x="605.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="304" width="6" height="6" fill="none"/>
  <rect x="599.0" y="304" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="304" width="6" height="6" fill="none"/>
  <rect x="605.0" y="304" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="310" width="6" height="6" fill="none"/>
  <rect x="599.0" y="310" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="310" width="6" height="6" fill="none"/>
  <rect x="605.0" y="310" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="316" width="6" height="6" fill="none"/>
  <rect x="599.0" y="316" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="316" width="6" height="6" fill="none"/>
  <rect x="605.0" y="316" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="322" width="6" height="6" fill="none"/>
  <rect x="599.0" y="322" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="322" width="6" height="6" fill="none"/>
  <rect x="605.0" y="322" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="328" width="6" height="6" fill="none"/>
  <rect x="599.0" y="328" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="328" width="6" height="6" fill="none"/>
  <rect x="605.0" y="328" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="334" width="6" height="6" fill="none"/>
  <rect x="599.0" y="334" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="334" width="6" height="6" fill="none"/>
  <rect x="605.0" y="334" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="244" width="12" height="96" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="630.0" y="353.0" text-anchor="end">C [4096, 512]</text>
  <text class="s" x="630.0" y="366.0" text-anchor="end">8 MiB</text>
  <line x1="618.0" y1="131.0" x2="608.0" y2="249.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-4-m3)"/>
  <text class="s" x="606.0" y="226" text-anchor="end" style="fill: var(--fig-1)">tl.store</text>
  <text class="s" x="276.0" y="296">for 循环沿 K 读块的顺序</text>
  <rect x="276.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.12" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="288.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.21" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="300.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.31" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="312.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.40" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="324.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.50" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="336.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.59" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="348.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.69" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="360.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.78" stroke-opacity=".4" stroke-width=".6"/>
  <text class="s" x="276.0" y="325">第 0 轮</text><text class="s" x="372.0" y="325" text-anchor="end">第 63 轮</text>
</svg>
<figcaption><strong>图 2-5</strong> matmul kernel（A 4096 × 2048，B 2048 × 512，fp32，BM = BN = 64，BK = 32），画法同<a href="#fig-t-1">图 2-1</a>。显存里的 A、B、C 按形状比例画全，一格是 256 × 256 个元素。program (1, 1) 读 A 的第 1 行和 B 的第 1 列，颜色由淡到浓是 for 循环沿 K 读块的顺序：第 0 轮读最淡的块，第 63 轮读最浓的；acc 一轮轮累积，最后铺满整条渐变，ReLU 后写回 C 的 (1, 1) 块。512 个 program 分到 170 个 SM，每个 SM 放 3 个，一批 510 个，剩下 2 个要等第二批。</figcaption>
</figure>

一个 program 负责一个 $C_{IJ}$，grid 是二维的（[图 2-5](#fig-t-4)）。program 沿 K 方向循环，第 $T$ 轮读 $A_{IT}$ 和 $B_{TJ}$，用 `tl.dot` 乘加到累加器上。写回之前在 register 里做 ReLU，激活函数就不需要单独一个 kernel 再读写一遍 C：

<div class="nv-key">

**shared memory 什么时候值得用**：一是同一份数据要被读很多次（复用），二是线程之间要交换大量数据。

matmul 两条都满足：A 的一块要和 B 块的 64 列都乘一遍，`tl.dot` 还要把 A、B 块重新分给不同的线程，所以 A、B 块放在 shared memory 里，每个 program 32 KiB。softmax 和 row sum 两条都不满足：每个数只被一个 thread 读一次，线程之间只交换 4 个 warp 的部分结果（16 B），把数据整块放进 shared memory 只会多搬一趟。代价是 shared memory 用得越多，一个 SM 能同时放的 program 越少，matmul 只放得下 3 个。

</div>

```python {role="program" tag="算 C 的第 (pid_m, pid_n) 块"}
@triton.jit
def matmul_relu_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       sam, sak, sbk, sbn, scm, scn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)   # 负责 C 的第 (pid_m, pid_n) 块
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak  # [BM, BK] 的指针
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn  # [BK, BN] 的指针
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):                 # 沿 K 一块一块乘加
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")  # fp32 默认会走 tf32，这里要求精确的 fp32
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    acc = tl.maximum(acc, 0.0)                # ReLU 在 register 里做完，不多一次读写
    c_ptrs = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
```

grid 是二维的，C 的每一块一个 program：

```python {role="grid" tag="C 的每一块一个 program，二维"}
def matmul_relu(a, b, BM=64, BN=64, BK=32):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_relu_kernel[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(), BM=BM, BN=BN, BK=BK)
    return c
```

A、B 块具体怎么进出 shared memory，看编译出的 TTGIR 和 PTX 就能数清楚（[图 2-6](#fig-t-6)）。`num_stages=3` 让编译器把循环改写成流水线：A、B 各开两个槽，循环前先发出第 0、1 轮的 cp.async；第 $T$ 轮先等第 $T$ 轮的数据到齐，用 ld.shared 读进 register 做乘加，这时第 $T+1$ 轮还在搬，读完马上往刚空出来的槽发第 $T+2$ 轮。A、B 各从显存搬 64 次，每次 8 KiB，一个 program 一共 1 MiB；每轮从 shared memory 读进 register 的是 192 KiB，同一个数平均被读 12 次，这就是前面说的复用。写回之前 acc 还要经 shared memory 换一次排布，让 tl.store 能按连续地址写。

<figure id="fig-t-6" class="fg-fig">
<svg class="fg" viewBox="0 0 640 346" width="100%" role="img" aria-label="matmul 的 shared memory 里 A、B 各有两个槽：循环前先发出第 0、1 轮的 cp.async；第 i 轮等第 i 轮到齐，ld.shared 读进 register 做 dot，第 i + 1 轮还在搬，读完就往同一个槽发第 i + 2 轮；第 62、63 轮不再搬；循环后 acc 经 shared memory 换排布再写回显存">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-t-6-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <text class="tb" x="129.5" y="18" text-anchor="middle">循环前</text>
  <text class="tb" x="196.5" y="18" text-anchor="middle">第 0 轮</text>
  <text class="tb" x="263.5" y="18" text-anchor="middle">第 1 轮</text>
  <text class="tb" x="330.5" y="18" text-anchor="middle">第 2 轮</text>
  <text class="tb" x="397.5" y="18" text-anchor="middle">⋯</text>
  <text class="tb" x="464.5" y="18" text-anchor="middle">第 62 轮</text>
  <text class="tb" x="531.5" y="18" text-anchor="middle">第 63 轮</text>
  <text class="tb" x="598.5" y="18" text-anchor="middle">循环后</text>
  <line x1="96" y1="26" x2="626" y2="26" stroke="currentColor" stroke-opacity=".5" stroke-width="1.2" marker-end="url(#fig-t-6-m0)"/>
  <text class="lab2" x="8" y="48">cp.async</text><text class="lab2" x="8" y="61">显存 → shared</text>
  <text class="s" x="129.5" y="54" text-anchor="middle" >第 0、1 轮</text>
  <text class="s" x="196.5" y="54" text-anchor="middle" >第 2 轮</text>
  <text class="s" x="263.5" y="54" text-anchor="middle" >第 3 轮</text>
  <text class="s" x="330.5" y="54" text-anchor="middle" >第 4 轮</text>
  <text class="s" x="464.5" y="54" text-anchor="middle" opacity=.6>不搬</text>
  <text class="s" x="531.5" y="54" text-anchor="middle" opacity=.6>不搬</text>
  <text class="s" x="598.5" y="54" text-anchor="middle" >st.shared</text>
  <text class="s" x="397.5" y="120" text-anchor="middle">⋯</text>
  <text class="lab" x="8" y="85">A 槽 0</text>
  <rect x="100.5" y="70" width="58.0" height="22" rx="3" fill="none"/><rect x="100.5" y="70" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="129.5" y="85" text-anchor="middle" >0</text>
  <rect x="167.5" y="70" width="27.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.12"/><rect x="167.5" y="70" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="2"/>
  <text class="s" x="181.0" y="85" text-anchor="middle" >0</text>
  <rect x="198.5" y="70" width="27.0" height="22" rx="3" fill="none"/><rect x="198.5" y="70" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="212.0" y="85" text-anchor="middle" >2</text>
  <rect x="234.5" y="70" width="58.0" height="22" rx="3" fill="none"/><rect x="234.5" y="70" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="263.5" y="85" text-anchor="middle" >2</text>
  <rect x="301.5" y="70" width="27.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.14"/><rect x="301.5" y="70" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="2"/>
  <text class="s" x="315.0" y="85" text-anchor="middle" >2</text>
  <rect x="332.5" y="70" width="27.0" height="22" rx="3" fill="none"/><rect x="332.5" y="70" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="346.0" y="85" text-anchor="middle" >4</text>
  <rect x="435.5" y="70" width="27.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.77"/><rect x="435.5" y="70" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="2"/>
  <text class="s" x="449.0" y="85" text-anchor="middle" style="fill: var(--fig-on-1)">62</text>
  <rect x="466.5" y="70" width="27.0" height="22" rx="3" fill="none"/><rect x="466.5" y="70" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1"/>
  <text class="s" x="480.0" y="85" text-anchor="middle" >—</text>
  <rect x="502.5" y="70" width="58.0" height="22" rx="3" fill="none"/><rect x="502.5" y="70" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1"/>
  <text class="s" x="531.5" y="85" text-anchor="middle" >—</text>
  <text class="lab" x="8" y="113">A 槽 1</text>
  <rect x="100.5" y="98" width="58.0" height="22" rx="3" fill="none"/><rect x="100.5" y="98" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="129.5" y="113" text-anchor="middle" >1</text>
  <rect x="167.5" y="98" width="58.0" height="22" rx="3" fill="none"/><rect x="167.5" y="98" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="196.5" y="113" text-anchor="middle" >1</text>
  <rect x="234.5" y="98" width="27.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.13"/><rect x="234.5" y="98" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="2"/>
  <text class="s" x="248.0" y="113" text-anchor="middle" >1</text>
  <rect x="265.5" y="98" width="27.0" height="22" rx="3" fill="none"/><rect x="265.5" y="98" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="279.0" y="113" text-anchor="middle" >3</text>
  <rect x="301.5" y="98" width="58.0" height="22" rx="3" fill="none"/><rect x="301.5" y="98" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="330.5" y="113" text-anchor="middle" >3</text>
  <rect x="435.5" y="98" width="58.0" height="22" rx="3" fill="none"/><rect x="435.5" y="98" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="464.5" y="113" text-anchor="middle" >63</text>
  <rect x="502.5" y="98" width="27.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="502.5" y="98" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="2"/>
  <text class="s" x="516.0" y="113" text-anchor="middle" style="fill: var(--fig-on-1)">63</text>
  <rect x="533.5" y="98" width="27.0" height="22" rx="3" fill="none"/><rect x="533.5" y="98" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1"/>
  <text class="s" x="547.0" y="113" text-anchor="middle" >—</text>
  <text class="lab" x="8" y="147">B 槽 0</text>
  <rect x="100.5" y="132" width="58.0" height="22" rx="3" fill="none"/><rect x="100.5" y="132" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="129.5" y="147" text-anchor="middle" >0</text>
  <rect x="167.5" y="132" width="27.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.12"/><rect x="167.5" y="132" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="2"/>
  <text class="s" x="181.0" y="147" text-anchor="middle" >0</text>
  <rect x="198.5" y="132" width="27.0" height="22" rx="3" fill="none"/><rect x="198.5" y="132" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="212.0" y="147" text-anchor="middle" >2</text>
  <rect x="234.5" y="132" width="58.0" height="22" rx="3" fill="none"/><rect x="234.5" y="132" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="263.5" y="147" text-anchor="middle" >2</text>
  <rect x="301.5" y="132" width="27.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.14"/><rect x="301.5" y="132" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="2"/>
  <text class="s" x="315.0" y="147" text-anchor="middle" >2</text>
  <rect x="332.5" y="132" width="27.0" height="22" rx="3" fill="none"/><rect x="332.5" y="132" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="346.0" y="147" text-anchor="middle" >4</text>
  <rect x="435.5" y="132" width="27.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.77"/><rect x="435.5" y="132" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="2"/>
  <text class="s" x="449.0" y="147" text-anchor="middle" style="fill: var(--fig-on-1)">62</text>
  <rect x="466.5" y="132" width="27.0" height="22" rx="3" fill="none"/><rect x="466.5" y="132" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1"/>
  <text class="s" x="480.0" y="147" text-anchor="middle" >—</text>
  <rect x="502.5" y="132" width="58.0" height="22" rx="3" fill="none"/><rect x="502.5" y="132" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1"/>
  <text class="s" x="531.5" y="147" text-anchor="middle" >—</text>
  <text class="lab" x="8" y="175">B 槽 1</text>
  <rect x="100.5" y="160" width="58.0" height="22" rx="3" fill="none"/><rect x="100.5" y="160" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="129.5" y="175" text-anchor="middle" >1</text>
  <rect x="167.5" y="160" width="58.0" height="22" rx="3" fill="none"/><rect x="167.5" y="160" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="196.5" y="175" text-anchor="middle" >1</text>
  <rect x="234.5" y="160" width="27.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.13"/><rect x="234.5" y="160" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="2"/>
  <text class="s" x="248.0" y="175" text-anchor="middle" >1</text>
  <rect x="265.5" y="160" width="27.0" height="22" rx="3" fill="none"/><rect x="265.5" y="160" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="279.0" y="175" text-anchor="middle" >3</text>
  <rect x="301.5" y="160" width="58.0" height="22" rx="3" fill="none"/><rect x="301.5" y="160" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="330.5" y="175" text-anchor="middle" >3</text>
  <rect x="435.5" y="160" width="58.0" height="22" rx="3" fill="none"/><rect x="435.5" y="160" width="58.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="464.5" y="175" text-anchor="middle" >63</text>
  <rect x="502.5" y="160" width="27.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="502.5" y="160" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="2"/>
  <text class="s" x="516.0" y="175" text-anchor="middle" style="fill: var(--fig-on-1)">63</text>
  <rect x="533.5" y="160" width="27.0" height="22" rx="3" fill="none"/><rect x="533.5" y="160" width="27.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1"/>
  <text class="s" x="547.0" y="175" text-anchor="middle" >—</text>
  <line x1="96" y1="126.0" x2="565" y2="126.0" class="grid"/>
  <rect x="568.5" y="70" width="60" height="112" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="598.5" y="110" text-anchor="middle">C 块</text>
  <text class="s" x="598.5" y="126" text-anchor="middle">64 × 64</text>
  <text class="s" x="598.5" y="142" text-anchor="middle">16 KiB</text>
  <text class="s" x="598.5" y="158" text-anchor="middle">换排布</text>
  <text class="lab2" x="8" y="210">ld.shared</text><text class="lab2" x="8" y="223">shared → register</text>
  <text class="lab2" x="8" y="248">计算单元</text>
  <text class="s" x="196.5" y="216" text-anchor="middle">第 0 轮</text>
  <text class="s" x="196.5" y="248" text-anchor="middle">dot</text>
  <text class="s" x="263.5" y="216" text-anchor="middle">第 1 轮</text>
  <text class="s" x="263.5" y="248" text-anchor="middle">dot</text>
  <text class="s" x="330.5" y="216" text-anchor="middle">第 2 轮</text>
  <text class="s" x="330.5" y="248" text-anchor="middle">dot</text>
  <text class="s" x="464.5" y="216" text-anchor="middle">第 62 轮</text>
  <text class="s" x="464.5" y="248" text-anchor="middle">dot</text>
  <text class="s" x="531.5" y="216" text-anchor="middle">第 63 轮</text>
  <text class="s" x="531.5" y="248" text-anchor="middle">dot</text>
  <text class="s" x="598.5" y="216" text-anchor="middle">C 块</text><text class="s" x="598.5" y="248" text-anchor="middle">relu</text>
  <rect x="8.0" y="261" width="22.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.17"/><rect x="8.0" y="261" width="22.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1"/>
  <text class="s" x="19.0" y="276" text-anchor="middle" >5</text>
  <text class="s" x="36" y="276">已在 shared memory</text>
  <rect x="150.0" y="261" width="22.0" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity="0.17"/><rect x="150.0" y="261" width="22.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="2"/>
  <text class="s" x="161.0" y="276" text-anchor="middle" >5</text>
  <text class="s" x="178" y="276">这一轮 ld.shared 读的槽</text>
  <rect x="318.0" y="261" width="22.0" height="22" rx="3" fill="none"/><rect x="318.0" y="261" width="22.0" height="22" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="329.0" y="276" text-anchor="middle" >5</text>
  <text class="s" x="346" y="276">cp.async 已发出，还在搬；数字是第几轮</text>
  <text class="lab2" x="8" y="300">显存 → shared memory：A、B 各搬 64 次，每次 8 KiB（每个 thread 4 条 16 B 的 cp.async），一个 program 共 1 MiB。</text>
  <text class="lab2" x="8" y="316">shared memory → register：每轮每个 thread 192 条 ld.shared、读 1536 B，128 个 thread 共读 192 KiB，</text>
  <text class="lab2" x="8" y="332">是 A、B 两块（16 KiB）的 12 倍。循环后 relu 过的 acc 先写进 shared memory 换排布，再读回 register，tl.store 写回显存。</text>
</svg>
<figcaption><strong>图 2-6</strong> matmul kernel 的 shared memory 流水线，按编译出的 TTGIR 和 PTX 画（<code>num_stages=3</code>，K = 2048 按 BK = 32 切成 64 轮）。A、B 各两个槽，共 32 KiB；颜色深浅是第几轮，和<a href="#fig-t-4">图 2-5</a> 一样。第 62、63 轮的 cp.async 被 mask 掉，不读显存。</figcaption>
</figure>

fp32 输入时 `tl.dot` 默认走 tf32 的 Tensor core，结果和 fp32 的矩阵乘差 0.1 左右；要和 PyTorch（关掉 tf32）对上，需要 `input_precision="ieee"`。这样 `tl.dot` 就不走 Tensor core 了，PTX 里是 `fma.rn.f32x2`，在 FP32 核上算，每个 thread 每轮 512 条，一条算两个 FMA。

第 2 节的四个例子是同一个模式：program 把一块数据读进 register，在片上做完能做的计算，再写回显存。[下一篇](/blog/gpu-training-analysis/)在 5090 上拆解一步训练的时间和显存，用的就是这个模式：把 RMSNorm 的几个 kernel 融合成一个，少读写一遍中间结果。

---

## 附录 A TPU：更大的矩阵单元，更简单的控制 {#tpu}

正文的 GPU 靠 kernel 自己分块来复用数据。TPU 面对同样的带宽问题，思路是把控制逻辑做轻，把矩阵乘单元做大。[图 A-1](#fig-h-3) 是 TPU v5p 的一个 TensorCore：Scalar unit 取指令、发给 VPU 和 MXU，整个核只有这一路控制，没有 warp 和调度器之间的切换；VPU 做逐元素运算和归约；矩阵乘交给 4 个 MXU，每个是 128 × 128 的 systolic array。片上存储只有 VMEM 和 VREG 两层，VMEM 是编译器管理的 scratchpad，不是硬件 cache，数据什么时候从 HBM 搬进来、搬多少，都由编译器（XLA）提前排好。

<figure id="fig-h-3" class="fg-fig">
<svg class="fg" viewBox="0 0 640 300" width="100%" role="img" aria-label="TPU v5p 的一个 TensorCore：Scalar unit 发指令，VMEM 是编译器管理的片上存储，VREG 供给 VPU 和 4 个 128×128 的 MXU；芯片外是 95 GB HBM，带宽 2.8e12 B/s">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-h-3-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-h-3-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-mute)"/></marker></defs>
  <rect x="10" y="22" width="452" height="250" rx="12" style="fill: color-mix(in srgb, var(--fig-1) 6%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="lab2" x="22" y="40">TensorCore（TPU v5p 每个芯片 2 个）</text>
  <rect x="24" y="50" width="424" height="30" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="t" x="236" y="70" text-anchor="middle">Scalar unit：取指令，发给 VPU 和 MXU（只有这一路控制）</text>
  <rect x="24" y="92" width="424" height="36" rx="6" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="t" x="236" y="115" text-anchor="middle">VMEM：片上 scratchpad，由编译器显式搬入搬出</text>
  <rect x="24" y="156" width="424" height="30" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="t" x="236" y="176" text-anchor="middle">VREG 256 KB（vector register）</text>
  <line x1="200.0" y1="129.0" x2="200.0" y2="154.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-3-m0)"/>
  <line x1="272.0" y1="155.0" x2="272.0" y2="130.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-3-m0)"/>
  <rect x="24" y="214" width="120" height="48" rx="6" style="fill: color-mix(in srgb, var(--fig-2) 30%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="tb" x="84" y="235" text-anchor="middle">VPU</text><text class="s" x="84" y="252" text-anchor="middle">8 × 128 lanes</text>
  <line x1="84.0" y1="187.0" x2="84.0" y2="212.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-3-m0)"/>
  <rect x="160" y="214" width="64" height="48" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="192" y="235" text-anchor="middle">MXU</text><text class="s" x="192" y="252" text-anchor="middle">128 × 128</text>
  <line x1="192.0" y1="187.0" x2="192.0" y2="212.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-3-m0)"/>
  <rect x="232" y="214" width="64" height="48" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="264" y="235" text-anchor="middle">MXU</text><text class="s" x="264" y="252" text-anchor="middle">128 × 128</text>
  <line x1="264.0" y1="187.0" x2="264.0" y2="212.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-3-m0)"/>
  <rect x="304" y="214" width="64" height="48" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="336" y="235" text-anchor="middle">MXU</text><text class="s" x="336" y="252" text-anchor="middle">128 × 128</text>
  <line x1="336.0" y1="187.0" x2="336.0" y2="212.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-3-m0)"/>
  <rect x="376" y="214" width="64" height="48" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="408" y="235" text-anchor="middle">MXU</text><text class="s" x="408" y="252" text-anchor="middle">128 × 128</text>
  <line x1="408.0" y1="187.0" x2="408.0" y2="212.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-3-m0)"/>
  <rect x="504" y="70" width="124" height="150" rx="10" style="fill: color-mix(in srgb, var(--fig-mute) 14%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="tb" x="566" y="124" text-anchor="middle">HBM</text><text class="t" x="566" y="144" text-anchor="middle">95 GB</text><text class="s" x="566" y="168" text-anchor="middle">带宽</text><text class="s" x="566" y="182" text-anchor="middle">2.8e12 B/s</text>
  <line x1="450.0" y1="104.0" x2="500.0" y2="104.0" style="stroke: var(--fig-mute)" stroke-width="1.8" marker-end="url(#fig-h-3-m1)"/>
  <line x1="500.0" y1="118.0" x2="450.0" y2="118.0" style="stroke: var(--fig-mute)" stroke-width="1.8" marker-end="url(#fig-h-3-m1)"/>
  <text class="lab2" x="10" y="292">没有 warp，也没有硬件管理的 L1 / L2：HBM → VMEM → VREG → VPU / MXU 每一步都由编译器排好</text>
</svg>
<figcaption><strong>图 A-1</strong> TPU v5p 一个 TensorCore 的结构示意，画法对应<a href="#fig-h-1">图 1-2</a>。规格来自 <a href="https://jax-ml.github.io/scaling-book/tpus/">How to Scale Your Model</a>，原书也有一张 TensorCore 的结构图。</figcaption>
</figure>

MXU 和 GPU 最大的不同在乘加单元怎么拿到数据（[图 A-2](#fig-h-4)）。GPU 上每个 CUDA core 做一次乘加，要从 register file 读两个数、写回一个；一个数被用很多次，得靠 kernel 自己分块，先搬进 shared memory 和 register 再反复读，[2.4 节](#triton-matmul)的 matmul 就是这样写的。systolic array 把复用做进了硬件：权重先装进 128 × 128 个格子，固定不动；输入 x 从左边进入，每拍向右传一格，经过的每一格把它乘上自己的权重，加到从上面传下来的部分和上，再把部分和往下传。一个 x 从 VMEM 读一次，在一行里被 128 个格子用到；稳定以后每拍从左边进 128 个数、从底部出 128 个结果，阵列里同时做 16384 次乘加，中间结果都不经过 register file。

<figure id="fig-h-4" class="fg-fig">
<svg class="fg" viewBox="0 0 640 334" width="100%" role="img" aria-label="GPU 的 CUDA core 每次乘加都从 register file 读两个数、写回一个；TPU 的 MXU 是 systolic array，权重固定在格子里，x 每拍向右流一格，部分和每拍向下流一格，只有边上读写">
  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-h-4-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-h-4-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker><marker id="fig-h-4-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-2)"/></marker></defs>
  <text class="ttl" x="20" y="22">GPU：CUDA core 从 register file 取数</text>
  <rect x="20" y="52" width="220" height="34" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="t" x="130" y="74" text-anchor="middle">register file</text>
  <rect x="20" y="150" width="44" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="42" y="172" text-anchor="middle">FMA</text>
  <line x1="30.0" y1="87.0" x2="30.0" y2="148.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.3" marker-end="url(#fig-h-4-m0)"/>
  <line x1="42.0" y1="87.0" x2="42.0" y2="148.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.3" marker-end="url(#fig-h-4-m0)"/>
  <line x1="54.0" y1="149.0" x2="54.0" y2="89.0" style="stroke: var(--fig-hi)" stroke-width="1.3" marker-end="url(#fig-h-4-m1)"/>
  <rect x="79" y="150" width="44" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="101" y="172" text-anchor="middle">FMA</text>
  <line x1="89.0" y1="87.0" x2="89.0" y2="148.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.3" marker-end="url(#fig-h-4-m0)"/>
  <line x1="101.0" y1="87.0" x2="101.0" y2="148.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.3" marker-end="url(#fig-h-4-m0)"/>
  <line x1="113.0" y1="149.0" x2="113.0" y2="89.0" style="stroke: var(--fig-hi)" stroke-width="1.3" marker-end="url(#fig-h-4-m1)"/>
  <rect x="138" y="150" width="44" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="160" y="172" text-anchor="middle">FMA</text>
  <line x1="148.0" y1="87.0" x2="148.0" y2="148.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.3" marker-end="url(#fig-h-4-m0)"/>
  <line x1="160.0" y1="87.0" x2="160.0" y2="148.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.3" marker-end="url(#fig-h-4-m0)"/>
  <line x1="172.0" y1="149.0" x2="172.0" y2="89.0" style="stroke: var(--fig-hi)" stroke-width="1.3" marker-end="url(#fig-h-4-m1)"/>
  <rect x="197" y="150" width="44" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="219" y="172" text-anchor="middle">FMA</text>
  <line x1="207.0" y1="87.0" x2="207.0" y2="148.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.3" marker-end="url(#fig-h-4-m0)"/>
  <line x1="219.0" y1="87.0" x2="219.0" y2="148.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.3" marker-end="url(#fig-h-4-m0)"/>
  <line x1="231.0" y1="149.0" x2="231.0" y2="89.0" style="stroke: var(--fig-hi)" stroke-width="1.3" marker-end="url(#fig-h-4-m1)"/>
  <text class="lab" x="20" y="212">每次乘加读 2 个数、写回 1 个</text>
  <text class="lab2" x="20" y="234">一个数要被用很多次，得靠 kernel 自己分块：</text>
  <text class="lab2" x="20" y="250">先搬进 shared memory / register，再反复读</text>
  <text class="ttl" x="300" y="22">TPU：MXU 是 systolic array（示意 4 × 4，实际 128 × 128）</text>
  <text class="lab2" x="300" y="40">W 先装进格子固定不动；x 每拍右移一格，部分和每拍下移一格</text>
  <rect x="372" y="52" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="390.0" y="74" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">00</tspan></text>
  <line x1="410.0" y1="70.0" x2="435.0" y2="70.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <line x1="390.0" y1="90.0" x2="390.0" y2="115.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <rect x="438" y="52" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="456.0" y="74" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">01</tspan></text>
  <line x1="476.0" y1="70.0" x2="501.0" y2="70.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <line x1="456.0" y1="90.0" x2="456.0" y2="115.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <rect x="504" y="52" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="522.0" y="74" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">02</tspan></text>
  <line x1="542.0" y1="70.0" x2="567.0" y2="70.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <line x1="522.0" y1="90.0" x2="522.0" y2="115.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <rect x="570" y="52" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="588.0" y="74" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">03</tspan></text>
  <line x1="588.0" y1="90.0" x2="588.0" y2="115.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <line x1="328.0" y1="70.0" x2="369.0" y2="70.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <text class="t" x="320" y="74.0" text-anchor="end">x<tspan baseline-shift="sub" font-size="9">0</tspan></text>
  <rect x="372" y="118" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="390.0" y="140" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">10</tspan></text>
  <line x1="410.0" y1="136.0" x2="435.0" y2="136.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <line x1="390.0" y1="156.0" x2="390.0" y2="181.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <rect x="438" y="118" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="456.0" y="140" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">11</tspan></text>
  <line x1="476.0" y1="136.0" x2="501.0" y2="136.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <line x1="456.0" y1="156.0" x2="456.0" y2="181.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <rect x="504" y="118" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="522.0" y="140" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">12</tspan></text>
  <line x1="542.0" y1="136.0" x2="567.0" y2="136.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <line x1="522.0" y1="156.0" x2="522.0" y2="181.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <rect x="570" y="118" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="588.0" y="140" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">13</tspan></text>
  <line x1="588.0" y1="156.0" x2="588.0" y2="181.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <line x1="328.0" y1="136.0" x2="369.0" y2="136.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <text class="t" x="320" y="140.0" text-anchor="end">x<tspan baseline-shift="sub" font-size="9">1</tspan></text>
  <rect x="372" y="184" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="390.0" y="206" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">20</tspan></text>
  <line x1="410.0" y1="202.0" x2="435.0" y2="202.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <line x1="390.0" y1="222.0" x2="390.0" y2="247.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <rect x="438" y="184" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="456.0" y="206" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">21</tspan></text>
  <line x1="476.0" y1="202.0" x2="501.0" y2="202.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <line x1="456.0" y1="222.0" x2="456.0" y2="247.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <rect x="504" y="184" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="522.0" y="206" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">22</tspan></text>
  <line x1="542.0" y1="202.0" x2="567.0" y2="202.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <line x1="522.0" y1="222.0" x2="522.0" y2="247.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <rect x="570" y="184" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="588.0" y="206" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">23</tspan></text>
  <line x1="588.0" y1="222.0" x2="588.0" y2="247.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <line x1="328.0" y1="202.0" x2="369.0" y2="202.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <text class="t" x="320" y="206.0" text-anchor="end">x<tspan baseline-shift="sub" font-size="9">2</tspan></text>
  <rect x="372" y="250" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="390.0" y="272" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">30</tspan></text>
  <line x1="410.0" y1="268.0" x2="435.0" y2="268.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <rect x="438" y="250" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="456.0" y="272" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">31</tspan></text>
  <line x1="476.0" y1="268.0" x2="501.0" y2="268.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <rect x="504" y="250" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="522.0" y="272" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">32</tspan></text>
  <line x1="542.0" y1="268.0" x2="567.0" y2="268.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <rect x="570" y="250" width="36" height="36" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="588.0" y="272" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">33</tspan></text>
  <line x1="328.0" y1="268.0" x2="369.0" y2="268.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-h-4-m2)"/>
  <text class="t" x="320" y="272.0" text-anchor="end">x<tspan baseline-shift="sub" font-size="9">3</tspan></text>
  <line x1="390.0" y1="288.0" x2="390.0" y2="310.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <text class="t" x="390.0" y="326" text-anchor="middle">y<tspan baseline-shift="sub" font-size="9">0</tspan></text>
  <line x1="456.0" y1="288.0" x2="456.0" y2="310.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <text class="t" x="456.0" y="326" text-anchor="middle">y<tspan baseline-shift="sub" font-size="9">1</tspan></text>
  <line x1="522.0" y1="288.0" x2="522.0" y2="310.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <text class="t" x="522.0" y="326" text-anchor="middle">y<tspan baseline-shift="sub" font-size="9">2</tspan></text>
  <line x1="588.0" y1="288.0" x2="588.0" y2="310.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-h-4-m1)"/>
  <text class="t" x="588.0" y="326" text-anchor="middle">y<tspan baseline-shift="sub" font-size="9">3</tspan></text>
</svg>
<figcaption><strong>图 A-2</strong> 乘加单元怎么拿到数据。左边是 GPU 的 CUDA core，右边是 TPU 的 MXU；x 按行错开一拍进入，y 是每列的输出。动画演示见 <a href="https://jax-ml.github.io/scaling-book/tpus/#appendix-b-how-does-a-systolic-array-work">How to Scale Your Model 附录 B</a>。</figcaption>
</figure>

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

## References {#references}

1. Gholami, Yao, Kim, Hooper, Mahoney, Keutzer. [AI and Memory Wall](https://arxiv.org/abs/2403.14123). IEEE Micro, 2024. 图 1-1。
2. NVIDIA. [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)（白皮书）。RTX 5090 的存储层级和 SM 结构，图 1-2、图 1-3。
3. NVIDIA. [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)、[NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)、[NVIDIA Blackwell Architecture](https://resources.nvidia.com/en-us-blackwell-architecture)。表 1-1 的 A100、H100、B200 一列。
4. NVIDIA. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)。各 compute capability 每 SM 的 warp、block、register 和 shared memory 上限，表 1-2，CUDA 执行层级。
5. NVIDIA. [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。warp 调度状态。
6. Austin et al. [How to Scale Your Model](https://jax-ml.github.io/scaling-book/gpus/), Google DeepMind, 2025. 附录 A 的 TPU v5p 规格、systolic array 的数据流和表 A-1。
7. Jouppi et al. [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760). ISCA 2017. 第一代 TPU 和它的 systolic array。
8. Kung. [Why Systolic Architectures?](https://doi.org/10.1109/MC.1982.1653825). IEEE Computer, 1982.
9. Tillet, Kung, Cox. [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://doi.org/10.1145/3315508.3329973). MAPL 2019.
10. [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)。第 2 节 softmax 和 matmul 例子的写法。
11. Hendrycks, Gimpel. [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415). arXiv:1606.08415, 2016. 第 2.1 节的 tanh 近似。
12. Milakov, Gimelshein. [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867). arXiv:1805.02867, 2018. 第 2.3 节提到的 online softmax。
