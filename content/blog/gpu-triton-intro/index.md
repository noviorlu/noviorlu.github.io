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

做性能分析时最常用的两个数是峰值算力和显存带宽，一个 kernel 受哪个限制，取决于它要搬多少数据、在哪一层读写。这一节先看数据存在哪几层（[1.1](#hw-memory)）和 SM 里有什么（[1.2](#hw-sm)），再看 CUDA 的 thread、warp、block 怎么落到这些硬件上（[1.3](#hw-model)）。

### 1.1 存储层级：越近越快，也越小 {#hw-memory}

处理器的频率早就不怎么涨了，算力的增长主要来自并行：更多的 SM、更宽的 Tensor core。显存带宽涨得慢得多：过去 20 年，峰值算力涨了约 6 万倍（每两年 3.0 倍），DRAM 带宽只涨了约 100 倍（每两年 1.6 倍），芯片之间的互联带宽约 30 倍（[图 1-1](#fig-memwall)）。带宽跟不上，越来越多的 op 受显存带宽限制（memory-bound），只能少搬：让读进来的数据留在离计算单元近的地方，多用几次。

<figure id="fig-memwall" class="fg-fig">
<img src="memory-wall.webp" alt="1997 到 2023 年峰值算力、DRAM 带宽和互联带宽的增长：算力每两年 3.0 倍，DRAM 带宽每两年 1.6 倍，互联带宽每两年 1.4 倍" loading="lazy" style="width: 100%; height: auto; border-radius: 8px; background: #fff;">
<figcaption><strong>图 1-1</strong> 峰值算力与 DRAM、互联带宽 20 年来的增长，纵轴是相对 1997 年的倍数（对数轴）。图来自 Gholami 等人的 <a href="https://arxiv.org/abs/2403.14123">AI and Memory Wall</a>（IEEE Micro，2024）。</figcaption>
</figure>

数据离计算单元越近，读写越快，容量也越小（[图 1-2](#fig-h-1)）。每个 SM 里有 register 和 L1 / shared memory，所有 SM 共享芯片上的 L2，显存在芯片外面。L1 和 L2 由硬件当作 cache 自动管理，kernel 能自己安排的只有 shared memory 和 register。所以 kernel 优化的套路都一样：把一个 block 的数据读进 shared memory 或 register，在片上多算几次，再写回显存。[第 2 节](#triton)的 block 矩阵乘、[下一篇](/blog/gpu-training-analysis/)的算子融合和 [FlashAttention](/blog/flashattention-1-to-4/) 都是这个思路。

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

每个 SM 分成 4 个 SMSP（SM sub-partition），每个 SMSP 有自己的 warp 调度器、register、32 个 CUDA core（做 FP32 和 INT32 运算）和一个 Tensor core，4 个 SMSP 共享 L1 / shared memory（[图 1-3](#fig-h-2)）。

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

[图 1-4](#fig-h-9) 把一个 SMSP 展开。[第 2 节](#triton) kernel 图里的「计算单元」指其中的 CUDA core 和 Tensor core，「register」就是这个 64 KB 的 register file。

<figure id="fig-h-9" class="fg-fig">
<svg class="fg" viewBox="0 0 640 322" width="100%" role="img" aria-label="RTX 5090 一个 SMSP 的结构：L0 指令 cache、warp scheduler 和 dispatch，64 KB register file，32 个 CUDA core（FP32/INT32）和 1 个 Tensor core，4 个 LD/ST 和 SFU；下面接 SM 共享的 128 KB L1 / shared memory">
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
  <text class="tb" x="132" y="152" text-anchor="middle">③ CUDA core</text><text class="s" x="132" y="166" text-anchor="middle">FP32 / INT32 × 32</text>
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
  <text class="lab2" x="400" y="125">CUDA core：一条加法、乘法或</text>
  <text class="lab2" x="400" y="140">FMA，32 个 thread 各用一个；</text>
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
<figcaption><strong>图 1-4</strong> RTX 5090 一个 SMSP 的结构示意，按 <a href="https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf">NVIDIA RTX Blackwell 白皮书</a>的图 5 画。Blackwell 的 CUDA core 既做 FP32 也做 INT32，同一拍只能做其中一种。</figcaption>
</figure>

这些是硬件提供的资源。register 按 thread 分配，shared memory 按 block 分配，要知道一个 kernel 怎么用这些资源，得先看 CUDA 怎么把 thread 组织起来。

### 1.3 编程模型：thread、warp、block、grid {#hw-model}

CUDA 的执行模型叫 SIMT（single instruction, multiple threads）：一个 warp 的 32 个 thread 同一时刻执行同一条指令。thread 往上分四层：thread、warp（32 个 thread）、block（也叫 CTA，若干个 warp）、grid（一次 kernel 启动的全部 block），每层跑在哪个硬件上见[图 1-5](#fig-h-5)。

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
  <text class="s" x="195" y="162" text-anchor="middle">shared memory：block 内共享，可以同步</text>
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

block 放到 SM 上以后就一直驻留，直到它的 thread 全部跑完。一个 SM 能同时驻留多少，由几条硬件上限决定（[表 1-1](#tab-1-1)），哪条先用完就卡在哪条。warp 的上限分摊到每个 SMSP 的调度器；block 的上限记在整个 SM 上，因为一个 block 的 shared memory 和同步由 4 个 SMSP 共用。

| 资源 | RTX 5090 每个 SM 的上限 |
|:--|:--|
| 驻留 warp | 48 个（每个 SMSP 12 个），即 1536 个 thread |
| 驻留 block | 24 个 |
| register | 65536 个 32-bit（256 KB），每个 thread 最多 255 个 |
| shared memory | 100 KiB，每个 block 另占 1 KiB |
{#tab-1-1 caption="**表 1-1** RTX 5090 一个 SM 的驻留上限" note="用 <code>cuDeviceGetAttribute</code> 在 5090 上读出，和 CUDA Programming Guide 里 compute capability 12.0 的数一致。"}

<details class="fold doc aside">
<summary>几代 NVIDIA GPU 的规格和驻留上限</summary>

| | A100 | H100 | B200 | RTX 5090 |
|:--|--:|--:|--:|--:|
| SM 数 | 108 | 132 | 148 | 170 |
| L2 | 40 MB | 50 MB | — | 96 MB |
| 显存 | 80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e | 32 GB GDDR7 |
| 显存带宽 | 2.0e12 B/s | 3.35e12 B/s | 8e12 B/s | 1.79e12 B/s |
| 每 SM 的 CUDA core | 64 | 128 | 128 | 128 |
| 每 SM 的 Tensor core | 4 | 4 | 4 | 4 |
| 每 SM 的 L1 + shared | 192 KB | 256 KB | 256 KB | 128 KB |
| 每 SM 的 register | 256 KB | 256 KB | 256 KB | 256 KB |
| 每 SMSP 最多驻留 warp | 16 | 16 | 16 | 12 |
| 每 SM 最多驻留 warp | 64 | 64 | 64 | 48 |
| 每 SM 最多驻留 block | 32 | 32 | 32 | 24 |
{#tab-1-1a caption="**表 1-1a** 几代 NVIDIA GPU 的规格和驻留上限" note="来自 NVIDIA 各代架构白皮书、产品规格（A100 80GB SXM，H100 SXM）和 CUDA Programming Guide 的 compute capability 表（8.0、9.0、10.0、12.0）。B200 的 L2 没有找到可靠的公开数字，暂缺。"}

5090 的 SM 最多，但每个 SM 能驻留的 warp 和 block 比数据中心卡少四分之一，L1 + shared 只有 H100、B200 的一半。

</details>

一个 block 的 warp 按编号轮流分到 4 个 SMSP。驻留不等于在跑：一个 warp 等显存数据要几百个周期，这段时间里调度器改发另一个就绪的 warp。所有驻留 warp 的 register 一直留在 register file 里，换 warp 不用保存或恢复状态，所以同时驻留的 warp 越多，访存延迟越容易藏住（[图 1-6](#fig-h-6)）。

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

block 之间不能直接共享数据，只能通过显存交换，而这正是最慢的一层。所以要尽量把会重复读的数据放进同一个 block 里处理，这就是 tiling。在 CUDA 里做 tiling 要自己管 shared memory 和 thread 同步，[第 2 节](#triton)的 Triton 把这部分交给编译器。

---

## 2 Triton：按 block 写 kernel {#triton}

Triton 按 block 写 kernel，正好对应 [1.3 节](#hw-model)说的 tiling。CUDA 要写清楚每个 thread 做什么，shared memory 和同步都要自己管；Triton 只写一个 block 做什么：把一个 block 的数据读进来，在片上算完，再写回显存，block 内怎么分给 thread、要不要经过 shared memory 由编译器决定。Triton 把一个 block 叫作一个 program（[表 2-1](#tab-2-1)）。

| CUDA | Triton | 写法 | 能否直接控制 |
|:--|:--|:--|:--|
| thread | 不暴露 | 写不到 `threadIdx` | 不能 |
| warp | 只给数量 | 启动时传 `num_warps=N`，默认 4 | 只能调数量 |
| block（CTA） | program | `@triton.jit` 函数体就是一个 program 的代码；`tl.program_id(axis)` ≈ `blockIdx`，`tl.num_programs(axis)` ≈ `gridDim` | 主要的编程层 |
| grid | grid | `kernel[grid](...)`，例如 `grid = (triton.cdiv(n, BLOCK),)` | 自己定 |
| shared memory | 由编译器分配 | 没有对应语句 | 不能 |
{#tab-2-1 caption="**表 2-1** CUDA 概念在 Triton 里的对应"}

下面四个例子由浅入深。grid 怎么切，看一个输出依赖哪些输入；program 里要不要循环，看这些输入能不能一次装进 register，装不下就要一个 block 一个 block 地读，用一个累加器（accumulator）存中间结果：

| op | 类型 | 一个输出依赖 | grid 怎么切 | program 里 |
|:--|:--|:--|:--|:--|
| GELU | 逐元素 | 同位置的一个输入 | 展平成一维，按 BLOCK 切段 | 不循环 |
| softmax | 按行归约 | 整行（max 和 sum） | 一行一个 program | 不循环，整行一次读进来 |
| row sum | 按行归约 | 整行（sum） | 一行一个 program | 沿行循环，累加器存部分和 |
| matmul | block 乘加 | A 的一行、B 的一列 | 二维，C 的一个 block 一个 program | 沿 K 循环，累加器存 C 的一个 block |
{#tab-2-2 caption="**表 2-2** 四个例子的 op 类型"}

代码都在 RTX 5090 上和 PyTorch 对照过。每个例子配一张数据流图：左边是 grid，右边从上到下是计算单元、register、shared memory 和显存（芯片外的 GDDR7）。`tl.load` 和 `tl.store` 在显存和 register 或 shared memory 之间直接搬数据，计算单元只读写 register。中间经过的 L2 和 L1 cache 由硬件自动管理，图里不画。

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
  <text class="s" x="10" y="329">thread 数先满</text>
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
<figcaption><strong>图 2-1</strong> GELU kernel（n = 8388608，fp32）。左边是 grid，数组按 BLOCK = 1024 切段，一段一个 program；下面是一个 SM 同时驻留的 program，12 格是上限（1536 个 thread）。右边是 program 1 的数据流，泳道和箭头的含义见正文。</figcaption>
</figure>

逐元素 op 最简单，每个元素各算各的，program 之间不需要配合。把 n 个元素切成每段 BLOCK 个，一段交给一个 program（[图 2-1](#fig-t-1)）。kernel 里先用 `tl.program_id` 算出自己负责的下标，读进来，算完写回：

```python {role="program" tag="处理第 pid 段的 BLOCK 个元素"}
@triton.jit
def gelu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)               # 第几个 program（≈ CUDA 的 blockIdx.x）
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # 这个 program 负责的 BLOCK 个下标
    mask = offsets < n                        # 最后一个 program 可能越界
    x = tl.load(x_ptr + offsets, mask=mask)   # 从显存读到 register
    # tanh 近似：0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))；tanh(a) = 1 − 2/(e^{2a} + 1)，e^{2a} 溢出成 inf 时仍得 1
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    e = tl.exp(2 * a)
    t = 1 - 2 / (e + 1)                       # tanh(a)
    y = 0.5 * x * (1 + t)
    tl.store(y_ptr + offsets, y, mask=mask)   # 写回显存
```

启动时只要定 grid，也就是 program 的个数，它和 SM 的个数无关。

```python {role="grid" tag="一段一个 program"}
def gelu(x):
    y = torch.empty_like(x)
    n, BLOCK = x.numel(), 1024                # 每个 program 处理 1024 个元素
    grid = (triton.cdiv(n, BLOCK),)           # program 个数，不是 SM 个数
    gelu_kernel[grid](x, y, n, BLOCK=BLOCK)
    return y
```

program 编译后就是 CUDA 的一个 block。一个 SM 能同时驻留几个，看[表 1-1](#tab-1-1) 里哪条上限先用完：GELU 一个 program 4 个 warp，SM 最多驻留 48 个 warp，所以同时放 12 个（[图 2-1](#fig-t-1) 左下），block 上限 24 用不满。

n = 8388608 时 Triton 版 45.2 µs，读写共 64 MiB，有效带宽 1.49e12 B/s，是显存带宽的 83%，和 PyTorch 的 `F.gelu(x, approximate="tanh")`（45.9 µs）持平。

<details class="fold doc aside">
<summary>8192 个 program 怎么分批跑：实测时间线</summary>

全卡一次放得下 170 × 12 = 2040 个 program，8192 个要 5 批，第 5 批只有 32 个。实测的时间线上批次之间却没有分界（[图 2-1a](#fig-t-5)）：一个 program 跑完，它的名额马上换下一个进来，第 2 批在 2.6 µs 就开始了，第 1 批到 15.4 µs 才全部跑完。同一批的 program 都在抢显存带宽，耗时从 4.2 µs 到 11.9 µs 不等（10% 和 90% 分位）。36.5 µs 时最后一个 program 开始，之后名额一个个空出来，在跑的 program 从 1400 个左右降到 0，这段约 10 µs，占整个 kernel 的 22%，叫 tail effect。打点的开销很小：打点版 46.9 µs，不打点 45.2 µs。

program 少而每个都重时尾巴更明显，[2.4 节](#triton-matmul)的 matmul 就是这样（[图 2-4a](#fig-t-7)）。

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
  <g class="m"><title>第 1 批：program 0 到 2039，共 2040 个</title><polygon points="74.0,142.8 76.4,52.0 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.0 95.4,52.0 97.7,52.0 100.1,52.0 102.5,52.1 104.9,52.1 107.2,53.0 109.6,54.7 112.0,57.4 114.4,59.8 116.7,62.5 119.1,70.4 121.5,78.5 123.9,88.3 126.2,99.1 128.6,108.8 131.0,119.8 133.3,129.6 135.7,136.5 138.1,146.1 140.5,154.5 142.8,160.9 145.2,164.0 147.6,168.3 150.0,172.9 152.3,176.0 154.7,176.8 157.1,177.1 159.5,178.4 161.8,178.7 164.2,178.7 166.6,180.0 169.0,182.7 171.3,184.7 173.7,185.6 176.1,187.0 178.5,188.3 180.8,188.8 183.2,189.3 185.6,189.4 187.9,189.6 190.3,190.6 192.7,191.0 195.1,191.5 197.4,191.7 199.8,192.2 202.2,192.9 204.6,193.4 206.9,194.0 209.3,194.5 211.7,195.1 214.1,195.4 216.4,196.0 218.8,196.0 221.2,196.3 223.6,197.3 225.9,197.5 228.3,197.5 230.7,197.5 233.1,198.2 235.4,198.6 237.8,198.7 240.2,199.1 242.5,199.3 244.9,199.6 247.3,199.6 249.7,199.8 252.0,199.9 254.4,200.0 256.8,200.0 259.2,200.0 261.5,200.0 263.9,200.0 266.3,200.0 268.7,200.0 271.0,200.0 273.4,200.0 275.8,200.0 278.2,200.0 280.5,200.0 282.9,200.0 285.3,200.0 287.7,200.0 290.0,200.0 292.4,200.0 294.8,200.0 297.1,200.0 299.5,200.0 301.9,200.0 304.3,200.0 306.6,200.0 309.0,200.0 311.4,200.0 313.8,200.0 316.1,200.0 318.5,200.0 320.9,200.0 323.3,200.0 325.6,200.0 328.0,200.0 330.4,200.0 332.8,200.0 335.1,200.0 337.5,200.0 339.9,200.0 342.3,200.0 344.6,200.0 347.0,200.0 349.4,200.0 351.7,200.0 354.1,200.0 356.5,200.0 358.9,200.0 361.2,200.0 363.6,200.0 366.0,200.0 368.4,200.0 370.7,200.0 373.1,200.0 375.5,200.0 377.9,200.0 380.2,200.0 382.6,200.0 385.0,200.0 387.4,200.0 389.7,200.0 392.1,200.0 394.5,200.0 396.9,200.0 399.2,200.0 401.6,200.0 404.0,200.0 406.3,200.0 408.7,200.0 411.1,200.0 413.5,200.0 415.8,200.0 418.2,200.0 420.6,200.0 423.0,200.0 425.3,200.0 427.7,200.0 430.1,200.0 432.5,200.0 434.8,200.0 437.2,200.0 439.6,200.0 442.0,200.0 444.3,200.0 446.7,200.0 449.1,200.0 451.5,200.0 453.8,200.0 456.2,200.0 458.6,200.0 460.9,200.0 463.3,200.0 465.7,200.0 468.1,200.0 470.4,200.0 472.8,200.0 475.2,200.0 477.6,200.0 479.9,200.0 482.3,200.0 484.7,200.0 487.1,200.0 489.4,200.0 491.8,200.0 494.2,200.0 496.6,200.0 498.9,200.0 501.3,200.0 503.7,200.0 506.1,200.0 508.4,200.0 510.8,200.0 513.2,200.0 515.5,200.0 517.9,200.0 520.3,200.0 522.7,200.0 525.0,200.0 527.4,200.0 529.8,200.0 532.2,200.0 534.5,200.0 536.9,200.0 539.3,200.0 541.7,200.0 544.0,200.0 546.4,200.0 548.8,200.0 551.2,200.0 553.5,200.0 555.9,200.0 558.3,200.0 560.7,200.0 563.0,200.0 565.4,200.0 567.8,200.0 570.1,200.0 572.5,200.0 574.9,200.0 577.3,200.0 579.6,200.0 582.0,200.0 584.4,200.0 586.8,200.0 589.1,200.0 591.5,200.0 593.9,200.0 596.3,200.0 598.6,200.0 601.0,200.0 603.4,200.0 605.8,200.0 608.1,200.0 610.5,200.0 612.9,200.0 615.3,200.0 617.6,200.0 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,200.0 563.0,200.0 560.7,200.0 558.3,200.0 555.9,200.0 553.5,200.0 551.2,200.0 548.8,200.0 546.4,200.0 544.0,200.0 541.7,200.0 539.3,200.0 536.9,200.0 534.5,200.0 532.2,200.0 529.8,200.0 527.4,200.0 525.0,200.0 522.7,200.0 520.3,200.0 517.9,200.0 515.5,200.0 513.2,200.0 510.8,200.0 508.4,200.0 506.1,200.0 503.7,200.0 501.3,200.0 498.9,200.0 496.6,200.0 494.2,200.0 491.8,200.0 489.4,200.0 487.1,200.0 484.7,200.0 482.3,200.0 479.9,200.0 477.6,200.0 475.2,200.0 472.8,200.0 470.4,200.0 468.1,200.0 465.7,200.0 463.3,200.0 460.9,200.0 458.6,200.0 456.2,200.0 453.8,200.0 451.5,200.0 449.1,200.0 446.7,200.0 444.3,200.0 442.0,200.0 439.6,200.0 437.2,200.0 434.8,200.0 432.5,200.0 430.1,200.0 427.7,200.0 425.3,200.0 423.0,200.0 420.6,200.0 418.2,200.0 415.8,200.0 413.5,200.0 411.1,200.0 408.7,200.0 406.3,200.0 404.0,200.0 401.6,200.0 399.2,200.0 396.9,200.0 394.5,200.0 392.1,200.0 389.7,200.0 387.4,200.0 385.0,200.0 382.6,200.0 380.2,200.0 377.9,200.0 375.5,200.0 373.1,200.0 370.7,200.0 368.4,200.0 366.0,200.0 363.6,200.0 361.2,200.0 358.9,200.0 356.5,200.0 354.1,200.0 351.7,200.0 349.4,200.0 347.0,200.0 344.6,200.0 342.3,200.0 339.9,200.0 337.5,200.0 335.1,200.0 332.8,200.0 330.4,200.0 328.0,200.0 325.6,200.0 323.3,200.0 320.9,200.0 318.5,200.0 316.1,200.0 313.8,200.0 311.4,200.0 309.0,200.0 306.6,200.0 304.3,200.0 301.9,200.0 299.5,200.0 297.1,200.0 294.8,200.0 292.4,200.0 290.0,200.0 287.7,200.0 285.3,200.0 282.9,200.0 280.5,200.0 278.2,200.0 275.8,200.0 273.4,200.0 271.0,200.0 268.7,200.0 266.3,200.0 263.9,200.0 261.5,200.0 259.2,200.0 256.8,200.0 254.4,200.0 252.0,200.0 249.7,200.0 247.3,200.0 244.9,200.0 242.5,200.0 240.2,200.0 237.8,200.0 235.4,200.0 233.1,200.0 230.7,200.0 228.3,200.0 225.9,200.0 223.6,200.0 221.2,200.0 218.8,200.0 216.4,200.0 214.1,200.0 211.7,200.0 209.3,200.0 206.9,200.0 204.6,200.0 202.2,200.0 199.8,200.0 197.4,200.0 195.1,200.0 192.7,200.0 190.3,200.0 187.9,200.0 185.6,200.0 183.2,200.0 180.8,200.0 178.5,200.0 176.1,200.0 173.7,200.0 171.3,200.0 169.0,200.0 166.6,200.0 164.2,200.0 161.8,200.0 159.5,200.0 157.1,200.0 154.7,200.0 152.3,200.0 150.0,200.0 147.6,200.0 145.2,200.0 142.8,200.0 140.5,200.0 138.1,200.0 135.7,200.0 133.3,200.0 131.0,200.0 128.6,200.0 126.2,200.0 123.9,200.0 121.5,200.0 119.1,200.0 116.7,200.0 114.4,200.0 112.0,200.0 109.6,200.0 107.2,200.0 104.9,200.0 102.5,200.0 100.1,200.0 97.7,200.0 95.4,200.0 93.0,200.0 90.6,200.0 88.2,200.0 85.9,200.0 83.5,200.0 81.1,200.0 78.7,200.0 76.4,200.0 74.0,200.0" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>第 2 批：program 2040 到 4079，共 2040 个</title><polygon points="74.0,142.8 76.4,52.0 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.0 95.4,52.0 97.7,52.0 100.1,52.0 102.5,52.1 104.9,52.1 107.2,52.9 109.6,54.5 112.0,56.8 114.4,58.5 116.7,60.1 119.1,66.0 121.5,68.6 123.9,71.6 126.2,73.8 128.6,76.9 131.0,81.2 133.3,81.9 135.7,82.2 138.1,86.5 140.5,89.4 142.8,90.7 145.2,89.4 147.6,91.8 150.0,93.9 152.3,94.2 154.7,94.6 157.1,93.9 159.5,95.9 161.8,95.6 164.2,95.3 166.6,97.0 169.0,98.6 171.3,100.0 173.7,97.9 176.1,99.4 178.5,96.6 180.8,94.1 183.2,94.1 185.6,92.1 187.9,92.7 190.3,92.1 192.7,90.3 195.1,89.8 197.4,89.8 199.8,89.8 202.2,90.0 204.6,89.7 206.9,88.7 209.3,88.8 211.7,90.2 214.1,91.5 216.4,93.3 218.8,93.6 221.2,94.7 223.6,97.6 225.9,98.6 228.3,99.2 230.7,100.0 233.1,102.9 235.4,105.0 237.8,106.3 240.2,108.4 242.5,111.3 244.9,113.3 247.3,115.6 249.7,117.5 252.0,120.7 254.4,123.9 256.8,125.1 259.2,126.5 261.5,128.4 263.9,129.3 266.3,130.6 268.7,131.5 271.0,133.9 273.4,136.8 275.8,139.9 278.2,142.9 280.5,144.6 282.9,145.2 285.3,147.5 287.7,148.6 290.0,150.7 292.4,151.8 294.8,153.1 297.1,154.7 299.5,156.5 301.9,157.4 304.3,159.4 306.6,161.8 309.0,165.3 311.4,166.6 313.8,167.9 316.1,171.5 318.5,173.2 320.9,174.0 323.3,176.4 325.6,177.0 328.0,177.5 330.4,178.3 332.8,179.0 335.1,179.4 337.5,181.1 339.9,182.5 342.3,185.5 344.6,188.5 347.0,190.5 349.4,191.4 351.7,192.1 354.1,192.6 356.5,193.0 358.9,193.5 361.2,194.5 363.6,195.2 366.0,196.2 368.4,196.4 370.7,197.0 373.1,197.3 375.5,197.4 377.9,197.8 380.2,198.3 382.6,198.5 385.0,198.9 387.4,199.3 389.7,199.4 392.1,199.6 394.5,199.6 396.9,199.8 399.2,199.9 401.6,199.9 404.0,199.9 406.3,199.9 408.7,200.0 411.1,200.0 413.5,200.0 415.8,200.0 418.2,200.0 420.6,200.0 423.0,200.0 425.3,200.0 427.7,200.0 430.1,200.0 432.5,200.0 434.8,200.0 437.2,200.0 439.6,200.0 442.0,200.0 444.3,200.0 446.7,200.0 449.1,200.0 451.5,200.0 453.8,200.0 456.2,200.0 458.6,200.0 460.9,200.0 463.3,200.0 465.7,200.0 468.1,200.0 470.4,200.0 472.8,200.0 475.2,200.0 477.6,200.0 479.9,200.0 482.3,200.0 484.7,200.0 487.1,200.0 489.4,200.0 491.8,200.0 494.2,200.0 496.6,200.0 498.9,200.0 501.3,200.0 503.7,200.0 506.1,200.0 508.4,200.0 510.8,200.0 513.2,200.0 515.5,200.0 517.9,200.0 520.3,200.0 522.7,200.0 525.0,200.0 527.4,200.0 529.8,200.0 532.2,200.0 534.5,200.0 536.9,200.0 539.3,200.0 541.7,200.0 544.0,200.0 546.4,200.0 548.8,200.0 551.2,200.0 553.5,200.0 555.9,200.0 558.3,200.0 560.7,200.0 563.0,200.0 565.4,200.0 567.8,200.0 570.1,200.0 572.5,200.0 574.9,200.0 577.3,200.0 579.6,200.0 582.0,200.0 584.4,200.0 586.8,200.0 589.1,200.0 591.5,200.0 593.9,200.0 596.3,200.0 598.6,200.0 601.0,200.0 603.4,200.0 605.8,200.0 608.1,200.0 610.5,200.0 612.9,200.0 615.3,200.0 617.6,200.0 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,200.0 563.0,200.0 560.7,200.0 558.3,200.0 555.9,200.0 553.5,200.0 551.2,200.0 548.8,200.0 546.4,200.0 544.0,200.0 541.7,200.0 539.3,200.0 536.9,200.0 534.5,200.0 532.2,200.0 529.8,200.0 527.4,200.0 525.0,200.0 522.7,200.0 520.3,200.0 517.9,200.0 515.5,200.0 513.2,200.0 510.8,200.0 508.4,200.0 506.1,200.0 503.7,200.0 501.3,200.0 498.9,200.0 496.6,200.0 494.2,200.0 491.8,200.0 489.4,200.0 487.1,200.0 484.7,200.0 482.3,200.0 479.9,200.0 477.6,200.0 475.2,200.0 472.8,200.0 470.4,200.0 468.1,200.0 465.7,200.0 463.3,200.0 460.9,200.0 458.6,200.0 456.2,200.0 453.8,200.0 451.5,200.0 449.1,200.0 446.7,200.0 444.3,200.0 442.0,200.0 439.6,200.0 437.2,200.0 434.8,200.0 432.5,200.0 430.1,200.0 427.7,200.0 425.3,200.0 423.0,200.0 420.6,200.0 418.2,200.0 415.8,200.0 413.5,200.0 411.1,200.0 408.7,200.0 406.3,200.0 404.0,200.0 401.6,200.0 399.2,200.0 396.9,200.0 394.5,200.0 392.1,200.0 389.7,200.0 387.4,200.0 385.0,200.0 382.6,200.0 380.2,200.0 377.9,200.0 375.5,200.0 373.1,200.0 370.7,200.0 368.4,200.0 366.0,200.0 363.6,200.0 361.2,200.0 358.9,200.0 356.5,200.0 354.1,200.0 351.7,200.0 349.4,200.0 347.0,200.0 344.6,200.0 342.3,200.0 339.9,200.0 337.5,200.0 335.1,200.0 332.8,200.0 330.4,200.0 328.0,200.0 325.6,200.0 323.3,200.0 320.9,200.0 318.5,200.0 316.1,200.0 313.8,200.0 311.4,200.0 309.0,200.0 306.6,200.0 304.3,200.0 301.9,200.0 299.5,200.0 297.1,200.0 294.8,200.0 292.4,200.0 290.0,200.0 287.7,200.0 285.3,200.0 282.9,200.0 280.5,200.0 278.2,200.0 275.8,200.0 273.4,200.0 271.0,200.0 268.7,200.0 266.3,200.0 263.9,200.0 261.5,200.0 259.2,200.0 256.8,200.0 254.4,200.0 252.0,199.9 249.7,199.8 247.3,199.6 244.9,199.6 242.5,199.3 240.2,199.1 237.8,198.7 235.4,198.6 233.1,198.2 230.7,197.5 228.3,197.5 225.9,197.5 223.6,197.3 221.2,196.3 218.8,196.0 216.4,196.0 214.1,195.4 211.7,195.1 209.3,194.5 206.9,194.0 204.6,193.4 202.2,192.9 199.8,192.2 197.4,191.7 195.1,191.5 192.7,191.0 190.3,190.6 187.9,189.6 185.6,189.4 183.2,189.3 180.8,188.8 178.5,188.3 176.1,187.0 173.7,185.6 171.3,184.7 169.0,182.7 166.6,180.0 164.2,178.7 161.8,178.7 159.5,178.4 157.1,177.1 154.7,176.8 152.3,176.0 150.0,172.9 147.6,168.3 145.2,164.0 142.8,160.9 140.5,154.5 138.1,146.1 135.7,136.5 133.3,129.6 131.0,119.8 128.6,108.8 126.2,99.1 123.9,88.3 121.5,78.5 119.1,70.4 116.7,62.5 114.4,59.8 112.0,57.4 109.6,54.7 107.2,53.0 104.9,52.1 102.5,52.1 100.1,52.0 97.7,52.0 95.4,52.0 93.0,52.0 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,52.0 74.0,142.8" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>第 3 批：program 4080 到 6119，共 2040 个</title><polygon points="74.0,142.8 76.4,52.0 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.0 95.4,52.0 97.7,52.0 100.1,52.0 102.5,52.1 104.9,52.1 107.2,52.9 109.6,54.5 112.0,56.8 114.4,58.5 116.7,60.1 119.1,66.0 121.5,68.6 123.9,71.6 126.2,73.8 128.6,76.9 131.0,81.2 133.3,81.9 135.7,82.2 138.1,86.5 140.5,89.4 142.8,90.7 145.2,89.4 147.6,91.8 150.0,93.9 152.3,94.2 154.7,94.6 157.1,93.9 159.5,95.9 161.8,95.6 164.2,95.3 166.6,97.0 169.0,98.6 171.3,100.0 173.7,97.9 176.1,99.4 178.5,96.6 180.8,94.1 183.2,94.1 185.6,92.1 187.9,92.7 190.3,92.1 192.7,90.3 195.1,89.8 197.4,89.8 199.8,89.8 202.2,90.0 204.6,89.7 206.9,88.7 209.3,88.8 211.7,90.2 214.1,90.2 216.4,89.1 218.8,86.1 221.2,86.8 223.6,87.4 225.9,85.2 228.3,84.9 230.7,85.1 233.1,83.5 235.4,82.7 237.8,81.5 240.2,80.5 242.5,80.1 244.9,78.6 247.3,76.8 249.7,76.7 252.0,76.7 254.4,77.2 256.8,76.3 259.2,77.0 261.5,77.7 263.9,78.3 266.3,79.1 268.7,79.4 271.0,80.9 273.4,81.4 275.8,82.3 278.2,83.6 280.5,83.3 282.9,83.8 285.3,85.6 287.7,85.5 290.0,86.8 292.4,87.0 294.8,87.5 297.1,88.6 299.5,89.4 301.9,90.5 304.3,91.5 306.6,93.4 309.0,95.2 311.4,94.4 313.8,95.0 316.1,96.5 318.5,97.3 320.9,97.2 323.3,98.9 325.6,98.4 328.0,98.1 330.4,99.0 332.8,99.0 335.1,99.4 337.5,100.7 339.9,101.2 342.3,103.0 344.6,102.5 347.0,98.9 349.4,96.6 351.7,97.0 354.1,97.0 356.5,97.3 358.9,96.8 361.2,97.5 363.6,96.3 366.0,97.5 368.4,95.7 370.7,96.6 373.1,96.5 375.5,95.8 377.9,97.9 380.2,102.9 382.6,105.6 385.0,110.5 387.4,114.5 389.7,118.0 392.1,121.4 394.5,123.3 396.9,126.7 399.2,128.8 401.6,131.2 404.0,133.8 406.3,135.2 408.7,137.8 411.1,141.5 413.5,142.8 415.8,144.9 418.2,148.6 420.6,150.6 423.0,152.9 425.3,155.7 427.7,156.9 430.1,159.9 432.5,162.2 434.8,163.8 437.2,166.4 439.6,168.2 442.0,170.2 444.3,170.8 446.7,172.6 449.1,176.1 451.5,179.5 453.8,180.3 456.2,182.6 458.6,183.7 460.9,184.7 463.3,186.6 465.7,187.8 468.1,188.7 470.4,189.7 472.8,190.7 475.2,191.1 477.6,192.0 479.9,192.8 482.3,194.3 484.7,195.0 487.1,195.4 489.4,195.7 491.8,196.0 494.2,196.3 496.6,196.9 498.9,197.3 501.3,197.8 503.7,198.0 506.1,198.5 508.4,198.8 510.8,198.9 513.2,199.1 515.5,199.1 517.9,199.2 520.3,199.3 522.7,199.4 525.0,199.4 527.4,199.4 529.8,199.6 532.2,199.6 534.5,199.6 536.9,199.7 539.3,199.7 541.7,199.7 544.0,199.8 546.4,199.9 548.8,199.9 551.2,199.9 553.5,199.9 555.9,199.9 558.3,199.9 560.7,199.9 563.0,199.9 565.4,199.9 567.8,200.0 570.1,200.0 572.5,200.0 574.9,200.0 577.3,200.0 579.6,200.0 582.0,200.0 584.4,200.0 586.8,200.0 589.1,200.0 591.5,200.0 593.9,200.0 596.3,200.0 598.6,200.0 601.0,200.0 603.4,200.0 605.8,200.0 608.1,200.0 610.5,200.0 612.9,200.0 615.3,200.0 617.6,200.0 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,200.0 563.0,200.0 560.7,200.0 558.3,200.0 555.9,200.0 553.5,200.0 551.2,200.0 548.8,200.0 546.4,200.0 544.0,200.0 541.7,200.0 539.3,200.0 536.9,200.0 534.5,200.0 532.2,200.0 529.8,200.0 527.4,200.0 525.0,200.0 522.7,200.0 520.3,200.0 517.9,200.0 515.5,200.0 513.2,200.0 510.8,200.0 508.4,200.0 506.1,200.0 503.7,200.0 501.3,200.0 498.9,200.0 496.6,200.0 494.2,200.0 491.8,200.0 489.4,200.0 487.1,200.0 484.7,200.0 482.3,200.0 479.9,200.0 477.6,200.0 475.2,200.0 472.8,200.0 470.4,200.0 468.1,200.0 465.7,200.0 463.3,200.0 460.9,200.0 458.6,200.0 456.2,200.0 453.8,200.0 451.5,200.0 449.1,200.0 446.7,200.0 444.3,200.0 442.0,200.0 439.6,200.0 437.2,200.0 434.8,200.0 432.5,200.0 430.1,200.0 427.7,200.0 425.3,200.0 423.0,200.0 420.6,200.0 418.2,200.0 415.8,200.0 413.5,200.0 411.1,200.0 408.7,200.0 406.3,199.9 404.0,199.9 401.6,199.9 399.2,199.9 396.9,199.8 394.5,199.6 392.1,199.6 389.7,199.4 387.4,199.3 385.0,198.9 382.6,198.5 380.2,198.3 377.9,197.8 375.5,197.4 373.1,197.3 370.7,197.0 368.4,196.4 366.0,196.2 363.6,195.2 361.2,194.5 358.9,193.5 356.5,193.0 354.1,192.6 351.7,192.1 349.4,191.4 347.0,190.5 344.6,188.5 342.3,185.5 339.9,182.5 337.5,181.1 335.1,179.4 332.8,179.0 330.4,178.3 328.0,177.5 325.6,177.0 323.3,176.4 320.9,174.0 318.5,173.2 316.1,171.5 313.8,167.9 311.4,166.6 309.0,165.3 306.6,161.8 304.3,159.4 301.9,157.4 299.5,156.5 297.1,154.7 294.8,153.1 292.4,151.8 290.0,150.7 287.7,148.6 285.3,147.5 282.9,145.2 280.5,144.6 278.2,142.9 275.8,139.9 273.4,136.8 271.0,133.9 268.7,131.5 266.3,130.6 263.9,129.3 261.5,128.4 259.2,126.5 256.8,125.1 254.4,123.9 252.0,120.7 249.7,117.5 247.3,115.6 244.9,113.3 242.5,111.3 240.2,108.4 237.8,106.3 235.4,105.0 233.1,102.9 230.7,100.0 228.3,99.2 225.9,98.6 223.6,97.6 221.2,94.7 218.8,93.6 216.4,93.3 214.1,91.5 211.7,90.2 209.3,88.8 206.9,88.7 204.6,89.7 202.2,90.0 199.8,89.8 197.4,89.8 195.1,89.8 192.7,90.3 190.3,92.1 187.9,92.7 185.6,92.1 183.2,94.1 180.8,94.1 178.5,96.6 176.1,99.4 173.7,97.9 171.3,100.0 169.0,98.6 166.6,97.0 164.2,95.3 161.8,95.6 159.5,95.9 157.1,93.9 154.7,94.6 152.3,94.2 150.0,93.9 147.6,91.8 145.2,89.4 142.8,90.7 140.5,89.4 138.1,86.5 135.7,82.2 133.3,81.9 131.0,81.2 128.6,76.9 126.2,73.8 123.9,71.6 121.5,68.6 119.1,66.0 116.7,60.1 114.4,58.5 112.0,56.8 109.6,54.5 107.2,52.9 104.9,52.1 102.5,52.1 100.1,52.0 97.7,52.0 95.4,52.0 93.0,52.0 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,52.0 74.0,142.8" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>第 4 批：program 6120 到 8159，共 2040 个</title><polygon points="74.0,142.8 76.4,52.0 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.0 95.4,52.0 97.7,52.0 100.1,52.0 102.5,52.1 104.9,52.1 107.2,52.9 109.6,54.5 112.0,56.8 114.4,58.5 116.7,60.1 119.1,66.0 121.5,68.6 123.9,71.6 126.2,73.8 128.6,76.9 131.0,81.2 133.3,81.9 135.7,82.2 138.1,86.5 140.5,89.4 142.8,90.7 145.2,89.4 147.6,91.8 150.0,93.9 152.3,94.2 154.7,94.6 157.1,93.9 159.5,95.9 161.8,95.6 164.2,95.3 166.6,97.0 169.0,98.6 171.3,100.0 173.7,97.9 176.1,99.4 178.5,96.6 180.8,94.1 183.2,94.1 185.6,92.1 187.9,92.7 190.3,92.1 192.7,90.3 195.1,89.8 197.4,89.8 199.8,89.8 202.2,90.0 204.6,89.7 206.9,88.7 209.3,88.8 211.7,90.2 214.1,90.2 216.4,89.1 218.8,86.1 221.2,86.8 223.6,87.4 225.9,85.2 228.3,84.9 230.7,85.1 233.1,83.5 235.4,82.7 237.8,81.5 240.2,80.5 242.5,80.1 244.9,78.6 247.3,76.8 249.7,76.7 252.0,76.7 254.4,77.2 256.8,76.3 259.2,77.0 261.5,77.7 263.9,78.3 266.3,79.1 268.7,79.4 271.0,80.9 273.4,81.4 275.8,82.3 278.2,83.6 280.5,83.3 282.9,83.8 285.3,85.6 287.7,85.5 290.0,86.8 292.4,87.0 294.8,87.5 297.1,88.6 299.5,89.4 301.9,90.5 304.3,91.5 306.6,93.4 309.0,95.2 311.4,94.4 313.8,95.0 316.1,96.5 318.5,97.3 320.9,97.2 323.3,98.9 325.6,98.4 328.0,98.1 330.4,99.0 332.8,99.0 335.1,99.4 337.5,100.7 339.9,101.2 342.3,103.0 344.6,102.5 347.0,98.9 349.4,96.6 351.7,97.0 354.1,97.0 356.5,97.3 358.9,96.8 361.2,97.5 363.6,96.3 366.0,97.5 368.4,95.7 370.7,96.6 373.1,96.5 375.5,95.8 377.9,97.7 380.2,97.7 382.6,95.4 385.0,96.4 387.4,92.1 389.7,91.4 392.1,90.4 394.5,90.5 396.9,90.8 399.2,89.2 401.6,88.3 404.0,86.7 406.3,85.3 408.7,86.5 411.1,86.5 413.5,85.7 415.8,87.3 418.2,87.9 420.6,87.2 423.0,87.2 425.3,88.1 427.7,88.2 430.1,89.8 432.5,87.6 434.8,87.6 437.2,89.2 439.6,89.4 442.0,88.9 444.3,87.7 446.7,87.7 449.1,89.9 451.5,90.5 453.8,89.0 456.2,89.0 458.6,88.9 460.9,87.8 463.3,89.9 465.7,90.4 468.1,89.7 470.4,90.2 472.8,91.2 475.2,92.0 477.6,93.3 479.9,93.1 482.3,95.4 484.7,95.5 487.1,94.7 489.4,95.2 491.8,95.7 494.2,96.0 496.6,98.2 498.9,99.0 501.3,101.9 503.7,104.7 506.1,110.3 508.4,113.4 510.8,116.4 513.2,118.7 515.5,121.2 517.9,123.5 520.3,127.5 522.7,129.7 525.0,132.8 527.4,135.6 529.8,137.4 532.2,141.8 534.5,145.6 536.9,147.6 539.3,151.7 541.7,153.6 544.0,157.3 546.4,159.7 548.8,162.1 551.2,163.2 553.5,163.9 555.9,167.1 558.3,167.7 560.7,168.4 563.0,169.3 565.4,170.9 567.8,173.4 570.1,177.2 572.5,180.6 574.9,181.5 577.3,183.0 579.6,183.8 582.0,185.2 584.4,187.1 586.8,187.6 589.1,188.7 591.5,189.3 593.9,190.7 596.3,192.0 598.6,195.2 601.0,197.1 603.4,197.9 605.8,198.4 608.1,198.9 610.5,199.0 612.9,199.3 615.3,199.3 617.6,199.6 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,199.9 563.0,199.9 560.7,199.9 558.3,199.9 555.9,199.9 553.5,199.9 551.2,199.9 548.8,199.9 546.4,199.9 544.0,199.8 541.7,199.7 539.3,199.7 536.9,199.7 534.5,199.6 532.2,199.6 529.8,199.6 527.4,199.4 525.0,199.4 522.7,199.4 520.3,199.3 517.9,199.2 515.5,199.1 513.2,199.1 510.8,198.9 508.4,198.8 506.1,198.5 503.7,198.0 501.3,197.8 498.9,197.3 496.6,196.9 494.2,196.3 491.8,196.0 489.4,195.7 487.1,195.4 484.7,195.0 482.3,194.3 479.9,192.8 477.6,192.0 475.2,191.1 472.8,190.7 470.4,189.7 468.1,188.7 465.7,187.8 463.3,186.6 460.9,184.7 458.6,183.7 456.2,182.6 453.8,180.3 451.5,179.5 449.1,176.1 446.7,172.6 444.3,170.8 442.0,170.2 439.6,168.2 437.2,166.4 434.8,163.8 432.5,162.2 430.1,159.9 427.7,156.9 425.3,155.7 423.0,152.9 420.6,150.6 418.2,148.6 415.8,144.9 413.5,142.8 411.1,141.5 408.7,137.8 406.3,135.2 404.0,133.8 401.6,131.2 399.2,128.8 396.9,126.7 394.5,123.3 392.1,121.4 389.7,118.0 387.4,114.5 385.0,110.5 382.6,105.6 380.2,102.9 377.9,97.9 375.5,95.8 373.1,96.5 370.7,96.6 368.4,95.7 366.0,97.5 363.6,96.3 361.2,97.5 358.9,96.8 356.5,97.3 354.1,97.0 351.7,97.0 349.4,96.6 347.0,98.9 344.6,102.5 342.3,103.0 339.9,101.2 337.5,100.7 335.1,99.4 332.8,99.0 330.4,99.0 328.0,98.1 325.6,98.4 323.3,98.9 320.9,97.2 318.5,97.3 316.1,96.5 313.8,95.0 311.4,94.4 309.0,95.2 306.6,93.4 304.3,91.5 301.9,90.5 299.5,89.4 297.1,88.6 294.8,87.5 292.4,87.0 290.0,86.8 287.7,85.5 285.3,85.6 282.9,83.8 280.5,83.3 278.2,83.6 275.8,82.3 273.4,81.4 271.0,80.9 268.7,79.4 266.3,79.1 263.9,78.3 261.5,77.7 259.2,77.0 256.8,76.3 254.4,77.2 252.0,76.7 249.7,76.7 247.3,76.8 244.9,78.6 242.5,80.1 240.2,80.5 237.8,81.5 235.4,82.7 233.1,83.5 230.7,85.1 228.3,84.9 225.9,85.2 223.6,87.4 221.2,86.8 218.8,86.1 216.4,89.1 214.1,90.2 211.7,90.2 209.3,88.8 206.9,88.7 204.6,89.7 202.2,90.0 199.8,89.8 197.4,89.8 195.1,89.8 192.7,90.3 190.3,92.1 187.9,92.7 185.6,92.1 183.2,94.1 180.8,94.1 178.5,96.6 176.1,99.4 173.7,97.9 171.3,100.0 169.0,98.6 166.6,97.0 164.2,95.3 161.8,95.6 159.5,95.9 157.1,93.9 154.7,94.6 152.3,94.2 150.0,93.9 147.6,91.8 145.2,89.4 142.8,90.7 140.5,89.4 138.1,86.5 135.7,82.2 133.3,81.9 131.0,81.2 128.6,76.9 126.2,73.8 123.9,71.6 121.5,68.6 119.1,66.0 116.7,60.1 114.4,58.5 112.0,56.8 109.6,54.5 107.2,52.9 104.9,52.1 102.5,52.1 100.1,52.0 97.7,52.0 95.4,52.0 93.0,52.0 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,52.0 74.0,142.8" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>第 5 批：program 8160 到 8191，共 32 个</title><polygon points="74.0,142.8 76.4,52.0 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.0 95.4,52.0 97.7,52.0 100.1,52.0 102.5,52.1 104.9,52.1 107.2,52.9 109.6,54.5 112.0,56.8 114.4,58.5 116.7,60.1 119.1,66.0 121.5,68.6 123.9,71.6 126.2,73.8 128.6,76.9 131.0,81.2 133.3,81.9 135.7,82.2 138.1,86.5 140.5,89.4 142.8,90.7 145.2,89.4 147.6,91.8 150.0,93.9 152.3,94.2 154.7,94.6 157.1,93.9 159.5,95.9 161.8,95.6 164.2,95.3 166.6,97.0 169.0,98.6 171.3,100.0 173.7,97.9 176.1,99.4 178.5,96.6 180.8,94.1 183.2,94.1 185.6,92.1 187.9,92.7 190.3,92.1 192.7,90.3 195.1,89.8 197.4,89.8 199.8,89.8 202.2,90.0 204.6,89.7 206.9,88.7 209.3,88.8 211.7,90.2 214.1,90.2 216.4,89.1 218.8,86.1 221.2,86.8 223.6,87.4 225.9,85.2 228.3,84.9 230.7,85.1 233.1,83.5 235.4,82.7 237.8,81.5 240.2,80.5 242.5,80.1 244.9,78.6 247.3,76.8 249.7,76.7 252.0,76.7 254.4,77.2 256.8,76.3 259.2,77.0 261.5,77.7 263.9,78.3 266.3,79.1 268.7,79.4 271.0,80.9 273.4,81.4 275.8,82.3 278.2,83.6 280.5,83.3 282.9,83.8 285.3,85.6 287.7,85.5 290.0,86.8 292.4,87.0 294.8,87.5 297.1,88.6 299.5,89.4 301.9,90.5 304.3,91.5 306.6,93.4 309.0,95.2 311.4,94.4 313.8,95.0 316.1,96.5 318.5,97.3 320.9,97.2 323.3,98.9 325.6,98.4 328.0,98.1 330.4,99.0 332.8,99.0 335.1,99.4 337.5,100.7 339.9,101.2 342.3,103.0 344.6,102.5 347.0,98.9 349.4,96.6 351.7,97.0 354.1,97.0 356.5,97.3 358.9,96.8 361.2,97.5 363.6,96.3 366.0,97.5 368.4,95.7 370.7,96.6 373.1,96.5 375.5,95.8 377.9,97.7 380.2,97.7 382.6,95.4 385.0,96.4 387.4,92.1 389.7,91.4 392.1,90.4 394.5,90.5 396.9,90.8 399.2,89.2 401.6,88.3 404.0,86.7 406.3,85.3 408.7,86.5 411.1,86.5 413.5,85.7 415.8,87.3 418.2,87.9 420.6,87.2 423.0,87.2 425.3,88.1 427.7,88.2 430.1,89.8 432.5,87.6 434.8,87.6 437.2,89.2 439.6,89.4 442.0,88.9 444.3,87.7 446.7,87.7 449.1,89.9 451.5,90.5 453.8,89.0 456.2,89.0 458.6,88.9 460.9,87.8 463.3,89.9 465.7,90.4 468.1,89.7 470.4,90.2 472.8,91.2 475.2,92.0 477.6,93.3 479.9,93.1 482.3,95.4 484.7,95.5 487.1,94.7 489.4,95.2 491.8,95.7 494.2,96.0 496.6,98.2 498.9,97.1 501.3,99.6 503.7,102.3 506.1,108.0 508.4,111.1 510.8,114.0 513.2,116.4 515.5,118.9 517.9,121.1 520.3,125.2 522.7,127.4 525.0,130.5 527.4,133.3 529.8,135.1 532.2,139.5 534.5,143.3 536.9,145.3 539.3,149.4 541.7,151.3 544.0,155.0 546.4,157.5 548.8,160.0 551.2,161.0 553.5,161.7 555.9,165.0 558.3,165.5 560.7,166.2 563.0,167.1 565.4,168.7 567.8,171.4 570.1,175.4 572.5,178.8 574.9,179.9 577.3,181.5 579.6,182.5 582.0,184.0 584.4,185.9 586.8,186.5 589.1,187.6 591.5,188.3 593.9,189.8 596.3,191.2 598.6,194.5 601.0,196.7 603.4,197.7 605.8,198.3 608.1,198.8 610.5,198.8 612.9,199.1 615.3,199.2 617.6,199.6 620.0,200.0 620.0,200.0 617.6,199.6 615.3,199.3 612.9,199.3 610.5,199.0 608.1,198.9 605.8,198.4 603.4,197.9 601.0,197.1 598.6,195.2 596.3,192.0 593.9,190.7 591.5,189.3 589.1,188.7 586.8,187.6 584.4,187.1 582.0,185.2 579.6,183.8 577.3,183.0 574.9,181.5 572.5,180.6 570.1,177.2 567.8,173.4 565.4,170.9 563.0,169.3 560.7,168.4 558.3,167.7 555.9,167.1 553.5,163.9 551.2,163.2 548.8,162.1 546.4,159.7 544.0,157.3 541.7,153.6 539.3,151.7 536.9,147.6 534.5,145.6 532.2,141.8 529.8,137.4 527.4,135.6 525.0,132.8 522.7,129.7 520.3,127.5 517.9,123.5 515.5,121.2 513.2,118.7 510.8,116.4 508.4,113.4 506.1,110.3 503.7,104.7 501.3,101.9 498.9,99.0 496.6,98.2 494.2,96.0 491.8,95.7 489.4,95.2 487.1,94.7 484.7,95.5 482.3,95.4 479.9,93.1 477.6,93.3 475.2,92.0 472.8,91.2 470.4,90.2 468.1,89.7 465.7,90.4 463.3,89.9 460.9,87.8 458.6,88.9 456.2,89.0 453.8,89.0 451.5,90.5 449.1,89.9 446.7,87.7 444.3,87.7 442.0,88.9 439.6,89.4 437.2,89.2 434.8,87.6 432.5,87.6 430.1,89.8 427.7,88.2 425.3,88.1 423.0,87.2 420.6,87.2 418.2,87.9 415.8,87.3 413.5,85.7 411.1,86.5 408.7,86.5 406.3,85.3 404.0,86.7 401.6,88.3 399.2,89.2 396.9,90.8 394.5,90.5 392.1,90.4 389.7,91.4 387.4,92.1 385.0,96.4 382.6,95.4 380.2,97.7 377.9,97.7 375.5,95.8 373.1,96.5 370.7,96.6 368.4,95.7 366.0,97.5 363.6,96.3 361.2,97.5 358.9,96.8 356.5,97.3 354.1,97.0 351.7,97.0 349.4,96.6 347.0,98.9 344.6,102.5 342.3,103.0 339.9,101.2 337.5,100.7 335.1,99.4 332.8,99.0 330.4,99.0 328.0,98.1 325.6,98.4 323.3,98.9 320.9,97.2 318.5,97.3 316.1,96.5 313.8,95.0 311.4,94.4 309.0,95.2 306.6,93.4 304.3,91.5 301.9,90.5 299.5,89.4 297.1,88.6 294.8,87.5 292.4,87.0 290.0,86.8 287.7,85.5 285.3,85.6 282.9,83.8 280.5,83.3 278.2,83.6 275.8,82.3 273.4,81.4 271.0,80.9 268.7,79.4 266.3,79.1 263.9,78.3 261.5,77.7 259.2,77.0 256.8,76.3 254.4,77.2 252.0,76.7 249.7,76.7 247.3,76.8 244.9,78.6 242.5,80.1 240.2,80.5 237.8,81.5 235.4,82.7 233.1,83.5 230.7,85.1 228.3,84.9 225.9,85.2 223.6,87.4 221.2,86.8 218.8,86.1 216.4,89.1 214.1,90.2 211.7,90.2 209.3,88.8 206.9,88.7 204.6,89.7 202.2,90.0 199.8,89.8 197.4,89.8 195.1,89.8 192.7,90.3 190.3,92.1 187.9,92.7 185.6,92.1 183.2,94.1 180.8,94.1 178.5,96.6 176.1,99.4 173.7,97.9 171.3,100.0 169.0,98.6 166.6,97.0 164.2,95.3 161.8,95.6 159.5,95.9 157.1,93.9 154.7,94.6 152.3,94.2 150.0,93.9 147.6,91.8 145.2,89.4 142.8,90.7 140.5,89.4 138.1,86.5 135.7,82.2 133.3,81.9 131.0,81.2 128.6,76.9 126.2,73.8 123.9,71.6 121.5,68.6 119.1,66.0 116.7,60.1 114.4,58.5 112.0,56.8 109.6,54.5 107.2,52.9 104.9,52.1 102.5,52.1 100.1,52.0 97.7,52.0 95.4,52.0 93.0,52.0 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,52.0 74.0,142.8" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <text class="lab2" x="74" y="234">SM 52 的 12 个名额，一条是一个 program 从第一条指令到最后一条 store 发出；这个 SM 一共跑了 49 个</text>
  <text class="tick" x="66" y="254" text-anchor="end">0</text>
  <g class="m"><title>program 62（第 1 批）：0.00 到 3.14 µs</title><rect x="74.0" y="247" width="35.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2059（第 2 批）：3.49 到 6.69 µs</title><rect x="114.6" y="247" width="36.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4138（第 3 批）：12.26 到 20.61 µs</title><rect x="216.6" y="247" width="96.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5612（第 3 批）：23.33 到 34.94 µs</title><rect x="345.5" y="247" width="134.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 402（第 1 批）：0.00 到 3.36 µs</title><rect x="74.0" y="256" width="38.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2068（第 2 批）：3.62 到 12.61 µs</title><rect x="116.1" y="256" width="104.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4450（第 3 批）：14.21 到 23.87 µs</title><rect x="239.4" y="256" width="111.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 6448（第 4 批）：26.98 到 36.99 µs</title><rect x="388.0" y="256" width="116.0" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 572（第 1 批）：0.00 到 4.00 µs</title><rect x="74.0" y="265" width="46.0" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2301（第 2 批）：4.35 到 11.84 µs</title><rect x="124.7" y="265" width="86.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4452（第 3 批）：14.21 到 23.17 µs</title><rect x="239.4" y="265" width="103.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 6203（第 4 批）：26.34 到 34.08 µs</title><rect x="380.5" y="265" width="89.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 8092（第 4 批）：35.97 到 44.93 µs</title><rect x="492.6" y="265" width="103.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <text class="tick" x="66" y="281" text-anchor="end">3</text>
  <g class="m"><title>program 232（第 1 批）：0.00 到 4.48 µs</title><rect x="74.0" y="274" width="51.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3145（第 2 批）：6.53 到 20.77 µs</title><rect x="150.0" y="274" width="165.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 5793（第 3 批）：23.87 到 34.11 µs</title><rect x="351.8" y="274" width="118.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 8098（第 4 批）：36.00 到 40.96 µs</title><rect x="493.0" y="274" width="57.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 742（第 1 批）：0.00 到 4.51 µs</title><rect x="74.0" y="283" width="51.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3147（第 2 批）：6.56 到 15.65 µs</title><rect x="150.4" y="283" width="105.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 5135（第 3 批）：20.32 到 34.75 µs</title><rect x="310.5" y="283" width="167.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 8170（第 5 批）：36.45 到 42.50 µs</title><rect x="498.2" y="283" width="69.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <g class="m"><title>program 1082（第 1 批）：0.06 到 4.19 µs</title><rect x="74.7" y="292" width="47.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2607（第 2 批）：4.96 到 14.53 µs</title><rect x="131.7" y="292" width="110.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4979（第 3 批）：18.46 到 29.41 µs</title><rect x="288.9" y="292" width="126.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7914（第 4 批）：34.88 到 42.08 µs</title><rect x="480.0" y="292" width="83.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <text class="tick" x="66" y="308" text-anchor="end">6</text>
  <g class="m"><title>program 1422（第 1 批）：0.06 到 4.26 µs</title><rect x="74.7" y="301" width="48.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2963（第 2 批）：5.76 到 13.76 µs</title><rect x="141.0" y="301" width="92.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4943（第 3 批）：18.02 到 24.99 µs</title><rect x="283.7" y="301" width="80.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 6461（第 4 批）：27.01 到 36.35 µs</title><rect x="388.3" y="301" width="108.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 912（第 1 批）：0.06 到 4.80 µs</title><rect x="74.7" y="310" width="54.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3195（第 2 批）：6.78 到 17.92 µs</title><rect x="153.0" y="310" width="129.0" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 5604（第 3 批）：23.33 到 29.50 µs</title><rect x="345.5" y="310" width="71.3" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7949（第 4 批）：35.07 到 42.05 µs</title><rect x="482.2" y="310" width="80.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 1252（第 1 批）：0.06 到 5.89 µs</title><rect x="74.7" y="319" width="67.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3342（第 2 批）：8.35 到 17.57 µs</title><rect x="171.2" y="319" width="106.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 5585（第 3 批）：23.30 到 28.54 µs</title><rect x="345.1" y="319" width="60.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7154（第 4 批）：30.78 到 39.68 µs</title><rect x="432.3" y="319" width="102.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <text class="tick" x="66" y="335" text-anchor="end">9</text>
  <g class="m"><title>program 1592（第 1 批）：0.06 到 6.62 µs</title><rect x="74.7" y="328" width="75.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 4134（第 3 批）：12.22 到 23.01 µs</title><rect x="216.3" y="328" width="124.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5857（第 3 批）：24.35 到 34.72 µs</title><rect x="357.4" y="328" width="120.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 8160（第 5 批）：36.42 到 44.96 µs</title><rect x="497.8" y="328" width="98.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <g class="m"><title>program 1762（第 1 批）：0.10 到 6.37 µs</title><rect x="75.1" y="337" width="72.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 4136（第 3 批）：12.22 到 17.54 µs</title><rect x="216.3" y="337" width="61.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5500（第 3 批）：23.10 到 28.54 µs</title><rect x="342.9" y="337" width="62.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7916（第 4 批）：34.88 到 38.91 µs</title><rect x="480.0" y="337" width="46.3" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 1932（第 1 批）：0.10 到 7.30 µs</title><rect x="75.1" y="346" width="83.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 4140（第 3 批）：12.26 到 20.83 µs</title><rect x="216.6" y="346" width="99.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5803（第 3 批）：23.94 到 31.97 µs</title><rect x="352.6" y="346" width="92.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 8094（第 4 批）：35.97 到 44.16 µs</title><rect x="492.6" y="346" width="94.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <text class="tick" x="74.0" y="369" text-anchor="middle">0</text>
  <line class="grid" x1="74.0" y1="52" x2="74.0" y2="200"/>
  <text class="tick" x="132.2" y="369" text-anchor="middle">5</text>
  <line class="grid" x1="132.2" y1="52" x2="132.2" y2="200"/>
  <text class="tick" x="190.4" y="369" text-anchor="middle">10</text>
  <line class="grid" x1="190.4" y1="52" x2="190.4" y2="200"/>
  <text class="tick" x="248.6" y="369" text-anchor="middle">15</text>
  <line class="grid" x1="248.6" y1="52" x2="248.6" y2="200"/>
  <text class="tick" x="306.8" y="369" text-anchor="middle">20</text>
  <line class="grid" x1="306.8" y1="52" x2="306.8" y2="200"/>
  <text class="tick" x="365.0" y="369" text-anchor="middle">25</text>
  <line class="grid" x1="365.0" y1="52" x2="365.0" y2="200"/>
  <text class="tick" x="423.2" y="369" text-anchor="middle">30</text>
  <line class="grid" x1="423.2" y1="52" x2="423.2" y2="200"/>
  <text class="tick" x="481.4" y="369" text-anchor="middle">35</text>
  <line class="grid" x1="481.4" y1="52" x2="481.4" y2="200"/>
  <text class="tick" x="539.6" y="369" text-anchor="middle">40</text>
  <line class="grid" x1="539.6" y1="52" x2="539.6" y2="200"/>
  <text class="tick" x="597.7" y="369" text-anchor="middle">45</text>
  <line class="grid" x1="597.7" y1="52" x2="597.7" y2="200"/>
  <text class="lab2" x="66" y="369" text-anchor="end">µs</text>
  <text class="lab2" x="30" y="304" text-anchor="middle" transform="rotate(-90 30 300)">名额</text>
  <text class="lab2" x="74" y="394">批次</text>
  <rect x="110" y="385" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.14"/><text class="lab" x="127" y="394">0–2039</text>
  <rect x="218" y="385" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.32"/><text class="lab" x="235" y="394">2040–4079</text>
  <rect x="326" y="385" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.5"/><text class="lab" x="343" y="394">4080–6119</text>
  <rect x="434" y="385" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.68"/><text class="lab" x="451" y="394">6120–8159</text>
  <rect x="542" y="385" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.9"/><text class="lab" x="559" y="394">8160–8191</text>
</svg>
<figcaption><strong>图 2-1a</strong> GELU 的 8192 个 program 实际怎么分批跑（RTX 5090，n = 8388608，fp32，每个 program 4 个 warp）。每个 program 开头读一次 <code>%globaltimer</code> 和 <code>%smid</code>，store 发出后再读一次时间，记录前先清空 L2。颜色由淡到浓是 program 编号所在的批次，一批 2040 个。悬停可看 program 编号和时间。</figcaption>
</figure>

</details>

<details class="fold doc aside">
<summary>GELU 编译成的 PTX</summary>

Triton 把 kernel 编译成 PTX，PTX 描述的是单个 thread 的指令。n = 8388608 时的节选：

```
mov.u32          %r17, %ctaid.x;                                // program 编号，即 blockIdx.x
mov.u32          %r20, %tid.x;                                  // thread 编号
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];     // 一条读 4 个 fp32（16 B），共 2 条
ex2.approx.f32   %r66, %r65;                                    // tl.exp 编成以 2 为底的指数
div.full.f32     %r90, %r89, %r81;                              // 2 / (e + 1)
@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r9, %r10, %r11, %r12 };  // 一条写 4 个，共 2 条
```

一个 program 处理 1024 个元素、128 个 thread，每个 thread 分到 8 个。n 是 16 的倍数，Triton 知道地址按 16 B 对齐，把 8 次 4 B 的读合并成 2 条 `ld.global.v4`，写也一样；n 不是 16 的倍数时（比如 n + 7），PTX 里就是 8 条 4 B 的 `ld.global`。

</details>

GELU 的每个元素互不依赖，怎么切都可以。softmax 的每个元素都要用到整行的最大值和总和，切分就得按行来。

### 2.2 Softmax：按行归约，一行一个 program {#triton-softmax}

softmax 对 `[M, N]` 的输入按行算，先减去每行的最大值 $m_i$ 防止 exp 溢出：

<p align="center">$m_i = \max_k x_{ik}, \qquad y_{ij} = \dfrac{e^{x_{ij} - m_i}}{\sum_{k} e^{x_{ik} - m_i}}$</p>

$y_{ij}$ 要用到第 $i$ 行全部 N 个数。eager 下 softmax 要 5 个 kernel（和[下一篇](/blog/gpu-training-analysis/#attention)里 attention 的 softmax 一样），一共读 5MN + 2M 个数，写 3MN + 2M 个：

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
<figcaption><strong>图 2-2</strong> softmax kernel（4096 × 4096，fp32），画法同<a href="#fig-t-1">图 2-1</a>。一行一个 program，中间结果都留在 register 里，显存只读写各一次；每个 program 用的 register 多，一个 SM 只放得下 9 个。</figcaption>
</figure>

如果一个 program 处理一整行（[图 2-2](#fig-t-2)），max、exp、求和、除都可以在 register 里做完，只读 MN、写 MN，读写量是原来的 1/4：

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

启动时一行一个 program：

```python {role="grid" tag="一行一个 program"}
def softmax(x):
    M, N = x.shape
    y = torch.empty_like(x)
    softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
```

4096 × 4096 的 fp32 输入，eager 版 246.9 µs，Triton 版 92.4 µs，快 2.7 倍，和 `torch.softmax`（92.3 µs）持平，有效带宽 1.45e12 B/s，是显存带宽的 81%。eager 版读写量是 4 倍，慢得却不到 4 倍：按读写量算它的带宽是 2.17e12 B/s，超过了显存带宽，说明 64 MiB 的中间结果有一部分直接在 96 MB 的 L2 里命中了。

[图 2-2](#fig-t-2) 里的 warp max、warp sum 是 shared memory 在这个 kernel 里唯一的用处：`tl.max`、`tl.sum` 先在每个 warp 内用 shuffle 归约，4 个 warp 的部分结果再经 shared memory 交换，一共 16 B。把一整个 block 的数据放进 shared memory 什么时候值得，[2.4 节](#triton-matmul)再讲。

这个写法要求 BLOCK ≥ N，也就是一行能一次装进一个 program。行再长，program 就只能沿行一个 block 一个 block 地读，读过的 block 要留下一个中间结果。下一节先看最简单的按行归约：求和。

### 2.3 Row sum：按行归约，按 block 累加 {#triton-rowsum}

row sum 对 `[M, N]` 的输入每行求和：

<p align="center">$s_i = \sum_{j=1}^{N} x_{ij} = \sum_{T} \ \sum_{j \in \mathcal{J}_T} x_{ij}$</p>

$\mathcal{J}_T$ 是第 $T$ 个 block 的 TILE 个列下标，代码里 TILE 取 1024。加法满足结合律，先把每个 block 加起来再相加，和一次加完是同一个数（浮点数下只差舍入误差），所以 program 可以沿行一个 block 一个 block 地读，每读一个 block 就加到累加器上（[图 2-3](#fig-t-3)）。

<figure id="fig-t-3" class="fg-fig">
<svg class="fg" viewBox="0 0 640 380" width="100%" role="img" aria-label="row sum kernel：一行一个 program；tl.load 一轮把一个 block 从显存直接读进 register，计算单元把它加到 acc 上，颜色深浅是 for 循环的顺序；acc 一直留在 register 里，最后归约成一个数，tl.store 写回">
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
  <text class="tb" x="630" y="16" text-anchor="end" style="fill: var(--fig-hi)">按行归约，按 block 累加</text>
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
  <text class="s" x="10" y="314">thread 数先满</text>
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
  <text class="lab2" x="152" y="348">图里 TILE = 4（代码里 1024），一行 16 个数分 4 个 block，颜色由淡到浓是 for 循环读的顺序；</text>
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
  <text class="s" x="210.0" y="107">block 0</text>
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
  <text class="s" x="373.3" y="107">block 3</text>
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
  <text class="s" x="232.0" y="298" text-anchor="middle">block 0</text>
  <text class="s" x="286.4" y="298" text-anchor="middle">block 1</text>
  <text class="s" x="340.9" y="298" text-anchor="middle">block 2</text>
  <text class="s" x="395.3" y="298" text-anchor="middle">block 3</text>
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
<figcaption><strong>图 2-3</strong> row sum kernel（4096 × 4096，fp32），画法同<a href="#fig-t-1">图 2-1</a>。图里 TILE = 4，一行 16 个数分 4 个 block，代码里 TILE = 1024。颜色由淡到浓是 for 循环读 block 的顺序：第 0 轮后 acc 只有 block 0 的数，最后的 acc 是 4 个 block 逐位相加 [7, 8, 4, 6]，铺满整条渐变；中间两轮用 ⋯ 省略，最后 <code>tl.sum</code> 加成 25 写回。</figcaption>
</figure>

累加器 `acc` 是一个长度为 TILE 的向量，不是一个数：循环里每个 thread 只加自己负责的几个位置，thread 之间不需要通信；循环结束后，`tl.sum` 再做一次跨 thread 归约，把 TILE 个部分和加成一个数。

```python {role="program" tag="沿第 row 行循环累加"}
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, TILE: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([TILE], dtype=tl.float32)  # TILE 路部分和，不是一个标量
    for start in range(0, N, TILE):           # 行比 TILE 长：一个 block 一个 block 串行读
        cols = start + tl.arange(0, TILE)
        acc += tl.load(x_ptr + row * N + cols, mask=cols < N, other=0.0)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))  # 最后只做一次跨 thread 归约
```

grid 和 softmax 一样，一行一个 program：

```python {role="grid" tag="一行一个 program"}
def row_sum(x, TILE=1024):
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, out, N, TILE=TILE)
    return out
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

<figure id="fig-t-4" class="fg-fig">
<svg class="fg" viewBox="0 0 640 435" width="100%" role="img" aria-label="matmul kernel（A 4096 × 2048，B 2048 × 512）：C 的每个 block 一个 program；program (1, 1) 沿 K 一轮用 cp.async 把 A、B 各一个 block 从显存直接搬进 shared memory，tl.dot 从 shared memory 读一小片进 register 做乘加，acc 留在 register 里，中间几轮用 ⋯ 省略，最后 ReLU 后 tl.store 写回 C 的 block (1, 1)">
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
  <text class="tb" x="630" y="16" text-anchor="end" style="fill: var(--fig-hi)">block 乘加</text>
  <rect x="10" y="28" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="50" y="28" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="10" y="55" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="50" y="55" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 30%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="68" y="70" text-anchor="middle">(1,1)</text>
  <rect x="10" y="82" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="50" y="82" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="10" y="109" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="50" y="109" width="36" height="23" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="lab2" x="10" y="148">每个 C block 一个 program</text>
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
  <text class="lab2" x="152" y="388">cp.async 把 A、B 的 block 从显存直接搬进 shared memory，tl.dot 从中读 a、b 到 register 做 FMA；</text>
  <text class="lab2" x="152" y="403">K 按 BK = 32 切成 64 轮（T = 0 … 63），只画 T = 0 和 63；虚线框是正在搬的下一轮；</text>
  <text class="lab2" x="152" y="418">写回前 acc 先经 shared memory 换一次排布，再从 register 写回显存。</text>
  <defs><linearGradient id="fig-t-4-acc" x1="0" x2="1" y1="0" y2="0"><stop offset="0.00" style="stop-color: var(--fig-1)" stop-opacity="0.12"/><stop offset="0.14" style="stop-color: var(--fig-1)" stop-opacity="0.21"/><stop offset="0.29" style="stop-color: var(--fig-1)" stop-opacity="0.31"/><stop offset="0.43" style="stop-color: var(--fig-1)" stop-opacity="0.40"/><stop offset="0.57" style="stop-color: var(--fig-1)" stop-opacity="0.50"/><stop offset="0.71" style="stop-color: var(--fig-1)" stop-opacity="0.59"/><stop offset="0.86" style="stop-color: var(--fig-1)" stop-opacity="0.69"/><stop offset="1.00" style="stop-color: var(--fig-1)" stop-opacity="0.78"/></linearGradient></defs>
  <rect x="195.0" y="158" width="46" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.12"/><rect x="195.0" y="158" width="46" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="218.0" y="171" text-anchor="middle" >A block</text>
  <rect x="239.0" y="190" width="46" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.12"/><rect x="239.0" y="190" width="46" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="262.0" y="203" text-anchor="middle" >B block</text>
  <rect x="249.0" y="111" width="34" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.12"/><rect x="249.0" y="111" width="34" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="266.0" y="124" text-anchor="middle" >a、b</text>
  <line x1="224.0" y1="156.0" x2="258.0" y2="131.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="268.0" y1="188.0" x2="272.0" y2="131.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="280.0" y1="109.0" x2="281.7" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <rect x="358.3" y="158" width="46" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="358.3" y="158" width="46" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="381.3" y="171" text-anchor="middle" style="fill: var(--fig-on-1)">A block</text>
  <rect x="402.3" y="190" width="46" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="402.3" y="190" width="46" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="425.3" y="203" text-anchor="middle" style="fill: var(--fig-on-1)">B block</text>
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
  <text class="s" x="291.7" y="41" text-anchor="end" style="fill: var(--fig-hi)">T = 0</text><text class="s" x="467.0" y="41" style="fill: var(--fig-hi)">T = 63</text>
  <rect x="585.0" y="111" width="46" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="585.0" y="111" width="46" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="608.0" y="124" text-anchor="middle" style="fill: var(--fig-on-1)">C block</text>
  <line x1="558.3" y1="86.0" x2="594.0" y2="109.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <rect x="563.0" y="176" width="46" height="18" rx="4" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="563.0" y="176" width="46" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="586.0" y="189" text-anchor="middle" style="fill: var(--fig-on-1)">C block</text>
  <line x1="591.0" y1="131.0" x2="591.0" y2="174.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="600.0" y1="174.0" x2="600.0" y2="131.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <rect x="285.7" y="172" width="74" height="18" rx="4" style="fill: none; stroke: var(--fig-1)" stroke-dasharray="3 2" stroke-opacity=".7"/>
  <text class="s" x="322.7" y="185" text-anchor="middle">T = 1 的 A、B</text><text class="s" x="322.7" y="203" text-anchor="middle">正在搬</text>
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
  <text class="s" x="276.0" y="296">沿 K 第 T 轮读的 block：T = 0 … 63</text>
  <rect x="276.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.12" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="288.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.21" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="300.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.31" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="312.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.40" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="324.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.50" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="336.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.59" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="348.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.69" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="360.0" y="302" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.78" stroke-opacity=".4" stroke-width=".6"/>
  <text class="s" x="276.0" y="325">T = 0</text><text class="s" x="372.0" y="325" text-anchor="end">T = 63</text>
  <text class="s" x="276.0" y="342">每轮读 A 的 64 × 32、B 的 32 × 64（BK = 32）；</text><text class="s" x="276.0" y="356">图里一格 256 列，是 8 轮</text>
</svg>
<figcaption><strong>图 2-4</strong> matmul kernel（A 4096 × 2048，B 2048 × 512，fp32，BM = BN = 64，BK = 32），画法同<a href="#fig-t-1">图 2-1</a>。显存里的矩阵按比例画全，一格是 256 × 256 个元素。program (1, 1) 读 A 的第 1 个 row block 和 B 的第 1 个 column block，颜色由淡到浓是沿 K 读 block 的顺序，acc 一轮轮累积，ReLU 后写回 C 的 block (1, 1)。</figcaption>
</figure>

一个 program 负责一个 $C_{IJ}$（[图 2-4](#fig-t-4)），沿 K 循环，第 $T$ 轮读 $A_{IT}$ 和 $B_{TJ}$，用 `tl.dot` 乘加到累加器上；写回前在 register 里做 ReLU，激活函数不用再单独一个 kernel 读写一遍 C：

```python {role="program" tag="算 C 的第 (pid_m, pid_n) 个 block"}
@triton.jit
def matmul_relu_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       sam, sak, sbk, sbn, scm, scn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)   # 负责 C 的第 (pid_m, pid_n) 个 block
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak  # [BM, BK] 的指针
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn  # [BK, BN] 的指针
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):                 # 沿 K 一个 block 一个 block 乘加
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")  # fp32 默认会走 tf32，这里要求精确的 fp32
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    acc = tl.maximum(acc, 0.0)                # ReLU 在 register 里做完，不多一次读写
    c_ptrs = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
```

grid 是二维的：

```python {role="grid" tag="C 的每个 block 一个 program，二维"}
def matmul_relu(a, b, BM=64, BN=64, BK=32):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_relu_kernel[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(), BM=BM, BN=BN, BK=BK)
    return c
```

<div class="nv-key">

**shared memory 什么时候值得用**：满足下面任一条。

- **复用**：同一份数据要被读很多次。matmul 里 A 的一个 block 要和 B 的 block 的 64 列都乘一遍；softmax 和 row sum 里每个数只被一个 thread 读一次。
- **交换**：thread 之间要交换大量数据。matmul 里 `tl.dot` 要把 A、B 的 block 重新分给不同的 thread；softmax 和 row sum 只交换 4 个 warp 的部分结果（16 B）。

所以 matmul 把 A、B 的 block 放在 shared memory 里，每个 program 32 KiB；softmax 和 row sum 两条都不满足，把整个 block 放进 shared memory 只会多搬一趟。代价是 shared memory 用得越多，一个 SM 能同时放的 program 越少：每个 SM 的上限是 100 KiB，matmul 每个 program 占 32 KiB，再加每个 block 保留的 1 KiB，只放得下 3 个（3 × 33 = 99 KiB）。

</div>

A 4096 × 2048、B 2048 × 512 时 Triton 版 203.1 µs，4.23e13 FLOPS，是 fp32 标称峰值 1.05e14 FLOPS 的 40%；cuBLAS（`torch.relu(A @ B)`，关掉 tf32）是 160.3 µs。差距主要来自 tail effect。

<details class="fold doc aside">
<summary>matmul 的 tail effect：512 个 program 分两批</summary>

fp32 标称峰值按 170 个 SM × 128 个 CUDA core × 2 FLOPs × 2.41e9 Hz 算。每个 SM 只放 3 个 program，全卡一批 510 个，512 个 program 多出 2 个。实测的时间线上（[图 2-4a](#fig-t-7)），510 个 program 在开头一起开始，各跑 125 到 152 µs；全卡最先跑完的两个 program 在 SM 68 和 SM 135 上，约 125 µs 结束，多出的 2 个这时才开始。152 µs 以后全卡只剩这 2 个 program，另外 168 个 SM 空着，一直等到 200 µs。这 2 个 program 后半段独占一个 SM，只跑了 75 µs，比第一批的中位数 135 µs 快得多。

A 换成 4032 × 2048 时 504 个 program 一批放下，同一个 kernel 只要 157.9 µs，和 cuBLAS 相当。4096 比 4032 的计算量只多 1.6%，耗时多 29%。打点的开销不大：打点版 210.8 µs 和 162.8 µs，不打点 203.1 µs 和 157.9 µs。

<figure id="fig-t-7" class="fg-fig">
<svg class="fg" viewBox="0 0 640 352" width="100%" role="img" aria-label="matmul 的 program 实测时间线：A 4096 × 2048 时 512 个 program，第一批 510 个一起开始；多出的 2 个等到 124.9 µs 才开始，152 µs 以后全卡只有这 2 个在跑，到 200 µs 结束。A 4032 × 2048 时 504 个一批放下，152 µs 结束">
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
  <text class="lab2" x="74" y="26">全卡同时在跑的 program 数；虚线是 170 个 SM × 3 = 510 个名额</text>
  <line class="grid" x1="74" y1="176.0" x2="620" y2="176.0"/><text class="tick" x="66" y="180.0" text-anchor="end">0</text>
  <line class="grid" x1="74" y1="151.8" x2="620" y2="151.8"/><text class="tick" x="66" y="155.8" text-anchor="end">100</text>
  <line class="grid" x1="74" y1="127.5" x2="620" y2="127.5"/><text class="tick" x="66" y="131.5" text-anchor="end">200</text>
  <line class="grid" x1="74" y1="103.3" x2="620" y2="103.3"/><text class="tick" x="66" y="107.3" text-anchor="end">300</text>
  <line class="grid" x1="74" y1="79.1" x2="620" y2="79.1"/><text class="tick" x="66" y="83.1" text-anchor="end">400</text>
  <line class="grid" x1="74" y1="52.4" x2="620" y2="52.4"/><text class="tick" x="66" y="56.4" text-anchor="end">510</text>
  <line class="ref" x1="74" y1="52.4" x2="620" y2="52.4"/>
  <rect x="470.4" y="50" width="124.0" height="126" style="fill: color-mix(in srgb, var(--fig-hi) 10%, transparent); stroke: var(--fig-hi)" stroke-width="0"/>
  <text class="s" x="532.4" y="80" text-anchor="middle" style="fill: var(--fig-hi)">只剩 2 个</text><text class="s" x="532.4" y="94" text-anchor="middle" style="fill: var(--fig-hi)">program</text>
  <polyline points="74.0,168.2 74.1,53.9 355.6,54.1 356.0,54.4 356.5,54.6 356.5,55.1 356.9,55.3 357.0,55.6 357.1,55.8 357.2,56.1 357.3,56.3 357.7,56.5 357.9,56.8 398.9,57.0 399.9,57.8 400.2,58.2 401.1,58.7 401.5,59.0 401.6,59.4 402.1,59.7 402.6,59.9 402.9,60.2 403.2,60.4 403.6,60.7 403.9,61.1 404.2,61.4 404.3,61.6 404.4,61.9 404.5,62.1 404.6,62.8 404.6,63.3 404.7,64.1 404.8,64.5 404.9,65.0 405.0,65.8 405.2,66.0 405.4,66.7 405.5,67.0 405.6,67.4 405.6,67.7 405.7,68.2 405.8,68.7 406.1,69.1 406.1,69.4 406.2,69.9 406.3,70.1 406.5,70.4 406.6,70.6 406.6,71.1 406.7,72.3 406.8,72.8 406.9,73.0 407.0,73.3 407.0,73.7 407.1,74.0 407.2,74.2 407.3,74.5 407.5,74.7 407.5,75.4 407.7,75.9 407.9,76.2 408.0,76.9 408.0,77.1 408.1,77.4 408.2,77.9 408.3,78.1 408.4,78.3 408.5,78.6 408.6,79.1 408.8,79.6 409.0,80.0 409.3,80.3 409.4,80.5 409.5,80.8 409.5,81.0 409.6,81.5 409.7,81.7 409.8,82.0 409.9,82.7 410.0,83.0 410.1,83.7 410.2,83.9 410.7,84.2 411.1,84.7 411.5,84.9 413.9,85.1 415.7,85.4 416.0,85.6 416.1,86.1 416.2,86.3 416.4,86.6 416.6,86.8 416.7,87.1 416.9,87.3 417.2,88.0 417.4,88.3 417.5,88.5 417.7,88.8 417.9,89.0 418.3,89.3 418.4,89.5 418.5,90.0 418.6,90.5 418.9,90.7 419.0,90.9 419.1,91.7 419.4,92.4 419.6,92.6 419.7,92.9 419.9,93.1 419.9,93.4 420.0,93.6 420.1,93.9 420.4,94.1 420.4,94.8 420.5,95.1 420.7,95.6 421.2,96.0 421.4,96.5 421.5,96.8 421.6,97.2 421.8,97.5 421.9,97.7 422.0,98.0 422.1,98.7 422.3,99.2 422.4,99.9 422.4,100.9 422.6,101.4 422.7,101.9 422.8,103.1 422.9,103.3 422.9,103.8 423.0,105.5 423.1,105.7 423.2,106.2 423.3,106.7 423.4,106.9 423.4,107.2 423.5,107.4 423.7,107.7 423.9,108.9 423.9,109.4 424.1,109.8 424.2,110.6 424.3,111.5 424.4,111.8 424.4,112.3 424.5,112.5 424.6,112.8 424.7,113.5 424.9,114.0 424.9,114.2 425.0,114.7 425.1,114.9 425.2,115.4 425.3,115.7 425.4,116.2 425.5,116.4 425.6,116.9 425.8,117.1 425.9,117.4 425.9,117.6 426.3,117.8 426.4,118.1 426.7,118.3 426.8,118.6 427.0,118.8 427.1,119.1 427.4,119.5 427.5,119.8 427.7,120.0 428.0,120.3 428.1,120.5 428.2,120.8 428.4,121.0 428.5,121.2 428.7,121.7 428.8,122.0 428.8,122.5 428.9,122.7 429.0,122.9 429.2,123.4 429.7,123.9 429.9,124.4 430.5,124.6 430.8,124.9 431.2,125.1 432.8,125.6 433.2,125.8 433.3,126.1 433.6,126.3 433.8,126.6 434.0,126.8 434.1,127.1 434.3,127.3 434.4,127.5 434.8,128.0 434.8,128.3 435.1,128.5 435.2,128.8 435.6,129.0 435.8,129.5 436.0,129.7 436.2,130.0 436.3,130.2 436.9,130.4 437.4,130.7 437.6,130.9 437.7,131.2 438.0,131.4 438.2,131.7 438.7,132.1 438.7,132.4 438.8,132.6 439.1,132.9 439.2,133.1 439.3,133.4 439.7,133.6 439.9,133.8 440.0,134.1 440.1,134.3 440.2,134.6 440.4,134.8 441.2,135.5 441.9,135.8 442.0,136.0 442.5,136.3 443.2,136.5 443.6,136.7 443.8,137.0 443.9,137.2 444.1,137.7 444.2,138.2 444.3,138.4 444.4,138.7 444.5,138.9 444.8,139.2 445.2,139.4 445.5,139.7 447.7,139.9 448.0,140.4 448.2,140.6 448.2,141.1 448.3,141.3 448.6,141.6 448.7,141.8 448.9,142.8 449.0,143.3 449.1,143.5 449.2,144.3 449.3,144.5 449.5,144.7 449.6,145.2 449.6,145.5 449.7,146.0 449.8,146.2 450.0,146.4 450.1,146.9 450.2,147.4 450.4,148.1 450.5,148.6 450.6,149.6 450.6,150.3 450.7,150.8 450.8,151.3 450.9,152.0 451.1,153.0 451.1,153.2 451.2,153.5 451.3,154.2 451.5,154.4 451.6,154.7 451.6,155.6 451.8,156.6 451.9,156.9 452.1,157.8 452.2,158.3 452.3,158.6 452.4,158.8 452.6,159.3 452.8,159.5 453.1,159.8 453.3,160.2 453.6,160.7 453.7,161.0 453.8,161.2 453.9,161.7 454.0,162.4 454.1,162.7 454.2,163.2 454.5,163.6 454.7,164.1 455.4,164.4 459.8,164.6 460.1,164.9 460.2,165.1 460.4,165.3 460.5,165.6 460.5,165.8 460.7,166.1 460.8,166.3 460.9,166.6 461.2,166.8 461.3,167.0 461.6,167.3 461.7,167.5 461.8,167.8 462.0,168.0 462.5,168.2 463.2,168.5 463.5,168.7 463.5,169.0 463.6,169.2 463.8,169.5 463.9,169.7 464.2,169.9 464.3,170.4 464.4,170.7 464.5,170.9 464.5,171.2 464.6,171.6 464.8,171.9 465.2,172.1 466.5,172.4 467.0,172.6 467.1,173.1 467.3,173.3 467.5,173.6 467.5,174.1 467.6,174.3 467.7,174.8 467.9,175.3 468.1,175.5 468.2,176.0" fill="none" style="stroke: var(--fig-2)" stroke-width="2" stroke-linejoin="round" stroke-dasharray="6 4"/>
  <line x1="98" y1="116" x2="120" y2="116" style="stroke: var(--fig-2)" stroke-width="2" stroke-dasharray="6 4"/><text class="lab" x="128" y="120">A 4032 × 2048，504 个 program</text>
  <polyline points="74.0,102.3 74.1,52.4 398.2,52.9 398.5,53.2 398.6,53.4 398.7,53.2 399.1,53.4 399.3,53.2 399.6,53.4 399.6,53.6 399.9,53.9 400.6,54.1 400.8,54.4 401.1,54.6 401.1,54.8 401.4,55.1 402.5,55.3 403.0,55.6 403.1,55.8 403.5,56.1 403.6,56.3 403.7,56.8 403.9,57.0 404.1,57.3 404.2,57.5 404.4,57.8 404.9,58.0 405.1,58.2 405.2,58.7 405.3,59.2 405.5,59.4 405.6,59.9 405.9,60.2 406.0,60.9 406.1,61.1 406.3,61.4 406.6,61.6 406.8,62.4 406.9,63.3 407.0,63.8 407.1,64.1 407.2,64.5 407.3,64.8 407.5,65.3 407.5,66.0 407.6,66.7 407.7,67.9 407.8,68.2 407.9,68.7 408.0,69.1 408.0,69.6 408.1,70.1 408.2,70.6 408.3,71.6 408.4,72.1 408.5,72.3 408.6,72.8 408.7,73.3 409.0,73.5 409.0,74.0 409.1,74.2 409.3,74.7 409.4,75.0 409.5,75.2 409.5,76.2 409.6,76.4 409.8,76.7 410.0,77.1 410.0,77.4 410.5,77.6 410.6,77.9 410.7,78.1 411.0,78.3 411.1,78.6 411.2,79.1 411.4,79.3 411.5,79.6 412.0,79.8 412.1,80.3 412.5,80.5 412.7,80.8 412.8,81.0 413.0,81.5 417.3,81.7 417.5,82.0 417.7,82.2 417.9,82.5 418.1,82.7 418.2,83.0 418.4,83.4 418.5,83.7 418.7,83.9 418.9,84.2 418.9,84.4 419.0,84.7 419.1,85.4 419.2,85.6 419.4,85.9 419.4,86.3 419.5,86.6 419.6,86.8 419.7,87.1 419.8,87.3 419.9,87.6 420.0,87.8 420.2,88.0 420.4,88.3 420.5,88.8 420.8,89.0 420.9,89.3 420.9,89.5 421.0,90.0 421.2,90.2 421.4,90.5 421.5,90.7 422.1,90.9 422.3,91.4 422.4,91.9 422.6,92.2 422.7,92.9 422.9,93.1 423.0,93.9 423.3,94.6 423.4,95.3 423.4,95.8 423.5,96.3 423.6,96.8 423.8,97.0 423.9,97.7 423.9,98.7 424.0,98.9 424.1,99.4 424.2,100.4 424.3,100.9 424.4,101.6 424.4,102.1 424.5,103.3 424.6,104.0 424.8,104.5 424.9,105.2 424.9,105.7 425.0,106.9 425.1,107.4 425.3,109.1 425.4,109.4 425.4,109.6 425.5,109.8 425.6,110.3 425.7,110.6 425.8,111.3 425.9,111.5 425.9,112.3 426.0,112.8 426.1,113.2 426.2,113.7 426.3,114.0 426.5,114.2 427.0,114.5 427.3,114.7 427.4,114.9 427.7,115.2 428.3,115.4 428.3,115.7 428.4,115.9 428.5,116.6 428.6,116.9 428.8,117.1 428.9,117.8 429.0,118.3 429.3,118.6 429.4,119.1 429.7,119.3 429.8,119.5 429.8,119.8 430.1,120.0 430.2,120.3 430.8,120.5 430.9,120.8 431.2,121.0 431.3,121.5 431.6,121.7 431.7,122.0 432.3,122.2 433.1,122.5 433.2,122.7 434.7,122.9 435.6,123.2 436.0,123.4 436.2,124.4 436.9,124.6 437.1,124.9 437.2,125.1 437.4,125.4 437.7,125.8 437.9,126.1 438.0,126.3 438.6,126.6 438.7,127.1 439.1,127.3 439.7,127.5 439.8,127.8 440.0,128.0 440.1,128.5 440.2,128.8 440.2,129.0 440.5,129.2 440.6,129.5 440.7,129.7 440.8,130.0 440.9,130.4 441.2,131.2 441.3,132.1 441.7,132.4 441.9,132.9 442.0,133.1 442.2,133.4 442.2,133.6 442.3,133.8 442.7,134.1 442.9,134.3 443.8,134.6 444.1,134.8 444.7,135.1 444.8,135.8 445.2,136.0 445.3,136.3 445.4,137.0 446.1,137.2 448.4,137.5 448.6,137.7 448.7,138.0 449.1,138.2 449.1,138.4 449.2,138.7 449.4,138.9 449.5,139.2 449.6,139.4 449.8,139.9 449.9,140.1 450.0,140.4 450.1,140.6 450.3,141.1 450.6,141.8 450.7,142.1 450.8,142.6 450.9,142.8 451.0,143.0 451.1,143.8 451.1,144.5 451.2,146.0 451.4,146.9 451.5,147.4 451.6,147.7 451.7,148.4 451.8,148.6 451.9,148.9 452.0,149.6 452.1,150.1 452.2,150.6 452.4,151.3 452.5,151.5 452.6,151.8 452.6,152.3 452.7,153.0 452.8,153.7 452.9,153.9 453.1,154.2 453.1,154.4 453.2,154.7 453.4,155.2 453.5,155.4 453.6,155.6 453.7,155.9 453.8,156.6 453.9,156.9 454.0,157.1 454.1,157.3 454.1,157.6 454.2,157.8 454.3,158.1 454.6,158.3 454.6,158.8 454.7,159.0 455.1,159.3 455.4,159.5 455.6,159.8 455.8,160.0 455.9,160.2 456.0,160.5 456.1,160.7 456.1,161.0 456.2,161.2 456.3,161.9 456.5,162.4 457.1,162.9 457.1,163.2 457.2,163.4 461.2,163.6 461.8,163.9 462.2,164.1 462.7,164.4 463.1,164.6 463.5,164.9 463.5,165.1 463.6,165.6 463.7,166.1 463.8,166.6 464.0,167.0 464.3,167.5 465.0,167.8 465.0,168.0 465.5,168.2 466.2,168.5 466.3,168.7 466.4,169.0 466.6,169.2 466.7,169.5 466.8,169.7 466.9,169.9 467.0,170.2 467.0,170.4 467.1,170.7 467.2,171.4 467.3,171.6 468.1,172.1 468.4,172.4 468.5,172.6 468.9,172.8 469.0,173.1 469.0,173.3 469.2,173.6 469.4,173.8 469.5,174.1 469.6,174.5 469.8,174.8 470.1,175.0 470.2,175.3 470.4,175.5 594.2,175.8 594.4,176.0" fill="none" style="stroke: var(--fig-1)" stroke-width="2" stroke-linejoin="round"/>
  <line x1="98" y1="134" x2="120" y2="134" style="stroke: var(--fig-1)" stroke-width="2"/><text class="lab" x="128" y="138">A 4096 × 2048，512 个 program</text>
  <text class="lab2" x="74" y="212">A 4096 × 2048 时跑了多出 2 个 program 的两个 SM，每个 3 个名额；一条是一个 program</text>
  <text class="lab" x="66" y="242" text-anchor="end">SM 135</text>
  <g class="m"><title>program 216：0.0 到 124.7 µs</title><rect x="74.0" y="223" width="323.6" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.45"/></g>
  <g class="m"><title>program 503：124.9 到 200.2 µs</title><rect x="398.7" y="223" width="195.1" height="9" rx="1.5" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>program 43：0.0 到 135.1 µs</title><rect x="74.0" y="234" width="350.7" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.45"/></g>
  <g class="m"><title>program 381：0.0 到 149.3 µs</title><rect x="74.1" y="245" width="387.5" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.45"/></g>
  <text class="lab" x="66" y="281" text-anchor="end">SM 68</text>
  <g class="m"><title>program 160：0.0 到 124.8 µs</title><rect x="74.0" y="262" width="323.9" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.45"/></g>
  <g class="m"><title>program 511：125.1 到 200.1 µs</title><rect x="399.3" y="262" width="194.3" height="9" rx="1.5" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>program 498：0.0 到 134.4 µs</title><rect x="74.0" y="273" width="348.8" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.45"/></g>
  <g class="m"><title>program 325：0.0 到 148.9 µs</title><rect x="74.1" y="284" width="386.5" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.45"/></g>
  <text class="tick" x="74.0" y="310" text-anchor="middle">0</text>
  <text class="tick" x="139.0" y="310" text-anchor="middle">25</text>
  <text class="tick" x="204.0" y="310" text-anchor="middle">50</text>
  <text class="tick" x="269.0" y="310" text-anchor="middle">75</text>
  <text class="tick" x="334.0" y="310" text-anchor="middle">100</text>
  <text class="tick" x="399.0" y="310" text-anchor="middle">125</text>
  <text class="tick" x="464.0" y="310" text-anchor="middle">150</text>
  <text class="tick" x="529.0" y="310" text-anchor="middle">175</text>
  <text class="tick" x="594.0" y="310" text-anchor="middle">200</text>
  <text class="lab2" x="66" y="310" text-anchor="end">µs</text>
  <rect x="74" y="324" width="14" height="9" rx="2" style="fill: var(--fig-1)" fill-opacity=".45"/><text class="s" x="94" y="332">第一批的 program</text>
  <rect x="214" y="324" width="14" height="9" rx="2" style="fill: var(--fig-hi)"/><text class="s" x="234" y="332">多出的 2 个，等名额空出来才开始</text>
</svg>
<figcaption><strong>图 2-4a</strong> matmul kernel 的 program 实际怎么分批跑（RTX 5090，fp32，B 2048 × 512）。每个 program 开头读一次 <code>%globaltimer</code> 和 <code>%smid</code>，写回 C 之后再读一次时间，记录前先清空 L2。悬停可看 program 编号和时间。</figcaption>
</figure>

</details>

Triton 默认 `num_stages=3`，编译器把这个循环改写成流水线：A、B 各开两个槽，算第 $T$ 轮时，第 $T+1$ 轮的 block 已经在搬了。

<details class="fold doc aside">
<summary>matmul 的 load 藏住了吗：PTX 计数和 Nsight Compute 实测</summary>

从编译出的 TTGIR 和 PTX 里能数清楚 A、B 的 block 怎么进出 shared memory。循环前先发出第 0、1 轮的 cp.async；第 $T$ 轮先等第 $T$ 轮的数据到齐，用 ld.shared 读进 register 做乘加，读完马上往刚空出来的槽发第 $T+2$ 轮。A、B 各搬 64 次，每次 8 KiB，一个 program 一共 1 MiB；512 个 program 加起来 512 MiB，而 A、B 本身只有 36 MiB，所以 L2 命中率有 91%，大部分读不用到显存。每轮从 shared memory 读进 register 的是 192 KiB，同一个数平均被读 12 次，这就是前面说的复用。写回之前 acc 还要经 shared memory 换一次排布，让 tl.store 能按连续地址写。

预取藏住了多少 load，要用 Nsight Compute 测（[图 2-4b](#fig-t-6)）。ncu 会统计一个 warp 每两条指令之间平均隔几拍，以及这几拍里在等什么。不预取（`num_stages=1`）时，4.59 拍里有 0.69 拍在等显存或 L2 的数据；预取以后只剩 0.06 拍，load 几乎全藏在上一轮的乘加后面。每个 warp 的等待少了 13%，耗时只少 6%（167.4 µs 到 157.9 µs），因为一个 warp 等 load 的时候，同一个 SMSP 上的其他 warp 本来就能发射，这段等待原本就被藏住了一部分。剩下的时间主要是等 ld.shared 的结果和轮不到发射：FMA 管线只忙了 66%，shared memory 读只用了 55%，两边都没满。每个 SMSP 上只有 3 个 warp，一个 warp 等 ld.shared 时，没有足够多的别的 warp 补上。

<figure id="fig-t-6" class="fg-fig">
<svg class="fg" viewBox="0 0 640 236" width="100%" role="img" aria-label="Nsight Compute 实测 matmul kernel 每个 warp 两条指令之间平均隔几拍、各在等什么：num_stages = 1 时 0.65 拍在等显存 / L2 的数据，num_stages = 3 预取后只剩 0.06 拍，剩下的时间主要花在等 shared memory 和轮不到发射上">
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
  <text class="lab2" x="150" y="18">一个 warp 两条指令之间平均隔几拍，按原因分（A 4032 × 2048，B 2048 × 512）</text>
  <text class="tick" x="150.0" y="160" text-anchor="middle">0</text>
  <line class="grid" x1="150.0" y1="32" x2="150.0" y2="66"/>
  <line class="grid" x1="150.0" y1="92" x2="150.0" y2="126"/>
  <text class="tick" x="238.0" y="160" text-anchor="middle">1</text>
  <line class="grid" x1="238.0" y1="32" x2="238.0" y2="66"/>
  <line class="grid" x1="238.0" y1="92" x2="238.0" y2="126"/>
  <text class="tick" x="326.0" y="160" text-anchor="middle">2</text>
  <line class="grid" x1="326.0" y1="32" x2="326.0" y2="66"/>
  <line class="grid" x1="326.0" y1="92" x2="326.0" y2="126"/>
  <text class="tick" x="414.0" y="160" text-anchor="middle">3</text>
  <line class="grid" x1="414.0" y1="32" x2="414.0" y2="66"/>
  <line class="grid" x1="414.0" y1="92" x2="414.0" y2="126"/>
  <text class="tick" x="502.0" y="160" text-anchor="middle">4</text>
  <line class="grid" x1="502.0" y1="32" x2="502.0" y2="66"/>
  <line class="grid" x1="502.0" y1="92" x2="502.0" y2="126"/>
  <text class="tick" x="590.0" y="160" text-anchor="middle">5</text>
  <line class="grid" x1="590.0" y1="32" x2="590.0" y2="66"/>
  <line class="grid" x1="590.0" y1="92" x2="590.0" y2="126"/>
  <line class="grid" x1="150" y1="148" x2="590" y2="148"/>
  <text class="lab2" x="590" y="174" text-anchor="end">拍</text>
  <text class="lab" x="8" y="47">num_stages = 1</text><text class="lab2" x="8" y="61">不预取</text>
  <g class="m"><title>等显存 / L2 的数据：0.69 拍（long_scoreboard 0.69）</title><rect x="150.0" y="36" width="60.7" height="26" style="fill: var(--fig-hi)"/></g>
  <text class="s" x="180.4" y="53" text-anchor="middle" style="fill: var(--fig-on-hi)">0.69</text>
  <g class="m"><title>等 shared memory 的数据：0.93 拍（short_scoreboard 0.88，mio_throttle 0.05）</title><rect x="210.7" y="36" width="81.8" height="26" style="fill: var(--fig-2)"/></g>
  <text class="s" x="251.6" y="53" text-anchor="middle" style="fill: var(--fig-on-2)">0.93</text>
  <g class="m"><title>其他（barrier、wait 等）：0.49 拍（barrier 0.19，wait 0.23，math_pipe_throttle 0.03，no_instruction 0.04）</title><rect x="292.6" y="36" width="43.1" height="26" style="fill: var(--fig-mute)"/></g>
  <text class="s" x="314.1" y="53" text-anchor="middle" style="fill: var(--fig-on-mute)">0.49</text>
  <g class="m"><title>没被调度发出（not selected、dispatch）：1.45 拍（not_selected 0.94，dispatch_stall 0.51）</title><rect x="335.7" y="36" width="127.6" height="26" style="fill: var(--fig-3)"/></g>
  <text class="s" x="399.5" y="53" text-anchor="middle" style="fill: var(--fig-on-3)">1.45</text>
  <g class="m"><title>发射：1.00 拍（selected 1.00）</title><rect x="463.3" y="36" width="88.0" height="26" style="fill: var(--fig-1)"/></g>
  <text class="s" x="507.3" y="53" text-anchor="middle" style="fill: var(--fig-on-1)">1.00</text>
  <text class="s" x="557.3" y="53">4.59</text>
  <text class="lab2" x="150" y="76">167.4 µs；FMA 管线忙 60%，shared memory 读 50%，L2 命中 91.4%</text>
  <text class="lab" x="8" y="107">num_stages = 3</text><text class="lab2" x="8" y="121">预取，图 2-4</text>
  <g class="m"><title>等显存 / L2 的数据：0.06 拍（long_scoreboard 0.06）</title><rect x="150.0" y="96" width="5.3" height="26" style="fill: var(--fig-hi)"/></g>
  <text class="s" x="158.3" y="92" style="fill: var(--fig-hi)">0.06</text>
  <g class="m"><title>等 shared memory 的数据：0.62 拍（short_scoreboard 0.58，mio_throttle 0.04）</title><rect x="155.3" y="96" width="54.6" height="26" style="fill: var(--fig-2)"/></g>
  <text class="s" x="182.6" y="113" text-anchor="middle" style="fill: var(--fig-on-2)">0.62</text>
  <g class="m"><title>其他（barrier、wait 等）：0.41 拍（barrier 0.22，wait 0.10，math_pipe_throttle 0.05，no_instruction 0.04）</title><rect x="209.8" y="96" width="36.1" height="26" style="fill: var(--fig-mute)"/></g>
  <text class="s" x="227.9" y="113" text-anchor="middle" style="fill: var(--fig-on-mute)">0.41</text>
  <g class="m"><title>没被调度发出（not selected、dispatch）：1.90 拍（not_selected 1.21，dispatch_stall 0.69）</title><rect x="245.9" y="96" width="167.2" height="26" style="fill: var(--fig-3)"/></g>
  <text class="s" x="329.5" y="113" text-anchor="middle" style="fill: var(--fig-on-3)">1.90</text>
  <g class="m"><title>发射：1.00 拍（selected 1.00）</title><rect x="413.1" y="96" width="88.0" height="26" style="fill: var(--fig-1)"/></g>
  <text class="s" x="457.1" y="113" text-anchor="middle" style="fill: var(--fig-on-1)">1.00</text>
  <text class="s" x="507.1" y="113">3.98</text>
  <text class="lab2" x="150" y="136">157.9 µs；FMA 管线忙 66%，shared memory 读 55%，L2 命中 91.5%</text>
  <rect x="8" y="189" width="10" height="10" rx="2" style="fill: var(--fig-hi)"/>
  <text class="lab" x="24" y="198">等显存 / L2 的数据</text>
  <rect x="153.6" y="189" width="10" height="10" rx="2" style="fill: var(--fig-2)"/>
  <text class="lab" x="169.6" y="198">等 shared memory 的数据</text>
  <rect x="334.6" y="189" width="10" height="10" rx="2" style="fill: var(--fig-mute)"/>
  <text class="lab" x="350.6" y="198">其他（barrier、wait 等）</text>
  <rect x="8" y="211" width="10" height="10" rx="2" style="fill: var(--fig-3); stroke: var(--fig-2)"/>
  <text class="lab" x="24" y="220">没被调度发出（not selected、dispatch）</text>
  <rect x="282.0" y="211" width="10" height="10" rx="2" style="fill: var(--fig-1)"/>
  <text class="lab" x="298.0" y="220">发射</text>
</svg>
<figcaption><strong>图 2-4b</strong> matmul kernel 的 warp 在等什么（RTX 5090，fp32，Nsight Compute 实测）。用 A 4032 × 2048 是为了让 504 个 program 一批跑完，不受 tail effect 影响；ncu 不锁频率（<code>--clock-control none</code>），耗时用 do_bench 另测。悬停可看每段包含的 ncu 停顿原因。</figcaption>
</figure>

</details>

fp32 输入时 `tl.dot` 默认走 tf32 的 Tensor core，结果和 fp32 矩阵乘差 0.1 左右；要和 PyTorch（关掉 tf32）对上，得加 `input_precision="ieee"`，这时 `tl.dot` 编译成 CUDA core 上的 `fma.rn.f32x2`，不走 Tensor core。

第 2 节的四个例子是同一个模式：program 把一个 block 的数据读进 register，在片上做完能做的计算，再写回显存。[下一篇](/blog/gpu-training-analysis/)在 5090 上拆解一步训练的时间和显存，用的就是这个模式：把 RMSNorm 的几个 kernel 融合成一个，少读写一遍中间结果。

---

## 附录 A TPU：更大的矩阵单元，更简单的控制 {#tpu}

正文的 GPU 靠 kernel 自己切 block 来复用数据。TPU 面对同样的带宽问题，思路是把控制逻辑做轻，把矩阵乘单元做大。

[图 A-1](#fig-h-3) 是 TPU v5p 的一个 TensorCore：Scalar unit 取指令、发给 VPU 和 MXU，整个核只有这一路控制，没有 warp 和调度器之间的切换；VPU 做逐元素运算和归约；矩阵乘交给 4 个 MXU，每个是 128 × 128 的 systolic array。片上存储只有 VMEM 和 VREG 两层，VMEM 是编译器管理的 scratchpad，不是硬件 cache，数据什么时候从 HBM 搬进来、搬多少，都由编译器（XLA）提前排好。

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

MXU 和 GPU 最大的不同在乘加单元怎么拿到数据（[图 A-2](#fig-h-4)）。GPU 上每个 CUDA core 做一次乘加，要从 register file 读两个数、写回一个；一个数被用很多次，得靠 kernel 自己切 block，先搬进 shared memory 和 register 再反复读，[2.4 节](#triton-matmul)的 matmul 就是这样写的。systolic array 把复用做进了硬件：权重先装进 128 × 128 个格子，固定不动；输入 x 从左边进入，每拍向右传一格，经过的每一格把它乘上自己的权重，加到从上面传下来的部分和上，再把部分和往下传。一个 x 从 VMEM 读一次，在一行里被 128 个格子用到；稳定以后每拍从左边进 128 个数、从底部出 128 个结果，阵列里同时做 16384 次乘加，中间结果都不经过 register file。

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
  <text class="lab2" x="20" y="234">一个数要被用很多次，得靠 kernel 自己切 block：</text>
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
3. NVIDIA. [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)、[NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)、[NVIDIA Blackwell Architecture](https://resources.nvidia.com/en-us-blackwell-architecture)。表 1-1a 的 A100、H100、B200 一列。
4. NVIDIA. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)。各 compute capability 每 SM 的 warp、block、register 和 shared memory 上限，表 1-1、表 1-1a，CUDA 执行层级。
5. NVIDIA. [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。warp 停顿原因的定义，图 2-4b。
6. Austin et al. [How to Scale Your Model](https://jax-ml.github.io/scaling-book/gpus/), Google DeepMind, 2025. 附录 A 的 TPU v5p 规格、systolic array 的数据流和表 A-1。
7. Jouppi et al. [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760). ISCA 2017. 第一代 TPU 和它的 systolic array。
8. Kung. [Why Systolic Architectures?](https://doi.org/10.1109/MC.1982.1653825). IEEE Computer, 1982.
9. Tillet, Kung, Cox. [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://doi.org/10.1145/3315508.3329973). MAPL 2019.
10. [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)。第 2 节 softmax 和 matmul 例子的写法。
11. Hendrycks, Gimpel. [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415). arXiv:1606.08415, 2016. 第 2.1 节的 tanh 近似。
12. Milakov, Gimelshein. [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867). arXiv:1805.02867, 2018. 第 2.3 节提到的 online softmax。
