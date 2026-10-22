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

这个系列讲怎么让 Transformer 训练在 GPU 上跑得更快，这一篇打基础：

- **[第 1 节](#hardware) GPU 硬件**：存储层级、SM 结构、CUDA 执行层级
- **[第 2 节](#triton) Triton 入门**：四个例子，后面的 kernel 都用它写
- **[第 3 节](#tune) 调度与调参**：warp 藏延迟、一个 SM 放几个 program、分批，`num_warps` / `num_stages` / block 大小
- **Appendix**：[A](#specs) 几代 GPU 规格，[B](#tpu) TPU 对照，[C](#tma) TMA 与异步同步

下一篇[单卡 GPU 优化方法](/blog/gpu-single-gpu-optimization/)把单卡上常用的优化方法放在一起，第三篇[谁偷走了 5090 的算力和显存](/blog/gpu-training-analysis/)用这些知识在 RTX 5090 上拆解一步训练的时间和显存，第四篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/) 讲 attention 的 IO-aware kernel。

> 环境：RTX 5090 32 GB，torch 2.11.0+cu130，Triton 3.6.0。文中的 KB、MB、GB 都按 1024 进位。

---

## 1 GPU 硬件 {#hardware}

一个 kernel 受算力还是显存带宽限制，取决于它要搬多少数据、在哪一层读写。

### 1.1 存储层级 {#hw-memory}

过去 20 年算力涨了约 6 万倍，显存带宽只涨了约 100 倍（[图 1-1](#fig-memwall)），所以越来越多的 op 卡在搬数据上（memory-bound）。

<figure id="fig-memwall" class="fg-fig">
<img src="memory-wall.webp" alt="1997 到 2023 年峰值算力、DRAM 带宽和互联带宽的增长：算力每两年 3.0 倍，DRAM 带宽每两年 1.6 倍，互联带宽每两年 1.4 倍" loading="lazy" style="width: 100%; height: auto; border-radius: 8px; background: #fff;">
<figcaption><strong>图 1-1</strong> 峰值算力与 DRAM、互联带宽 20 年来的增长，纵轴是相对 1997 年的倍数（对数轴）。图来自 Gholami 等人的 <a href="https://arxiv.org/abs/2403.14123">AI and Memory Wall</a>（IEEE Micro，2024）。</figcaption>
</figure>

离计算单元越近越快、越小（[图 1-2](#fig-h-1)）。L1、L2 是硬件管的 cache，kernel 能自己安排的只有 register 和 shared memory。优化的核心就一条：把要反复用的数据留在这两层，多算几次再写回显存。

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

#### Memory access {#hw-move}

计算单元只读写 register。register 归一个 thread，shared memory 归一个 block（[1.3 节](#hw-model)），L2 和显存全卡共享。[表 1-1](#tab-1-1) 是各层之间怎么搬，指令名用的是 PTX（GPU 上的汇编，CUDA 和 Triton 都先编译成它）的写法。

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

显存到 shared memory 有三种搬法，差别在经不经过 register、谁来算地址（[图 1-3](#fig-h-10)）。

<figure id="fig-h-10" class="fg-fig">
<svg class="fg" viewBox="0 0 640 150" width="100%" role="img" aria-label="显存到 shared memory 的三种搬法：ld.global 加 st.shared 经过 L2 和 register；cp.async 和 TMA 经过 L2，直接写进 shared memory，不经过 register">
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
  <defs><marker id="fig-h-10-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <text class="lab2" x="200" y="20" text-anchor="middle">显存</text>
  <text class="lab2" x="310" y="20" text-anchor="middle">L2</text>
  <text class="lab2" x="420" y="20" text-anchor="middle">register</text>
  <text class="lab2" x="548" y="20" text-anchor="middle">shared memory</text>
  <text class="lab" x="8" y="43">ld.global + st.shared</text><text class="s" x="8" y="57">每个 thread 算地址</text>
  <rect x="160.0" y="32" width="80" height="26" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 14%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="270.0" y="32" width="80" height="26" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 45%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <rect x="380.0" y="32" width="80" height="26" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="493.0" y="32" width="110" height="26" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <line x1="242.0" y1="45.0" x2="268.0" y2="45.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-10-m0)"/>
  <line x1="352.0" y1="45.0" x2="378.0" y2="45.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-10-m0)"/>
  <line x1="462.0" y1="45.0" x2="491.0" y2="45.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-10-m0)"/>
  <text class="lab" x="8" y="81">cp.async</text><text class="s" x="8" y="95">每个 thread 算地址</text>
  <rect x="160.0" y="70" width="80" height="26" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 14%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="270.0" y="70" width="80" height="26" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 45%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <rect x="493.0" y="70" width="110" height="26" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <line x1="242.0" y1="83.0" x2="268.0" y2="83.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-10-m0)"/>
  <line x1="352.0" y1="83.0" x2="491.0" y2="83.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-10-m0)"/>
  <text class="lab" x="8" y="119">TMA</text><text class="s" x="8" y="133">1 个 thread 发，整块搬</text>
  <rect x="160.0" y="108" width="80" height="26" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 14%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <rect x="270.0" y="108" width="80" height="26" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 45%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <rect x="493.0" y="108" width="110" height="26" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <line x1="242.0" y1="121.0" x2="268.0" y2="121.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-10-m0)"/>
  <line x1="352.0" y1="121.0" x2="491.0" y2="121.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-10-m0)"/>
</svg>
<figcaption><strong>图 1-3</strong> 显存到 shared memory 的三种搬法。ld.global 先把数据读进 register，再用 st.shared 写进 shared memory；cp.async（Ampere 起）和 TMA（Hopper 起，见 <a href="#tma">Appendix C</a>）不经过 register，发出后 thread 接着往下跑，要用时再等。</figcaption>
</figure>

Triton 代码里只有 `tl.load` 和 `tl.store`，走[表 1-1](#tab-1-1)的哪一行由编译器决定：[第 2 节](#triton)的 GELU、softmax、row sum 都是显存和 register 之间直接读写；matmul 的 A、B 要经过 shared memory，编出来用的是 cp.async（[3.3 节](#stages)）；用 tensor descriptor 写的 load 才会用 TMA（[Appendix C](#tma)）。

register 和 L1 / shared memory 都在 SM 里面，下面看 SM 本身的结构。

### 1.2 SM 的结构 {#hw-sm}

每个 SM 分成 4 个 SMSP（SM sub-partition），每个 SMSP 有自己的 warp 调度器、register、32 个 CUDA core（做 FP32 和 INT32 运算）、一个 Tensor core 和 4 个 LD/ST 单元，4 个 SMSP 共享 L1 / shared memory（[图 1-4](#fig-h-2)）。load / store 从 LD/ST 发出，先到 L1 / shared memory，不在这里的再经片上互联去 L2 和显存。L1 和 shared memory 是同一块 128 KB 的片上存储，shared memory 最多切到 100 KB，下面的 notes 讲两者的区别和几种切法。

<figure id="fig-h-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 392" width="100%" role="img" aria-label="RTX 5090 的一个 SM：4 个 SMSP，每个有 1 个 warp 调度器、64 KB register、32 个 CUDA core、1 个 Tensor core 和 4 个 LD/ST；LD/ST 把请求交给 4 个 SMSP 共享的 128 KB L1/shared memory；TMA 接在 shared memory 和片上互联之间；请求经片上互联到 L2，再到显存">
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
  <defs><marker id="fig-h-2-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <rect x="10" y="10" width="620" height="306" rx="12" style="fill: color-mix(in srgb, var(--fig-1) 6%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="tb" x="24" y="32">SM（每颗 5090 有 170 个）</text>
  <rect x="24" y="44" width="141" height="148" rx="8" class="op"/>
  <text class="tb" x="94" y="62" text-anchor="middle">SMSP 0</text>
  <rect x="34" y="70" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="94" y="84" text-anchor="middle">warp 调度器</text>
  <rect x="34" y="94" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="94" y="108" text-anchor="middle">register 64 KB</text>
  <rect x="34" y="118" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="94" y="132" text-anchor="middle">CUDA core ×32</text>
  <rect x="34" y="142" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="94" y="156" text-anchor="middle">Tensor core ×1</text>
  <rect x="34" y="166" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 22%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="s" x="94" y="180" text-anchor="middle">LD/ST ×4</text>
  <line x1="89.0" y1="192.0" x2="89.0" y2="214.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <line x1="99.0" y1="214.0" x2="99.0" y2="192.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <rect x="175" y="44" width="141" height="148" rx="8" class="op"/>
  <text class="tb" x="245" y="62" text-anchor="middle">SMSP 1</text>
  <rect x="185" y="70" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="245" y="84" text-anchor="middle">warp 调度器</text>
  <rect x="185" y="94" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="245" y="108" text-anchor="middle">register 64 KB</text>
  <rect x="185" y="118" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="245" y="132" text-anchor="middle">CUDA core ×32</text>
  <rect x="185" y="142" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="245" y="156" text-anchor="middle">Tensor core ×1</text>
  <rect x="185" y="166" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 22%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="s" x="245" y="180" text-anchor="middle">LD/ST ×4</text>
  <line x1="240.0" y1="192.0" x2="240.0" y2="214.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <line x1="250.0" y1="214.0" x2="250.0" y2="192.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <rect x="326" y="44" width="141" height="148" rx="8" class="op"/>
  <text class="tb" x="396" y="62" text-anchor="middle">SMSP 2</text>
  <rect x="336" y="70" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="396" y="84" text-anchor="middle">warp 调度器</text>
  <rect x="336" y="94" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="396" y="108" text-anchor="middle">register 64 KB</text>
  <rect x="336" y="118" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="396" y="132" text-anchor="middle">CUDA core ×32</text>
  <rect x="336" y="142" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="396" y="156" text-anchor="middle">Tensor core ×1</text>
  <rect x="336" y="166" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 22%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="s" x="396" y="180" text-anchor="middle">LD/ST ×4</text>
  <line x1="391.0" y1="192.0" x2="391.0" y2="214.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <line x1="401.0" y1="214.0" x2="401.0" y2="192.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <rect x="477" y="44" width="141" height="148" rx="8" class="op"/>
  <text class="tb" x="547" y="62" text-anchor="middle">SMSP 3</text>
  <rect x="487" y="70" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="547" y="84" text-anchor="middle">warp 调度器</text>
  <rect x="487" y="94" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="547" y="108" text-anchor="middle">register 64 KB</text>
  <rect x="487" y="118" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="547" y="132" text-anchor="middle">CUDA core ×32</text>
  <rect x="487" y="142" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 22%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="547" y="156" text-anchor="middle">Tensor core ×1</text>
  <rect x="487" y="166" width="121" height="20" rx="4" style="fill: color-mix(in srgb, var(--fig-mute) 22%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="s" x="547" y="180" text-anchor="middle">LD/ST ×4</text>
  <line x1="542.0" y1="192.0" x2="542.0" y2="214.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <line x1="552.0" y1="214.0" x2="552.0" y2="192.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <rect x="24" y="214" width="594" height="38" rx="8" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="t" x="321" y="238" text-anchor="middle">L1 cache / shared memory 128 KB（4 个 SMSP 共享）</text>
  <rect x="447" y="274" width="171" height="34" rx="8" style="fill: color-mix(in srgb, var(--fig-hi) 14%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="tb" x="532" y="289" text-anchor="middle">TMA</text><text class="s" x="532" y="302" text-anchor="middle">整块搬，自己算地址</text>
  <line x1="527.0" y1="252.0" x2="527.0" y2="274.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <line x1="537.0" y1="274.0" x2="537.0" y2="252.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <rect x="24" y="346" width="594" height="30" rx="8" style="fill: color-mix(in srgb, var(--fig-mute) 14%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/>
  <text class="t" x="321" y="366" text-anchor="middle">片上互联（crossbar）→ L2（所有 SM 共享）→ 显存</text>
  <line x1="195.0" y1="252.0" x2="195.0" y2="346.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <line x1="205.0" y1="346.0" x2="205.0" y2="252.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <line x1="527.0" y1="308.0" x2="527.0" y2="346.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <line x1="537.0" y1="346.0" x2="537.0" y2="308.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-h-2-m0)"/>
  <text class="s" x="214" y="300">LD/ST 的请求</text><text class="s" x="434" y="330" text-anchor="end">TMA 的请求</text>
</svg>
<figcaption><strong>图 1-4</strong> RTX 5090 一个 SM 的结构示意（SMSP 即 SM sub-partition）。每个 SMSP 有自己的 warp 调度器、register、32 个 CUDA core 和 1 个 Tensor core，4 个 SMSP 共享 L1 / shared memory。普通的 load / store 由 SMSP 里的 LD/ST 单元按每个 thread 的地址发出，经 L1 / shared memory、片上互联到 L2。TMA 是专门整块搬数据的单元，和 LD/ST 走同一条路，见 Appendix C。</figcaption>
</figure>

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

[图 1-4](#fig-h-2) 右下角的 TMA（Tensor Memory Accelerator）是 Hopper 起 SM 里专门在显存和 shared memory 之间整块搬数据的硬件单元。它和 LD/ST 走同一条路，区别在于只要一个 thread 交给它一块数据的描述，地址由它自己算。正文的 kernel 用不到它，细节放在 [Appendix C](#tma)。

再往里看一层，一个 SMSP 里有两种计算单元（[图 1-5](#fig-h-9)）。CUDA core 一次算一个数。最常用的是 FMA（fused multiply-add）指令，算 $d = a \times b + c$，乘和加在一条指令里做完、只舍入一次：一条 FMA 指令里 warp 的 32 个 thread 各用一个 CUDA core，各做一次乘加（2 FLOPs）。5090 一共 170 × 128 个 CUDA core，fp32 峰值 1.05e14 FLOPS。Tensor core 一次算一个小矩阵乘：一条指令由整个 warp 一起发出，算一小块 A × B 再加到累加器上。它只吃 bf16、fp16、tf32、fp8 这类低精度输入，bf16 的峰值是 2.1e14 FLOPS（fp32 累加），fp8 更高。第 2 节的 `tl.dot` 默认走 Tensor core，加减乘除在 CUDA core 上算，exp 这类函数在 SFU 上算。

SMSP 里另一块要紧的是 register file。每个 SMSP 有 16384 个 32-bit register（64 KB），一个 SM 共 65536 个。每个 thread 用几个 register 在编译时就定了，最多 255 个；block 放上 SM 时，它的每个 thread 从所在 SMSP 的 register file 里分到这么多个，只归自己读写，block 跑完才还回去。所以每个 thread 用的 register 越多，一个 SM 能同时放的 thread 越少，[3.1 节](#sched-warp)会算这笔账。

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
  <text class="lab2" x="400" y="87">register：按 thread 切给驻留的 thread，</text>
  <text class="lab2" x="400" y="102">每个 thread 最多 255 个，只归自己用</text>
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
<figcaption><strong>图 1-5</strong> RTX 5090 一个 SMSP 的结构示意，按 <a href="https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf">NVIDIA RTX Blackwell 白皮书</a>的图 5 画。Blackwell 的 CUDA core 既做 FP32 也做 INT32，同一拍只能做其中一种。</figcaption>
</figure>

[第 2 节](#triton) kernel 图里的「register」就是这块 register file，「计算单元」指其中的 CUDA core 和 Tensor core。几代 GPU 的规格对比见 [Appendix A](#specs)。

这些是硬件提供的资源。register 按 thread 分配，shared memory 按 block 分配，要知道一个 kernel 怎么用这些资源，得先看 CUDA 怎么把 thread 组织起来。

### 1.3 编程模型 {#hw-model}

CUDA 的执行模型叫 SIMT（single instruction, multiple threads）：一个 warp 的 32 个 thread 同一时刻执行同一条指令。thread 往上分四层：thread、warp（32 个 thread）、block（也叫 CTA，若干个 warp）、grid（一次 kernel 启动的全部 block），每层跑在哪个硬件上见[图 1-6](#fig-h-5)。

<figure id="fig-h-5" class="fg-fig">
<svg class="fg" viewBox="0 0 820 352" width="100%" style="--fg-minw: 692px" role="img" aria-label="CUDA 的执行层级：grid 由 block 组成，block 由 warp 组成，warp 由 32 个 thread 组成；分别跑在整张 GPU、一个 SM、一个 SMSP 和一条 lane 上；在 Triton 里，grid 自己定，block 叫 program，warp 只能定个数，thread 写不到">
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
  <text class="ttl" x="20" y="18">软件：一次 kernel 启动</text><text class="ttl" x="432" y="18">硬件：跑在哪里（RTX 5090）</text><text class="ttl" x="648" y="18">Triton 里</text>
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
  <rect x="648" y="30" width="166" height="66" rx="8" style="fill: var(--fig-bg); stroke: var(--fig-hi)" stroke-opacity=".7" stroke-width="1.2"/>
  <text class="tb" x="731" y="51.0" text-anchor="middle">grid</text>
  <text class="s" x="731" y="66.0" text-anchor="middle">kernel[grid](...) 自己定</text>
  <text class="s" x="731" y="80.0" text-anchor="middle">program 之间不能同步</text>
  <rect x="648" y="126" width="166" height="66" rx="8" style="fill: var(--fig-bg); stroke: var(--fig-hi)" stroke-opacity=".7" stroke-width="1.2"/>
  <text class="tb" x="731" y="147.0" text-anchor="middle">program</text>
  <text class="s" x="731" y="162.0" text-anchor="middle">tl.program_id 是它的编号</text>
  <text class="s" x="731" y="176.0" text-anchor="middle">shared memory 由编译器分配</text>
  <rect x="648" y="222" width="166" height="50" rx="8" style="fill: var(--fig-bg); stroke: var(--fig-hi)" stroke-opacity=".7" stroke-width="1.2"/>
  <text class="tb" x="731" y="242.0" text-anchor="middle">只给个数</text>
  <text class="s" x="731" y="257.0" text-anchor="middle">num_warps，默认 4</text>
  <rect x="648" y="302" width="166" height="42" rx="8" style="fill: var(--fig-bg); stroke: var(--fig-hi)" stroke-opacity=".7" stroke-width="1.2"/>
  <text class="tb" x="731" y="318.0" text-anchor="middle">写不到</text>
  <text class="s" x="731" y="333.0" text-anchor="middle">怎么分给 thread 由编译器定</text>
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
  <text class="lab" x="32" y="328">thread：自己的 register，按编号算出读哪个数</text>
</svg>
<figcaption><strong>图 1-6</strong> CUDA 的执行层级。左边从上到下逐层放大：grid 里的一个 block、block 里的一个 warp、warp 里的一个 thread；中间是每一层跑在哪个硬件上，对应<a href="#fig-h-2">图 1-3</a>；右边是这一层在 Triton 里怎么写。</figcaption>
</figure>

block 放到 SM 上以后一直驻留到跑完，一个 SM 可以同时驻留多个 block。一个 SM 能放几个 block、驻留的 warp 怎么轮流执行、block 多于全卡的名额时怎么分批跑，[第 3 节](#tune)再讲。

block 之间不能直接共享数据，只能通过显存交换，而这正是最慢的一层。所以要尽量把会重复读的数据放进同一个 block 里处理，这就是 tiling。在 CUDA 里做 tiling 要自己管 shared memory 和 thread 同步，[第 2 节](#triton)的 Triton 把这部分交给编译器。

---

## 2 Triton 入门 {#triton}

第 1 节说过，kernel 优化的套路是把数据读进片上，算完能算的再写回显存。Triton 就是按这个思路写 kernel 的语言：你只写一个 block 的数据怎么处理，也就是 [1.3 节](#hw-model)说的 tiling。Triton 把这个 block 叫作一个 program，block 内怎么分给 thread、要不要经过 shared memory 由编译器决定，CUDA 的各层在 Triton 里怎么写见[图 1-6](#fig-h-5)。先看这样写能换来什么，[表 2-1](#tab-2-1) 把这一节的四个例子各写三遍：用 PyTorch 的张量运算直接写（eager），同一份代码包一层 `torch.compile`，以及用 Triton 手写一个 kernel。

| | [GELU](#triton-gelu) | [softmax](#triton-softmax) | [row sum](#triton-rowsum) | [matmul + ReLU](#triton-matmul) |
|:--|--:|--:|--:|--:|
| eager | 194.7 | 244.0 | 51.3 | 158.5 |
| torch.compile | 47.3（4.1×） | 91.0（2.7×） | 48.2（1.1×） | 164.7（1.0×） |
| Triton | 46.6（4.2×） | 95.4（2.6×） | 48.9（1.0×） | 173.0（0.9×） |
{#tab-2-1 caption="**表 2-1** 同一个 op 的三种写法（µs，括号里是相对 eager 的加速）" note="RTX 5090，fp32，形状和各小节一样：GELU n = 8388608，softmax 和 row sum 4096 × 4096，matmul A 4096 × 2048、B 2048 × 512。eager 分别是按 tanh 公式逐个运算、<code>softmax_naive</code>、<code>x.sum(1)</code>、<code>torch.relu(A @ B)</code>，matmul 关掉了 tf32。Triton 一行是各小节的 kernel 套上 <code>triton.autotune</code> 以后的结果，候选和选中的配置见 3.4 节，其中 softmax 选中的配置比第 2 节的固定参数（90.8 µs）慢 5%；<code>torch.compile</code> 也会自己挑 block 大小和 <code>num_warps</code>，matmul 的另外两行直接调 cuBLAS。<code>triton.testing.do_bench</code> 跑 5 次取中位数。换一张卡数字会变，三行之间的对比关系一般不变。"}

GELU 和 softmax 的 eager 写法里，每个运算都单独跑一遍，中间结果要写回显存、下一个运算再读回来；`torch.compile` 和 Triton 都把它们合成一个 kernel，中间结果留在 register 里，所以 GELU 快了 4 倍多，softmax 快了 2.6 倍左右。两者差不多快，因为 `torch.compile` 在 GPU 上生成的也是 Triton kernel。row sum 本来只有一步求和，三种写法差不多。matmul + ReLU 的另外两行都调 cuBLAS，Triton 版慢 9%，原因在 [3.4 节](#tile)最后。

下文 program 就是 CUDA 的 block；单说 block 时指一块数据，比如 A 的 block、C 的 block；代码里的 `BLOCK`、`TILE` 是一段数据的长度。四个例子由浅入深（[图 2-1](#fig-o-1)）：grid 怎么切，看一个输出依赖哪些输入；program 里要不要循环，看这些输入能不能一次装进 register，装不下就一个 block 一个 block 地读，用累加器（accumulator）存中间结果。

<figure id="fig-o-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 452" width="100%" role="img" aria-label="四个例子的数据怎么分给 program：GELU 把 x 展平按 BLOCK 切段，一段一个 program，不循环；softmax 一行一个 program，整行一次读进来；row sum 一行一个 program，沿行一个 block 一个 block 读，累加器存部分和；matmul 每个 C block 一个 program，沿 K 循环，读 A 的一个 row block 和 B 的一个 column block">
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
  <defs><marker id="fig-o-1-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <defs><pattern id="fig-o-1-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-1)" stroke-opacity=".55" stroke-width="1.3"/></pattern></defs>
  <rect x="10" y="8" width="306" height="206" rx="10" class="band"/>
  <text class="tb" x="22" y="30">GELU</text><text class="s" x="304" y="30" text-anchor="end" style="fill: var(--fig-hi)">逐元素</text>
  <text class="s" x="22" y="182">x 展平，按 BLOCK 切段，一段一个 program</text>
  <text class="s" x="22" y="197">不循环：一段一次读进 register</text>
  <text class="lab2" x="22" y="66">x</text><text class="lab2" x="22" y="136">y</text>
  <rect x="40.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="40.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="54.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="54.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="68.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="68.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="82.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="82.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="96.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="96.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="110.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="110.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="124.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="124.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="138.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="138.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="152.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="152.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="166.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="166.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="180.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="180.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="194.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="194.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="208.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="208.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="222.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="222.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="236.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="236.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="250.0" y="54.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="250.0" y="124.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <text class="s" x="68.0" y="84" text-anchor="middle">段 0</text>
  <text class="s" x="124.0" y="84" text-anchor="middle">pid=1</text>
  <text class="s" x="180.0" y="84" text-anchor="middle">段 2</text>
  <text class="s" x="236.0" y="84" text-anchor="middle">段 3</text>
  <line x1="124.0" y1="90.0" x2="124.0" y2="120.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-o-1-m0)"/>
  <rect x="324" y="8" width="306" height="206" rx="10" class="band"/>
  <text class="tb" x="336" y="30">softmax</text><text class="s" x="618" y="30" text-anchor="end" style="fill: var(--fig-hi)">按行归约</text>
  <text class="s" x="336" y="182">一行一个 program</text>
  <text class="s" x="336" y="197">不循环：整行（max 和 sum 都要用）一次读进来</text>
  <rect x="338.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="352.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="366.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="380.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="394.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="408.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="422.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="338.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="352.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="366.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="380.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="394.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="408.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="422.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="338.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="352.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="366.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="380.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="394.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="408.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="422.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="338.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="352.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="366.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="380.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="394.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="408.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="422.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="338.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="352.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="366.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="380.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="394.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="408.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="422.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <text class="lab2" x="338" y="134">x</text>
  <line x1="442.0" y1="69.0" x2="474.0" y2="69.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-o-1-m0)"/>
  <rect x="480.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="494.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="508.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="522.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="536.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="550.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="564.0" y="48.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="480.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="494.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="508.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="522.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="536.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="550.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="564.0" y="62.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="480.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="494.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="508.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="522.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="536.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="550.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="564.0" y="76.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="480.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="494.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="508.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="522.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="536.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="550.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="564.0" y="90.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="480.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="494.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="508.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="522.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="536.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="550.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="564.0" y="104.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <text class="lab2" x="480" y="134">y</text>
  <text class="s" x="582" y="73.0">pid=1</text>
  <rect x="10" y="222" width="306" height="206" rx="10" class="band"/>
  <text class="tb" x="22" y="244">row sum</text><text class="s" x="304" y="244" text-anchor="end" style="fill: var(--fig-hi)">按行归约</text>
  <text class="s" x="22" y="396">一行一个 program</text>
  <text class="s" x="22" y="411">沿行循环：一轮读一个 block，累加器存部分和</text>
  <rect x="24.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="38.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="52.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="66.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="80.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="94.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="108.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="122.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="136.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="150.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="164.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="178.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="24.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="38.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="52.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="66.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="80.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="94.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="108.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="122.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="136.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="150.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="164.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="178.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="24.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="38.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="52.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="66.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="80.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="94.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="108.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="122.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="136.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="150.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="164.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="178.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="24.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="38.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="52.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="66.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="80.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="94.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="108.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="122.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="136.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="150.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="164.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="178.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="24.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="38.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="52.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="66.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="80.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="94.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="108.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="122.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="136.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="150.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="164.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="178.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="66.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="80.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="94.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="108.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="122.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="136.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="150.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="164.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="178.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <text class="lab2" x="24" y="348">x</text>
  <line x1="198.0" y1="283.0" x2="232.0" y2="283.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-o-1-m0)"/>
  <rect x="238.0" y="262.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="238.0" y="276.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="238.0" y="290.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="238.0" y="304.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="238.0" y="318.0" width="12.5" height="12.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <text class="lab2" x="238" y="348">s</text>
  <text class="s" x="258" y="287.0">pid=1</text>
  <rect x="324" y="222" width="306" height="206" rx="10" class="band"/>
  <text class="tb" x="336" y="244">matmul</text><text class="s" x="618" y="244" text-anchor="end" style="fill: var(--fig-hi)">block 乘加</text>
  <text class="s" x="336" y="396">C 的一个 block 一个 program，grid 是二维</text>
  <text class="s" x="336" y="411">沿 K 循环：一轮读 A、B 各一个 block，累加器存 C 的 block</text>
  <text class="tb" x="374.0" y="278" text-anchor="middle">C</text>
  <text class="tb" x="468.0" y="278" text-anchor="middle">A</text>
  <text class="tb" x="562.0" y="278" text-anchor="middle">B</text>
  <text class="t" x="421.0" y="325.0" text-anchor="middle">=</text>
  <text class="t" x="515.0" y="325.0" text-anchor="middle">@</text>
  <rect x="342.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="436.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="530.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="358.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="452.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="546.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="374.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="468.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="562.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="390.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="484.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="578.0" y="288.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="342.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="436.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="530.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="358.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 40%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="452.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="546.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="374.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="468.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="562.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="390.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="484.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="578.0" y="304.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="342.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="436.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="530.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="358.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="452.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="546.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="374.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="468.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="562.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="390.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="484.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="578.0" y="320.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="342.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="436.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="530.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="358.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="452.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="546.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>
  <rect x="374.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="468.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="562.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="390.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="484.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="578.0" y="336.0" width="14.5" height="14.5" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <text class="s" x="374.0" y="368" text-anchor="middle">pid = (1, 1)</text>
  <text class="s" x="515.0" y="368" text-anchor="middle">实心是第 0 轮读的 block，斜线是之后几轮</text>
</svg>
<figcaption><strong>图 2-1</strong> 四个例子的数据怎么分给 program。深蓝是 pid = 1 的 program 要读的数据，红色是它写回的输出；row sum 和 matmul 要循环，实心是第 0 轮读的 block，斜线是之后几轮要读的。matmul 那格一格是一个 block。每个例子的完整数据流图在后面各小节。</figcaption>
</figure>

每个例子配一张数据流图：左边是 grid，右边从上到下是计算单元、register、shared memory 和显存。`tl.load`、`tl.store` 画成显存和 register 或 shared memory 之间的一条箭头，中间经过的 L2 不画；计算单元那一行从左往右是执行顺序，register 和 shared memory 里的一块存储只画一次，横向跨过它存在的那段时间，块上标的是它的大小。

### 2.1 GELU {#triton-gelu}

GELU 对每个元素单独算，$\Phi$ 是标准正态分布的累积分布函数，实际用 tanh 近似：

<p align="center">$y_i = \mathrm{GELU}(x_i) = x_i\,\Phi(x_i) \approx \tfrac{1}{2}\,x_i\left(1 + \tanh\left(\sqrt{2/\pi}\,(x_i + 0.044715\,x_i^3)\right)\right)$</p>

<figure id="fig-t-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 356" width="100%" role="img" aria-label="GELU kernel：数组展平成一维按 BLOCK 切段，一段一个 program；tl.load 把一段从显存读进 register，计算单元算 GELU，tl.store 从 register 直接写回显存">
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
  <rect class="band" x="146" y="26" width="490" height="66" rx="8"/>
  <text class="lab2" x="152" y="56.5">计算</text>
  <text class="lab2" x="152" y="69.5">单元</text>
  <rect class="band" x="146" y="100" width="490" height="76" rx="8"/>
  <text class="lab2" x="152" y="142.0">register</text>
  <rect class="band" x="146" y="184" width="490" height="32" rx="8"/>
  <text class="lab2" x="152" y="197.5">shared</text>
  <text class="lab2" x="152" y="210.5">memory</text>
  <text class="s" x="204" y="204.0" style="fill: var(--fig-1); opacity: 1; font-weight: 600">用 0 B</text>
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
  <rect x="298.0" y="110" width="50" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="323.0" y="122" text-anchor="middle">offsets</text>
  <text class="s" x="323.0" y="134" text-anchor="middle" style="font-size: 9.5px; fill: var(--fig-1); opacity: 1">4 KB</text>
  <line x1="323.0" y1="86.0" x2="323.0" y2="108.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-1-m1)"/>
  <rect x="373.0" y="110" width="44" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="395.0" y="122" text-anchor="middle">x 段</text>
  <text class="s" x="395.0" y="134" text-anchor="middle" style="font-size: 9.5px; fill: var(--fig-1); opacity: 1">4 KB</text>
  <line x1="395.0" y1="250.0" x2="395.0" y2="140.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-1-m2)"/>
  <text class="s" x="399.0" y="199.0" text-anchor="start" style="fill: var(--fig-1)">tl.load</text>
  <line x1="409.0" y1="108.0" x2="451.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-1-m1)"/>
  <rect x="517.0" y="110" width="44" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 26%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="539.0" y="122" text-anchor="middle">y 段</text>
  <text class="s" x="539.0" y="134" text-anchor="middle" style="font-size: 9.5px; fill: var(--fig-1); opacity: 1">4 KB</text>
  <line x1="483.0" y1="86.0" x2="525.0" y2="108.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-1-m1)"/>
  <line x1="539.0" y1="140.0" x2="539.0" y2="250.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-1-m2)"/>
  <text class="s" x="543.0" y="199.0" text-anchor="start" style="fill: var(--fig-1)">tl.store</text>
  <text class="s" x="390.0" y="247" text-anchor="end" style="fill: var(--fig-1); opacity: 1">读 4 KB</text>
  <rect x="355.0" y="252" width="80.0" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <rect x="385.0" y="252.0" width="20.0" height="28.0" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="395.0" y="293" text-anchor="middle">整个 x：32 MB</text>
  <text class="s" x="534.0" y="247" text-anchor="end" style="fill: var(--fig-1); opacity: 1">写 4 KB</text>
  <rect x="499.0" y="252" width="80.0" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <rect x="529.0" y="252.0" width="20.0" height="28.0" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="539.0" y="293" text-anchor="middle">整个 y：32 MB</text>
</svg>
<figcaption><strong>图 2-2</strong> GELU kernel（n = 8388608，fp32）。左边是 grid，数组按 BLOCK = 1024 切段，一段一个 program。右边是 program 1 的数据流，泳道和箭头的含义见正文。显存泳道里的整条是整个 x、y（各 8388608 个 fp32，32 MB），深色一段是 program 1 读写的 1024 个（4 KB）；按比例只有整条的 1/8192，图里放大了。</figcaption>
</figure>

逐元素 op 最简单，每个元素各算各的，program 之间不需要配合。把 n 个元素切成每段 BLOCK 个，一段交给一个 program（[图 2-2](#fig-t-1)）。kernel 里先用 `tl.program_id` 算出自己负责的下标，读进来，算完写回：

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
def softmax_naive(x):                         # x: [M, N]，按行
    m = x.max(dim=1, keepdim=True).values     # 读 MN，写 M
    z = x - m                                 # 读 MN + M，写 MN
    e = torch.exp(z)                          # 读 MN，写 MN
    s = e.sum(dim=1, keepdim=True)            # 读 MN，写 M
    return e / s                              # 读 MN + M，写 MN
```

<figure id="fig-t-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 371" width="100%" role="img" aria-label="softmax kernel：一行一个 program；tl.load 把一整行从显存读进 register，计算单元减最大值、exp、求和、归一化，中间结果都在 register 里，跨 warp 的归约经 shared memory，最后 tl.store 写回">
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
  <text class="lab2" x="152" y="324">x、x − m、e、y 是这一行在不同时刻的值，都在同一块 register 里，每个 thread 存 32 个数；</text>
  <text class="lab2" x="152" y="339">tl.max、tl.sum 先在 warp 内归约，各 warp 的部分结果经 shared memory 交换（同一块 16 B）；</text>
  <text class="lab2" x="152" y="354">显存只读 x 一次、写 y 一次。</text>
  <rect x="224.0" y="112" width="392.0" height="32" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 14%, transparent); stroke: var(--fig-2)" stroke-width="1.2"/>
  <line x1="326.0" y1="115" x2="326.0" y2="129" style="stroke: var(--fig-2)" stroke-dasharray="2 2"/>
  <line x1="420.0" y1="115" x2="420.0" y2="129" style="stroke: var(--fig-2)" stroke-dasharray="2 2"/>
  <line x1="514.0" y1="115" x2="514.0" y2="129" style="stroke: var(--fig-2)" stroke-dasharray="2 2"/>
  <text class="s" x="290.0" y="125" text-anchor="middle">x</text>
  <text class="s" x="373.0" y="125" text-anchor="middle">x − m</text>
  <text class="s" x="467.0" y="125" text-anchor="middle">e</text>
  <text class="s" x="550.0" y="125" text-anchor="middle">y</text>
  <text class="s" x="420.0" y="139" text-anchor="middle" style="font-size: 9.5px; fill: var(--fig-1); opacity: 1">整行，16 KB</text>
  <line x1="254.0" y1="250.0" x2="254.0" y2="146.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-2-m1)"/>
  <text class="s" x="250.0" y="202.0" text-anchor="end" style="fill: var(--fig-1)">tl.load</text>
  <line x1="304.0" y1="110.0" x2="304.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <line x1="348.0" y1="86.0" x2="348.0" y2="110.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <line x1="398.0" y1="110.0" x2="398.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <line x1="442.0" y1="86.0" x2="442.0" y2="110.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <line x1="492.0" y1="110.0" x2="492.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <line x1="536.0" y1="86.0" x2="536.0" y2="110.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-2-m2)"/>
  <line x1="586.0" y1="146.0" x2="586.0" y2="250.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-2-m1)"/>
  <text class="s" x="590.0" y="202.0" text-anchor="start" style="fill: var(--fig-1)">tl.store</text>
  <text class="s" x="249.0" y="247" text-anchor="end" style="fill: var(--fig-1); opacity: 1">读 1 行，16 KB</text>
  <rect x="218.0" y="252" width="72.0" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <rect x="218.0" y="260.4" width="72.0" height="7.0" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="254.0" y="293" text-anchor="middle">整个 x：64 MB</text>
  <text class="s" x="581.0" y="247" text-anchor="end" style="fill: var(--fig-1); opacity: 1">写 1 行，16 KB</text>
  <rect x="550.0" y="252" width="72.0" height="28" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <rect x="550.0" y="260.4" width="72.0" height="7.0" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="586.0" y="293" text-anchor="middle">整个 y：64 MB</text>
  <rect x="286.0" y="191" width="80" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 40%, transparent); stroke: var(--fig-3)" stroke-width="1.4"/><text class="s" x="326.0" y="204" text-anchor="middle">warp max，16 B</text>
  <line x1="321.0" y1="86.0" x2="321.0" y2="189.0" style="stroke: var(--nv-bg, #fff)" stroke-width="5"/>
  <line x1="321.0" y1="86.0" x2="321.0" y2="189.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-2-m2)"/>
  <line x1="331.0" y1="189.0" x2="331.0" y2="86.0" style="stroke: var(--nv-bg, #fff)" stroke-width="5"/>
  <line x1="331.0" y1="189.0" x2="331.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-2-m2)"/>
  <rect x="474.0" y="191" width="80" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 40%, transparent); stroke: var(--fig-3)" stroke-width="1.4"/><text class="s" x="514.0" y="204" text-anchor="middle">warp sum，16 B</text>
  <line x1="509.0" y1="86.0" x2="509.0" y2="189.0" style="stroke: var(--nv-bg, #fff)" stroke-width="5"/>
  <line x1="509.0" y1="86.0" x2="509.0" y2="189.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-2-m2)"/>
  <line x1="519.0" y1="189.0" x2="519.0" y2="86.0" style="stroke: var(--nv-bg, #fff)" stroke-width="5"/>
  <line x1="519.0" y1="189.0" x2="519.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-2-m2)"/>
</svg>
<figcaption><strong>图 2-3</strong> softmax kernel（4096 × 4096，fp32），画法同<a href="#fig-t-1">图 2-2</a>。一行一个 program，中间结果都留在 register 里，显存只读写各一次。显存泳道里的方块是整个 x、y（各 64 MB），深色一条是 program 1 读写的那一行（4096 个 fp32，16 KB），按比例只有 1/4096，图里放大了。</figcaption>
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

启动时一行一个 program：

```python {role="grid" tag="一行一个 program"}
def softmax(x):
    M, N = x.shape
    y = torch.empty_like(x)
    softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK=triton.next_power_of_2(N))
    return y
```

[图 2-3](#fig-t-2) 里的 warp max、warp sum 是 shared memory 在这个 kernel 里唯一的用处：`tl.max`、`tl.sum` 分三步归约。每个 thread 先把自己手里的 32 个数归约成一个。再在 warp 内用 shuffle 归约，shuffle 让每个 thread 直接读同一个 warp 里另一个 thread 的 register，不经过 shared memory（[表 1-1](#tab-1-1)）。编出来的 PTX 用的是 `shfl.sync.bfly`：每一步 thread $i$ 读 thread $i \oplus d$（按位异或）的值并和自己的合并，$d$ 依次取 16、8、4、2、1，5 步以后 warp 里每个 thread 都拿到这个 warp 的最大值或和。最后 4 个 warp 的部分结果经 shared memory 交换（一共 16 B），再用 $d$ 取 2、1 的两步 shuffle 合并成整行的结果。把一整个 block 的数据放进 shared memory 什么时候值得，[2.4 节](#triton-matmul)再讲。

这个写法要求 BLOCK ≥ N，也就是一行能一次装进一个 program。行再长，program 就只能沿行一个 block 一个 block 地读，读过的 block 要留下一个中间结果。下一节先看最简单的按行归约：求和。

### 2.3 Row sum {#triton-rowsum}

row sum 对 `[M, N]` 的输入每行求和：

<p align="center">$s_i = \sum_{j=1}^{N} x_{ij} = \sum_{T} \ \sum_{j \in \mathcal{J}_T} x_{ij}$</p>

$\mathcal{J}_T$ 是第 $T$ 个 block 的 TILE 个列下标，代码里 TILE 取 1024。加法满足结合律，先把每个 block 加起来再相加，和一次加完是同一个数（浮点数下只差舍入误差），所以 program 可以沿行一个 block 一个 block 地读，每读一个 block 就加到累加器上（[图 2-4](#fig-t-3)）。

<figure id="fig-t-3" class="fg-fig">
<svg class="fg" viewBox="0 0 640 395" width="100%" role="img" aria-label="row sum kernel：一行一个 program；tl.load 一轮把一个 block 从显存读进 register，计算单元把它加到 acc 上，颜色深浅是 for 循环的顺序；acc 一直留在 register 里，最后归约成一个数，tl.store 写回">
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
  <text class="lab2" x="152" y="348">图里 TILE = 4（代码里 1024，标的大小按 1024 算），一行 16 个数分 4 个 block；</text>
  <text class="lab2" x="152" y="363">x、acc 各是一块 register，横向是时间：每轮 tl.load 把新的 block 覆盖进 x，acc 原地累加；</text>
  <text class="lab2" x="152" y="378">最后的 tl.sum 跨 4 个 warp，经 shared memory 交换。</text>
  <defs><linearGradient id="fig-t-3-acc" x1="0" x2="1" y1="0" y2="0"><stop offset="0.00" style="stop-color: var(--fig-1)" stop-opacity="0.12"/><stop offset="0.33" style="stop-color: var(--fig-1)" stop-opacity="0.34"/><stop offset="0.67" style="stop-color: var(--fig-1)" stop-opacity="0.56"/><stop offset="1.00" style="stop-color: var(--fig-1)" stop-opacity="0.78"/></linearGradient></defs>
  <rect x="279.7" y="102" width="250.7" height="32" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 14%, transparent); stroke: var(--fig-2)" stroke-width="1.2"/>
  <rect x="198.0" y="140" width="231.3" height="32" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 14%, transparent); stroke: var(--fig-2)" stroke-width="1.2"/>
  <text class="s" x="342.3" y="113" style="font-size: 9.5px; fill: var(--fig-1); opacity: 1">acc，4 KB</text><text class="s" x="260.0" y="151" style="font-size: 9.5px; fill: var(--fig-1); opacity: 1">x，4 KB</text>
  <rect x="210.0" y="154" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="210.0" y="154" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="215.5" y="165.5" text-anchor="middle" style="opacity: .9">1</text>
  <rect x="221.0" y="154" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="221.0" y="154" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="226.5" y="165.5" text-anchor="middle" style="opacity: .9">2</text>
  <rect x="232.0" y="154" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="232.0" y="154" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="237.5" y="165.5" text-anchor="middle" style="opacity: .9">0</text>
  <rect x="243.0" y="154" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="243.0" y="154" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="248.5" y="165.5" text-anchor="middle" style="opacity: .9">3</text>
  <line x1="232.0" y1="254.0" x2="232.0" y2="174.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-3-m1)"/>
  <text class="s" x="236.0" y="218.0" text-anchor="start" style="fill: var(--fig-1)">tl.load</text>
  <rect x="373.3" y="154" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="373.3" y="154" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="378.8" y="165.5" text-anchor="middle" style="fill: var(--fig-on-1)">0</text>
  <rect x="384.3" y="154" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="384.3" y="154" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="389.8" y="165.5" text-anchor="middle" style="fill: var(--fig-on-1)">3</text>
  <rect x="395.3" y="154" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="395.3" y="154" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="400.8" y="165.5" text-anchor="middle" style="fill: var(--fig-on-1)">1</text>
  <rect x="406.3" y="154" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="406.3" y="154" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="411.8" y="165.5" text-anchor="middle" style="fill: var(--fig-on-1)">2</text>
  <line x1="395.3" y1="254.0" x2="395.3" y2="174.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-3-m1)"/>
  <text class="s" x="399.3" y="218.0" text-anchor="start" style="fill: var(--fig-1)">tl.load</text>
  <text class="tb" x="354.3" y="167" text-anchor="middle">⋯</text><text class="tb" x="360.3" y="129" text-anchor="middle">⋯</text>
  <line x1="238.0" y1="138.0" x2="281.7" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <line x1="409.3" y1="138.0" x2="445.0" y2="86.0" style="stroke: var(--nv-bg, #fff)" stroke-width="5"/>
  <line x1="409.3" y1="138.0" x2="445.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <rect x="297.7" y="116" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="297.7" y="116" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="303.2" y="127.5" text-anchor="middle" style="opacity: .9">1</text>
  <rect x="308.7" y="116" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="308.7" y="116" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="314.2" y="127.5" text-anchor="middle" style="opacity: .9">2</text>
  <rect x="319.7" y="116" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="319.7" y="116" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="325.2" y="127.5" text-anchor="middle" style="opacity: .9">0</text>
  <rect x="330.7" y="116" width="11" height="16" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="330.7" y="116" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="336.2" y="127.5" text-anchor="middle" style="opacity: .9">3</text>
  <rect x="461.0" y="116" width="44" height="16" fill="url(#fig-t-3-acc)"/>
  <rect x="461.0" y="116" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="466.5" y="127.5" text-anchor="middle" style="opacity: .9">7</text>
  <rect x="472.0" y="116" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="477.5" y="127.5" text-anchor="middle" style="opacity: .9">8</text>
  <rect x="483.0" y="116" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="488.5" y="127.5" text-anchor="middle" style="fill: var(--fig-on-1)">4</text>
  <rect x="494.0" y="116" width="11" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="499.5" y="127.5" text-anchor="middle" style="fill: var(--fig-on-1)">6</text>
  <line x1="305.7" y1="86.0" x2="309.7" y2="100.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <line x1="469.0" y1="86.0" x2="473.0" y2="100.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <line x1="520.3" y1="100.0" x2="520.3" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <path d="M297.7,44 L297.7,30 L461.0,30 L461.0,43" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.5" marker-end="url(#fig-t-3-m3)"/>
  <text class="s halo" x="379.3" y="34" text-anchor="middle">acc 带到下一轮</text>
  <text class="s" x="379.3" y="34" text-anchor="middle" style="fill: var(--fig-hi)">acc 带到下一轮</text>
  <rect x="596.0" y="112" width="24" height="16" style="fill: var(--fig-1)" fill-opacity=".78"/><text class="s" x="608.0" y="123.5" text-anchor="middle" style="fill: var(--fig-on-1)">25</text>
  <line x1="564.3" y1="86.0" x2="598.0" y2="108.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-3-m2)"/>
  <rect x="502.3" y="191" width="80" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 40%, transparent); stroke: var(--fig-3)" stroke-width="1.4"/><text class="s" x="542.3" y="204" text-anchor="middle">warp sum，16 B</text>
  <line x1="540.3" y1="86.0" x2="540.3" y2="189.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-3-m2)"/>
  <line x1="550.3" y1="189.0" x2="550.3" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.3" marker-end="url(#fig-t-3-m2)"/>
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
  <text class="s" x="313.7" y="312" text-anchor="middle">整个 x [4096, 4096]：64 MB；program 1 只读行 1，16 KB</text>
  <rect x="600.0" y="238" width="16" height="16" fill="none"/><rect x="600.0" y="238" width="16" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <rect x="600.0" y="254" width="16" height="16" style="fill: var(--fig-1)" fill-opacity=".78"/><rect x="600.0" y="254" width="16" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <rect x="600.0" y="270" width="16" height="16" fill="none"/><rect x="600.0" y="270" width="16" height="16" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>
  <text class="s" x="608.0" y="265.5" text-anchor="middle" style="fill: var(--fig-on-1)">25</text>
  <rect x="600.0" y="238" width="16" height="48" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="608.0" y="299" text-anchor="middle">out</text>
  <line x1="608.0" y1="130.0" x2="608.0" y2="253.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-3-m1)"/>
  <text class="s" x="604.0" y="152.0" text-anchor="end" style="fill: var(--fig-1)">tl.store</text>
</svg>
<figcaption><strong>图 2-4</strong> row sum kernel（4096 × 4096，fp32），画法同<a href="#fig-t-1">图 2-2</a>。图里 TILE = 4，一行 16 个数分 4 个 block，代码里 TILE = 1024。颜色由淡到浓是 for 循环读 block 的顺序：第 0 轮后 acc 只有 block 0 的数，最后的 acc 是 4 个 block 逐位相加 [7, 8, 4, 6]，铺满整条渐变；中间两轮用 ⋯ 省略，最后 <code>tl.sum</code> 加成 25 写回。</figcaption>
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

softmax 不能直接这样按 block 读。$e^{x_{ij} - m_i}$ 要先知道整行的最大值 $m_i$，读完整行才知道。按 block 读时，累加器里要同时存目前的最大值和部分和，最大值变大时把已有的部分和乘上 $e^{m_{\text{old}} - m_{\text{new}}}$ 修正，这就是 online softmax，[FlashAttention](/blog/flashattention-1-to-4/) 那篇会讲。

前面三个例子里每个数只读一次，切 block 只是为了把工作分给不同的 program。矩阵乘不同：A 的每个元素要和 B 的一整列相乘，切 block 是为了让读进片上的数据被多次使用。它也要沿 K 循环，累加器是 C 的一个 block。

### 2.4 Matmul + ReLU {#triton-matmul}

`[M, K] × [K, N]` 的矩阵乘再接 ReLU：

<p align="center">$C_{ij} = \max\left(0,\ \sum_{k=1}^{K} A_{ik} B_{kj}\right)$</p>

把 M、N、K 分别按 BM、BN、BK 切成 block，记 $A_{IT}$ 是 A 的第 $I$ 个 row block 和第 $T$ 个 column block 交叉处的 block（BM × BK），$B_{TJ}$、$C_{IJ}$ 同理。$C_{IJ}$ 只用到 A 的第 $I$ 个 row block 和 B 的第 $J$ 个 column block，不同的 $C_{IJ}$ 可以交给不同的 program。对 $k$ 的求和再按 BK 分段，$\mathcal{K}_T$ 是第 $T$ 段的 BK 个下标：

<p align="center">$\sum_{k=1}^{K} A_{ik} B_{kj} = \sum_{T=1}^{K/\mathrm{BK}} \ \sum_{k \in \mathcal{K}_T} A_{ik} B_{kj}, \qquad C_{IJ} = \max\left(0,\ \sum_{T} A_{IT}\, B_{TJ}\right)$</p>

和 row sum 一样，分段加完再相加等于一次加完，所以每轮只读 $A_{IT}$、$B_{TJ}$ 两个 block，乘完累加，最后就是完整的 $C_{IJ}$。ReLU 逐元素作用在累加完的结果上，放在最后做。

切 block 能把每个输出从显存读的数降到原来的 1/T，这笔账在[单卡 GPU 优化方法](/blog/gpu-single-gpu-optimization/#tiling)里算。BM、BN、BK 都得是 2 的幂，这是 `tl.arange` 的要求；`tl.dot` 还要求 BK 至少 16。

<figure id="fig-t-4" class="fg-fig">
<svg class="fg" viewBox="0 0 640 447" width="100%" role="img" aria-label="matmul kernel（A 4096 × 2048，B 2048 × 512）：C 的每个 block 一个 program；program (1, 1) 沿 K 一轮用 cp.async 把 A、B 各一个 block 从显存经 L2 搬进 shared memory，tl.dot 从 shared memory 读一小片进 register 做乘加，acc 留在 register 里，中间几轮用 ⋯ 省略，最后 ReLU 后 tl.store 写回 C 的 block (1, 1)">
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
  <rect class="band" x="146" y="26" width="490" height="66" rx="8"/>
  <text class="lab2" x="152" y="56.5">计算</text>
  <text class="lab2" x="152" y="69.5">单元</text>
  <rect class="band" x="146" y="100" width="490" height="52" rx="8"/>
  <text class="lab2" x="152" y="130.0">register</text>
  <rect class="band" x="146" y="160" width="490" height="76" rx="8"/>
  <text class="lab2" x="152" y="195.5">shared</text>
  <text class="lab2" x="152" y="208.5">memory</text>
  <text class="s" x="152" y="228.0" style="fill: var(--fig-1); opacity: 1; font-weight: 600">用 32 KB</text>
  <rect class="band" x="146" y="244" width="490" height="140" rx="8"/>
  <text class="lab2" x="152" y="318.0">显存</text>
  <text class="s" x="152" y="332.0">GDDR7</text>
  <line x1="90.0" y1="66.0" x2="144.0" y2="66.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-t-4-m0)"/>
  <rect class="op" x="262.0" y="46" width="71.3" height="38" rx="6"/>
  <text class="tb" x="297.7" y="62" text-anchor="middle">dot</text><text class="s" x="297.7" y="77" text-anchor="middle">tl.dot</text>
  <text class="tb" x="354.3" y="69.0" text-anchor="middle">⋯</text>
  <rect class="op" x="425.3" y="46" width="71.3" height="38" rx="6"/>
  <text class="tb" x="461.0" y="62" text-anchor="middle">dot</text><text class="s" x="461.0" y="77" text-anchor="middle">tl.dot</text>
  <rect class="op" x="506.7" y="46" width="71.3" height="38" rx="6"/>
  <text class="tb" x="542.3" y="62" text-anchor="middle">relu</text><text class="s" x="542.3" y="77" text-anchor="middle">tl.maximum</text>
  <text class="lab2" x="152" y="400">cp.async 把 A、B 的 block 从显存经 L2 搬进 shared memory，tl.dot 从中读 a、b 到 register 做 FMA；</text>
  <text class="lab2" x="152" y="415">K 按 BK = 32 切成 64 轮，只画 T = 0 和 63；两个 slot 轮流用，算第 T 轮时第 T + 1 轮在搬；</text>
  <text class="lab2" x="152" y="430">tl.store 先把 C block 经 shared memory 换一次排布（A、B 已用完，复用同一块），再写回显存。</text>
  <defs><linearGradient id="fig-t-4-acc" x1="0" x2="1" y1="0" y2="0"><stop offset="0.00" style="stop-color: var(--fig-1)" stop-opacity="0.12"/><stop offset="0.14" style="stop-color: var(--fig-1)" stop-opacity="0.21"/><stop offset="0.29" style="stop-color: var(--fig-1)" stop-opacity="0.31"/><stop offset="0.43" style="stop-color: var(--fig-1)" stop-opacity="0.40"/><stop offset="0.57" style="stop-color: var(--fig-1)" stop-opacity="0.50"/><stop offset="0.71" style="stop-color: var(--fig-1)" stop-opacity="0.59"/><stop offset="0.86" style="stop-color: var(--fig-1)" stop-opacity="0.69"/><stop offset="1.00" style="stop-color: var(--fig-1)" stop-opacity="0.78"/></linearGradient></defs>
  <rect x="285.7" y="103" width="344.3" height="21" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 14%, transparent); stroke: var(--fig-2)" stroke-width="1.2"/>
  <rect x="226.0" y="128" width="223.0" height="21" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 14%, transparent); stroke: var(--fig-2)" stroke-width="1.2"/>
  <rect x="542.3" y="103" width="87.7" height="21" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 45%, transparent); stroke: var(--fig-2)" stroke-width="1.2"/>
  <text class="s" x="354.3" y="117" text-anchor="middle">acc <tspan style="font-size: 9.5px; fill: var(--fig-1); opacity: 1">16 KB</tspan></text>
  <text class="s" x="586.2" y="117" text-anchor="middle">C block <tspan style="font-size: 9.5px; fill: var(--fig-1); opacity: 1">16 KB</tspan></text>
  <text class="s" x="346.3" y="142" text-anchor="middle">a、b <tspan style="font-size: 9.5px; fill: var(--fig-1); opacity: 1">各 8 KB</tspan></text>
  <line x1="275.7" y1="126.0" x2="275.7" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="311.7" y1="86.0" x2="311.7" y2="101.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="439.0" y1="126.0" x2="439.0" y2="86.0" style="stroke: var(--nv-bg, #fff)" stroke-width="5"/>
  <line x1="439.0" y1="126.0" x2="439.0" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="475.0" y1="86.0" x2="475.0" y2="101.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="528.3" y1="101.0" x2="528.3" y2="86.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="556.3" y1="86.0" x2="556.3" y2="101.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <path d="M297.7,44 L297.7,30 L461.0,30 L461.0,43" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.5" marker-end="url(#fig-t-4-m2)"/>
  <text class="s halo" x="379.3" y="34" text-anchor="middle">acc 带到下一轮</text>
  <text class="s" x="379.3" y="34" text-anchor="middle" style="fill: var(--fig-hi)">acc 带到下一轮</text>
  <text class="s" x="289.7" y="34" text-anchor="end" style="fill: var(--fig-hi)">T = 0</text><text class="s" x="471.0" y="34" style="fill: var(--fig-hi)">T = 63</text>
  <rect x="210.0" y="166" width="396.0" height="66" rx="5" style="fill: color-mix(in srgb, var(--fig-2) 14%, transparent); stroke: var(--fig-2)" stroke-width="1.2"/>
  <rect x="216.0" y="202" width="34" height="26" rx="3" style="fill: var(--fig-1)" fill-opacity="0.12"/><rect x="216.0" y="202" width="34" height="26" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="233.0" y="213" text-anchor="middle" >A</text><text class="s" x="233.0" y="224" text-anchor="middle" style="font-size: 9.5px" >8 KB</text>
  <rect x="250.0" y="202" width="34" height="26" rx="3" style="fill: var(--fig-1)" fill-opacity="0.12"/><rect x="250.0" y="202" width="34" height="26" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="267.0" y="213" text-anchor="middle" >B</text><text class="s" x="267.0" y="224" text-anchor="middle" style="font-size: 9.5px" >8 KB</text>
  <rect x="373.3" y="170" width="34" height="26" rx="3" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="373.3" y="170" width="34" height="26" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="390.3" y="181" text-anchor="middle" style="fill: var(--fig-on-1)">A</text><text class="s" x="390.3" y="192" text-anchor="middle" style="font-size: 9.5px" style="fill: var(--fig-on-1)">8 KB</text>
  <rect x="407.3" y="170" width="34" height="26" rx="3" style="fill: var(--fig-1)" fill-opacity="0.78"/><rect x="407.3" y="170" width="34" height="26" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"/>
  <text class="s" x="424.3" y="181" text-anchor="middle" style="fill: var(--fig-on-1)">B</text><text class="s" x="424.3" y="192" text-anchor="middle" style="font-size: 9.5px" style="fill: var(--fig-on-1)">8 KB</text>
  <rect x="294.0" y="170" width="68" height="26" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7" stroke-dasharray="3 2"/>
  <text class="s" x="328.0" y="181" text-anchor="middle" style="font-size: 9.5px">T = 1，16 KB</text><text class="s" x="328.0" y="192" text-anchor="middle" style="font-size: 9.5px">正在搬</text>
  <text class="s" x="465.0" y="186">slot 1</text><text class="s" x="465.0" y="218">slot 0</text>
  <line x1="250.0" y1="200.0" x2="250.0" y2="151.0" style="stroke: var(--nv-bg, #fff)" stroke-width="5"/>
  <line x1="250.0" y1="200.0" x2="250.0" y2="151.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="407.3" y1="168.0" x2="407.3" y2="151.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <rect x="538.3" y="170" width="39.7" height="58" rx="3" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <text class="s" x="558.2" y="194" text-anchor="middle" style="fill: var(--fig-on-1)">tl.store</text><text class="s" x="558.2" y="208" text-anchor="middle" style="fill: var(--fig-on-1)">换排布</text>
  <line x1="556.0" y1="126.0" x2="556.0" y2="168.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
  <line x1="568.0" y1="168.0" x2="568.0" y2="126.0" style="stroke: var(--fig-2)" stroke-width="1.4" marker-end="url(#fig-t-4-m1)"/>
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
  <rect x="202.0" y="262" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="202.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="262" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.2142857142857143"/>
  <rect x="208.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="262" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.3085714285714286"/>
  <rect x="214.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="262" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.40285714285714286"/>
  <rect x="220.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="262" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.49714285714285716"/>
  <rect x="226.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="262" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.5914285714285714"/>
  <rect x="232.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="262" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.6857142857142857"/>
  <rect x="238.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="262" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.78"/>
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
  <rect x="202.0" y="340" width="6" height="6" fill="none"/>
  <rect x="202.0" y="340" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="340" width="6" height="6" fill="none"/>
  <rect x="208.0" y="340" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="340" width="6" height="6" fill="none"/>
  <rect x="214.0" y="340" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="340" width="6" height="6" fill="none"/>
  <rect x="220.0" y="340" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="340" width="6" height="6" fill="none"/>
  <rect x="226.0" y="340" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="340" width="6" height="6" fill="none"/>
  <rect x="232.0" y="340" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="340" width="6" height="6" fill="none"/>
  <rect x="238.0" y="340" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="340" width="6" height="6" fill="none"/>
  <rect x="244.0" y="340" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="346" width="6" height="6" fill="none"/>
  <rect x="202.0" y="346" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="208.0" y="346" width="6" height="6" fill="none"/>
  <rect x="208.0" y="346" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="214.0" y="346" width="6" height="6" fill="none"/>
  <rect x="214.0" y="346" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="220.0" y="346" width="6" height="6" fill="none"/>
  <rect x="220.0" y="346" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="226.0" y="346" width="6" height="6" fill="none"/>
  <rect x="226.0" y="346" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="232.0" y="346" width="6" height="6" fill="none"/>
  <rect x="232.0" y="346" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="238.0" y="346" width="6" height="6" fill="none"/>
  <rect x="238.0" y="346" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="244.0" y="346" width="6" height="6" fill="none"/>
  <rect x="244.0" y="346" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="202.0" y="256" width="48" height="96" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="226.0" y="365.0" text-anchor="middle">A [4096, 2048]</text>
  <text class="s" x="226.0" y="378.0" text-anchor="middle">整个 32 MB</text>
  <rect x="258.0" y="256" width="6" height="6" fill="none"/>
  <rect x="258.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="256" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.12"/>
  <rect x="264.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="262" width="6" height="6" fill="none"/>
  <rect x="258.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="262" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.2142857142857143"/>
  <rect x="264.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="268" width="6" height="6" fill="none"/>
  <rect x="258.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="268" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.3085714285714286"/>
  <rect x="264.0" y="268" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="274" width="6" height="6" fill="none"/>
  <rect x="258.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="274" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.40285714285714286"/>
  <rect x="264.0" y="274" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="280" width="6" height="6" fill="none"/>
  <rect x="258.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="280" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.49714285714285716"/>
  <rect x="264.0" y="280" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="286" width="6" height="6" fill="none"/>
  <rect x="258.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="286" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.5914285714285714"/>
  <rect x="264.0" y="286" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="292" width="6" height="6" fill="none"/>
  <rect x="258.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="292" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.6857142857142857"/>
  <rect x="264.0" y="292" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="298" width="6" height="6" fill="none"/>
  <rect x="258.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="264.0" y="298" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.78"/>
  <rect x="264.0" y="298" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="258.0" y="256" width="12" height="48" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="276.0" y="270.0" text-anchor="start">B [2048, 512]</text>
  <text class="s" x="276.0" y="283.0" text-anchor="start">整个 4 MB</text>
  <line x1="205.0" y1="262.0" x2="233.0" y2="230.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-4-m3)"/>
  <line x1="267.0" y1="256.0" x2="267.0" y2="230.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-4-m3)"/>
  <text class="s" x="271.0" y="248.0" text-anchor="start" style="fill: var(--fig-1)">cp.async</text>
  <rect x="599.0" y="256" width="6" height="6" fill="none"/>
  <rect x="599.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="256" width="6" height="6" fill="none"/>
  <rect x="605.0" y="256" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="262" width="6" height="6" fill="none"/>
  <rect x="599.0" y="262" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="262" width="6" height="6" style="fill: var(--fig-1)" fill-opacity="0.78"/>
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
  <rect x="599.0" y="340" width="6" height="6" fill="none"/>
  <rect x="599.0" y="340" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="340" width="6" height="6" fill="none"/>
  <rect x="605.0" y="340" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="346" width="6" height="6" fill="none"/>
  <rect x="599.0" y="346" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="605.0" y="346" width="6" height="6" fill="none"/>
  <rect x="605.0" y="346" width="6" height="6" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>
  <rect x="599.0" y="256" width="12" height="96" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="630.0" y="365.0" text-anchor="end">C [4096, 512]</text>
  <text class="s" x="630.0" y="378.0" text-anchor="end">整个 8 MB</text>
  <line x1="618.0" y1="126.0" x2="608.0" y2="261.0" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#fig-t-4-m3)"/>
  <text class="s" x="606.0" y="248" text-anchor="end" style="fill: var(--fig-1)">tl.store：写 16 KB</text>
  <text class="s" x="276.0" y="308">沿 K 第 T 轮读的 block：T = 0 … 63</text>
  <rect x="276.0" y="314" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.12" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="288.0" y="314" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.21" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="300.0" y="314" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.31" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="312.0" y="314" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.40" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="324.0" y="314" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.50" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="336.0" y="314" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.59" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="348.0" y="314" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.69" stroke-opacity=".4" stroke-width=".6"/>
  <rect x="360.0" y="314" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.78" stroke-opacity=".4" stroke-width=".6"/>
  <text class="s" x="276.0" y="337">T = 0</text><text class="s" x="372.0" y="337" text-anchor="end">T = 63</text>
  <text class="s" x="276.0" y="354">每轮读 A 的 64 × 32、B 的 32 × 64，各 8 KB；</text><text class="s" x="276.0" y="368">图里一格 256 列，是 8 轮</text>
</svg>
<figcaption><strong>图 2-5</strong> matmul kernel（A 4096 × 2048，B 2048 × 512，fp32，BM = BN = 64，BK = 32），画法同<a href="#fig-t-1">图 2-2</a>。显存里的矩阵按比例画全，一格是 256 × 256 个元素。program (1, 1) 读 A 的第 1 个 row block 和 B 的第 1 个 column block，颜色由淡到浓是沿 K 读 block 的顺序。acc 就是 C 的 block (1, 1)，64 轮 dot 在它上面一轮轮累加，ReLU 之后就是要写回的值，图里 ReLU 后的那段叫 C block。A、B 的 block 由 cp.async 从显存经 L2 搬进 shared memory，不经过 register。shared memory 里 A、B 各有 2 个 slot，因为默认 <code>num_stages=3</code>：第 T 轮的 block 在第 T − 2 轮算完时开始搬，第 T − 1 轮期间在路上，第 T 轮被读出来算；算完 slot 就空出来，接着搬第 T + 2 轮。所以偶数轮用 slot 0、奇数轮用 slot 1，算第 T 轮时另一个 slot 里是第 T + 1 轮，见 <a href="#stages">3.3 节</a>。</figcaption>
</figure>

一个 program 负责一个 $C_{IJ}$（[图 2-5](#fig-t-4)），沿 K 循环，第 $T$ 轮读 $A_{IT}$ 和 $B_{TJ}$，用 `tl.dot` 乘加到累加器上；写回前在 register 里做 ReLU，激活函数不用再单独一个 kernel 读写一遍 C：

```python {role="program" tag="算 C 的第 (pid_m, pid_n) 个 block"}
@triton.jit
def matmul_relu_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  # A [M, K]，B [K, N]，C [M, N]
                       stride_am, stride_ak,          # A[i, k] 的地址 = a_ptr + i·stride_am + k·stride_ak
                       stride_bk, stride_bn,          # B、C 同理；按行连续存时 stride_am = K、stride_ak = 1
                       stride_cm, stride_cn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):  # C 的 block 是 BM × BN，K 每轮走 BK
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)  # 这个 program 负责 C 的第 (pid_m, pid_n) 个 block
    offs_m = pid_m * BM + tl.arange(0, BM)    # 这个 block 的 BM 个行号
    offs_n = pid_n * BN + tl.arange(0, BN)    # 这个 block 的 BN 个列号
    offs_k = tl.arange(0, BK)                 # 一轮里 K 方向的 BK 个下标
    # [:, None] 变成列、[None, :] 变成行，相加广播成二维：A 的 [BM, BK] 个地址、B 的 [BK, BN] 个地址
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros([BM, BN], dtype=tl.float32)  # C 的这个 block，累加在 register 里
    for k in range(0, K, BK):                 # 第 k // BK 轮：读 A、B 各一个 block，乘加
        # mask 把越界的位置读成 0：M、N 不是 BM、BN 的倍数时边上的 block，K 不是 BK 的倍数时最后一轮
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")  # fp32 默认会走 tf32，这里要求精确的 fp32
        a_ptrs += BK * stride_ak              # A 的 block 往右挪 BK 列
        b_ptrs += BK * stride_bk              # B 的 block 往下挪 BK 行
    acc = tl.maximum(acc, 0.0)                # ReLU 在 register 里做完，不多一次读写
    # 写回 C 的这个 block，越界的位置不写
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
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

<figure id="fig-h-6" class="fg-fig">
<svg class="fg" viewBox="0 0 640 300" width="100%" role="img" aria-label="一个 SM 上驻留 3 个 program、每个 4 个 warp 时的某一拍：每个 SMSP 有 3 个 warp，分别来自 3 个 program；每个调度器这一拍最多发射一个，SMSP 2 的 3 个都在等数据，调度器空转">
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
  <text class="lab2" x="616" y="32" text-anchor="end">整个 SM 最多驻留 24 个 program、48 个 warp</text>
  <rect x="24" y="44" width="141" height="164" rx="8" class="op"/>
  <text class="tb" x="94" y="62" text-anchor="middle">SMSP 0</text>
  <rect x="34" y="70" width="121" height="22" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="94" y="85" text-anchor="middle">warp 调度器</text>
  <text class="s" x="94" y="110" text-anchor="middle">最多驻留 12 个 warp</text>
  <rect x="31" y="118" width="28" height="22" rx="3" style="fill: var(--fig-hi)"/>
  <text class="s" x="45" y="133" text-anchor="middle">P0</text>
  <rect x="64" y="118" width="28" height="22" rx="3" style="fill: var(--fig-2)"/>
  <text class="s" x="78" y="133" text-anchor="middle">P1</text>
  <rect x="97" y="118" width="28" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="111" y="133" text-anchor="middle">P2</text>
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
  <text class="s" x="196" y="133" text-anchor="middle">P0</text>
  <rect x="215" y="118" width="28" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="229" y="133" text-anchor="middle">P1</text>
  <rect x="248" y="118" width="28" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="262" y="133" text-anchor="middle">P2</text>
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
  <text class="s" x="347" y="133" text-anchor="middle">P0</text>
  <rect x="366" y="118" width="28" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="380" y="133" text-anchor="middle">P1</text>
  <rect x="399" y="118" width="28" height="22" rx="3" style="fill: var(--fig-1)" fill-opacity=".55"/>
  <text class="s" x="413" y="133" text-anchor="middle">P2</text>
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
  <text class="s" x="498" y="133" text-anchor="middle">P0</text>
  <rect x="517" y="118" width="28" height="22" rx="3" style="fill: var(--fig-2)"/>
  <text class="s" x="531" y="133" text-anchor="middle">P1</text>
  <rect x="550" y="118" width="28" height="22" rx="3" style="fill: var(--fig-2)"/>
  <text class="s" x="564" y="133" text-anchor="middle">P2</text>
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
  <text class="s" x="321" y="236" text-anchor="middle">P0、P1、P2 各有 4 个 warp：warp 0 到 3 依次分到 SMSP 0 到 3</text>
  <rect x="96" y="272" width="17" height="10" rx="2" style="fill: var(--fig-hi)"/><text class="lab2" x="120" y="281">正在发射</text>
  <rect x="200" y="272" width="17" height="10" rx="2" style="fill: var(--fig-2)"/><text class="lab2" x="224" y="281">就绪，没被选中</text>
  <rect x="343" y="272" width="17" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity=".55"/><text class="lab2" x="367" y="281">等数据</text>
  <rect x="434" y="272" width="17" height="10" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"/><text class="lab2" x="458" y="281">空名额</text>
</svg>
<figcaption><strong>图 3-1</strong> 一个 SM 上驻留 3 个 program（P0、P1、P2，每个 4 个 warp）时的某一拍。每个 program 的 4 个 warp 分到 4 个 SMSP，格子里写的是这个 warp 属于哪个 program；每个 SMSP 有 12 个格子，整个 SM 共 48 个，就是<a href="#tab-3-1">表 3-1</a> 的 warp 名额。每个调度器每拍最多发射一个 warp，SMSP 2 的 3 个 warp 都在等数据，这一拍空转。</figcaption>
</figure>

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
  <text class="lab2" x="74" y="28">某一时刻全卡已开始、未结束的 program 数，按批次堆叠；虚线是 170 × 12 = 2040 个名额</text>
  <text class="lab2" x="16" y="126.0" text-anchor="middle" transform="rotate(-90 16 126.0)">同时在跑的 program 数</text>
  <line class="ref" x1="74" y1="52.0" x2="620" y2="52.0"/>
  <g class="m"><title>第 1 批：program 0 到 2039，共 2040 个</title><polygon points="74.0,142.8 76.4,52.0 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.0 95.4,52.0 97.7,52.0 100.1,52.0 102.5,52.1 104.9,52.1 107.2,53.0 109.6,54.7 112.0,57.4 114.4,59.8 116.7,62.5 119.1,70.4 121.5,78.5 123.9,88.3 126.2,99.1 128.6,108.8 131.0,119.8 133.3,129.6 135.7,136.5 138.1,146.1 140.5,154.5 142.8,160.9 145.2,164.0 147.6,168.3 150.0,172.9 152.3,176.0 154.7,176.8 157.1,177.1 159.5,178.4 161.8,178.7 164.2,178.7 166.6,180.0 169.0,182.7 171.3,184.7 173.7,185.6 176.1,187.0 178.5,188.3 180.8,188.8 183.2,189.3 185.6,189.4 187.9,189.6 190.3,190.6 192.7,191.0 195.1,191.5 197.4,191.7 199.8,192.2 202.2,192.9 204.6,193.4 206.9,194.0 209.3,194.5 211.7,195.1 214.1,195.4 216.4,196.0 218.8,196.0 221.2,196.3 223.6,197.3 225.9,197.5 228.3,197.5 230.7,197.5 233.1,198.2 235.4,198.6 237.8,198.7 240.2,199.1 242.5,199.3 244.9,199.6 247.3,199.6 249.7,199.8 252.0,199.9 254.4,200.0 256.8,200.0 259.2,200.0 261.5,200.0 263.9,200.0 266.3,200.0 268.7,200.0 271.0,200.0 273.4,200.0 275.8,200.0 278.2,200.0 280.5,200.0 282.9,200.0 285.3,200.0 287.7,200.0 290.0,200.0 292.4,200.0 294.8,200.0 297.1,200.0 299.5,200.0 301.9,200.0 304.3,200.0 306.6,200.0 309.0,200.0 311.4,200.0 313.8,200.0 316.1,200.0 318.5,200.0 320.9,200.0 323.3,200.0 325.6,200.0 328.0,200.0 330.4,200.0 332.8,200.0 335.1,200.0 337.5,200.0 339.9,200.0 342.3,200.0 344.6,200.0 347.0,200.0 349.4,200.0 351.7,200.0 354.1,200.0 356.5,200.0 358.9,200.0 361.2,200.0 363.6,200.0 366.0,200.0 368.4,200.0 370.7,200.0 373.1,200.0 375.5,200.0 377.9,200.0 380.2,200.0 382.6,200.0 385.0,200.0 387.4,200.0 389.7,200.0 392.1,200.0 394.5,200.0 396.9,200.0 399.2,200.0 401.6,200.0 404.0,200.0 406.3,200.0 408.7,200.0 411.1,200.0 413.5,200.0 415.8,200.0 418.2,200.0 420.6,200.0 423.0,200.0 425.3,200.0 427.7,200.0 430.1,200.0 432.5,200.0 434.8,200.0 437.2,200.0 439.6,200.0 442.0,200.0 444.3,200.0 446.7,200.0 449.1,200.0 451.5,200.0 453.8,200.0 456.2,200.0 458.6,200.0 460.9,200.0 463.3,200.0 465.7,200.0 468.1,200.0 470.4,200.0 472.8,200.0 475.2,200.0 477.6,200.0 479.9,200.0 482.3,200.0 484.7,200.0 487.1,200.0 489.4,200.0 491.8,200.0 494.2,200.0 496.6,200.0 498.9,200.0 501.3,200.0 503.7,200.0 506.1,200.0 508.4,200.0 510.8,200.0 513.2,200.0 515.5,200.0 517.9,200.0 520.3,200.0 522.7,200.0 525.0,200.0 527.4,200.0 529.8,200.0 532.2,200.0 534.5,200.0 536.9,200.0 539.3,200.0 541.7,200.0 544.0,200.0 546.4,200.0 548.8,200.0 551.2,200.0 553.5,200.0 555.9,200.0 558.3,200.0 560.7,200.0 563.0,200.0 565.4,200.0 567.8,200.0 570.1,200.0 572.5,200.0 574.9,200.0 577.3,200.0 579.6,200.0 582.0,200.0 584.4,200.0 586.8,200.0 589.1,200.0 591.5,200.0 593.9,200.0 596.3,200.0 598.6,200.0 601.0,200.0 603.4,200.0 605.8,200.0 608.1,200.0 610.5,200.0 612.9,200.0 615.3,200.0 617.6,200.0 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,200.0 563.0,200.0 560.7,200.0 558.3,200.0 555.9,200.0 553.5,200.0 551.2,200.0 548.8,200.0 546.4,200.0 544.0,200.0 541.7,200.0 539.3,200.0 536.9,200.0 534.5,200.0 532.2,200.0 529.8,200.0 527.4,200.0 525.0,200.0 522.7,200.0 520.3,200.0 517.9,200.0 515.5,200.0 513.2,200.0 510.8,200.0 508.4,200.0 506.1,200.0 503.7,200.0 501.3,200.0 498.9,200.0 496.6,200.0 494.2,200.0 491.8,200.0 489.4,200.0 487.1,200.0 484.7,200.0 482.3,200.0 479.9,200.0 477.6,200.0 475.2,200.0 472.8,200.0 470.4,200.0 468.1,200.0 465.7,200.0 463.3,200.0 460.9,200.0 458.6,200.0 456.2,200.0 453.8,200.0 451.5,200.0 449.1,200.0 446.7,200.0 444.3,200.0 442.0,200.0 439.6,200.0 437.2,200.0 434.8,200.0 432.5,200.0 430.1,200.0 427.7,200.0 425.3,200.0 423.0,200.0 420.6,200.0 418.2,200.0 415.8,200.0 413.5,200.0 411.1,200.0 408.7,200.0 406.3,200.0 404.0,200.0 401.6,200.0 399.2,200.0 396.9,200.0 394.5,200.0 392.1,200.0 389.7,200.0 387.4,200.0 385.0,200.0 382.6,200.0 380.2,200.0 377.9,200.0 375.5,200.0 373.1,200.0 370.7,200.0 368.4,200.0 366.0,200.0 363.6,200.0 361.2,200.0 358.9,200.0 356.5,200.0 354.1,200.0 351.7,200.0 349.4,200.0 347.0,200.0 344.6,200.0 342.3,200.0 339.9,200.0 337.5,200.0 335.1,200.0 332.8,200.0 330.4,200.0 328.0,200.0 325.6,200.0 323.3,200.0 320.9,200.0 318.5,200.0 316.1,200.0 313.8,200.0 311.4,200.0 309.0,200.0 306.6,200.0 304.3,200.0 301.9,200.0 299.5,200.0 297.1,200.0 294.8,200.0 292.4,200.0 290.0,200.0 287.7,200.0 285.3,200.0 282.9,200.0 280.5,200.0 278.2,200.0 275.8,200.0 273.4,200.0 271.0,200.0 268.7,200.0 266.3,200.0 263.9,200.0 261.5,200.0 259.2,200.0 256.8,200.0 254.4,200.0 252.0,200.0 249.7,200.0 247.3,200.0 244.9,200.0 242.5,200.0 240.2,200.0 237.8,200.0 235.4,200.0 233.1,200.0 230.7,200.0 228.3,200.0 225.9,200.0 223.6,200.0 221.2,200.0 218.8,200.0 216.4,200.0 214.1,200.0 211.7,200.0 209.3,200.0 206.9,200.0 204.6,200.0 202.2,200.0 199.8,200.0 197.4,200.0 195.1,200.0 192.7,200.0 190.3,200.0 187.9,200.0 185.6,200.0 183.2,200.0 180.8,200.0 178.5,200.0 176.1,200.0 173.7,200.0 171.3,200.0 169.0,200.0 166.6,200.0 164.2,200.0 161.8,200.0 159.5,200.0 157.1,200.0 154.7,200.0 152.3,200.0 150.0,200.0 147.6,200.0 145.2,200.0 142.8,200.0 140.5,200.0 138.1,200.0 135.7,200.0 133.3,200.0 131.0,200.0 128.6,200.0 126.2,200.0 123.9,200.0 121.5,200.0 119.1,200.0 116.7,200.0 114.4,200.0 112.0,200.0 109.6,200.0 107.2,200.0 104.9,200.0 102.5,200.0 100.1,200.0 97.7,200.0 95.4,200.0 93.0,200.0 90.6,200.0 88.2,200.0 85.9,200.0 83.5,200.0 81.1,200.0 78.7,200.0 76.4,200.0 74.0,200.0" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>第 2 批：program 2040 到 4079，共 2040 个</title><polygon points="74.0,142.8 76.4,52.0 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.0 95.4,52.0 97.7,52.0 100.1,52.0 102.5,52.1 104.9,52.1 107.2,52.9 109.6,54.5 112.0,56.8 114.4,58.5 116.7,60.1 119.1,66.0 121.5,68.6 123.9,71.6 126.2,73.8 128.6,76.9 131.0,81.2 133.3,81.9 135.7,82.2 138.1,86.5 140.5,89.4 142.8,90.7 145.2,89.4 147.6,91.8 150.0,93.9 152.3,94.2 154.7,94.6 157.1,93.9 159.5,95.9 161.8,95.6 164.2,95.3 166.6,97.0 169.0,98.6 171.3,100.0 173.7,97.9 176.1,99.4 178.5,96.6 180.8,94.1 183.2,94.1 185.6,92.1 187.9,92.7 190.3,92.1 192.7,90.3 195.1,89.8 197.4,89.8 199.8,89.8 202.2,90.0 204.6,89.7 206.9,88.7 209.3,88.8 211.7,90.2 214.1,91.5 216.4,93.3 218.8,93.6 221.2,94.7 223.6,97.6 225.9,98.6 228.3,99.2 230.7,100.0 233.1,102.9 235.4,105.0 237.8,106.3 240.2,108.4 242.5,111.3 244.9,113.3 247.3,115.6 249.7,117.5 252.0,120.7 254.4,123.9 256.8,125.1 259.2,126.5 261.5,128.4 263.9,129.3 266.3,130.6 268.7,131.5 271.0,133.9 273.4,136.8 275.8,139.9 278.2,142.9 280.5,144.6 282.9,145.2 285.3,147.5 287.7,148.6 290.0,150.7 292.4,151.8 294.8,153.1 297.1,154.7 299.5,156.5 301.9,157.4 304.3,159.4 306.6,161.8 309.0,165.3 311.4,166.6 313.8,167.9 316.1,171.5 318.5,173.2 320.9,174.0 323.3,176.4 325.6,177.0 328.0,177.5 330.4,178.3 332.8,179.0 335.1,179.4 337.5,181.1 339.9,182.5 342.3,185.5 344.6,188.5 347.0,190.5 349.4,191.4 351.7,192.1 354.1,192.6 356.5,193.0 358.9,193.5 361.2,194.5 363.6,195.2 366.0,196.2 368.4,196.4 370.7,197.0 373.1,197.3 375.5,197.4 377.9,197.8 380.2,198.3 382.6,198.5 385.0,198.9 387.4,199.3 389.7,199.4 392.1,199.6 394.5,199.6 396.9,199.8 399.2,199.9 401.6,199.9 404.0,199.9 406.3,199.9 408.7,200.0 411.1,200.0 413.5,200.0 415.8,200.0 418.2,200.0 420.6,200.0 423.0,200.0 425.3,200.0 427.7,200.0 430.1,200.0 432.5,200.0 434.8,200.0 437.2,200.0 439.6,200.0 442.0,200.0 444.3,200.0 446.7,200.0 449.1,200.0 451.5,200.0 453.8,200.0 456.2,200.0 458.6,200.0 460.9,200.0 463.3,200.0 465.7,200.0 468.1,200.0 470.4,200.0 472.8,200.0 475.2,200.0 477.6,200.0 479.9,200.0 482.3,200.0 484.7,200.0 487.1,200.0 489.4,200.0 491.8,200.0 494.2,200.0 496.6,200.0 498.9,200.0 501.3,200.0 503.7,200.0 506.1,200.0 508.4,200.0 510.8,200.0 513.2,200.0 515.5,200.0 517.9,200.0 520.3,200.0 522.7,200.0 525.0,200.0 527.4,200.0 529.8,200.0 532.2,200.0 534.5,200.0 536.9,200.0 539.3,200.0 541.7,200.0 544.0,200.0 546.4,200.0 548.8,200.0 551.2,200.0 553.5,200.0 555.9,200.0 558.3,200.0 560.7,200.0 563.0,200.0 565.4,200.0 567.8,200.0 570.1,200.0 572.5,200.0 574.9,200.0 577.3,200.0 579.6,200.0 582.0,200.0 584.4,200.0 586.8,200.0 589.1,200.0 591.5,200.0 593.9,200.0 596.3,200.0 598.6,200.0 601.0,200.0 603.4,200.0 605.8,200.0 608.1,200.0 610.5,200.0 612.9,200.0 615.3,200.0 617.6,200.0 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,200.0 563.0,200.0 560.7,200.0 558.3,200.0 555.9,200.0 553.5,200.0 551.2,200.0 548.8,200.0 546.4,200.0 544.0,200.0 541.7,200.0 539.3,200.0 536.9,200.0 534.5,200.0 532.2,200.0 529.8,200.0 527.4,200.0 525.0,200.0 522.7,200.0 520.3,200.0 517.9,200.0 515.5,200.0 513.2,200.0 510.8,200.0 508.4,200.0 506.1,200.0 503.7,200.0 501.3,200.0 498.9,200.0 496.6,200.0 494.2,200.0 491.8,200.0 489.4,200.0 487.1,200.0 484.7,200.0 482.3,200.0 479.9,200.0 477.6,200.0 475.2,200.0 472.8,200.0 470.4,200.0 468.1,200.0 465.7,200.0 463.3,200.0 460.9,200.0 458.6,200.0 456.2,200.0 453.8,200.0 451.5,200.0 449.1,200.0 446.7,200.0 444.3,200.0 442.0,200.0 439.6,200.0 437.2,200.0 434.8,200.0 432.5,200.0 430.1,200.0 427.7,200.0 425.3,200.0 423.0,200.0 420.6,200.0 418.2,200.0 415.8,200.0 413.5,200.0 411.1,200.0 408.7,200.0 406.3,200.0 404.0,200.0 401.6,200.0 399.2,200.0 396.9,200.0 394.5,200.0 392.1,200.0 389.7,200.0 387.4,200.0 385.0,200.0 382.6,200.0 380.2,200.0 377.9,200.0 375.5,200.0 373.1,200.0 370.7,200.0 368.4,200.0 366.0,200.0 363.6,200.0 361.2,200.0 358.9,200.0 356.5,200.0 354.1,200.0 351.7,200.0 349.4,200.0 347.0,200.0 344.6,200.0 342.3,200.0 339.9,200.0 337.5,200.0 335.1,200.0 332.8,200.0 330.4,200.0 328.0,200.0 325.6,200.0 323.3,200.0 320.9,200.0 318.5,200.0 316.1,200.0 313.8,200.0 311.4,200.0 309.0,200.0 306.6,200.0 304.3,200.0 301.9,200.0 299.5,200.0 297.1,200.0 294.8,200.0 292.4,200.0 290.0,200.0 287.7,200.0 285.3,200.0 282.9,200.0 280.5,200.0 278.2,200.0 275.8,200.0 273.4,200.0 271.0,200.0 268.7,200.0 266.3,200.0 263.9,200.0 261.5,200.0 259.2,200.0 256.8,200.0 254.4,200.0 252.0,199.9 249.7,199.8 247.3,199.6 244.9,199.6 242.5,199.3 240.2,199.1 237.8,198.7 235.4,198.6 233.1,198.2 230.7,197.5 228.3,197.5 225.9,197.5 223.6,197.3 221.2,196.3 218.8,196.0 216.4,196.0 214.1,195.4 211.7,195.1 209.3,194.5 206.9,194.0 204.6,193.4 202.2,192.9 199.8,192.2 197.4,191.7 195.1,191.5 192.7,191.0 190.3,190.6 187.9,189.6 185.6,189.4 183.2,189.3 180.8,188.8 178.5,188.3 176.1,187.0 173.7,185.6 171.3,184.7 169.0,182.7 166.6,180.0 164.2,178.7 161.8,178.7 159.5,178.4 157.1,177.1 154.7,176.8 152.3,176.0 150.0,172.9 147.6,168.3 145.2,164.0 142.8,160.9 140.5,154.5 138.1,146.1 135.7,136.5 133.3,129.6 131.0,119.8 128.6,108.8 126.2,99.1 123.9,88.3 121.5,78.5 119.1,70.4 116.7,62.5 114.4,59.8 112.0,57.4 109.6,54.7 107.2,53.0 104.9,52.1 102.5,52.1 100.1,52.0 97.7,52.0 95.4,52.0 93.0,52.0 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,52.0 74.0,142.8" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>第 3 批：program 4080 到 6119，共 2040 个</title><polygon points="74.0,142.8 76.4,52.0 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.0 95.4,52.0 97.7,52.0 100.1,52.0 102.5,52.1 104.9,52.1 107.2,52.9 109.6,54.5 112.0,56.8 114.4,58.5 116.7,60.1 119.1,66.0 121.5,68.6 123.9,71.6 126.2,73.8 128.6,76.9 131.0,81.2 133.3,81.9 135.7,82.2 138.1,86.5 140.5,89.4 142.8,90.7 145.2,89.4 147.6,91.8 150.0,93.9 152.3,94.2 154.7,94.6 157.1,93.9 159.5,95.9 161.8,95.6 164.2,95.3 166.6,97.0 169.0,98.6 171.3,100.0 173.7,97.9 176.1,99.4 178.5,96.6 180.8,94.1 183.2,94.1 185.6,92.1 187.9,92.7 190.3,92.1 192.7,90.3 195.1,89.8 197.4,89.8 199.8,89.8 202.2,90.0 204.6,89.7 206.9,88.7 209.3,88.8 211.7,90.2 214.1,90.2 216.4,89.1 218.8,86.1 221.2,86.8 223.6,87.4 225.9,85.2 228.3,84.9 230.7,85.1 233.1,83.5 235.4,82.7 237.8,81.5 240.2,80.5 242.5,80.1 244.9,78.6 247.3,76.8 249.7,76.7 252.0,76.7 254.4,77.2 256.8,76.3 259.2,77.0 261.5,77.7 263.9,78.3 266.3,79.1 268.7,79.4 271.0,80.9 273.4,81.4 275.8,82.3 278.2,83.6 280.5,83.3 282.9,83.8 285.3,85.6 287.7,85.5 290.0,86.8 292.4,87.0 294.8,87.5 297.1,88.6 299.5,89.4 301.9,90.5 304.3,91.5 306.6,93.4 309.0,95.2 311.4,94.4 313.8,95.0 316.1,96.5 318.5,97.3 320.9,97.2 323.3,98.9 325.6,98.4 328.0,98.1 330.4,99.0 332.8,99.0 335.1,99.4 337.5,100.7 339.9,101.2 342.3,103.0 344.6,102.5 347.0,98.9 349.4,96.6 351.7,97.0 354.1,97.0 356.5,97.3 358.9,96.8 361.2,97.5 363.6,96.3 366.0,97.5 368.4,95.7 370.7,96.6 373.1,96.5 375.5,95.8 377.9,97.9 380.2,102.9 382.6,105.6 385.0,110.5 387.4,114.5 389.7,118.0 392.1,121.4 394.5,123.3 396.9,126.7 399.2,128.8 401.6,131.2 404.0,133.8 406.3,135.2 408.7,137.8 411.1,141.5 413.5,142.8 415.8,144.9 418.2,148.6 420.6,150.6 423.0,152.9 425.3,155.7 427.7,156.9 430.1,159.9 432.5,162.2 434.8,163.8 437.2,166.4 439.6,168.2 442.0,170.2 444.3,170.8 446.7,172.6 449.1,176.1 451.5,179.5 453.8,180.3 456.2,182.6 458.6,183.7 460.9,184.7 463.3,186.6 465.7,187.8 468.1,188.7 470.4,189.7 472.8,190.7 475.2,191.1 477.6,192.0 479.9,192.8 482.3,194.3 484.7,195.0 487.1,195.4 489.4,195.7 491.8,196.0 494.2,196.3 496.6,196.9 498.9,197.3 501.3,197.8 503.7,198.0 506.1,198.5 508.4,198.8 510.8,198.9 513.2,199.1 515.5,199.1 517.9,199.2 520.3,199.3 522.7,199.4 525.0,199.4 527.4,199.4 529.8,199.6 532.2,199.6 534.5,199.6 536.9,199.7 539.3,199.7 541.7,199.7 544.0,199.8 546.4,199.9 548.8,199.9 551.2,199.9 553.5,199.9 555.9,199.9 558.3,199.9 560.7,199.9 563.0,199.9 565.4,199.9 567.8,200.0 570.1,200.0 572.5,200.0 574.9,200.0 577.3,200.0 579.6,200.0 582.0,200.0 584.4,200.0 586.8,200.0 589.1,200.0 591.5,200.0 593.9,200.0 596.3,200.0 598.6,200.0 601.0,200.0 603.4,200.0 605.8,200.0 608.1,200.0 610.5,200.0 612.9,200.0 615.3,200.0 617.6,200.0 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,200.0 563.0,200.0 560.7,200.0 558.3,200.0 555.9,200.0 553.5,200.0 551.2,200.0 548.8,200.0 546.4,200.0 544.0,200.0 541.7,200.0 539.3,200.0 536.9,200.0 534.5,200.0 532.2,200.0 529.8,200.0 527.4,200.0 525.0,200.0 522.7,200.0 520.3,200.0 517.9,200.0 515.5,200.0 513.2,200.0 510.8,200.0 508.4,200.0 506.1,200.0 503.7,200.0 501.3,200.0 498.9,200.0 496.6,200.0 494.2,200.0 491.8,200.0 489.4,200.0 487.1,200.0 484.7,200.0 482.3,200.0 479.9,200.0 477.6,200.0 475.2,200.0 472.8,200.0 470.4,200.0 468.1,200.0 465.7,200.0 463.3,200.0 460.9,200.0 458.6,200.0 456.2,200.0 453.8,200.0 451.5,200.0 449.1,200.0 446.7,200.0 444.3,200.0 442.0,200.0 439.6,200.0 437.2,200.0 434.8,200.0 432.5,200.0 430.1,200.0 427.7,200.0 425.3,200.0 423.0,200.0 420.6,200.0 418.2,200.0 415.8,200.0 413.5,200.0 411.1,200.0 408.7,200.0 406.3,199.9 404.0,199.9 401.6,199.9 399.2,199.9 396.9,199.8 394.5,199.6 392.1,199.6 389.7,199.4 387.4,199.3 385.0,198.9 382.6,198.5 380.2,198.3 377.9,197.8 375.5,197.4 373.1,197.3 370.7,197.0 368.4,196.4 366.0,196.2 363.6,195.2 361.2,194.5 358.9,193.5 356.5,193.0 354.1,192.6 351.7,192.1 349.4,191.4 347.0,190.5 344.6,188.5 342.3,185.5 339.9,182.5 337.5,181.1 335.1,179.4 332.8,179.0 330.4,178.3 328.0,177.5 325.6,177.0 323.3,176.4 320.9,174.0 318.5,173.2 316.1,171.5 313.8,167.9 311.4,166.6 309.0,165.3 306.6,161.8 304.3,159.4 301.9,157.4 299.5,156.5 297.1,154.7 294.8,153.1 292.4,151.8 290.0,150.7 287.7,148.6 285.3,147.5 282.9,145.2 280.5,144.6 278.2,142.9 275.8,139.9 273.4,136.8 271.0,133.9 268.7,131.5 266.3,130.6 263.9,129.3 261.5,128.4 259.2,126.5 256.8,125.1 254.4,123.9 252.0,120.7 249.7,117.5 247.3,115.6 244.9,113.3 242.5,111.3 240.2,108.4 237.8,106.3 235.4,105.0 233.1,102.9 230.7,100.0 228.3,99.2 225.9,98.6 223.6,97.6 221.2,94.7 218.8,93.6 216.4,93.3 214.1,91.5 211.7,90.2 209.3,88.8 206.9,88.7 204.6,89.7 202.2,90.0 199.8,89.8 197.4,89.8 195.1,89.8 192.7,90.3 190.3,92.1 187.9,92.7 185.6,92.1 183.2,94.1 180.8,94.1 178.5,96.6 176.1,99.4 173.7,97.9 171.3,100.0 169.0,98.6 166.6,97.0 164.2,95.3 161.8,95.6 159.5,95.9 157.1,93.9 154.7,94.6 152.3,94.2 150.0,93.9 147.6,91.8 145.2,89.4 142.8,90.7 140.5,89.4 138.1,86.5 135.7,82.2 133.3,81.9 131.0,81.2 128.6,76.9 126.2,73.8 123.9,71.6 121.5,68.6 119.1,66.0 116.7,60.1 114.4,58.5 112.0,56.8 109.6,54.5 107.2,52.9 104.9,52.1 102.5,52.1 100.1,52.0 97.7,52.0 95.4,52.0 93.0,52.0 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,52.0 74.0,142.8" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>第 4 批：program 6120 到 8159，共 2040 个</title><polygon points="74.0,142.8 76.4,52.0 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.0 95.4,52.0 97.7,52.0 100.1,52.0 102.5,52.1 104.9,52.1 107.2,52.9 109.6,54.5 112.0,56.8 114.4,58.5 116.7,60.1 119.1,66.0 121.5,68.6 123.9,71.6 126.2,73.8 128.6,76.9 131.0,81.2 133.3,81.9 135.7,82.2 138.1,86.5 140.5,89.4 142.8,90.7 145.2,89.4 147.6,91.8 150.0,93.9 152.3,94.2 154.7,94.6 157.1,93.9 159.5,95.9 161.8,95.6 164.2,95.3 166.6,97.0 169.0,98.6 171.3,100.0 173.7,97.9 176.1,99.4 178.5,96.6 180.8,94.1 183.2,94.1 185.6,92.1 187.9,92.7 190.3,92.1 192.7,90.3 195.1,89.8 197.4,89.8 199.8,89.8 202.2,90.0 204.6,89.7 206.9,88.7 209.3,88.8 211.7,90.2 214.1,90.2 216.4,89.1 218.8,86.1 221.2,86.8 223.6,87.4 225.9,85.2 228.3,84.9 230.7,85.1 233.1,83.5 235.4,82.7 237.8,81.5 240.2,80.5 242.5,80.1 244.9,78.6 247.3,76.8 249.7,76.7 252.0,76.7 254.4,77.2 256.8,76.3 259.2,77.0 261.5,77.7 263.9,78.3 266.3,79.1 268.7,79.4 271.0,80.9 273.4,81.4 275.8,82.3 278.2,83.6 280.5,83.3 282.9,83.8 285.3,85.6 287.7,85.5 290.0,86.8 292.4,87.0 294.8,87.5 297.1,88.6 299.5,89.4 301.9,90.5 304.3,91.5 306.6,93.4 309.0,95.2 311.4,94.4 313.8,95.0 316.1,96.5 318.5,97.3 320.9,97.2 323.3,98.9 325.6,98.4 328.0,98.1 330.4,99.0 332.8,99.0 335.1,99.4 337.5,100.7 339.9,101.2 342.3,103.0 344.6,102.5 347.0,98.9 349.4,96.6 351.7,97.0 354.1,97.0 356.5,97.3 358.9,96.8 361.2,97.5 363.6,96.3 366.0,97.5 368.4,95.7 370.7,96.6 373.1,96.5 375.5,95.8 377.9,97.7 380.2,97.7 382.6,95.4 385.0,96.4 387.4,92.1 389.7,91.4 392.1,90.4 394.5,90.5 396.9,90.8 399.2,89.2 401.6,88.3 404.0,86.7 406.3,85.3 408.7,86.5 411.1,86.5 413.5,85.7 415.8,87.3 418.2,87.9 420.6,87.2 423.0,87.2 425.3,88.1 427.7,88.2 430.1,89.8 432.5,87.6 434.8,87.6 437.2,89.2 439.6,89.4 442.0,88.9 444.3,87.7 446.7,87.7 449.1,89.9 451.5,90.5 453.8,89.0 456.2,89.0 458.6,88.9 460.9,87.8 463.3,89.9 465.7,90.4 468.1,89.7 470.4,90.2 472.8,91.2 475.2,92.0 477.6,93.3 479.9,93.1 482.3,95.4 484.7,95.5 487.1,94.7 489.4,95.2 491.8,95.7 494.2,96.0 496.6,98.2 498.9,99.0 501.3,101.9 503.7,104.7 506.1,110.3 508.4,113.4 510.8,116.4 513.2,118.7 515.5,121.2 517.9,123.5 520.3,127.5 522.7,129.7 525.0,132.8 527.4,135.6 529.8,137.4 532.2,141.8 534.5,145.6 536.9,147.6 539.3,151.7 541.7,153.6 544.0,157.3 546.4,159.7 548.8,162.1 551.2,163.2 553.5,163.9 555.9,167.1 558.3,167.7 560.7,168.4 563.0,169.3 565.4,170.9 567.8,173.4 570.1,177.2 572.5,180.6 574.9,181.5 577.3,183.0 579.6,183.8 582.0,185.2 584.4,187.1 586.8,187.6 589.1,188.7 591.5,189.3 593.9,190.7 596.3,192.0 598.6,195.2 601.0,197.1 603.4,197.9 605.8,198.4 608.1,198.9 610.5,199.0 612.9,199.3 615.3,199.3 617.6,199.6 620.0,200.0 620.0,200.0 617.6,200.0 615.3,200.0 612.9,200.0 610.5,200.0 608.1,200.0 605.8,200.0 603.4,200.0 601.0,200.0 598.6,200.0 596.3,200.0 593.9,200.0 591.5,200.0 589.1,200.0 586.8,200.0 584.4,200.0 582.0,200.0 579.6,200.0 577.3,200.0 574.9,200.0 572.5,200.0 570.1,200.0 567.8,200.0 565.4,199.9 563.0,199.9 560.7,199.9 558.3,199.9 555.9,199.9 553.5,199.9 551.2,199.9 548.8,199.9 546.4,199.9 544.0,199.8 541.7,199.7 539.3,199.7 536.9,199.7 534.5,199.6 532.2,199.6 529.8,199.6 527.4,199.4 525.0,199.4 522.7,199.4 520.3,199.3 517.9,199.2 515.5,199.1 513.2,199.1 510.8,198.9 508.4,198.8 506.1,198.5 503.7,198.0 501.3,197.8 498.9,197.3 496.6,196.9 494.2,196.3 491.8,196.0 489.4,195.7 487.1,195.4 484.7,195.0 482.3,194.3 479.9,192.8 477.6,192.0 475.2,191.1 472.8,190.7 470.4,189.7 468.1,188.7 465.7,187.8 463.3,186.6 460.9,184.7 458.6,183.7 456.2,182.6 453.8,180.3 451.5,179.5 449.1,176.1 446.7,172.6 444.3,170.8 442.0,170.2 439.6,168.2 437.2,166.4 434.8,163.8 432.5,162.2 430.1,159.9 427.7,156.9 425.3,155.7 423.0,152.9 420.6,150.6 418.2,148.6 415.8,144.9 413.5,142.8 411.1,141.5 408.7,137.8 406.3,135.2 404.0,133.8 401.6,131.2 399.2,128.8 396.9,126.7 394.5,123.3 392.1,121.4 389.7,118.0 387.4,114.5 385.0,110.5 382.6,105.6 380.2,102.9 377.9,97.9 375.5,95.8 373.1,96.5 370.7,96.6 368.4,95.7 366.0,97.5 363.6,96.3 361.2,97.5 358.9,96.8 356.5,97.3 354.1,97.0 351.7,97.0 349.4,96.6 347.0,98.9 344.6,102.5 342.3,103.0 339.9,101.2 337.5,100.7 335.1,99.4 332.8,99.0 330.4,99.0 328.0,98.1 325.6,98.4 323.3,98.9 320.9,97.2 318.5,97.3 316.1,96.5 313.8,95.0 311.4,94.4 309.0,95.2 306.6,93.4 304.3,91.5 301.9,90.5 299.5,89.4 297.1,88.6 294.8,87.5 292.4,87.0 290.0,86.8 287.7,85.5 285.3,85.6 282.9,83.8 280.5,83.3 278.2,83.6 275.8,82.3 273.4,81.4 271.0,80.9 268.7,79.4 266.3,79.1 263.9,78.3 261.5,77.7 259.2,77.0 256.8,76.3 254.4,77.2 252.0,76.7 249.7,76.7 247.3,76.8 244.9,78.6 242.5,80.1 240.2,80.5 237.8,81.5 235.4,82.7 233.1,83.5 230.7,85.1 228.3,84.9 225.9,85.2 223.6,87.4 221.2,86.8 218.8,86.1 216.4,89.1 214.1,90.2 211.7,90.2 209.3,88.8 206.9,88.7 204.6,89.7 202.2,90.0 199.8,89.8 197.4,89.8 195.1,89.8 192.7,90.3 190.3,92.1 187.9,92.7 185.6,92.1 183.2,94.1 180.8,94.1 178.5,96.6 176.1,99.4 173.7,97.9 171.3,100.0 169.0,98.6 166.6,97.0 164.2,95.3 161.8,95.6 159.5,95.9 157.1,93.9 154.7,94.6 152.3,94.2 150.0,93.9 147.6,91.8 145.2,89.4 142.8,90.7 140.5,89.4 138.1,86.5 135.7,82.2 133.3,81.9 131.0,81.2 128.6,76.9 126.2,73.8 123.9,71.6 121.5,68.6 119.1,66.0 116.7,60.1 114.4,58.5 112.0,56.8 109.6,54.5 107.2,52.9 104.9,52.1 102.5,52.1 100.1,52.0 97.7,52.0 95.4,52.0 93.0,52.0 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,52.0 74.0,142.8" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>第 5 批：program 8160 到 8191，共 32 个</title><polygon points="74.0,142.8 76.4,52.0 78.7,52.0 81.1,52.0 83.5,52.0 85.9,52.0 88.2,52.0 90.6,52.0 93.0,52.0 95.4,52.0 97.7,52.0 100.1,52.0 102.5,52.1 104.9,52.1 107.2,52.9 109.6,54.5 112.0,56.8 114.4,58.5 116.7,60.1 119.1,66.0 121.5,68.6 123.9,71.6 126.2,73.8 128.6,76.9 131.0,81.2 133.3,81.9 135.7,82.2 138.1,86.5 140.5,89.4 142.8,90.7 145.2,89.4 147.6,91.8 150.0,93.9 152.3,94.2 154.7,94.6 157.1,93.9 159.5,95.9 161.8,95.6 164.2,95.3 166.6,97.0 169.0,98.6 171.3,100.0 173.7,97.9 176.1,99.4 178.5,96.6 180.8,94.1 183.2,94.1 185.6,92.1 187.9,92.7 190.3,92.1 192.7,90.3 195.1,89.8 197.4,89.8 199.8,89.8 202.2,90.0 204.6,89.7 206.9,88.7 209.3,88.8 211.7,90.2 214.1,90.2 216.4,89.1 218.8,86.1 221.2,86.8 223.6,87.4 225.9,85.2 228.3,84.9 230.7,85.1 233.1,83.5 235.4,82.7 237.8,81.5 240.2,80.5 242.5,80.1 244.9,78.6 247.3,76.8 249.7,76.7 252.0,76.7 254.4,77.2 256.8,76.3 259.2,77.0 261.5,77.7 263.9,78.3 266.3,79.1 268.7,79.4 271.0,80.9 273.4,81.4 275.8,82.3 278.2,83.6 280.5,83.3 282.9,83.8 285.3,85.6 287.7,85.5 290.0,86.8 292.4,87.0 294.8,87.5 297.1,88.6 299.5,89.4 301.9,90.5 304.3,91.5 306.6,93.4 309.0,95.2 311.4,94.4 313.8,95.0 316.1,96.5 318.5,97.3 320.9,97.2 323.3,98.9 325.6,98.4 328.0,98.1 330.4,99.0 332.8,99.0 335.1,99.4 337.5,100.7 339.9,101.2 342.3,103.0 344.6,102.5 347.0,98.9 349.4,96.6 351.7,97.0 354.1,97.0 356.5,97.3 358.9,96.8 361.2,97.5 363.6,96.3 366.0,97.5 368.4,95.7 370.7,96.6 373.1,96.5 375.5,95.8 377.9,97.7 380.2,97.7 382.6,95.4 385.0,96.4 387.4,92.1 389.7,91.4 392.1,90.4 394.5,90.5 396.9,90.8 399.2,89.2 401.6,88.3 404.0,86.7 406.3,85.3 408.7,86.5 411.1,86.5 413.5,85.7 415.8,87.3 418.2,87.9 420.6,87.2 423.0,87.2 425.3,88.1 427.7,88.2 430.1,89.8 432.5,87.6 434.8,87.6 437.2,89.2 439.6,89.4 442.0,88.9 444.3,87.7 446.7,87.7 449.1,89.9 451.5,90.5 453.8,89.0 456.2,89.0 458.6,88.9 460.9,87.8 463.3,89.9 465.7,90.4 468.1,89.7 470.4,90.2 472.8,91.2 475.2,92.0 477.6,93.3 479.9,93.1 482.3,95.4 484.7,95.5 487.1,94.7 489.4,95.2 491.8,95.7 494.2,96.0 496.6,98.2 498.9,97.1 501.3,99.6 503.7,102.3 506.1,108.0 508.4,111.1 510.8,114.0 513.2,116.4 515.5,118.9 517.9,121.1 520.3,125.2 522.7,127.4 525.0,130.5 527.4,133.3 529.8,135.1 532.2,139.5 534.5,143.3 536.9,145.3 539.3,149.4 541.7,151.3 544.0,155.0 546.4,157.5 548.8,160.0 551.2,161.0 553.5,161.7 555.9,165.0 558.3,165.5 560.7,166.2 563.0,167.1 565.4,168.7 567.8,171.4 570.1,175.4 572.5,178.8 574.9,179.9 577.3,181.5 579.6,182.5 582.0,184.0 584.4,185.9 586.8,186.5 589.1,187.6 591.5,188.3 593.9,189.8 596.3,191.2 598.6,194.5 601.0,196.7 603.4,197.7 605.8,198.3 608.1,198.8 610.5,198.8 612.9,199.1 615.3,199.2 617.6,199.6 620.0,200.0 620.0,200.0 617.6,199.6 615.3,199.3 612.9,199.3 610.5,199.0 608.1,198.9 605.8,198.4 603.4,197.9 601.0,197.1 598.6,195.2 596.3,192.0 593.9,190.7 591.5,189.3 589.1,188.7 586.8,187.6 584.4,187.1 582.0,185.2 579.6,183.8 577.3,183.0 574.9,181.5 572.5,180.6 570.1,177.2 567.8,173.4 565.4,170.9 563.0,169.3 560.7,168.4 558.3,167.7 555.9,167.1 553.5,163.9 551.2,163.2 548.8,162.1 546.4,159.7 544.0,157.3 541.7,153.6 539.3,151.7 536.9,147.6 534.5,145.6 532.2,141.8 529.8,137.4 527.4,135.6 525.0,132.8 522.7,129.7 520.3,127.5 517.9,123.5 515.5,121.2 513.2,118.7 510.8,116.4 508.4,113.4 506.1,110.3 503.7,104.7 501.3,101.9 498.9,99.0 496.6,98.2 494.2,96.0 491.8,95.7 489.4,95.2 487.1,94.7 484.7,95.5 482.3,95.4 479.9,93.1 477.6,93.3 475.2,92.0 472.8,91.2 470.4,90.2 468.1,89.7 465.7,90.4 463.3,89.9 460.9,87.8 458.6,88.9 456.2,89.0 453.8,89.0 451.5,90.5 449.1,89.9 446.7,87.7 444.3,87.7 442.0,88.9 439.6,89.4 437.2,89.2 434.8,87.6 432.5,87.6 430.1,89.8 427.7,88.2 425.3,88.1 423.0,87.2 420.6,87.2 418.2,87.9 415.8,87.3 413.5,85.7 411.1,86.5 408.7,86.5 406.3,85.3 404.0,86.7 401.6,88.3 399.2,89.2 396.9,90.8 394.5,90.5 392.1,90.4 389.7,91.4 387.4,92.1 385.0,96.4 382.6,95.4 380.2,97.7 377.9,97.7 375.5,95.8 373.1,96.5 370.7,96.6 368.4,95.7 366.0,97.5 363.6,96.3 361.2,97.5 358.9,96.8 356.5,97.3 354.1,97.0 351.7,97.0 349.4,96.6 347.0,98.9 344.6,102.5 342.3,103.0 339.9,101.2 337.5,100.7 335.1,99.4 332.8,99.0 330.4,99.0 328.0,98.1 325.6,98.4 323.3,98.9 320.9,97.2 318.5,97.3 316.1,96.5 313.8,95.0 311.4,94.4 309.0,95.2 306.6,93.4 304.3,91.5 301.9,90.5 299.5,89.4 297.1,88.6 294.8,87.5 292.4,87.0 290.0,86.8 287.7,85.5 285.3,85.6 282.9,83.8 280.5,83.3 278.2,83.6 275.8,82.3 273.4,81.4 271.0,80.9 268.7,79.4 266.3,79.1 263.9,78.3 261.5,77.7 259.2,77.0 256.8,76.3 254.4,77.2 252.0,76.7 249.7,76.7 247.3,76.8 244.9,78.6 242.5,80.1 240.2,80.5 237.8,81.5 235.4,82.7 233.1,83.5 230.7,85.1 228.3,84.9 225.9,85.2 223.6,87.4 221.2,86.8 218.8,86.1 216.4,89.1 214.1,90.2 211.7,90.2 209.3,88.8 206.9,88.7 204.6,89.7 202.2,90.0 199.8,89.8 197.4,89.8 195.1,89.8 192.7,90.3 190.3,92.1 187.9,92.7 185.6,92.1 183.2,94.1 180.8,94.1 178.5,96.6 176.1,99.4 173.7,97.9 171.3,100.0 169.0,98.6 166.6,97.0 164.2,95.3 161.8,95.6 159.5,95.9 157.1,93.9 154.7,94.6 152.3,94.2 150.0,93.9 147.6,91.8 145.2,89.4 142.8,90.7 140.5,89.4 138.1,86.5 135.7,82.2 133.3,81.9 131.0,81.2 128.6,76.9 126.2,73.8 123.9,71.6 121.5,68.6 119.1,66.0 116.7,60.1 114.4,58.5 112.0,56.8 109.6,54.5 107.2,52.9 104.9,52.1 102.5,52.1 100.1,52.0 97.7,52.0 95.4,52.0 93.0,52.0 90.6,52.0 88.2,52.0 85.9,52.0 83.5,52.0 81.1,52.0 78.7,52.0 76.4,52.0 74.0,142.8" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <polygon points="102.5,52.1 104.9,52.1 107.2,52.9 109.6,54.5 112.0,56.8 114.4,58.5 116.7,60.1 119.1,66.0 121.5,68.6 123.9,71.6 126.2,73.8 128.6,76.9 131.0,81.2 133.3,81.9 135.7,82.2 138.1,86.5 140.5,89.4 142.8,90.7 145.2,89.4 147.6,91.8 150.0,93.9 152.3,94.2 154.7,94.6 157.1,93.9 159.5,95.9 161.8,95.6 164.2,95.3 166.6,97.0 169.0,98.6 171.3,100.0 173.7,97.9 176.1,99.4 178.5,96.6 180.8,94.1 183.2,94.1 185.6,92.1 187.9,92.7 190.3,92.1 192.7,90.3 195.1,89.8 197.4,89.8 199.8,89.8 202.2,90.0 204.6,89.7 206.9,88.7 209.3,88.8 211.7,90.2 214.1,90.2 216.4,89.1 218.8,86.1 221.2,86.8 223.6,87.4 225.9,85.2 228.3,84.9 230.7,85.1 233.1,83.5 235.4,82.7 237.8,81.5 240.2,80.5 242.5,80.1 244.9,78.6 247.3,76.8 249.7,76.7 252.0,76.7 254.4,77.2 256.8,76.3 259.2,77.0 261.5,77.7 263.9,78.3 266.3,79.1 268.7,79.4 271.0,80.9 273.4,81.4 275.8,82.3 278.2,83.6 280.5,83.3 282.9,83.8 285.3,85.6 287.7,85.5 290.0,86.8 292.4,87.0 294.8,87.5 297.1,88.6 299.5,89.4 301.9,90.5 304.3,91.5 306.6,93.4 309.0,95.2 311.4,94.4 313.8,95.0 316.1,96.5 318.5,97.3 320.9,97.2 323.3,98.9 325.6,98.4 328.0,98.1 330.4,99.0 332.8,99.0 335.1,99.4 337.5,100.7 339.9,101.2 342.3,103.0 344.6,102.5 347.0,98.9 349.4,96.6 351.7,97.0 354.1,97.0 356.5,97.3 358.9,96.8 361.2,97.5 363.6,96.3 366.0,97.5 368.4,95.7 370.7,96.6 373.1,96.5 375.5,95.8 377.9,97.7 380.2,97.7 382.6,95.4 385.0,96.4 387.4,92.1 389.7,91.4 392.1,90.4 394.5,90.5 396.9,90.8 399.2,89.2 401.6,88.3 404.0,86.7 406.3,85.3 408.7,86.5 411.1,86.5 413.5,85.7 415.8,87.3 418.2,87.9 420.6,87.2 423.0,87.2 425.3,88.1 427.7,88.2 430.1,89.8 432.5,87.6 434.8,87.6 437.2,89.2 439.6,89.4 442.0,88.9 444.3,87.7 446.7,87.7 449.1,89.9 451.5,90.5 453.8,89.0 456.2,89.0 458.6,88.9 460.9,87.8 463.3,89.9 465.7,90.4 468.1,89.7 470.4,90.2 472.8,91.2 475.2,92.0 477.6,93.3 479.9,93.1 482.3,95.4 484.7,95.5 487.1,94.7 489.4,95.2 491.8,95.7 494.2,96.0 496.6,98.2 498.9,97.1 499.3,52.0 102.3,52.0" style="fill: var(--fig-hi)" fill-opacity=".28"/>
  <text class="s halo" x="342.3" y="81.5" text-anchor="middle">bubble</text><text class="s" x="342.3" y="81.5" text-anchor="middle" style="fill: var(--fig-hi)">bubble</text>
  <path d="M499.3,48 L499.3,43 L620.0,43 L620.0,48" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.2"/>
  <text class="s" x="559.7" y="40" text-anchor="middle" style="fill: var(--fig-hi)">tail</text>
  <text class="lab2" x="74" y="234">SM 52 的 12 个名额，一条是一个 program 从第一条指令到最后一条 store 发出；这个 SM 一共跑了 49 个</text>
  <text class="tick" x="66" y="254" text-anchor="end">0</text>
  <g class="m"><title>program 62（第 1 批）：0.00 到 3.14 µs</title><rect x="74.0" y="247" width="35.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2068（第 2 批）：3.62 到 12.61 µs</title><rect x="116.1" y="247" width="104.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4450（第 3 批）：14.21 到 23.87 µs</title><rect x="239.4" y="247" width="111.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5793（第 3 批）：23.87 到 34.11 µs</title><rect x="351.8" y="247" width="118.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 8094（第 4 批）：35.97 到 44.16 µs</title><rect x="492.6" y="247" width="94.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 402（第 1 批）：0.00 到 3.36 µs</title><rect x="74.0" y="256" width="38.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2059（第 2 批）：3.49 到 6.69 µs</title><rect x="114.6" y="256" width="36.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 3195（第 2 批）：6.78 到 17.92 µs</title><rect x="153.0" y="256" width="129.0" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4943（第 3 批）：18.02 到 24.99 µs</title><rect x="283.7" y="256" width="80.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 6203（第 4 批）：26.34 到 34.08 µs</title><rect x="380.5" y="256" width="89.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 8092（第 4 批）：35.97 到 44.93 µs</title><rect x="492.6" y="256" width="103.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 572（第 1 批）：0.00 到 4.00 µs</title><rect x="74.0" y="265" width="46.0" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 6461（第 4 批）：27.01 到 36.35 µs</title><rect x="388.3" y="265" width="108.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 8160（第 5 批）：36.42 到 44.96 µs</title><rect x="497.8" y="265" width="98.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <text class="tick" x="66" y="281" text-anchor="end">3</text>
  <g class="m"><title>program 232（第 1 批）：0.00 到 4.48 µs</title><rect x="74.0" y="274" width="51.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 4138（第 3 批）：12.26 到 20.61 µs</title><rect x="216.6" y="274" width="96.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5803（第 3 批）：23.94 到 31.97 µs</title><rect x="352.6" y="274" width="92.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 8098（第 4 批）：36.00 到 40.96 µs</title><rect x="493.0" y="274" width="57.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 742（第 1 批）：0.00 到 4.51 µs</title><rect x="74.0" y="283" width="51.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2963（第 2 批）：5.76 到 13.76 µs</title><rect x="141.0" y="283" width="92.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4452（第 3 批）：14.21 到 23.17 µs</title><rect x="239.4" y="283" width="103.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5585（第 3 批）：23.30 到 28.54 µs</title><rect x="345.1" y="283" width="60.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 1082（第 1 批）：0.06 到 4.19 µs</title><rect x="74.7" y="292" width="47.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 4140（第 3 批）：12.26 到 20.83 µs</title><rect x="216.6" y="292" width="99.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5604（第 3 批）：23.33 到 29.50 µs</title><rect x="345.5" y="292" width="71.3" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7154（第 4 批）：30.78 到 39.68 µs</title><rect x="432.3" y="292" width="102.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <text class="tick" x="66" y="308" text-anchor="end">6</text>
  <g class="m"><title>program 1422（第 1 批）：0.06 到 4.26 µs</title><rect x="74.7" y="301" width="48.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2301（第 2 批）：4.35 到 11.84 µs</title><rect x="124.7" y="301" width="86.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4136（第 3 批）：12.22 到 17.54 µs</title><rect x="216.3" y="301" width="61.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5135（第 3 批）：20.32 到 34.75 µs</title><rect x="310.5" y="301" width="167.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7916（第 4 批）：34.88 到 38.91 µs</title><rect x="480.0" y="301" width="46.3" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 912（第 1 批）：0.06 到 4.80 µs</title><rect x="74.7" y="310" width="54.5" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 2607（第 2 批）：4.96 到 14.53 µs</title><rect x="131.7" y="310" width="110.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 6448（第 4 批）：26.98 到 36.99 µs</title><rect x="388.0" y="310" width="116.0" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 1252（第 1 批）：0.06 到 5.89 µs</title><rect x="74.7" y="319" width="67.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3147（第 2 批）：6.56 到 15.65 µs</title><rect x="150.4" y="319" width="105.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 5857（第 3 批）：24.35 到 34.72 µs</title><rect x="357.4" y="319" width="120.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7914（第 4 批）：34.88 到 42.08 µs</title><rect x="480.0" y="319" width="83.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <text class="tick" x="66" y="335" text-anchor="end">9</text>
  <g class="m"><title>program 1592（第 1 批）：0.06 到 6.62 µs</title><rect x="74.7" y="328" width="75.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 4134（第 3 批）：12.22 到 23.01 µs</title><rect x="216.3" y="328" width="124.9" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 5500（第 3 批）：23.10 到 28.54 µs</title><rect x="342.9" y="328" width="62.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 1762（第 1 批）：0.10 到 6.37 µs</title><rect x="75.1" y="337" width="72.4" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3145（第 2 批）：6.53 到 20.77 µs</title><rect x="150.0" y="337" width="165.1" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 5612（第 3 批）：23.33 到 34.94 µs</title><rect x="345.5" y="337" width="134.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 7949（第 4 批）：35.07 到 42.05 µs</title><rect x="482.2" y="337" width="80.6" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.68"/></g>
  <g class="m"><title>program 1932（第 1 批）：0.10 到 7.30 µs</title><rect x="75.1" y="346" width="83.2" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.14"/></g>
  <g class="m"><title>program 3342（第 2 批）：8.35 到 17.57 µs</title><rect x="171.2" y="346" width="106.7" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.32"/></g>
  <g class="m"><title>program 4979（第 3 批）：18.46 到 29.41 µs</title><rect x="288.9" y="346" width="126.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.5"/></g>
  <g class="m"><title>program 8170（第 5 批）：36.45 到 42.50 µs</title><rect x="498.2" y="346" width="69.8" height="7" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <rect x="118.6" y="263" width="271.8" height="11" rx="3" style="fill: none; stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="s" x="254.4" y="272" text-anchor="middle" style="fill: var(--fig-hi)">bubble</text>
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
<figcaption><strong>图 3-2</strong> GELU 的 8192 个 program 实际怎么分批跑（RTX 5090，n = 8388608，fp32，每个 program 4 个 warp）。每个 program 开头读一次 <code>%globaltimer</code> 和 <code>%smid</code>，store 发出后再读一次时间，记录前先清空 L2。颜色由淡到浓是 program 编号所在的批次，一批 2040 个。浅红色是 bubble，最后一个 program 开始以后是 tail。下面的名额是按开始、结束时间排出来的，红框是这个 SM 上最长的一段空档。打点版 46.9 µs，不打点 45.2 µs。悬停可看 program 编号和时间。</figcaption>
</figure>

matmul 一批 510 个，512 个 program 多出 2 个，最后 168 个 SM 空等这 2 个跑完，这叫 tail effect（[图 3-3](#fig-t-7)）。A 换成 4032 × 2048 时，504 个 program 一批放下，耗时从 203.3 µs 降到 158.3 µs，按每秒的 FLOP 算和 cuBLAS 一样快（4096 行 160.5 µs）。所以 Triton 版慢在多出的第 2 批，不在 kernel 本身。

<figure id="fig-t-7" class="fg-fig">
<svg class="fg" viewBox="0 0 640 372" width="100%" role="img" aria-label="matmul 的 512 个 program 在 170 个 SM 上的实测时间线：上面是全卡同时在跑的 program 数，按批次堆叠，最多 510 个；第 1 批 510 个一起开始，第 2 批只有 2 个，等到 125 µs 才开始，152 µs 以后全卡只有这 2 个在跑。虚线是 A 4032 × 2048 时 504 个 program 一批跑完的对照。下面是跑了第 2 批的两个 SM 的 3 个名额">
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
  <line class="grid" x1="74" y1="186.0" x2="620" y2="186.0"/><text class="tick" x="66" y="190.0" text-anchor="end">0</text>
  <line class="grid" x1="74" y1="159.7" x2="620" y2="159.7"/><text class="tick" x="66" y="163.7" text-anchor="end">100</text>
  <line class="grid" x1="74" y1="133.5" x2="620" y2="133.5"/><text class="tick" x="66" y="137.5" text-anchor="end">200</text>
  <line class="grid" x1="74" y1="107.2" x2="620" y2="107.2"/><text class="tick" x="66" y="111.2" text-anchor="end">300</text>
  <line class="grid" x1="74" y1="80.9" x2="620" y2="80.9"/><text class="tick" x="66" y="84.9" text-anchor="end">400</text>
  <line class="grid" x1="74" y1="52.0" x2="620" y2="52.0"/><text class="tick" x="66" y="56.0" text-anchor="end">510</text>
  <text class="lab2" x="74" y="28">某一时刻全卡已开始、未结束的 program 数，按批次堆叠；虚线是 170 × 3 = 510 个名额</text>
  <text class="lab2" x="16" y="119.0" text-anchor="middle" transform="rotate(-90 16 119.0)">同时在跑的 program 数</text>
  <line class="ref" x1="74" y1="52.0" x2="620" y2="52.0"/>
  <g class="m"><title>第 1 批：program 0 到 509，共 510 个</title><polygon points="74.0,106.1 75.2,52.0 76.5,52.0 77.7,52.0 79.0,52.0 80.2,52.0 81.4,52.0 82.7,52.0 83.9,52.0 85.2,52.0 86.4,52.0 87.6,52.0 88.9,52.0 90.1,52.0 91.3,52.0 92.6,52.0 93.8,52.0 95.1,52.0 96.3,52.0 97.5,52.0 98.8,52.0 100.0,52.0 101.3,52.0 102.5,52.0 103.7,52.0 105.0,52.0 106.2,52.0 107.5,52.0 108.7,52.0 109.9,52.0 111.2,52.0 112.4,52.0 113.7,52.0 114.9,52.0 116.1,52.0 117.4,52.0 118.6,52.0 119.8,52.0 121.1,52.0 122.3,52.0 123.6,52.0 124.8,52.0 126.0,52.0 127.3,52.0 128.5,52.0 129.8,52.0 131.0,52.0 132.2,52.0 133.5,52.0 134.7,52.0 136.0,52.0 137.2,52.0 138.4,52.0 139.7,52.0 140.9,52.0 142.1,52.0 143.4,52.0 144.6,52.0 145.9,52.0 147.1,52.0 148.3,52.0 149.6,52.0 150.8,52.0 152.1,52.0 153.3,52.0 154.5,52.0 155.8,52.0 157.0,52.0 158.3,52.0 159.5,52.0 160.7,52.0 162.0,52.0 163.2,52.0 164.5,52.0 165.7,52.0 166.9,52.0 168.2,52.0 169.4,52.0 170.6,52.0 171.9,52.0 173.1,52.0 174.4,52.0 175.6,52.0 176.8,52.0 178.1,52.0 179.3,52.0 180.6,52.0 181.8,52.0 183.0,52.0 184.3,52.0 185.5,52.0 186.8,52.0 188.0,52.0 189.2,52.0 190.5,52.0 191.7,52.0 193.0,52.0 194.2,52.0 195.4,52.0 196.7,52.0 197.9,52.0 199.1,52.0 200.4,52.0 201.6,52.0 202.9,52.0 204.1,52.0 205.3,52.0 206.6,52.0 207.8,52.0 209.1,52.0 210.3,52.0 211.5,52.0 212.8,52.0 214.0,52.0 215.3,52.0 216.5,52.0 217.7,52.0 219.0,52.0 220.2,52.0 221.5,52.0 222.7,52.0 223.9,52.0 225.2,52.0 226.4,52.0 227.6,52.0 228.9,52.0 230.1,52.0 231.4,52.0 232.6,52.0 233.8,52.0 235.1,52.0 236.3,52.0 237.6,52.0 238.8,52.0 240.0,52.0 241.3,52.0 242.5,52.0 243.8,52.0 245.0,52.0 246.2,52.0 247.5,52.0 248.7,52.0 250.0,52.0 251.2,52.0 252.4,52.0 253.7,52.0 254.9,52.0 256.1,52.0 257.4,52.0 258.6,52.0 259.9,52.0 261.1,52.0 262.3,52.0 263.6,52.0 264.8,52.0 266.1,52.0 267.3,52.0 268.5,52.0 269.8,52.0 271.0,52.0 272.3,52.0 273.5,52.0 274.7,52.0 276.0,52.0 277.2,52.0 278.4,52.0 279.7,52.0 280.9,52.0 282.2,52.0 283.4,52.0 284.6,52.0 285.9,52.0 287.1,52.0 288.4,52.0 289.6,52.0 290.8,52.0 292.1,52.0 293.3,52.0 294.6,52.0 295.8,52.0 297.0,52.0 298.3,52.0 299.5,52.0 300.8,52.0 302.0,52.0 303.2,52.0 304.5,52.0 305.7,52.0 306.9,52.0 308.2,52.0 309.4,52.0 310.7,52.0 311.9,52.0 313.1,52.0 314.4,52.0 315.6,52.0 316.9,52.0 318.1,52.0 319.3,52.0 320.6,52.0 321.8,52.0 323.1,52.0 324.3,52.0 325.5,52.0 326.8,52.0 328.0,52.0 329.3,52.0 330.5,52.0 331.7,52.0 333.0,52.0 334.2,52.0 335.4,52.0 336.7,52.0 337.9,52.0 339.2,52.0 340.4,52.0 341.6,52.0 342.9,52.0 344.1,52.0 345.4,52.0 346.6,52.0 347.8,52.0 349.1,52.0 350.3,52.0 351.6,52.0 352.8,52.0 354.0,52.0 355.3,52.0 356.5,52.0 357.8,52.0 359.0,52.0 360.2,52.0 361.5,52.0 362.7,52.0 363.9,52.0 365.2,52.0 366.4,52.0 367.7,52.0 368.9,52.0 370.1,52.0 371.4,52.0 372.6,52.0 373.9,52.0 375.1,52.0 376.3,52.0 377.6,52.0 378.8,52.0 380.1,52.0 381.3,52.0 382.5,52.0 383.8,52.0 385.0,52.0 386.2,52.0 387.5,52.0 388.7,52.0 390.0,52.0 391.2,52.0 392.4,52.0 393.7,52.0 394.9,52.0 396.2,52.0 397.4,52.0 398.6,52.8 399.9,53.8 401.1,54.9 402.4,55.4 403.6,56.7 404.8,58.3 406.1,62.0 407.3,65.9 408.6,74.1 409.8,78.5 411.0,80.4 412.3,82.7 413.5,84.1 414.7,84.1 416.0,84.1 417.2,84.1 418.5,86.2 419.7,90.1 420.9,92.7 422.2,94.3 423.4,99.0 424.7,108.5 425.9,116.6 427.1,119.8 428.4,121.1 429.6,124.8 430.9,126.4 432.1,127.9 433.3,128.7 434.6,128.7 435.8,129.2 437.1,130.8 438.3,132.7 439.5,133.7 440.8,136.3 442.0,140.0 443.2,141.3 444.5,141.9 445.7,144.2 447.0,144.5 448.2,144.5 449.4,146.3 450.7,149.5 451.9,157.1 453.2,163.1 454.4,167.1 455.6,168.9 456.9,171.8 458.1,172.9 459.4,172.9 460.6,172.9 461.8,173.4 463.1,173.9 464.3,177.3 465.6,178.1 466.8,179.7 468.0,181.8 469.3,183.9 470.5,186.0 471.7,186.0 473.0,186.0 474.2,186.0 475.5,186.0 476.7,186.0 477.9,186.0 479.2,186.0 480.4,186.0 481.7,186.0 482.9,186.0 484.1,186.0 485.4,186.0 486.6,186.0 487.9,186.0 489.1,186.0 490.3,186.0 491.6,186.0 492.8,186.0 494.1,186.0 495.3,186.0 496.5,186.0 497.8,186.0 499.0,186.0 500.2,186.0 501.5,186.0 502.7,186.0 504.0,186.0 505.2,186.0 506.4,186.0 507.7,186.0 508.9,186.0 510.2,186.0 511.4,186.0 512.6,186.0 513.9,186.0 515.1,186.0 516.4,186.0 517.6,186.0 518.8,186.0 520.1,186.0 521.3,186.0 522.5,186.0 523.8,186.0 525.0,186.0 526.3,186.0 527.5,186.0 528.7,186.0 530.0,186.0 531.2,186.0 532.5,186.0 533.7,186.0 534.9,186.0 536.2,186.0 537.4,186.0 538.7,186.0 539.9,186.0 541.1,186.0 542.4,186.0 543.6,186.0 544.9,186.0 546.1,186.0 547.3,186.0 548.6,186.0 549.8,186.0 551.0,186.0 552.3,186.0 553.5,186.0 554.8,186.0 556.0,186.0 557.2,186.0 558.5,186.0 559.7,186.0 561.0,186.0 562.2,186.0 563.4,186.0 564.7,186.0 565.9,186.0 567.2,186.0 568.4,186.0 569.6,186.0 570.9,186.0 572.1,186.0 573.4,186.0 574.6,186.0 575.8,186.0 577.1,186.0 578.3,186.0 579.5,186.0 580.8,186.0 582.0,186.0 583.3,186.0 584.5,186.0 585.7,186.0 587.0,186.0 588.2,186.0 589.5,186.0 590.7,186.0 591.9,186.0 593.2,186.0 594.4,186.0 594.4,186.0 593.2,186.0 591.9,186.0 590.7,186.0 589.5,186.0 588.2,186.0 587.0,186.0 585.7,186.0 584.5,186.0 583.3,186.0 582.0,186.0 580.8,186.0 579.5,186.0 578.3,186.0 577.1,186.0 575.8,186.0 574.6,186.0 573.4,186.0 572.1,186.0 570.9,186.0 569.6,186.0 568.4,186.0 567.2,186.0 565.9,186.0 564.7,186.0 563.4,186.0 562.2,186.0 561.0,186.0 559.7,186.0 558.5,186.0 557.2,186.0 556.0,186.0 554.8,186.0 553.5,186.0 552.3,186.0 551.0,186.0 549.8,186.0 548.6,186.0 547.3,186.0 546.1,186.0 544.9,186.0 543.6,186.0 542.4,186.0 541.1,186.0 539.9,186.0 538.7,186.0 537.4,186.0 536.2,186.0 534.9,186.0 533.7,186.0 532.5,186.0 531.2,186.0 530.0,186.0 528.7,186.0 527.5,186.0 526.3,186.0 525.0,186.0 523.8,186.0 522.5,186.0 521.3,186.0 520.1,186.0 518.8,186.0 517.6,186.0 516.4,186.0 515.1,186.0 513.9,186.0 512.6,186.0 511.4,186.0 510.2,186.0 508.9,186.0 507.7,186.0 506.4,186.0 505.2,186.0 504.0,186.0 502.7,186.0 501.5,186.0 500.2,186.0 499.0,186.0 497.8,186.0 496.5,186.0 495.3,186.0 494.1,186.0 492.8,186.0 491.6,186.0 490.3,186.0 489.1,186.0 487.9,186.0 486.6,186.0 485.4,186.0 484.1,186.0 482.9,186.0 481.7,186.0 480.4,186.0 479.2,186.0 477.9,186.0 476.7,186.0 475.5,186.0 474.2,186.0 473.0,186.0 471.7,186.0 470.5,186.0 469.3,186.0 468.0,186.0 466.8,186.0 465.6,186.0 464.3,186.0 463.1,186.0 461.8,186.0 460.6,186.0 459.4,186.0 458.1,186.0 456.9,186.0 455.6,186.0 454.4,186.0 453.2,186.0 451.9,186.0 450.7,186.0 449.4,186.0 448.2,186.0 447.0,186.0 445.7,186.0 444.5,186.0 443.2,186.0 442.0,186.0 440.8,186.0 439.5,186.0 438.3,186.0 437.1,186.0 435.8,186.0 434.6,186.0 433.3,186.0 432.1,186.0 430.9,186.0 429.6,186.0 428.4,186.0 427.1,186.0 425.9,186.0 424.7,186.0 423.4,186.0 422.2,186.0 420.9,186.0 419.7,186.0 418.5,186.0 417.2,186.0 416.0,186.0 414.7,186.0 413.5,186.0 412.3,186.0 411.0,186.0 409.8,186.0 408.6,186.0 407.3,186.0 406.1,186.0 404.8,186.0 403.6,186.0 402.4,186.0 401.1,186.0 399.9,186.0 398.6,186.0 397.4,186.0 396.2,186.0 394.9,186.0 393.7,186.0 392.4,186.0 391.2,186.0 390.0,186.0 388.7,186.0 387.5,186.0 386.2,186.0 385.0,186.0 383.8,186.0 382.5,186.0 381.3,186.0 380.1,186.0 378.8,186.0 377.6,186.0 376.3,186.0 375.1,186.0 373.9,186.0 372.6,186.0 371.4,186.0 370.1,186.0 368.9,186.0 367.7,186.0 366.4,186.0 365.2,186.0 363.9,186.0 362.7,186.0 361.5,186.0 360.2,186.0 359.0,186.0 357.8,186.0 356.5,186.0 355.3,186.0 354.0,186.0 352.8,186.0 351.6,186.0 350.3,186.0 349.1,186.0 347.8,186.0 346.6,186.0 345.4,186.0 344.1,186.0 342.9,186.0 341.6,186.0 340.4,186.0 339.2,186.0 337.9,186.0 336.7,186.0 335.4,186.0 334.2,186.0 333.0,186.0 331.7,186.0 330.5,186.0 329.3,186.0 328.0,186.0 326.8,186.0 325.5,186.0 324.3,186.0 323.1,186.0 321.8,186.0 320.6,186.0 319.3,186.0 318.1,186.0 316.9,186.0 315.6,186.0 314.4,186.0 313.1,186.0 311.9,186.0 310.7,186.0 309.4,186.0 308.2,186.0 306.9,186.0 305.7,186.0 304.5,186.0 303.2,186.0 302.0,186.0 300.8,186.0 299.5,186.0 298.3,186.0 297.0,186.0 295.8,186.0 294.6,186.0 293.3,186.0 292.1,186.0 290.8,186.0 289.6,186.0 288.4,186.0 287.1,186.0 285.9,186.0 284.6,186.0 283.4,186.0 282.2,186.0 280.9,186.0 279.7,186.0 278.4,186.0 277.2,186.0 276.0,186.0 274.7,186.0 273.5,186.0 272.3,186.0 271.0,186.0 269.8,186.0 268.5,186.0 267.3,186.0 266.1,186.0 264.8,186.0 263.6,186.0 262.3,186.0 261.1,186.0 259.9,186.0 258.6,186.0 257.4,186.0 256.1,186.0 254.9,186.0 253.7,186.0 252.4,186.0 251.2,186.0 250.0,186.0 248.7,186.0 247.5,186.0 246.2,186.0 245.0,186.0 243.8,186.0 242.5,186.0 241.3,186.0 240.0,186.0 238.8,186.0 237.6,186.0 236.3,186.0 235.1,186.0 233.8,186.0 232.6,186.0 231.4,186.0 230.1,186.0 228.9,186.0 227.6,186.0 226.4,186.0 225.2,186.0 223.9,186.0 222.7,186.0 221.5,186.0 220.2,186.0 219.0,186.0 217.7,186.0 216.5,186.0 215.3,186.0 214.0,186.0 212.8,186.0 211.5,186.0 210.3,186.0 209.1,186.0 207.8,186.0 206.6,186.0 205.3,186.0 204.1,186.0 202.9,186.0 201.6,186.0 200.4,186.0 199.1,186.0 197.9,186.0 196.7,186.0 195.4,186.0 194.2,186.0 193.0,186.0 191.7,186.0 190.5,186.0 189.2,186.0 188.0,186.0 186.8,186.0 185.5,186.0 184.3,186.0 183.0,186.0 181.8,186.0 180.6,186.0 179.3,186.0 178.1,186.0 176.8,186.0 175.6,186.0 174.4,186.0 173.1,186.0 171.9,186.0 170.6,186.0 169.4,186.0 168.2,186.0 166.9,186.0 165.7,186.0 164.5,186.0 163.2,186.0 162.0,186.0 160.7,186.0 159.5,186.0 158.3,186.0 157.0,186.0 155.8,186.0 154.5,186.0 153.3,186.0 152.1,186.0 150.8,186.0 149.6,186.0 148.3,186.0 147.1,186.0 145.9,186.0 144.6,186.0 143.4,186.0 142.1,186.0 140.9,186.0 139.7,186.0 138.4,186.0 137.2,186.0 136.0,186.0 134.7,186.0 133.5,186.0 132.2,186.0 131.0,186.0 129.8,186.0 128.5,186.0 127.3,186.0 126.0,186.0 124.8,186.0 123.6,186.0 122.3,186.0 121.1,186.0 119.8,186.0 118.6,186.0 117.4,186.0 116.1,186.0 114.9,186.0 113.7,186.0 112.4,186.0 111.2,186.0 109.9,186.0 108.7,186.0 107.5,186.0 106.2,186.0 105.0,186.0 103.7,186.0 102.5,186.0 101.3,186.0 100.0,186.0 98.8,186.0 97.5,186.0 96.3,186.0 95.1,186.0 93.8,186.0 92.6,186.0 91.3,186.0 90.1,186.0 88.9,186.0 87.6,186.0 86.4,186.0 85.2,186.0 83.9,186.0 82.7,186.0 81.4,186.0 80.2,186.0 79.0,186.0 77.7,186.0 76.5,186.0 75.2,186.0 74.0,186.0" style="fill: var(--fig-1)" fill-opacity="0.3"/></g>
  <g class="m"><title>第 2 批：program 510 到 511，共 2 个</title><polygon points="74.0,106.1 75.2,52.0 76.5,52.0 77.7,52.0 79.0,52.0 80.2,52.0 81.4,52.0 82.7,52.0 83.9,52.0 85.2,52.0 86.4,52.0 87.6,52.0 88.9,52.0 90.1,52.0 91.3,52.0 92.6,52.0 93.8,52.0 95.1,52.0 96.3,52.0 97.5,52.0 98.8,52.0 100.0,52.0 101.3,52.0 102.5,52.0 103.7,52.0 105.0,52.0 106.2,52.0 107.5,52.0 108.7,52.0 109.9,52.0 111.2,52.0 112.4,52.0 113.7,52.0 114.9,52.0 116.1,52.0 117.4,52.0 118.6,52.0 119.8,52.0 121.1,52.0 122.3,52.0 123.6,52.0 124.8,52.0 126.0,52.0 127.3,52.0 128.5,52.0 129.8,52.0 131.0,52.0 132.2,52.0 133.5,52.0 134.7,52.0 136.0,52.0 137.2,52.0 138.4,52.0 139.7,52.0 140.9,52.0 142.1,52.0 143.4,52.0 144.6,52.0 145.9,52.0 147.1,52.0 148.3,52.0 149.6,52.0 150.8,52.0 152.1,52.0 153.3,52.0 154.5,52.0 155.8,52.0 157.0,52.0 158.3,52.0 159.5,52.0 160.7,52.0 162.0,52.0 163.2,52.0 164.5,52.0 165.7,52.0 166.9,52.0 168.2,52.0 169.4,52.0 170.6,52.0 171.9,52.0 173.1,52.0 174.4,52.0 175.6,52.0 176.8,52.0 178.1,52.0 179.3,52.0 180.6,52.0 181.8,52.0 183.0,52.0 184.3,52.0 185.5,52.0 186.8,52.0 188.0,52.0 189.2,52.0 190.5,52.0 191.7,52.0 193.0,52.0 194.2,52.0 195.4,52.0 196.7,52.0 197.9,52.0 199.1,52.0 200.4,52.0 201.6,52.0 202.9,52.0 204.1,52.0 205.3,52.0 206.6,52.0 207.8,52.0 209.1,52.0 210.3,52.0 211.5,52.0 212.8,52.0 214.0,52.0 215.3,52.0 216.5,52.0 217.7,52.0 219.0,52.0 220.2,52.0 221.5,52.0 222.7,52.0 223.9,52.0 225.2,52.0 226.4,52.0 227.6,52.0 228.9,52.0 230.1,52.0 231.4,52.0 232.6,52.0 233.8,52.0 235.1,52.0 236.3,52.0 237.6,52.0 238.8,52.0 240.0,52.0 241.3,52.0 242.5,52.0 243.8,52.0 245.0,52.0 246.2,52.0 247.5,52.0 248.7,52.0 250.0,52.0 251.2,52.0 252.4,52.0 253.7,52.0 254.9,52.0 256.1,52.0 257.4,52.0 258.6,52.0 259.9,52.0 261.1,52.0 262.3,52.0 263.6,52.0 264.8,52.0 266.1,52.0 267.3,52.0 268.5,52.0 269.8,52.0 271.0,52.0 272.3,52.0 273.5,52.0 274.7,52.0 276.0,52.0 277.2,52.0 278.4,52.0 279.7,52.0 280.9,52.0 282.2,52.0 283.4,52.0 284.6,52.0 285.9,52.0 287.1,52.0 288.4,52.0 289.6,52.0 290.8,52.0 292.1,52.0 293.3,52.0 294.6,52.0 295.8,52.0 297.0,52.0 298.3,52.0 299.5,52.0 300.8,52.0 302.0,52.0 303.2,52.0 304.5,52.0 305.7,52.0 306.9,52.0 308.2,52.0 309.4,52.0 310.7,52.0 311.9,52.0 313.1,52.0 314.4,52.0 315.6,52.0 316.9,52.0 318.1,52.0 319.3,52.0 320.6,52.0 321.8,52.0 323.1,52.0 324.3,52.0 325.5,52.0 326.8,52.0 328.0,52.0 329.3,52.0 330.5,52.0 331.7,52.0 333.0,52.0 334.2,52.0 335.4,52.0 336.7,52.0 337.9,52.0 339.2,52.0 340.4,52.0 341.6,52.0 342.9,52.0 344.1,52.0 345.4,52.0 346.6,52.0 347.8,52.0 349.1,52.0 350.3,52.0 351.6,52.0 352.8,52.0 354.0,52.0 355.3,52.0 356.5,52.0 357.8,52.0 359.0,52.0 360.2,52.0 361.5,52.0 362.7,52.0 363.9,52.0 365.2,52.0 366.4,52.0 367.7,52.0 368.9,52.0 370.1,52.0 371.4,52.0 372.6,52.0 373.9,52.0 375.1,52.0 376.3,52.0 377.6,52.0 378.8,52.0 380.1,52.0 381.3,52.0 382.5,52.0 383.8,52.0 385.0,52.0 386.2,52.0 387.5,52.0 388.7,52.0 390.0,52.0 391.2,52.0 392.4,52.0 393.7,52.0 394.9,52.0 396.2,52.0 397.4,52.0 398.6,52.8 399.9,53.3 401.1,54.4 402.4,54.9 403.6,56.2 404.8,57.8 406.1,61.5 407.3,65.4 408.6,73.5 409.8,78.0 411.0,79.9 412.3,82.2 413.5,83.5 414.7,83.5 416.0,83.5 417.2,83.5 418.5,85.6 419.7,89.6 420.9,92.2 422.2,93.8 423.4,98.5 424.7,108.0 425.9,116.1 427.1,119.3 428.4,120.6 429.6,124.3 430.9,125.8 432.1,127.4 433.3,128.2 434.6,128.2 435.8,128.7 437.1,130.3 438.3,132.1 439.5,133.2 440.8,135.8 442.0,139.5 443.2,140.8 444.5,141.3 445.7,143.7 447.0,144.0 448.2,144.0 449.4,145.8 450.7,149.0 451.9,156.6 453.2,162.6 454.4,166.6 455.6,168.4 456.9,171.3 458.1,172.3 459.4,172.3 460.6,172.3 461.8,172.9 463.1,173.4 464.3,176.8 465.6,177.6 466.8,179.2 468.0,181.3 469.3,183.4 470.5,185.5 471.7,185.5 473.0,185.5 474.2,185.5 475.5,185.5 476.7,185.5 477.9,185.5 479.2,185.5 480.4,185.5 481.7,185.5 482.9,185.5 484.1,185.5 485.4,185.5 486.6,185.5 487.9,185.5 489.1,185.5 490.3,185.5 491.6,185.5 492.8,185.5 494.1,185.5 495.3,185.5 496.5,185.5 497.8,185.5 499.0,185.5 500.2,185.5 501.5,185.5 502.7,185.5 504.0,185.5 505.2,185.5 506.4,185.5 507.7,185.5 508.9,185.5 510.2,185.5 511.4,185.5 512.6,185.5 513.9,185.5 515.1,185.5 516.4,185.5 517.6,185.5 518.8,185.5 520.1,185.5 521.3,185.5 522.5,185.5 523.8,185.5 525.0,185.5 526.3,185.5 527.5,185.5 528.7,185.5 530.0,185.5 531.2,185.5 532.5,185.5 533.7,185.5 534.9,185.5 536.2,185.5 537.4,185.5 538.7,185.5 539.9,185.5 541.1,185.5 542.4,185.5 543.6,185.5 544.9,185.5 546.1,185.5 547.3,185.5 548.6,185.5 549.8,185.5 551.0,185.5 552.3,185.5 553.5,185.5 554.8,185.5 556.0,185.5 557.2,185.5 558.5,185.5 559.7,185.5 561.0,185.5 562.2,185.5 563.4,185.5 564.7,185.5 565.9,185.5 567.2,185.5 568.4,185.5 569.6,185.5 570.9,185.5 572.1,185.5 573.4,185.5 574.6,185.5 575.8,185.5 577.1,185.5 578.3,185.5 579.5,185.5 580.8,185.5 582.0,185.5 583.3,185.5 584.5,185.5 585.7,185.5 587.0,185.5 588.2,185.5 589.5,185.5 590.7,185.5 591.9,185.5 593.2,185.5 594.4,186.0 594.4,186.0 593.2,186.0 591.9,186.0 590.7,186.0 589.5,186.0 588.2,186.0 587.0,186.0 585.7,186.0 584.5,186.0 583.3,186.0 582.0,186.0 580.8,186.0 579.5,186.0 578.3,186.0 577.1,186.0 575.8,186.0 574.6,186.0 573.4,186.0 572.1,186.0 570.9,186.0 569.6,186.0 568.4,186.0 567.2,186.0 565.9,186.0 564.7,186.0 563.4,186.0 562.2,186.0 561.0,186.0 559.7,186.0 558.5,186.0 557.2,186.0 556.0,186.0 554.8,186.0 553.5,186.0 552.3,186.0 551.0,186.0 549.8,186.0 548.6,186.0 547.3,186.0 546.1,186.0 544.9,186.0 543.6,186.0 542.4,186.0 541.1,186.0 539.9,186.0 538.7,186.0 537.4,186.0 536.2,186.0 534.9,186.0 533.7,186.0 532.5,186.0 531.2,186.0 530.0,186.0 528.7,186.0 527.5,186.0 526.3,186.0 525.0,186.0 523.8,186.0 522.5,186.0 521.3,186.0 520.1,186.0 518.8,186.0 517.6,186.0 516.4,186.0 515.1,186.0 513.9,186.0 512.6,186.0 511.4,186.0 510.2,186.0 508.9,186.0 507.7,186.0 506.4,186.0 505.2,186.0 504.0,186.0 502.7,186.0 501.5,186.0 500.2,186.0 499.0,186.0 497.8,186.0 496.5,186.0 495.3,186.0 494.1,186.0 492.8,186.0 491.6,186.0 490.3,186.0 489.1,186.0 487.9,186.0 486.6,186.0 485.4,186.0 484.1,186.0 482.9,186.0 481.7,186.0 480.4,186.0 479.2,186.0 477.9,186.0 476.7,186.0 475.5,186.0 474.2,186.0 473.0,186.0 471.7,186.0 470.5,186.0 469.3,183.9 468.0,181.8 466.8,179.7 465.6,178.1 464.3,177.3 463.1,173.9 461.8,173.4 460.6,172.9 459.4,172.9 458.1,172.9 456.9,171.8 455.6,168.9 454.4,167.1 453.2,163.1 451.9,157.1 450.7,149.5 449.4,146.3 448.2,144.5 447.0,144.5 445.7,144.2 444.5,141.9 443.2,141.3 442.0,140.0 440.8,136.3 439.5,133.7 438.3,132.7 437.1,130.8 435.8,129.2 434.6,128.7 433.3,128.7 432.1,127.9 430.9,126.4 429.6,124.8 428.4,121.1 427.1,119.8 425.9,116.6 424.7,108.5 423.4,99.0 422.2,94.3 420.9,92.7 419.7,90.1 418.5,86.2 417.2,84.1 416.0,84.1 414.7,84.1 413.5,84.1 412.3,82.7 411.0,80.4 409.8,78.5 408.6,74.1 407.3,65.9 406.1,62.0 404.8,58.3 403.6,56.7 402.4,55.4 401.1,54.9 399.9,53.8 398.6,52.8 397.4,52.0 396.2,52.0 394.9,52.0 393.7,52.0 392.4,52.0 391.2,52.0 390.0,52.0 388.7,52.0 387.5,52.0 386.2,52.0 385.0,52.0 383.8,52.0 382.5,52.0 381.3,52.0 380.1,52.0 378.8,52.0 377.6,52.0 376.3,52.0 375.1,52.0 373.9,52.0 372.6,52.0 371.4,52.0 370.1,52.0 368.9,52.0 367.7,52.0 366.4,52.0 365.2,52.0 363.9,52.0 362.7,52.0 361.5,52.0 360.2,52.0 359.0,52.0 357.8,52.0 356.5,52.0 355.3,52.0 354.0,52.0 352.8,52.0 351.6,52.0 350.3,52.0 349.1,52.0 347.8,52.0 346.6,52.0 345.4,52.0 344.1,52.0 342.9,52.0 341.6,52.0 340.4,52.0 339.2,52.0 337.9,52.0 336.7,52.0 335.4,52.0 334.2,52.0 333.0,52.0 331.7,52.0 330.5,52.0 329.3,52.0 328.0,52.0 326.8,52.0 325.5,52.0 324.3,52.0 323.1,52.0 321.8,52.0 320.6,52.0 319.3,52.0 318.1,52.0 316.9,52.0 315.6,52.0 314.4,52.0 313.1,52.0 311.9,52.0 310.7,52.0 309.4,52.0 308.2,52.0 306.9,52.0 305.7,52.0 304.5,52.0 303.2,52.0 302.0,52.0 300.8,52.0 299.5,52.0 298.3,52.0 297.0,52.0 295.8,52.0 294.6,52.0 293.3,52.0 292.1,52.0 290.8,52.0 289.6,52.0 288.4,52.0 287.1,52.0 285.9,52.0 284.6,52.0 283.4,52.0 282.2,52.0 280.9,52.0 279.7,52.0 278.4,52.0 277.2,52.0 276.0,52.0 274.7,52.0 273.5,52.0 272.3,52.0 271.0,52.0 269.8,52.0 268.5,52.0 267.3,52.0 266.1,52.0 264.8,52.0 263.6,52.0 262.3,52.0 261.1,52.0 259.9,52.0 258.6,52.0 257.4,52.0 256.1,52.0 254.9,52.0 253.7,52.0 252.4,52.0 251.2,52.0 250.0,52.0 248.7,52.0 247.5,52.0 246.2,52.0 245.0,52.0 243.8,52.0 242.5,52.0 241.3,52.0 240.0,52.0 238.8,52.0 237.6,52.0 236.3,52.0 235.1,52.0 233.8,52.0 232.6,52.0 231.4,52.0 230.1,52.0 228.9,52.0 227.6,52.0 226.4,52.0 225.2,52.0 223.9,52.0 222.7,52.0 221.5,52.0 220.2,52.0 219.0,52.0 217.7,52.0 216.5,52.0 215.3,52.0 214.0,52.0 212.8,52.0 211.5,52.0 210.3,52.0 209.1,52.0 207.8,52.0 206.6,52.0 205.3,52.0 204.1,52.0 202.9,52.0 201.6,52.0 200.4,52.0 199.1,52.0 197.9,52.0 196.7,52.0 195.4,52.0 194.2,52.0 193.0,52.0 191.7,52.0 190.5,52.0 189.2,52.0 188.0,52.0 186.8,52.0 185.5,52.0 184.3,52.0 183.0,52.0 181.8,52.0 180.6,52.0 179.3,52.0 178.1,52.0 176.8,52.0 175.6,52.0 174.4,52.0 173.1,52.0 171.9,52.0 170.6,52.0 169.4,52.0 168.2,52.0 166.9,52.0 165.7,52.0 164.5,52.0 163.2,52.0 162.0,52.0 160.7,52.0 159.5,52.0 158.3,52.0 157.0,52.0 155.8,52.0 154.5,52.0 153.3,52.0 152.1,52.0 150.8,52.0 149.6,52.0 148.3,52.0 147.1,52.0 145.9,52.0 144.6,52.0 143.4,52.0 142.1,52.0 140.9,52.0 139.7,52.0 138.4,52.0 137.2,52.0 136.0,52.0 134.7,52.0 133.5,52.0 132.2,52.0 131.0,52.0 129.8,52.0 128.5,52.0 127.3,52.0 126.0,52.0 124.8,52.0 123.6,52.0 122.3,52.0 121.1,52.0 119.8,52.0 118.6,52.0 117.4,52.0 116.1,52.0 114.9,52.0 113.7,52.0 112.4,52.0 111.2,52.0 109.9,52.0 108.7,52.0 107.5,52.0 106.2,52.0 105.0,52.0 103.7,52.0 102.5,52.0 101.3,52.0 100.0,52.0 98.8,52.0 97.5,52.0 96.3,52.0 95.1,52.0 93.8,52.0 92.6,52.0 91.3,52.0 90.1,52.0 88.9,52.0 87.6,52.0 86.4,52.0 85.2,52.0 83.9,52.0 82.7,52.0 81.4,52.0 80.2,52.0 79.0,52.0 77.7,52.0 76.5,52.0 75.2,52.0 74.0,106.1" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <polyline points="74.0,186.0 74.0,177.6 74.1,53.6 355.6,53.8 356.0,54.1 356.5,54.4 356.5,54.9 356.9,55.2 357.0,55.4 357.1,55.7 357.2,55.9 357.3,56.2 357.7,56.5 357.9,56.7 398.9,57.0 399.9,57.8 400.2,58.3 401.1,58.8 401.5,59.1 401.6,59.6 402.1,59.9 402.6,60.1 402.9,60.4 403.2,60.7 403.6,60.9 403.9,61.5 404.2,61.7 404.3,62.0 404.4,62.2 404.5,62.5 404.6,63.3 404.6,63.8 404.7,64.6 404.8,65.1 404.9,65.7 405.0,66.5 405.2,66.7 405.4,67.5 405.5,67.8 405.6,68.3 405.6,68.6 405.7,69.1 405.8,69.6 406.1,70.1 406.1,70.4 406.2,70.9 406.3,71.2 406.5,71.4 406.6,71.7 406.6,72.2 406.7,73.5 406.8,74.1 406.9,74.3 407.0,74.6 407.0,75.1 407.1,75.4 407.2,75.6 407.3,75.9 407.5,76.2 407.5,77.0 407.7,77.5 407.9,77.7 408.0,78.5 408.0,78.8 408.1,79.1 408.2,79.6 408.3,79.9 408.4,80.1 408.5,80.4 408.6,80.9 408.8,81.4 409.0,82.0 409.3,82.2 409.4,82.5 409.5,82.7 409.5,83.0 409.6,83.5 409.7,83.8 409.8,84.1 409.9,84.8 410.0,85.1 410.1,85.9 410.2,86.2 410.7,86.4 411.1,86.9 411.5,87.2 413.9,87.5 415.7,87.7 416.0,88.0 416.1,88.5 416.2,88.8 416.4,89.0 416.6,89.3 416.7,89.6 416.9,89.8 417.2,90.6 417.4,90.9 417.5,91.1 417.7,91.4 417.9,91.7 418.3,91.9 418.4,92.2 418.5,92.7 418.6,93.3 418.9,93.5 419.0,93.8 419.1,94.6 419.4,95.4 419.6,95.6 419.7,95.9 419.9,96.1 419.9,96.4 420.0,96.7 420.1,96.9 420.4,97.2 420.4,98.0 420.5,98.2 420.7,98.8 421.2,99.3 421.4,99.8 421.5,100.1 421.6,100.6 421.8,100.9 421.9,101.1 422.0,101.4 422.1,102.2 422.3,102.7 422.4,103.5 422.4,104.5 422.6,105.1 422.7,105.6 422.8,106.9 422.9,107.2 422.9,107.7 423.0,109.5 423.1,109.8 423.2,110.3 423.3,110.9 423.4,111.1 423.4,111.4 423.5,111.6 423.7,111.9 423.9,113.2 423.9,113.7 424.1,114.3 424.2,115.1 424.3,116.1 424.4,116.4 424.4,116.9 424.5,117.2 424.6,117.4 424.7,118.2 424.9,118.7 424.9,119.0 425.0,119.5 425.1,119.8 425.2,120.3 425.3,120.6 425.4,121.1 425.5,121.4 425.6,121.9 425.8,122.2 425.9,122.4 425.9,122.7 426.3,122.9 426.4,123.2 426.7,123.5 426.8,123.7 427.0,124.0 427.1,124.3 427.4,124.8 427.5,125.0 427.7,125.3 428.0,125.6 428.1,125.8 428.2,126.1 428.4,126.4 428.5,126.6 428.7,127.1 428.8,127.4 428.8,127.9 428.9,128.2 429.0,128.5 429.2,129.0 429.7,129.5 429.9,130.0 430.5,130.3 430.8,130.6 431.2,130.8 432.8,131.3 433.2,131.6 433.3,131.9 433.6,132.1 433.8,132.4 434.0,132.7 434.1,132.9 434.3,133.2 434.4,133.5 434.8,134.0 434.8,134.2 435.1,134.5 435.2,134.8 435.6,135.0 435.8,135.6 436.0,135.8 436.2,136.1 436.3,136.3 436.9,136.6 437.4,136.9 437.6,137.1 437.7,137.4 438.0,137.7 438.2,137.9 438.7,138.4 438.7,138.7 438.8,139.0 439.1,139.2 439.2,139.5 439.3,139.8 439.7,140.0 439.9,140.3 440.0,140.5 440.1,140.8 440.2,141.1 440.4,141.3 441.2,142.1 441.9,142.4 442.0,142.6 442.5,142.9 443.2,143.2 443.6,143.4 443.8,143.7 443.9,144.0 444.1,144.5 444.2,145.0 444.3,145.3 444.4,145.5 444.5,145.8 444.8,146.1 445.2,146.3 445.5,146.6 447.7,146.9 448.0,147.4 448.2,147.6 448.2,148.2 448.3,148.4 448.6,148.7 448.7,149.0 448.9,150.0 449.0,150.5 449.1,150.8 449.2,151.6 449.3,151.8 449.5,152.1 449.6,152.6 449.6,152.9 449.7,153.4 449.8,153.7 450.0,153.9 450.1,154.5 450.2,155.0 450.4,155.8 450.5,156.3 450.6,157.4 450.6,158.1 450.7,158.7 450.8,159.2 450.9,160.0 451.1,161.0 451.1,161.3 451.2,161.6 451.3,162.4 451.5,162.6 451.6,162.9 451.6,163.9 451.8,165.0 451.9,165.2 452.1,166.3 452.2,166.8 452.3,167.1 452.4,167.3 452.6,167.9 452.8,168.1 453.1,168.4 453.3,168.9 453.6,169.4 453.7,169.7 453.8,170.0 453.9,170.5 454.0,171.3 454.1,171.5 454.2,172.1 454.5,172.6 454.7,173.1 455.4,173.4 459.8,173.7 460.1,173.9 460.2,174.2 460.4,174.4 460.5,174.7 460.5,175.0 460.7,175.2 460.8,175.5 460.9,175.8 461.2,176.0 461.3,176.3 461.6,176.5 461.7,176.8 461.8,177.1 462.0,177.3 462.5,177.6 463.2,177.9 463.5,178.1 463.5,178.4 463.6,178.6 463.8,178.9 463.9,179.2 464.2,179.4 464.3,180.0 464.4,180.2 464.5,180.5 464.5,180.7 464.6,181.3 464.8,181.5 465.2,181.8 466.5,182.1 467.0,182.3 467.1,182.8 467.3,183.1 467.5,183.4 467.5,183.9 467.6,184.2 467.7,184.7 467.9,185.2 468.1,185.5 468.2,186.0" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6" stroke-dasharray="5 3"/>
  <line x1="398.7" y1="184.0" x2="594.4" y2="184.0" style="stroke: var(--fig-1)" stroke-opacity="0.9" stroke-width="4"/>
  <rect x="394.7" y="176.0" width="203.7" height="16" rx="5" style="fill: none; stroke: var(--fig-hi)" stroke-width="1.8"/>
  <line x1="532.4" y1="146.0" x2="532.4" y2="175.0" style="stroke: var(--fig-hi)" stroke-width="1.2"/>
  <text class="s" x="532.4" y="128.0" text-anchor="middle" style="fill: var(--fig-hi)">第 2 批（510–511）：2 个 program</text><text class="s" x="532.4" y="142.0" text-anchor="middle" style="fill: var(--fig-hi)">125 µs 开始，跑到 200 µs</text>
  <text class="lab2" x="74" y="222">跑了第 2 批的两个 SM，每个 3 个名额；一条是一个 program 从第一条指令到写完 C</text>
  <text class="lab" x="66" y="252" text-anchor="end">SM 135</text>
  <g class="m"><title>program 27（第 1 批）：0.0 到 124.7 µs</title><rect x="74.0" y="233" width="323.6" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.3"/></g>
  <g class="m"><title>program 510（第 2 批）：124.9 到 200.2 µs</title><rect x="398.7" y="233" width="195.1" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <g class="m"><title>program 197（第 1 批）：0.0 到 135.1 µs</title><rect x="74.0" y="244" width="350.7" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.3"/></g>
  <g class="m"><title>program 367（第 1 批）：0.0 到 149.3 µs</title><rect x="74.1" y="255" width="387.5" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.3"/></g>
  <text class="lab" x="66" y="291" text-anchor="end">SM 68</text>
  <g class="m"><title>program 20（第 1 批）：0.0 到 124.8 µs</title><rect x="74.0" y="272" width="323.9" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.3"/></g>
  <g class="m"><title>program 511（第 2 批）：125.1 到 200.1 µs</title><rect x="399.3" y="272" width="194.3" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.9"/></g>
  <g class="m"><title>program 190（第 1 批）：0.0 到 134.4 µs</title><rect x="74.0" y="283" width="348.8" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.3"/></g>
  <g class="m"><title>program 360（第 1 批）：0.0 到 148.9 µs</title><rect x="74.1" y="294" width="386.5" height="9" rx="1.5" style="fill: var(--fig-1)" fill-opacity="0.3"/></g>
  <text class="tick" x="74.0" y="320" text-anchor="middle">0</text>
  <line class="grid" x1="74.0" y1="52" x2="74.0" y2="186"/>
  <text class="tick" x="139.0" y="320" text-anchor="middle">25</text>
  <line class="grid" x1="139.0" y1="52" x2="139.0" y2="186"/>
  <text class="tick" x="204.0" y="320" text-anchor="middle">50</text>
  <line class="grid" x1="204.0" y1="52" x2="204.0" y2="186"/>
  <text class="tick" x="269.0" y="320" text-anchor="middle">75</text>
  <line class="grid" x1="269.0" y1="52" x2="269.0" y2="186"/>
  <text class="tick" x="334.0" y="320" text-anchor="middle">100</text>
  <line class="grid" x1="334.0" y1="52" x2="334.0" y2="186"/>
  <text class="tick" x="399.0" y="320" text-anchor="middle">125</text>
  <line class="grid" x1="399.0" y1="52" x2="399.0" y2="186"/>
  <text class="tick" x="464.0" y="320" text-anchor="middle">150</text>
  <line class="grid" x1="464.0" y1="52" x2="464.0" y2="186"/>
  <text class="tick" x="529.0" y="320" text-anchor="middle">175</text>
  <line class="grid" x1="529.0" y1="52" x2="529.0" y2="186"/>
  <text class="tick" x="594.0" y="320" text-anchor="middle">200</text>
  <line class="grid" x1="594.0" y1="52" x2="594.0" y2="186"/>
  <text class="lab2" x="66" y="320" text-anchor="end">µs</text>
  <text class="lab2" x="74" y="346">批次</text>
  <rect x="110" y="337" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.3"/><text class="lab" x="127" y="346">0–509</text>
  <rect x="210" y="337" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="0.9"/><text class="lab" x="227" y="346">510–511</text>
  <line x1="324" y1="342" x2="346" y2="342" style="stroke: var(--fig-hi)" stroke-width="1.6" stroke-dasharray="5 3"/><text class="lab" x="354" y="346">A 4032 × 2048：504 个，一批</text>
</svg>
<figcaption><strong>图 3-3</strong> matmul kernel 的 512 个 program 实际怎么分批跑（RTX 5090，fp32，A 4096 × 2048，B 2048 × 512，每个 SM 3 个 program），画法同<a href="#fig-t-5">图 3-2</a>。每个 program 开头读一次 <code>%globaltimer</code> 和 <code>%smid</code>，写回 C 之后再读一次时间，记录前先清空 L2。颜色由淡到浓是 program 的发射顺序所在的批次，一批 510 个。第 2 批只有 2 个，按比例只有 1 像素高，图里画成底边上的一条粗线并用红框圈出。打点版 210.8 µs，不打点 203.3 µs。悬停可看 program 编号和时间。</figcaption>
</figure>

### 3.3 num_warps 和 num_stages {#stages}

两个参数都在启动 kernel 时传：`kernel[grid](..., num_warps=4, num_stages=3)`，这是 Triton 的默认值。

`num_warps` 是一个 program 的 warp 数。block 的大小不变时，warp 多了，每个 thread 分到的数据就少，所以它是在 [3.1 节](#sched-warp)的前两种办法之间做分配。GELU 的 `num_warps` 从 1 到 16 耗时差不多；取 32 时一个 SM 只放得下 1 个 program，每个 thread 只剩 1 个数，才明显变慢。逐元素和按行归约的 kernel 用默认值就够。

`num_stages` 是第三种办法。matmul 每轮把 A、B 的 block 搬进 shared memory 里的 slot，再读进 register 乘加。`num_stages` 取 S 时，第 $k$ 轮的 load 在第 $k-(S-1)$ 轮算完时就用 cp.async 发出（[1.1 节](#hw-move)），算第 $k$ 轮时，下一轮的数据已经在另一个 slot 里（[图 3-4](#fig-t-9)）。代价是每多一级多一个 slot 的 shared memory（[表 3-2](#tab-3-2)）。

<figure id="fig-t-9" class="fg-fig">
<svg class="fg" viewBox="0 0 640 470" width="100%" role="img" aria-label="matmul 循环按 shared memory slot 画的甘特图，三栏共用一根时间轴，都只画前 4 轮。num_stages = 1 只有 1 个 slot：每轮先搬数据、等 852 拍，再从 slot 里读出来算。num_stages = 3 有 2 个 slot：第 k 轮的 load 在第 k − 2 轮算完后发出，算第 k 轮时，第 k + 1 轮已经在另一个 slot 里。num_stages = 4 有 3 个 slot，load 提前两整轮发出，每轮更短，但一个 program 占 49 KB，每个 SM 只放得下 2 个，504 个 program 要分两批">
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
  <text class="tb" x="8" y="30">num_stages = 1</text><text class="lab2" x="184" y="30">1 个 slot，不预取：每轮 5601 拍，其中等显存 852、让给别的 warp 1145</text>
  <text class="lab" x="8" y="55.0">slot (shmem)</text>
  <text class="lab" x="8" y="77.0">warp</text>
  <g class="m"><title>第 0 轮：load 在路上，852 拍</title><rect x="120.0" y="43" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 0 轮：在 slot 里被读</title><rect x="137.8" y="43" width="99.1" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="187.3" y="55.0" text-anchor="middle">第 0 轮</text>
  <g class="m"><title>第 0 轮：等数据 852 拍</title><rect x="120.0" y="65" width="17.8" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 0 轮：自己在跑 3604 拍（发射、等 shared memory、barrier 等）</title><rect x="137.8" y="65" width="75.2" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="175.4" y="77.0" text-anchor="middle">算第 0 轮</text>
  <g class="m"><title>第 0 轮：让给别的 warp 1145 拍</title><rect x="213.0" y="65" width="23.9" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <g class="m"><title>第 1 轮：load 在路上，852 拍</title><rect x="236.9" y="43" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 1 轮：在 slot 里被读</title><rect x="254.7" y="43" width="99.1" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="304.2" y="55.0" text-anchor="middle">第 1 轮</text>
  <g class="m"><title>第 1 轮：等数据 852 拍</title><rect x="236.9" y="65" width="17.8" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 1 轮：自己在跑 3604 拍（发射、等 shared memory、barrier 等）</title><rect x="254.7" y="65" width="75.2" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="292.3" y="77.0" text-anchor="middle">算第 1 轮</text>
  <g class="m"><title>第 1 轮：让给别的 warp 1145 拍</title><rect x="329.9" y="65" width="23.9" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <g class="m"><title>第 2 轮：load 在路上，852 拍</title><rect x="353.8" y="43" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 2 轮：在 slot 里被读</title><rect x="371.6" y="43" width="99.1" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="421.1" y="55.0" text-anchor="middle">第 2 轮</text>
  <g class="m"><title>第 2 轮：等数据 852 拍</title><rect x="353.8" y="65" width="17.8" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 2 轮：自己在跑 3604 拍（发射、等 shared memory、barrier 等）</title><rect x="371.6" y="65" width="75.2" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="409.2" y="77.0" text-anchor="middle">算第 2 轮</text>
  <g class="m"><title>第 2 轮：让给别的 warp 1145 拍</title><rect x="446.8" y="65" width="23.9" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <g class="m"><title>第 3 轮：load 在路上，852 拍</title><rect x="470.7" y="43" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 3 轮：在 slot 里被读</title><rect x="488.5" y="43" width="99.1" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="538.0" y="55.0" text-anchor="middle">第 3 轮</text>
  <g class="m"><title>第 3 轮：等数据 852 拍</title><rect x="470.7" y="65" width="17.8" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 3 轮：自己在跑 3604 拍（发射、等 shared memory、barrier 等）</title><rect x="488.5" y="65" width="75.2" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="526.1" y="77.0" text-anchor="middle">算第 3 轮</text>
  <g class="m"><title>第 3 轮：让给别的 warp 1145 拍</title><rect x="563.7" y="65" width="23.9" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <text class="tb" x="8" y="124">num_stages = 3</text><text class="lab2" x="184" y="124">2 个 slot，提前一轮发 load：每轮 4901 拍，其中等显存 74、让给别的 warp 1486</text>
  <text class="lab" x="8" y="149.0">slot 0 (shmem)</text>
  <text class="lab" x="8" y="171.0">slot 1 (shmem)</text>
  <text class="lab" x="8" y="193.0">warp</text>
  <g class="m"><title>第 0 轮：load 在路上，约 852 拍</title><rect x="120.0" y="137" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 0 轮：已到 slot 里，等着被读</title><rect x="137.8" y="137" width="0.8" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.28"/></g>
  <g class="m"><title>第 0 轮：在 slot 里被读</title><rect x="137.8" y="137" width="102.3" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="188.9" y="149.0" text-anchor="middle">第 0 轮</text>
  <g class="m"><title>第 1 轮：load 在路上，约 852 拍</title><rect x="120.0" y="159" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 1 轮：已到 slot 里，等着被读</title><rect x="137.8" y="159" width="102.3" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.28"/></g>
  <g class="m"><title>第 1 轮：在 slot 里被读</title><rect x="240.1" y="159" width="102.3" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="291.2" y="171.0" text-anchor="middle">第 1 轮</text>
  <g class="m"><title>第 2 轮：load 在路上，约 852 拍</title><rect x="240.1" y="137" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 2 轮：已到 slot 里，等着被读</title><rect x="257.9" y="137" width="84.5" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.28"/></g>
  <g class="m"><title>第 2 轮：在 slot 里被读</title><rect x="342.3" y="137" width="102.3" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="393.5" y="149.0" text-anchor="middle">第 2 轮</text>
  <g class="m"><title>第 3 轮：load 在路上，约 852 拍</title><rect x="342.3" y="159" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 3 轮：已到 slot 里，等着被读</title><rect x="360.1" y="159" width="84.5" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.28"/></g>
  <g class="m"><title>第 3 轮：在 slot 里被读</title><rect x="444.6" y="159" width="102.3" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="495.8" y="171.0" text-anchor="middle">第 3 轮</text>
  <g class="m"><title>第 0 轮：等数据 852 拍</title><rect x="120.0" y="181" width="17.8" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 0 轮：等数据 74 拍</title><rect x="137.8" y="181" width="1.5" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 0 轮：自己在跑 3341 拍（发射、等 shared memory、barrier 等）</title><rect x="139.3" y="181" width="69.7" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="174.2" y="193.0" text-anchor="middle">算第 0 轮</text>
  <g class="m"><title>第 0 轮：让给别的 warp 1486 拍</title><rect x="209.1" y="181" width="31.0" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <g class="m"><title>第 1 轮：等数据 74 拍</title><rect x="240.1" y="181" width="1.5" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 1 轮：自己在跑 3341 拍（发射、等 shared memory、barrier 等）</title><rect x="241.6" y="181" width="69.7" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="276.5" y="193.0" text-anchor="middle">算第 1 轮</text>
  <g class="m"><title>第 1 轮：让给别的 warp 1486 拍</title><rect x="311.3" y="181" width="31.0" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <g class="m"><title>第 2 轮：等数据 74 拍</title><rect x="342.3" y="181" width="1.5" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 2 轮：自己在跑 3341 拍（发射、等 shared memory、barrier 等）</title><rect x="343.9" y="181" width="69.7" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="378.7" y="193.0" text-anchor="middle">算第 2 轮</text>
  <g class="m"><title>第 2 轮：让给别的 warp 1486 拍</title><rect x="413.6" y="181" width="31.0" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <g class="m"><title>第 3 轮：等数据 74 拍</title><rect x="444.6" y="181" width="1.5" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 3 轮：自己在跑 3341 拍（发射、等 shared memory、barrier 等）</title><rect x="446.2" y="181" width="69.7" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="481.0" y="193.0" text-anchor="middle">算第 3 轮</text>
  <g class="m"><title>第 3 轮：让给别的 warp 1486 拍</title><rect x="515.9" y="181" width="31.0" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <line x1="383.3" y1="130" x2="383.3" y2="204" stroke="currentColor" stroke-opacity=".55" stroke-dasharray="4 3"/>
  <text class="s" x="120" y="218">竖虚线这一刻：第 2 轮在 slot 0 里被读、在算；第 3 轮已经到了 slot 1</text>
  <text class="tb" x="8" y="252">num_stages = 4</text><text class="lab2" x="184" y="252">3 个 slot，提前两轮发 load：每轮 3770 拍，其中等显存 111、让给别的 warp 776</text>
  <text class="lab" x="8" y="277.0">slot 0 (shmem)</text>
  <text class="lab" x="8" y="299.0">slot 1 (shmem)</text>
  <text class="lab" x="8" y="321.0">slot 2 (shmem)</text>
  <text class="lab" x="8" y="343.0">warp</text>
  <g class="m"><title>第 0 轮：load 在路上，约 852 拍</title><rect x="120.0" y="265" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 0 轮：已到 slot 里，等着被读</title><rect x="137.8" y="265" width="0.8" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.28"/></g>
  <g class="m"><title>第 0 轮：在 slot 里被读</title><rect x="137.8" y="265" width="78.7" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="177.1" y="277.0" text-anchor="middle">第 0 轮</text>
  <g class="m"><title>第 1 轮：load 在路上，约 852 拍</title><rect x="120.0" y="287" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 1 轮：已到 slot 里，等着被读</title><rect x="137.8" y="287" width="78.7" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.28"/></g>
  <g class="m"><title>第 1 轮：在 slot 里被读</title><rect x="216.5" y="287" width="78.7" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="255.8" y="299.0" text-anchor="middle">第 1 轮</text>
  <g class="m"><title>第 2 轮：load 在路上，约 852 拍</title><rect x="120.0" y="309" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 2 轮：已到 slot 里，等着被读</title><rect x="137.8" y="309" width="157.4" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.28"/></g>
  <g class="m"><title>第 2 轮：在 slot 里被读</title><rect x="295.2" y="309" width="78.7" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="334.5" y="321.0" text-anchor="middle">第 2 轮</text>
  <g class="m"><title>第 3 轮：load 在路上，约 852 拍</title><rect x="216.5" y="265" width="17.8" height="16" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".85" stroke-dasharray="3 2"/></g>
  <g class="m"><title>第 3 轮：已到 slot 里，等着被读</title><rect x="234.3" y="265" width="139.6" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.28"/></g>
  <g class="m"><title>第 3 轮：在 slot 里被读</title><rect x="373.8" y="265" width="78.7" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/></g>
  <text class="s" x="413.2" y="277.0" text-anchor="middle">第 3 轮</text>
  <g class="m"><title>第 0 轮：等数据 852 拍</title><rect x="120.0" y="331" width="17.8" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 0 轮：等数据 111 拍</title><rect x="137.8" y="331" width="2.3" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 0 轮：自己在跑 2883 拍（发射、等 shared memory、barrier 等）</title><rect x="140.1" y="331" width="60.2" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="170.2" y="343.0" text-anchor="middle">算第 0 轮</text>
  <g class="m"><title>第 0 轮：让给别的 warp 776 拍</title><rect x="200.3" y="331" width="16.2" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <g class="m"><title>第 1 轮：等数据 111 拍</title><rect x="216.5" y="331" width="2.3" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 1 轮：自己在跑 2883 拍（发射、等 shared memory、barrier 等）</title><rect x="218.8" y="331" width="60.2" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="248.9" y="343.0" text-anchor="middle">算第 1 轮</text>
  <g class="m"><title>第 1 轮：让给别的 warp 776 拍</title><rect x="279.0" y="331" width="16.2" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <g class="m"><title>第 2 轮：等数据 111 拍</title><rect x="295.2" y="331" width="2.3" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 2 轮：自己在跑 2883 拍（发射、等 shared memory、barrier 等）</title><rect x="297.5" y="331" width="60.2" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="327.6" y="343.0" text-anchor="middle">算第 2 轮</text>
  <g class="m"><title>第 2 轮：让给别的 warp 776 拍</title><rect x="357.6" y="331" width="16.2" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <g class="m"><title>第 3 轮：等数据 111 拍</title><rect x="373.8" y="331" width="2.3" height="16" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/></g>
  <g class="m"><title>第 3 轮：自己在跑 2883 拍（发射、等 shared memory、barrier 等）</title><rect x="376.2" y="331" width="60.2" height="16" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/></g>
  <text class="s" x="406.2" y="343.0" text-anchor="middle">算第 3 轮</text>
  <g class="m"><title>第 3 轮：让给别的 warp 776 拍</title><rect x="436.3" y="331" width="16.2" height="16" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/></g>
  <text class="s" x="120" y="368">每轮更短，主要短在让给别的 warp：每个 SM 只剩 2 个 program、8 个 warp</text>
  <line class="grid" x1="120.0" y1="40" x2="120.0" y2="374"/>
  <text class="tick" x="120.0" y="388" text-anchor="middle">0</text>
  <line class="grid" x1="224.3" y1="40" x2="224.3" y2="374"/>
  <text class="tick" x="224.3" y="388" text-anchor="middle">5000</text>
  <line class="grid" x1="328.7" y1="40" x2="328.7" y2="374"/>
  <text class="tick" x="328.7" y="388" text-anchor="middle">10000</text>
  <line class="grid" x1="433.0" y1="40" x2="433.0" y2="374"/>
  <text class="tick" x="433.0" y="388" text-anchor="middle">15000</text>
  <line class="grid" x1="537.4" y1="40" x2="537.4" y2="374"/>
  <text class="tick" x="537.4" y="388" text-anchor="middle">20000</text>
  <text class="lab2" x="112" y="388" text-anchor="end">拍</text>
  <line x1="587.6" y1="38" x2="587.6" y2="86" style="stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="s" x="592.6" y="78" style="fill: var(--fig-hi)">4 轮算完</text>
  <text class="s" x="592.6" y="92" style="fill: var(--fig-hi)">22406 拍</text>
  <line x1="546.9" y1="132" x2="546.9" y2="202" style="stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="s" x="551.9" y="194" style="fill: var(--fig-hi)">4 轮算完</text>
  <text class="s" x="551.9" y="208" style="fill: var(--fig-hi)">20456 拍</text>
  <line x1="452.5" y1="260" x2="452.5" y2="352" style="stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="s" x="457.5" y="344" style="fill: var(--fig-hi)">4 轮算完</text>
  <text class="s" x="457.5" y="358" style="fill: var(--fig-hi)">15933 拍</text>
  <rect x="8" y="401" width="20" height="11" rx="2" style="fill: none; stroke: var(--fig-1)" stroke-dasharray="3 2"/><text class="s" x="34" y="410">load 在路上</text>
  <rect x="119" y="401" width="20" height="11" rx="2" style="fill: var(--fig-2)" fill-opacity="0.28"/><text class="s" x="145" y="410">已到 slot 里，等着被读</text>
  <rect x="291" y="401" width="20" height="11" rx="2" style="fill: var(--fig-1)" fill-opacity="0.22"/><text class="s" x="317" y="410">在 slot 里被读</text>
  <rect x="8" y="421" width="20" height="11" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/><text class="s" x="34" y="430">自己在跑</text>
  <rect x="100" y="421" width="20" height="11" rx="2" style="fill: var(--fig-2)" fill-opacity="0.75"/><text class="s" x="126" y="430">让给别的 warp</text>
  <rect x="222" y="421" width="20" height="11" rx="2" style="fill: var(--fig-hi)" fill-opacity="1"/><text class="s" x="248" y="430">等显存</text>
</svg>
<figcaption><strong>图 3-4</strong> matmul 的循环按 shared memory slot 看（RTX 5090，fp32，A 4032 × 2048，B 2048 × 512，BM = BN = 64，BK = 32，4 个 warp）。1 级和 3 级时 504 个 program 一批跑完，4 级时每个 SM 只放得下 2 个，要分两批，这里画的是第一批里的一个 program。条长按 Nsight Compute 实测（<code>--clock-control none</code>）画，都是 64 轮的平均值：每轮的拍数 = 一个 warp 每条指令平均隔几拍 × 每轮的指令数；把这几拍按 warp 当时的状态拆开，等 L2 / 显存的记作「等显存」，自己已经就绪、但调度器这一拍发了别的 warp 的记作「让给别的 warp」，其余（发射、等 shared memory、barrier 等）记作「自己在跑」。三种状态在一轮里是穿插的，这里按比例拼在一起画。load 的延迟没法直接测，按不预取时每轮等数据的拍数画。三栏共用一根时间轴，都只画前 4 轮，红线是 4 轮算完的时刻。</figcaption>
</figure>

| num_stages | A、B 各几个 slot (shmem) | load 提前几整轮 | shared memory | 每个 SM 放几个 program | 耗时 |
|:--|--:|--:|--:|--:|--:|
| 1 | 1 | — | 16 KB | 5 | 181.4 µs |
| 2 | 1 | 0 | 16 KB | 5 | 175.8 µs |
| 3（默认） | 2 | 1 | 32 KB | 3 | 169.3 µs |
| 4 | 3 | 2 | 48 KB | 2 | 210.0 µs |
{#tab-3-2 caption="**表 3-2** matmul 的 num_stages 改了什么（A 4032 × 2048，B 2048 × 512，BM = BN = 64，BK = 32，4 个 warp）" note="slot 数读自编译出的 TTGIR，提前几轮读自 PTX 里 cp.async 的发出位置，shared memory 读自编译结果的 metadata。用 4032 行是为了让 504 个 program 正好一批跑完，避开 3.2 节的 tail effect。耗时是 <code>do_bench</code> 跑 5 次的中位数，四档是同一次测量。"}

默认的 3 级已经把每轮等数据的时间从 852 拍压到 74 拍，再加一级也没有多少等待可藏。`num_stages=4` 反而慢了 24%。它每一轮其实更短（[图 3-4](#fig-t-9) 最下面一栏），慢在 shared memory：每多一级多一个 16 KB 的 slot，一个 program 占 48 KB，加上保留的 1 KB 是 49 KB，100 KB 只放得下 2 个（3 级时 33 KB，放得下 3 个）。全卡一批从 510 个降到 340 个，504 个 program 要分两批，第 2 批 164 个（[3.2 节](#sched-wave)，[图 3-5](#fig-s-1)）。四个例子里只有 matmul 的循环用得上 `num_stages`。它也改不了 program 的个数：4096 行时取 1 到 4，耗时都在 209 到 214 µs。

<figure id="fig-s-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 350" width="100%" role="img" aria-label="num_stages = 3 和 4 时一个 SM 的 shared memory 和整个 kernel 的时间。3 级每个 program 占 33 KB，100 KB 放得下 3 个，504 个 program 一批跑完；4 级每个 program 占 49 KB，只放得下 2 个，要分两批，第二批每个 SM 最多 1 个">
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
  <text class="tb" x="8" y="20">一个 SM 的 shared memory（100 KB）</text>
  <text class="lab" x="8" y="61">num_stages = 3</text>
  <rect x="130.0" y="46" width="470.0" height="22" rx="2" fill="none" stroke="currentColor" stroke-opacity=".35"/>
  <g class="m"><title>P0 的 slot 0：A、B 各 8 KB</title><rect x="130.5" y="47" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="167.6" y="61" text-anchor="middle" style="fill: #fff; opacity: 1">slot 0</text>
  <g class="m"><title>P0 的 slot 1：A、B 各 8 KB</title><rect x="205.7" y="47" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="242.8" y="61" text-anchor="middle" style="fill: #fff; opacity: 1">slot 1</text>
  <rect x="280.4" y="47" width="4.7" height="20" style="fill: var(--fig-mute)"/>
  <text class="s" x="207.6" y="43" text-anchor="middle">P0：33 KB</text>
  <g class="m"><title>P1 的 slot 0：A、B 各 8 KB</title><rect x="285.6" y="47" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.35"/></g>
  <text class="s" x="322.7" y="61" text-anchor="middle" style="fill: #fff; opacity: 1">slot 0</text>
  <g class="m"><title>P1 的 slot 1：A、B 各 8 KB</title><rect x="360.8" y="47" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.35"/></g>
  <text class="s" x="397.9" y="61" text-anchor="middle" style="fill: #fff; opacity: 1">slot 1</text>
  <rect x="435.5" y="47" width="4.7" height="20" style="fill: var(--fig-mute)"/>
  <text class="s" x="362.7" y="43" text-anchor="middle">P1：33 KB</text>
  <g class="m"><title>P2 的 slot 0：A、B 各 8 KB</title><rect x="440.7" y="47" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="477.8" y="61" text-anchor="middle" style="fill: #fff; opacity: 1">slot 0</text>
  <g class="m"><title>P2 的 slot 1：A、B 各 8 KB</title><rect x="515.9" y="47" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="553.0" y="61" text-anchor="middle" style="fill: #fff; opacity: 1">slot 1</text>
  <rect x="590.6" y="47" width="4.7" height="20" style="fill: var(--fig-mute)"/>
  <text class="s" x="517.8" y="43" text-anchor="middle">P2：33 KB</text>
  <text class="s" x="606" y="55">放 3 个</text><text class="s" x="606" y="68">剩 1 KB</text>
  <text class="lab" x="8" y="105">num_stages = 4</text>
  <rect x="130.0" y="90" width="470.0" height="22" rx="2" fill="none" stroke="currentColor" stroke-opacity=".35"/>
  <g class="m"><title>P0 的 slot 0：A、B 各 8 KB</title><rect x="130.5" y="91" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="167.6" y="105" text-anchor="middle" style="fill: #fff; opacity: 1">slot 0</text>
  <g class="m"><title>P0 的 slot 1：A、B 各 8 KB</title><rect x="205.7" y="91" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="242.8" y="105" text-anchor="middle" style="fill: #fff; opacity: 1">slot 1</text>
  <g class="m"><title>P0 的 slot 2：A、B 各 8 KB</title><rect x="280.9" y="91" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="318.0" y="105" text-anchor="middle" style="fill: #fff; opacity: 1">slot 2</text>
  <rect x="355.6" y="91" width="4.7" height="20" style="fill: var(--fig-mute)"/>
  <text class="s" x="245.1" y="87" text-anchor="middle">P0：49 KB</text>
  <g class="m"><title>P1 的 slot 0：A、B 各 8 KB</title><rect x="360.8" y="91" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.35"/></g>
  <text class="s" x="397.9" y="105" text-anchor="middle" style="fill: #fff; opacity: 1">slot 0</text>
  <g class="m"><title>P1 的 slot 1：A、B 各 8 KB</title><rect x="436.0" y="91" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.35"/></g>
  <text class="s" x="473.1" y="105" text-anchor="middle" style="fill: #fff; opacity: 1">slot 1</text>
  <g class="m"><title>P1 的 slot 2：A、B 各 8 KB</title><rect x="511.2" y="91" width="74.2" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="0.35"/></g>
  <text class="s" x="548.3" y="105" text-anchor="middle" style="fill: #fff; opacity: 1">slot 2</text>
  <rect x="585.9" y="91" width="4.7" height="20" style="fill: var(--fig-mute)"/>
  <text class="s" x="475.4" y="87" text-anchor="middle">P1：49 KB</text>
  <text class="s" x="606" y="99">放 2 个</text><text class="s" x="606" y="112">剩 2 KB</text>
  <line class="grid" x1="130" y1="296" x2="600" y2="296"/>
  <line class="grid" x1="130.0" y1="154" x2="130.0" y2="296"/>
  <line class="grid" x1="220.4" y1="154" x2="220.4" y2="296"/>
  <line class="grid" x1="310.8" y1="154" x2="310.8" y2="296"/>
  <line class="grid" x1="401.2" y1="154" x2="401.2" y2="296"/>
  <line class="grid" x1="491.5" y1="154" x2="491.5" y2="296"/>
  <line class="grid" x1="581.9" y1="154" x2="581.9" y2="296"/>
  <text class="tb" x="8" y="146">整个 kernel：一个 SM 上同时跑的 program（504 个 program，170 个 SM）</text>
  <text class="lab" x="8" y="189.0">num_stages = 3</text>
  <g class="m"><title>每个 SM 3 个 program，504 个一批跑完</title><rect x="130.0" y="160" width="320.4" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="136.0" y="171.0" style="fill: #fff; opacity: 1">第 1 批 P0：64 轮 × 4901 拍</text>
  <g class="m"><title>每个 SM 3 个 program，504 个一批跑完</title><rect x="130.0" y="178" width="320.4" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="136.0" y="189.0" style="fill: #fff; opacity: 1">第 1 批 P1：64 轮 × 4901 拍</text>
  <g class="m"><title>每个 SM 3 个 program，504 个一批跑完</title><rect x="130.0" y="196" width="320.4" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="136.0" y="207.0" style="fill: #fff; opacity: 1">第 1 批 P2：64 轮 × 4901 拍</text>
  <text class="lab" x="8" y="256">num_stages = 4</text>
  <g class="m"><title>第 1 批 340 个 program，每个 SM 2 个</title><rect x="130.0" y="236" width="218.1" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="136.0" y="247.0" style="fill: #fff; opacity: 1">第 1 批 P0：64 轮 × 3770 拍</text>
  <g class="m"><title>第 1 批 340 个 program，每个 SM 2 个</title><rect x="130.0" y="254" width="218.1" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.55"/></g>
  <text class="s" x="136.0" y="265.0" style="fill: #fff; opacity: 1">第 1 批 P1：64 轮 × 3770 拍</text>
  <g class="m"><title>第 2 批 164 个 program，每个 SM 最多 1 个</title><rect x="348.1" y="236" width="227.6" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.35"/></g>
  <text class="s" x="354.1" y="247.0" style="fill: #fff; opacity: 1">第 2 批：64 轮</text>
  <rect x="348.1" y="254" width="227.6" height="14" rx="2" fill="none" stroke="currentColor" stroke-opacity=".45" stroke-dasharray="3 2"/>
  <text class="s" x="354.1" y="265.0">空着</text>
  <line x1="450.4" y1="154" x2="450.4" y2="278" style="stroke: var(--fig-hi)" stroke-width="1.4" stroke-dasharray="4 3"/>
  <line x1="575.7" y1="154" x2="575.7" y2="278" style="stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="454.4" y="224" style="fill: var(--fig-hi); opacity: 1">3 级结束：354432 拍</text>
  <text class="s" x="571.7" y="290" text-anchor="end" style="fill: var(--fig-hi); opacity: 1">4 级结束：493138 拍（1.39 倍）</text>
  <text class="tick" x="130.0" y="310" text-anchor="middle">0</text>
  <text class="tick" x="220.4" y="310" text-anchor="middle">1e5</text>
  <text class="tick" x="310.8" y="310" text-anchor="middle">2e5</text>
  <text class="tick" x="401.2" y="310" text-anchor="middle">3e5</text>
  <text class="tick" x="491.5" y="310" text-anchor="middle">4e5</text>
  <text class="tick" x="581.9" y="310" text-anchor="middle">5e5</text>
  <text class="lab2" x="122" y="310" text-anchor="end">拍</text>
</svg>
<figcaption><strong>图 3-5</strong> num_stages = 3 和 4 时，一个 SM 放几个 program、整个 kernel 跑几批（RTX 5090，形状同<a href="#tab-3-2">表 3-2</a>）。上：一个 slot 存一轮的 A、B block，各 8 KB，共 16 KB；灰色是每个 program 保留的 1 KB。下：每轮的拍数和整个 kernel 的拍数都按 Nsight Compute 实测（<code>--clock-control none</code>），整个 kernel 的拍数 = kernel 耗时 × SM 时钟频率；4 级第 1 批按 64 轮 × 每轮拍数画，第 2 批画到 kernel 结束。</figcaption>
</figure>

### 3.4 block 大小和 autotune {#tile}

block 的大小同时决定 program 的个数、每个 program 占的资源和每个 thread 存几个 acc。我们在正文形状和 4096 × 4096 × 4096 上各扫了 198 组 BM、BN、BK、`num_warps` 和 `num_stages`。正文形状下最快的是 BM = BN = 128、BK = 32、8 个 warp，176.9 µs，比 64 × 64 的 203.0 µs 少用 13% 的时间（[图 3-6](#fig-d-1)）。它只有 128 个 program，空着 42 个 SM，但没有哪个 SM 要多跑一个 program，每个 thread 存的 acc 也多一倍。

<figure id="fig-d-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 262" width="100%" role="img" aria-label="正文形状下最快的配置（BM = BN = 128，8 个 warp）和正文用的 64 × 64、4 个 warp，在一个 SM 上各用了多少：最快的只驻留 8 个 warp，register 用掉 88%">
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
  <rect x="10" y="10" width="620" height="100" rx="10" style="fill: color-mix(in srgb, var(--fig-1) 5%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="tb" x="22" y="28">最快：BM = 128，BN = 128，BK = 32，8 个 warp</text>
  <text class="lab2" x="22" y="44">每个 thread 64 个 acc，220 个 register；shared memory 64 KB；128 个 program</text>
  <text class="tb" x="618" y="28" text-anchor="end" style="fill: var(--fig-hi)">176.9 µs</text>
  <text class="s" x="22" y="63">warp 名额</text><text class="s" x="150" y="63" text-anchor="end">48</text>
  <rect x="156" y="54" width="312" height="11" rx="2" fill="currentColor" fill-opacity=".06" stroke="currentColor" stroke-opacity=".3" stroke-width=".8"/>
  <rect x="156.00" y="54" width="52.00" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <text class="s" x="476" y="63">用了 17%</text>
  <text class="s" x="22" y="77">program 名额</text><text class="s" x="150" y="77" text-anchor="end">24</text>
  <rect x="156" y="68" width="312" height="11" rx="2" fill="currentColor" fill-opacity=".06" stroke="currentColor" stroke-opacity=".3" stroke-width=".8"/>
  <rect x="156.00" y="68" width="13.00" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <text class="s" x="476" y="77">用了 4%</text>
  <text class="s" x="22" y="91">register</text><text class="s" x="150" y="91" text-anchor="end">65536</text>
  <rect x="156" y="82" width="312" height="11" rx="2" fill="currentColor" fill-opacity=".06" stroke="currentColor" stroke-opacity=".3" stroke-width=".8"/>
  <rect x="156.00" y="82" width="273.00" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <rect x="154" y="80" width="316" height="15" rx="3" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="s" style="fill: var(--fig-hi); opacity: 1; font-weight: 600" x="476" y="91">用了 88%</text>
  <text class="s" x="22" y="105">shared memory</text><text class="s" x="150" y="105" text-anchor="end">100 KB</text>
  <rect x="156" y="96" width="312" height="11" rx="2" fill="currentColor" fill-opacity=".06" stroke="currentColor" stroke-opacity=".3" stroke-width=".8"/>
  <rect x="156.00" y="96" width="202.80" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <text class="s" x="476" y="105">用了 65%</text>
  <rect x="10" y="122" width="620" height="100" rx="10" style="fill: color-mix(in srgb, var(--fig-1) 5%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="tb" x="22" y="140">正文用的：BM = 64，BN = 64，BK = 32，4 个 warp</text>
  <text class="lab2" x="22" y="156">每个 thread 32 个 acc，96 个 register；shared memory 32 KB；512 个 program</text>
  <text class="tb" x="618" y="140" text-anchor="end" style="fill: var(--fig-hi)">203.0 µs</text>
  <text class="s" x="22" y="175">warp 名额</text><text class="s" x="150" y="175" text-anchor="end">48</text>
  <rect x="156" y="166" width="312" height="11" rx="2" fill="currentColor" fill-opacity=".06" stroke="currentColor" stroke-opacity=".3" stroke-width=".8"/>
  <rect x="156.00" y="166" width="26.00" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <rect x="182.00" y="166" width="26.00" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <rect x="208.00" y="166" width="26.00" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <text class="s" x="476" y="175">用了 25%</text>
  <text class="s" x="22" y="189">program 名额</text><text class="s" x="150" y="189" text-anchor="end">24</text>
  <rect x="156" y="180" width="312" height="11" rx="2" fill="currentColor" fill-opacity=".06" stroke="currentColor" stroke-opacity=".3" stroke-width=".8"/>
  <rect x="156.00" y="180" width="13.00" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <rect x="169.00" y="180" width="13.00" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <rect x="182.00" y="180" width="13.00" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <text class="s" x="476" y="189">用了 12%</text>
  <text class="s" x="22" y="203">register</text><text class="s" x="150" y="203" text-anchor="end">65536</text>
  <rect x="156" y="194" width="312" height="11" rx="2" fill="currentColor" fill-opacity=".06" stroke="currentColor" stroke-opacity=".3" stroke-width=".8"/>
  <rect x="156.00" y="194" width="58.50" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <rect x="214.50" y="194" width="58.50" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <rect x="273.00" y="194" width="58.50" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <text class="s" x="476" y="203">用了 56%</text>
  <text class="s" x="22" y="217">shared memory</text><text class="s" x="150" y="217" text-anchor="end">100 KB</text>
  <rect x="156" y="208" width="312" height="11" rx="2" fill="currentColor" fill-opacity=".06" stroke="currentColor" stroke-opacity=".3" stroke-width=".8"/>
  <rect x="156.00" y="208" width="102.96" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <rect x="258.96" y="208" width="102.96" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <rect x="361.92" y="208" width="102.96" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="0.8"/>
  <rect x="154" y="206" width="316" height="15" rx="3" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="s" style="fill: var(--fig-hi); opacity: 1; font-weight: 600" x="476" y="217">用了 99%</text>
  <rect x="22" y="240" width="22" height="10" style="fill: var(--fig-1)"/><text class="lab" x="52" y="249">一格是一个 program 占的量</text>
  <rect x="230" y="239" width="22" height="12" rx="2" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6"/><text class="lab" x="260" y="249">先用完的那条</text>
</svg>
<figcaption><strong>图 3-6</strong> 正文形状（A 4096 × 2048，B 2048 × 512，fp32）下最快的配置和正文用的配置，在一个 SM 上各用了多少。四根条对应<a href="#tab-3-1">表 3-1</a> 的四条上限，条长是这条上限的总量。</figcaption>
</figure>

两个形状的规律一样（[图 3-7](#fig-d-2)）：每个 thread 存 32 到 64 个 acc、每个 SM 只驻留 8 到 16 个 warp 时最快。acc 越多，每从 shared memory 读一个数能做的乘加越多（[单卡优化篇第 7 节](/blog/gpu-single-gpu-optimization/#tiling)）；驻留的 warp 一多，每个 thread 分到的 register 就放不下这么多 acc。所以 matmul 靠第二、三种办法藏延迟，不靠多驻留 warp。

<figure id="fig-d-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 300" width="100%" role="img" aria-label="按每个 thread 的 acc 个数、每个 SM 驻留的 warp 数分组，每组最快的配置比整体最快慢多少：acc 在 32 到 64 个、warp 在 8 到 16 个时最快，warp 驻留满 48 个反而慢一倍">
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
  <rect x="214.0" y="40" width="52.0" height="190" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 14%, transparent); stroke: var(--fig-2)" stroke-width="0"/>
  <line class="grid" x1="60" y1="230.0" x2="300" y2="230.0"/><text class="tick" x="54" y="234.0" text-anchor="end">1.0×</text>
  <line class="grid" x1="60" y1="162.1" x2="300" y2="162.1"/><text class="tick" x="54" y="166.1" text-anchor="end">1.5×</text>
  <line class="grid" x1="60" y1="94.3" x2="300" y2="94.3"/><text class="tick" x="54" y="98.3" text-anchor="end">2.0×</text>
  <text class="tick" x="60.0" y="246" text-anchor="middle">2</text>
  <text class="tick" x="100.0" y="246" text-anchor="middle">4</text>
  <text class="tick" x="140.0" y="246" text-anchor="middle">8</text>
  <text class="tick" x="180.0" y="246" text-anchor="middle">16</text>
  <text class="tick" x="220.0" y="246" text-anchor="middle">32</text>
  <text class="tick" x="260.0" y="246" text-anchor="middle">64</text>
  <text class="tick" x="300.0" y="246" text-anchor="middle">128</text>
  <text class="lab2" x="180.0" y="264" text-anchor="middle">每个 thread 的 acc 个数</text>
  <polyline points="60.0,80.7 100.0,144.3 140.0,195.1 180.0,216.6 220.0,226.9 260.0,230.0 300.0,133.3" fill="none" style="stroke: var(--fig-1)" stroke-width="2" stroke-linejoin="round"/>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 thread 的 acc 个数 2：最快 371.4 µs，是整体最快的 2.10 倍</title><circle cx="60.0" cy="80.7" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 thread 的 acc 个数 4：最快 288.5 µs，是整体最快的 1.63 倍</title><circle cx="100.0" cy="144.3" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 thread 的 acc 个数 8：最快 222.4 µs，是整体最快的 1.26 倍</title><circle cx="140.0" cy="195.1" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 thread 的 acc 个数 16：最快 194.3 µs，是整体最快的 1.10 倍</title><circle cx="180.0" cy="216.6" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 thread 的 acc 个数 32：最快 180.9 µs，是整体最快的 1.02 倍</title><circle cx="220.0" cy="226.9" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 thread 的 acc 个数 64：最快 176.9 µs，是整体最快的 1.00 倍</title><circle cx="260.0" cy="230.0" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 thread 的 acc 个数 128：最快 303.0 µs，是整体最快的 1.71 倍</title><circle cx="300.0" cy="133.3" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <polyline points="60.0,40.0 100.0,102.5 140.0,175.3 180.0,195.3 220.0,223.9 260.0,230.0 300.0,216.3" fill="none" style="stroke: var(--fig-hi)" stroke-width="2" stroke-linejoin="round"/>
  <g class="m"><title>4096 × 4096 × 4096，每个 thread 的 acc 个数 2：最快 5811.9 µs，是整体最快的 2.40 倍</title><circle cx="60.0" cy="40.0" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 thread 的 acc 个数 4：最快 4689.2 µs，是整体最快的 1.94 倍</title><circle cx="100.0" cy="102.5" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 thread 的 acc 个数 8：最快 3391.7 µs，是整体最快的 1.40 倍</title><circle cx="140.0" cy="175.3" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 thread 的 acc 个数 16：最快 3035.1 µs，是整体最快的 1.26 倍</title><circle cx="180.0" cy="195.3" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 thread 的 acc 个数 32：最快 2525.8 µs，是整体最快的 1.04 倍</title><circle cx="220.0" cy="223.9" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 thread 的 acc 个数 64：最快 2417.6 µs，是整体最快的 1.00 倍</title><circle cx="260.0" cy="230.0" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 thread 的 acc 个数 128：最快 2661.4 µs，是整体最快的 1.10 倍</title><circle cx="300.0" cy="216.3" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <rect x="395.8" y="40" width="55.6" height="190" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 14%, transparent); stroke: var(--fig-2)" stroke-width="0"/>
  <line class="grid" x1="380" y1="230.0" x2="620" y2="230.0"/><text class="tick" x="374" y="234.0" text-anchor="end">1.0×</text>
  <line class="grid" x1="380" y1="162.1" x2="620" y2="162.1"/><text class="tick" x="374" y="166.1" text-anchor="end">1.5×</text>
  <line class="grid" x1="380" y1="94.3" x2="620" y2="94.3"/><text class="tick" x="374" y="98.3" text-anchor="end">2.0×</text>
  <text class="tick" x="380.0" y="246" text-anchor="middle">4</text>
  <text class="tick" x="401.8" y="246" text-anchor="middle">8</text>
  <text class="tick" x="423.6" y="246" text-anchor="middle">12</text>
  <text class="tick" x="445.5" y="246" text-anchor="middle">16</text>
  <text class="tick" x="489.1" y="246" text-anchor="middle">24</text>
  <text class="tick" x="532.7" y="246" text-anchor="middle">32</text>
  <text class="tick" x="576.4" y="246" text-anchor="middle">40</text>
  <text class="tick" x="620.0" y="246" text-anchor="middle">48</text>
  <text class="lab2" x="500.0" y="264" text-anchor="middle">每个 SM 驻留的 warp 数</text>
  <polyline points="380.0,196.5 401.8,230.0 423.6,226.2 445.5,226.9 467.3,219.4 489.1,202.9 510.9,211.5 532.7,206.2 554.5,182.3 576.4,184.6 620.0,123.6" fill="none" style="stroke: var(--fig-1)" stroke-width="2" stroke-linejoin="round"/>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 SM 驻留的 warp 数 4：最快 220.5 µs，是整体最快的 1.25 倍</title><circle cx="380.0" cy="196.5" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 SM 驻留的 warp 数 8：最快 176.9 µs，是整体最快的 1.00 倍</title><circle cx="401.8" cy="230.0" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 SM 驻留的 warp 数 12：最快 181.8 µs，是整体最快的 1.03 倍</title><circle cx="423.6" cy="226.2" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 SM 驻留的 warp 数 16：最快 180.9 µs，是整体最快的 1.02 倍</title><circle cx="445.5" cy="226.9" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 SM 驻留的 warp 数 20：最快 190.7 µs，是整体最快的 1.08 倍</title><circle cx="467.3" cy="219.4" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 SM 驻留的 warp 数 24：最快 212.2 µs，是整体最快的 1.20 倍</title><circle cx="489.1" cy="202.9" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 SM 驻留的 warp 数 28：最快 201.0 µs，是整体最快的 1.14 倍</title><circle cx="510.9" cy="211.5" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 SM 驻留的 warp 数 32：最快 207.9 µs，是整体最快的 1.18 倍</title><circle cx="532.7" cy="206.2" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 SM 驻留的 warp 数 36：最快 239.0 µs，是整体最快的 1.35 倍</title><circle cx="554.5" cy="182.3" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 SM 驻留的 warp 数 40：最快 236.0 µs，是整体最快的 1.33 倍</title><circle cx="576.4" cy="184.6" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>正文形状 4096 × 2048 × 512，每个 SM 驻留的 warp 数 48：最快 315.5 µs，是整体最快的 1.78 倍</title><circle cx="620.0" cy="123.6" r="3.5" class="ring" style="fill: var(--fig-1)"/></g>
  <polyline points="380.0,212.3 401.8,228.6 423.6,230.0 445.5,223.9 467.3,212.6 489.1,189.1 510.9,187.7 532.7,195.3 554.5,165.3 576.4,171.7 620.0,93.7" fill="none" style="stroke: var(--fig-hi)" stroke-width="2" stroke-linejoin="round"/>
  <g class="m"><title>4096 × 4096 × 4096，每个 SM 驻留的 warp 数 4：最快 2733.7 µs，是整体最快的 1.13 倍</title><circle cx="380.0" cy="212.3" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 SM 驻留的 warp 数 8：最快 2442.8 µs，是整体最快的 1.01 倍</title><circle cx="401.8" cy="228.6" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 SM 驻留的 warp 数 12：最快 2417.6 µs，是整体最快的 1.00 倍</title><circle cx="423.6" cy="230.0" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 SM 驻留的 warp 数 16：最快 2525.8 µs，是整体最快的 1.04 倍</title><circle cx="445.5" cy="223.9" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 SM 驻留的 warp 数 20：最快 2727.3 µs，是整体最快的 1.13 倍</title><circle cx="467.3" cy="212.6" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 SM 驻留的 warp 数 24：最快 3146.5 µs，是整体最快的 1.30 倍</title><circle cx="489.1" cy="189.1" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 SM 驻留的 warp 数 28：最快 3171.5 µs，是整体最快的 1.31 倍</title><circle cx="510.9" cy="187.7" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 SM 驻留的 warp 数 32：最快 3035.1 µs，是整体最快的 1.26 倍</title><circle cx="532.7" cy="195.3" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 SM 驻留的 warp 数 36：最快 3570.7 µs，是整体最快的 1.48 倍</title><circle cx="554.5" cy="165.3" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 SM 驻留的 warp 数 40：最快 3456.4 µs，是整体最快的 1.43 倍</title><circle cx="576.4" cy="171.7" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>4096 × 4096 × 4096，每个 SM 驻留的 warp 数 48：最快 4845.8 µs，是整体最快的 2.00 倍</title><circle cx="620.0" cy="93.7" r="3.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <text class="lab2" x="60" y="22">比最快的配置慢几倍（每组取最快的，不算 spill 的配置）；底色是最快的区间</text>
  <line x1="60" y1="286" x2="82" y2="286" style="stroke: var(--fig-1)" stroke-width="2"/><text class="lab" x="90" y="290">正文形状 4096 × 2048 × 512</text>
  <line x1="310" y1="286" x2="332" y2="286" style="stroke: var(--fig-hi)" stroke-width="2"/><text class="lab" x="340" y="290">4096 × 4096 × 4096</text>
</svg>
<figcaption><strong>图 3-7</strong> 两个形状各扫 198 组配置（RTX 5090，fp32，<code>input_precision="ieee"</code>）。左：按每个 thread 存的 acc 个数（BM × BN / (32 × num_warps)）分组；右：按每个 SM 驻留的 warp 数分组。每组取最快的一个，纵轴是它比这个形状下整体最快的配置慢几倍；有 spill 的配置不算在内。悬停可看时间。</figcaption>
</figure>

形状一变，最快的配置就变，Triton 用 autotune 自动挑：

```python {role="program" tag="matmul 的 kernel 不变，外面套一层 autotune"}
configs = [triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s)
           for bm, bn in ((64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128))
           for bk in (16, 32) for w in (4, 8) for s in (2, 3)]
matmul_relu_tuned = triton.autotune(configs=configs, key=["M", "N", "K"])(matmul_relu_kernel)


def matmul_relu_auto(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))  # grid 跟着选中的 BM、BN 变
    matmul_relu_tuned[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride())
    return c
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
<figcaption><strong>图 B-1</strong> TPU v5p 一个 TensorCore 的结构示意，画法对应<a href="#fig-h-1">图 1-2</a>。规格来自 <a href="https://jax-ml.github.io/scaling-book/tpus/">How to Scale Your Model</a>，原书也有一张 TensorCore 的结构图。</figcaption>
</figure>

MXU 和 GPU 最大的不同在乘加单元怎么拿到数据（[图 B-2](#fig-h-4)）。GPU 上每个 CUDA core 做一次乘加，要从 register file 读两个数、写回一个；一个数被用很多次，得靠 kernel 自己切 block，先搬进 shared memory 和 register 再反复读，[2.4 节](#triton-matmul)的 matmul 就是这样写的。systolic array 把复用做进了硬件：权重先装进 128 × 128 个格子，固定不动；输入 x 从左边进入，每拍向右传一格，经过的每一格把它乘上自己的权重，加到从上面传下来的部分和上，再把部分和往下传。一个 x 从 VMEM 读一次，在一行里被 128 个格子用到；稳定以后每拍从左边进 128 个数、从底部出 128 个结果，阵列里同时做 16384 次乘加，中间结果都不经过 register file。

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
<figcaption><strong>图 B-2</strong> 乘加单元怎么拿到数据。左边是 GPU 的 CUDA core，右边是 TPU 的 MXU；x 按行错开一拍进入，y 是每列的输出。动画演示见 <a href="https://jax-ml.github.io/scaling-book/tpus/#appendix-b-how-does-a-systolic-array-work">How to Scale Your Model 的 Appendix B</a>。</figcaption>
</figure>

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

<figure id="fig-b-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 440" width="100%" role="img" aria-label="两种 barrier 的甘特图。上下搬的是同样的 16 KB，放行时刻一样。上：asynchronous barrier 只数 thread 的 arrive，4 个 thread 各自搬完自己那份才 arrive，几乎没空做别的。下：asynchronous transaction barrier 还数字节，thread 0 发起 TMA 并登记 16 KB，4 个 thread 一开始就 arrive、去做别的活，TMA 分 4 段写进 shared memory，字节数到 16 KB 时放行">
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
  <text class="tb" x="8" y="28">asynchronous barrier（Ampere 起）：每个 thread 搬完自己那份再 arrive，只数 arrive</text>
  <text class="lab" x="8" y="54">thread 0</text>
  <rect x="92.0" y="43" width="160.8" height="14" rx="2" style="fill: var(--fig-2)" fill-opacity="0.55"/>
  <text class="s" x="172.4" y="54.0" text-anchor="middle">搬自己那份</text>
  <rect x="252.8" y="43" width="32.2" height="14" rx="2" style="fill: var(--fig-3)" fill-opacity="0.9"/>
  <rect x="285.0" y="43" width="21.4" height="14" rx="2" style="fill: none; stroke: var(--fig-hi)" stroke-dasharray="3 2"/>
  <rect x="306.4" y="43" width="117.9" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/>
  <text class="s" x="365.4" y="54.0" text-anchor="middle">用数据</text>
  <path d="M252.8 42 L257.8 50.0 L252.8 58 L247.8 50.0 Z" style="fill: var(--fig-1)"/>
  <text class="lab" x="8" y="74">thread 1</text>
  <rect x="92.0" y="63" width="214.4" height="14" rx="2" style="fill: var(--fig-2)" fill-opacity="0.55"/>
  <text class="s" x="199.2" y="74.0" text-anchor="middle">搬自己那份</text>
  <rect x="306.4" y="63" width="117.9" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/>
  <text class="s" x="365.4" y="74.0" text-anchor="middle">用数据</text>
  <path d="M306.4 62 L311.4 70.0 L306.4 78 L301.4 70.0 Z" style="fill: var(--fig-1)"/>
  <text class="lab" x="8" y="94">thread 2</text>
  <rect x="92.0" y="83" width="139.4" height="14" rx="2" style="fill: var(--fig-2)" fill-opacity="0.55"/>
  <text class="s" x="161.7" y="94.0" text-anchor="middle">搬自己那份</text>
  <rect x="231.4" y="83" width="42.9" height="14" rx="2" style="fill: var(--fig-3)" fill-opacity="0.9"/>
  <rect x="274.2" y="83" width="32.2" height="14" rx="2" style="fill: none; stroke: var(--fig-hi)" stroke-dasharray="3 2"/>
  <rect x="306.4" y="83" width="117.9" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/>
  <text class="s" x="365.4" y="94.0" text-anchor="middle">用数据</text>
  <path d="M231.4 82 L236.4 90.0 L231.4 98 L226.4 90.0 Z" style="fill: var(--fig-1)"/>
  <text class="lab" x="8" y="114">thread 3</text>
  <rect x="92.0" y="103" width="182.2" height="14" rx="2" style="fill: var(--fig-2)" fill-opacity="0.55"/>
  <text class="s" x="183.1" y="114.0" text-anchor="middle">搬自己那份</text>
  <rect x="274.2" y="103" width="21.4" height="14" rx="2" style="fill: var(--fig-3)" fill-opacity="0.9"/>
  <rect x="295.7" y="103" width="10.7" height="14" rx="2" style="fill: none; stroke: var(--fig-hi)" stroke-dasharray="3 2"/>
  <rect x="306.4" y="103" width="117.9" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/>
  <text class="s" x="365.4" y="114.0" text-anchor="middle">用数据</text>
  <path d="M274.2 102 L279.2 110.0 L274.2 118 L269.2 110.0 Z" style="fill: var(--fig-1)"/>
  <line x1="306.4" y1="38" x2="306.4" y2="130" stroke="currentColor" stroke-opacity=".6" stroke-dasharray="4 3"/>
  <text class="s" x="312.4" y="132">最后一个 thread arrive，放行</text>
  <text class="tb" x="8" y="168">asynchronous transaction barrier（Hopper 起）：同样 16 KB 交给 TMA 搬，还要数字节</text>
  <text class="lab" x="8" y="194">thread 0</text>
  <rect x="92.0" y="183" width="21.4" height="14" rx="2" style="fill: var(--fig-hi)" fill-opacity="0.7"/>
  <rect x="113.4" y="183" width="171.5" height="14" rx="2" style="fill: var(--fig-3)" fill-opacity="0.9"/>
  <text class="s" x="199.2" y="194.0" text-anchor="middle">别的活</text>
  <rect x="285.0" y="183" width="21.4" height="14" rx="2" style="fill: none; stroke: var(--fig-hi)" stroke-dasharray="3 2"/>
  <rect x="306.4" y="183" width="117.9" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/>
  <text class="s" x="365.4" y="194.0" text-anchor="middle">用数据</text>
  <path d="M113.4 182 L118.4 190.0 L113.4 198 L108.4 190.0 Z" style="fill: var(--fig-1)"/>
  <text class="lab" x="8" y="214">thread 1</text>
  <rect x="97.4" y="203" width="166.2" height="14" rx="2" style="fill: var(--fig-3)" fill-opacity="0.9"/>
  <text class="s" x="180.4" y="214.0" text-anchor="middle">别的活</text>
  <rect x="263.5" y="203" width="42.9" height="14" rx="2" style="fill: none; stroke: var(--fig-hi)" stroke-dasharray="3 2"/>
  <rect x="306.4" y="203" width="117.9" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/>
  <text class="s" x="365.4" y="214.0" text-anchor="middle">用数据</text>
  <path d="M97.4 202 L102.4 210.0 L97.4 218 L92.4 210.0 Z" style="fill: var(--fig-1)"/>
  <text class="lab" x="8" y="234">thread 2</text>
  <rect x="97.4" y="223" width="209.0" height="14" rx="2" style="fill: var(--fig-3)" fill-opacity="0.9"/>
  <text class="s" x="201.9" y="234.0" text-anchor="middle">别的活</text>
  <rect x="306.4" y="223" width="117.9" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/>
  <text class="s" x="365.4" y="234.0" text-anchor="middle">用数据</text>
  <path d="M97.4 222 L102.4 230.0 L97.4 238 L92.4 230.0 Z" style="fill: var(--fig-1)"/>
  <text class="lab" x="8" y="254">thread 3</text>
  <rect x="97.4" y="243" width="155.4" height="14" rx="2" style="fill: var(--fig-3)" fill-opacity="0.9"/>
  <text class="s" x="175.1" y="254.0" text-anchor="middle">别的活</text>
  <rect x="252.8" y="243" width="53.6" height="14" rx="2" style="fill: none; stroke: var(--fig-hi)" stroke-dasharray="3 2"/>
  <rect x="306.4" y="243" width="117.9" height="14" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/>
  <text class="s" x="365.4" y="254.0" text-anchor="middle">用数据</text>
  <path d="M97.4 242 L102.4 250.0 L97.4 258 L92.4 250.0 Z" style="fill: var(--fig-1)"/>
  <text class="lab" x="8" y="274">TMA</text><text class="s" x="312.4" y="274">写进 shared memory，共 16 KB，分 4 段</text>
  <rect x="113.4" y="263" width="48.2" height="14" rx="2" style="fill: var(--fig-hi)" fill-opacity="0.35"/>
  <rect x="161.7" y="263" width="48.2" height="14" rx="2" style="fill: var(--fig-hi)" fill-opacity="0.35"/>
  <rect x="209.9" y="263" width="48.2" height="14" rx="2" style="fill: var(--fig-hi)" fill-opacity="0.35"/>
  <rect x="258.2" y="263" width="48.2" height="14" rx="2" style="fill: var(--fig-hi)" fill-opacity="0.35"/>
  <text class="lab" x="8" y="294">字节计数</text>
  <text class="s" x="158.7" y="294" text-anchor="end">4 KB</text>
  <text class="s" x="206.9" y="294" text-anchor="end">8 KB</text>
  <text class="s" x="255.2" y="294" text-anchor="end">12 KB</text>
  <text class="s" x="303.4" y="294" text-anchor="end">16 KB</text>
  <line x1="306.4" y1="178" x2="306.4" y2="310" stroke="currentColor" stroke-opacity=".6" stroke-dasharray="4 3"/>
  <text class="s" x="312.4" y="312">thread 都 arrive 过、字节到齐，放行</text>
  <rect x="8" y="329" width="18" height="11" rx="2" style="fill: var(--fig-2)" fill-opacity="0.55"/><text class="s" x="32" y="338">thread 自己搬数据</text>
  <rect x="147.1" y="329" width="18" height="11" rx="2" style="fill: var(--fig-3)" fill-opacity="0.9"/><text class="s" x="171.1" y="338">不依赖这批数据的活</text>
  <rect x="287.6" y="329" width="18" height="11" rx="2" style="fill: var(--fig-1)" fill-opacity="0.6"/><text class="s" x="311.6" y="338">用数据</text>
  <rect x="8" y="349" width="18" height="11" rx="2" style="fill: none; stroke: var(--fig-hi)" stroke-dasharray="3 2"/><text class="s" x="32" y="358">等（A100 空转，H100 可以睡眠）</text>
  <rect x="217.0" y="349" width="18" height="11" rx="2" style="fill: var(--fig-hi)" fill-opacity="0.7"/><text class="s" x="241.0" y="358">发 TMA、登记 16 KB</text>
  <path d="M13 369 L18 375 L13 381 L8 375 Z" style="fill: var(--fig-1)"/><text class="s" x="32" y="378">arrive：告诉 barrier 自己这份做完了，不阻塞</text>
</svg>
<figcaption><strong>图 C-1</strong> 两种 barrier 怎么等数据（示意，横轴是时间，长度不按比例；画法参考 <a href="https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf">H100 白皮书</a> Figure 20）。上下搬的是同样的 16 KB，显存带宽一样，所以放行得一样早；区别在 thread 的时间花在哪。上：asynchronous barrier 只数 thread 的 arrive，每个 thread 要自己搬完自己那份才 arrive，几乎没有空做别的。下：asynchronous transaction barrier 还数字节，发起 TMA 的 thread 先登记这次要等 16 KB，TMA 每写进 shared memory 一段就把字节数加到 barrier 上；thread 一开始就 arrive，搬运期间都在做别的活，thread 都 arrive 过、字节数也到齐才放行。白皮书说 A100 上等着的 thread 在 barrier 上空转，H100 上可以睡眠。</figcaption>
</figure>

### C.3 有了 TMA 以后怎么分工 {#tma-ws}

[H100 白皮书](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)说，有了 TMA，只要很少几个 thread 就能用满显存带宽，其余的 thread 可以专心计算。常见的写法叫 warp specialization：把一个 program 的 warp 分成两种，producer warp 只发 TMA，把后面几轮的 block 搬进 slot；consumer warp 只用 Tensor core 算。两边按 slot 交接：每个 slot 配一个 transaction barrier 表示「满了」，TMA 搬完时它放行，consumer 等它再开始算；再配一个只数 arrive 的 barrier 表示「空了」，consumer 算完时 arrive，producer 等它再往这个 slot 搬下一批。这和 [3.3 节](#stages)的流水线是同一件事，只是搬和算交给了不同的 warp，不再是同一个 warp 里先发 load 再算。FlashAttention-3 在 Hopper 上就是这么写的，见 [FlashAttention 1–4](/blog/flashattention-1-to-4/)。

白皮书里还有 thread block cluster：一组保证同时调度、分布在几个 SM 上的 block。transaction barrier 可以在 cluster 里跨 SM 同步，TMA 也能在 cluster 里的 block 之间直接搬数据。这篇用不到，不展开。

---

## References {#references}

1. Gholami, Yao, Kim, Hooper, Mahoney, Keutzer. [AI and Memory Wall](https://arxiv.org/abs/2403.14123). IEEE Micro, 2024. 图 1-1。
2. NVIDIA. [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)（白皮书，[备份](https://web.archive.org/web/20261008150304/https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)）。RTX 5090 的存储层级和 SM 结构，图 1-2、图 1-4。
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
16. NVIDIA. [PTX ISA: cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) 和 [cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)。Appendix C 里 cp.async、wait_group 和 TMA、mbarrier 指令的语义。
