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

文章分四步走：[第 1 节](#basics)准备工具 roofline；[第 2 节](#step)算一步训练的总账，找出时间和显存的大头；[第 3 节](#rmsnorm)拿最简单的 RMSNorm 逐个 op 拆开，看每一步算了多少、读写了多少显存、为反向存了什么；[第 4 节](#attention)用同样的方法拆 attention；[第 5 节](#savings)试 bf16 和 activation checkpoint 能省多少。怎么把 S、P 彻底去掉，留给下一篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/)。

模型是我自己写的 Transformer LM（RMSNorm、RoPE、SwiGLU，pre-norm），一共五档：small 0.13B、medium 0.42B、large 0.97B、xl 3.41B，以及 10B（实际 12.83B 参数）。

> 环境：RTX 5090 32 GB，torch 2.11.0+cu130。fp32 基准关掉 tf32（`allow_tf32=False`）。除注明外 batch 4、seq 512，预热 5 步、计时 10 步。「full step」指前向、反向加 optimizer 的一整步，「前向 + 反向」不含 optimizer。显存一律 GiB = 2³⁰ B（`max_memory_allocated() / 1024³`）。

---

## 1 一把尺子：roofline {#basics}

一个 op 跑多快，取决于它要算多少和要搬多少。

算的量用 FLOPs（浮点运算次数）数。矩阵乘 `[M, K] × [K, N]` 要 2·M·N·K 次（M·N 个输出，每个做 K 次乘加）；逐元素 op（加、乘、exp、mask）每个元素只算一到几十次，比同样大小的矩阵乘少几个数量级。GPU 每秒最多能做的浮点运算次数叫峰值算力，记作 $\pi$，单位 FLOPS。5090 的 fp32 峰值是 $\pi$ = 1.05e14 FLOPS，用 bf16 Tensor core 时翻倍到 2.1e14。

搬的量是 op 读写显存的字节数：输入要从显存读进来，结果要写回去。显存每秒最多能读写的字节数叫带宽，记作 $\beta$，5090 是 $\beta$ = 1.79e12 B/s。一个 op 的耗时不会低于算的时间和搬的时间中较大的那个：

<p align="center">$t \ge \max\left(\dfrac{\mathrm{FLOPs}}{\pi},\ \dfrac{\mathrm{bytes}}{\beta}\right)$</p>

算和搬的比值叫**算术强度** $I = \mathrm{FLOPs} / \mathrm{bytes}$，即每搬 1 字节做多少次运算。把上式改写成「最多能跑到多少 FLOPS」，就是 **roofline**：

<p align="center">$\mathrm{FLOPS}_{\max}(I) = \min(\pi,\ I \cdot \beta)$</p>

在 log-log 坐标上它像一个屋顶（[图 1-1](#fig-1-1)）：左边是斜坡，被带宽限制；右边是平顶，被算力限制；拐角 $I^* = \pi / \beta$ 叫 **ridge point**，5090 fp32 是 58 FLOPs/B。落在拐角左边的 op 是 **memory-bound**，耗时由搬数据决定，减少 FLOPs 没有用；落在右边的是 **compute-bound**，耗时由算力决定。衡量 op 跑得好不好也分两种：compute-bound 的看 MFU（实际 FLOPS / $\pi$），memory-bound 的看 MBU（实际带宽 / $\beta$）。

<figure id="fig-1-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 392" width="100%" role="img" aria-label="RTX 5090 fp32 的 roofline：带宽斜线和峰值平线交于 58 FLOPs/B；attention 的 op 都在斜线上，只有 Linear 在平线下">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <text class="tick" x="70.0" y="357" text-anchor="middle">0.1</text>
  <line class="grid" x1="180.0" y1="20" x2="180.0" y2="340"/>
  <text class="tick" x="180.0" y="357" text-anchor="middle">1</text>
  <line class="grid" x1="290.0" y1="20" x2="290.0" y2="340"/>
  <text class="tick" x="290.0" y="357" text-anchor="middle">10</text>
  <line class="grid" x1="400.0" y1="20" x2="400.0" y2="340"/>
  <text class="tick" x="400.0" y="357" text-anchor="middle">100</text>
  <line class="grid" x1="510.0" y1="20" x2="510.0" y2="340"/>
  <text class="tick" x="510.0" y="357" text-anchor="middle">1000</text>
  <line class="grid" x1="620.0" y1="20" x2="620.0" y2="340"/>
  <text class="tick" x="620.0" y="357" text-anchor="middle">10000</text>
  <text class="tick" x="62" y="344.0" text-anchor="end">1e11</text>
  <line class="grid" x1="70" y1="251.1" x2="620" y2="251.1"/>
  <text class="tick" x="62" y="255.1" text-anchor="end">1e12</text>
  <line class="grid" x1="70" y1="162.2" x2="620" y2="162.2"/>
  <text class="tick" x="62" y="166.2" text-anchor="end">1e13</text>
  <line class="grid" x1="70" y1="73.3" x2="620" y2="73.3"/>
  <text class="tick" x="62" y="77.3" text-anchor="end">1e14</text>
  <line class="axis" x1="70" y1="340" x2="620" y2="340"/><line class="axis" x1="70" y1="20" x2="70" y2="340"/>
  <text class="lab2" x="345" y="380" text-anchor="middle">算术强度 I（FLOPs/B，对数轴）</text>
  <text class="lab2" transform="translate(16 180) rotate(-90)" text-anchor="middle">可达算力（FLOPS，对数轴）</text>
  <line x1="374.4" y1="71.5" x2="374.4" y2="340" stroke="currentColor" stroke-opacity=".25"/>
  <line x1="374.4" y1="71.5" x2="620" y2="71.5" style="stroke: var(--fig-1)" stroke-width="2.2"/>
  <g class="m"><title>fp32：ridge point = 1.05e14 / 1.792e12 = 58 FLOPs/B</title><circle cx="374.4" cy="71.5" r="3" style="fill: var(--fig-1)"/></g>
  <text class="lab" x="374.4" y="61.5" text-anchor="middle">ridge point I* = 58</text>
  <text class="lab" x="618" y="85.5" text-anchor="end">峰值 π = 1.05e14 FLOPS</text>
  <line x1="70" y1="317.5" x2="374.4" y2="71.5" style="stroke: var(--fig-1)" stroke-width="2.2"/>
  <text class="lab" transform="translate(230 180.2) rotate(-38.9)" text-anchor="middle">带宽 β = 1.79e12 B/s</text>
  <text class="lab2" x="279" y="326" text-anchor="middle">memory-bound</text>
  <text class="lab2" x="529" y="326" text-anchor="middle">compute-bound</text>
  <g class="m"><title>Linear（FFN w1）：I = 341 FLOPs/B，实测 6.87e13 FLOPS，MFU 64%</title><circle cx="458.6" cy="87.8" r="12" fill="transparent"/><circle cx="458.6" cy="87.8" r="5" class="ring" style="fill: var(--fig-hi)"/></g>
  <text class="lab" x="448.6" y="105.8" text-anchor="end">Linear</text>
  <g class="m"><title>S = QKᵀ：I = 28.4 FLOPs/B，实测 2.86e13 FLOPS，MBU 57%</title><circle cx="339.9" cy="121.6" r="12" fill="transparent"/><circle cx="339.9" cy="121.6" r="5" class="ring" style="fill: var(--fig-hi)"/></g>
  <text class="lab" x="349.9" y="135.6" text-anchor="start">QKᵀ</text>
  <g class="m"><title>O = PV：I = 30.1 FLOPs/B，实测 3.73e13 FLOPS，MBU 70%</title><circle cx="342.7" cy="111.4" r="12" fill="transparent"/><circle cx="342.7" cy="111.4" r="5" class="ring" style="fill: var(--fig-hi)"/></g>
  <text class="lab" x="352.7" y="109.4" text-anchor="start">PV</text>
  <g class="m"><title>softmax（5 个 kernel）：I = 0.844 FLOPs/B，实测 1.27e12 FLOPS，MBU 84%</title><circle cx="171.9" cy="242.0" r="12" fill="transparent"/><circle cx="171.9" cy="242.0" r="5" class="ring" style="fill: var(--fig-hi)"/></g>
  <text class="lab" x="181.9" y="258.0" text-anchor="start">softmax</text>
  <g class="m"><title>S / √d：I = 0.125 FLOPs/B，实测 1.92e11 FLOPS，MBU 86%</title><circle cx="80.7" cy="314.9" r="12" fill="transparent"/><circle cx="80.7" cy="314.9" r="5" class="ring" style="fill: var(--fig-hi)"/></g>
  <text class="lab" x="90.7" y="326.9" text-anchor="start">S / √d</text>
</svg>
<figcaption><strong>图 1-1</strong> RTX 5090 fp32 的 roofline，以及 medium、seq 1024 时一层 attention 里实测的 op（另放一个 Linear 作对照）。causal mask 没有 FLOPs，不在图上；悬停可看数值。</figcaption>
</figure>

一个 op 在拐角哪一边，用张量形状就能估。逐元素 op 在 fp32 下每个元素算 1 次、读写 8 字节，$I \approx 0.13$，远在拐角左边。矩阵乘的 $I$ 由 M、N、K 里最小的那个决定，fp32 下不超过它的一半。Linear 的三个维度都上千（medium、seq 1024 时 FFN 第一层是 `[4096, 1024] × [1024, 4096]`），$I \approx 340$，在拐角右边；attention 里的 QKᵀ 和 PV 都有一个维度是 d_head（64），$I$ 只有 28，在拐角左边。所以 QKᵀ 和 PV 虽然是矩阵乘，在 attention 里也是 memory-bound。图 1-1 里的点是实测，[第 4 节](#attention)再细看。

---

## 2 一步训练的总账 {#step}

### 2.1 时间：矩阵乘没在偷懒 {#time}

Transformer 的 FLOPs 几乎都在 Linear 上。一个 Linear 前向只做一次矩阵乘 $X_L = X_{L-1} W_L$；反向收到误差 $\nabla X_L$ 后要做两次：算参数梯度 $\nabla W_L = X_{L-1}^{\top} \nabla X_L$，再算传给上一层的 $\nabla X_{L-1} = \nabla X_L W_L^{\top}$，每次都和前向一样大（[图 2-1](#fig-2-1)）。

<figure id="fig-2-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 444" width="100%" role="img" aria-label="一个 Linear 的前向与反向：前向从输入 X_{L-1} 算出 X_L，进入深层；反向收到误差后，用留下来的 X_{L-1} 算参数梯度，用 W_L 算激活梯度，再传给浅层">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <defs><marker id="fig-2-1-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-2-1-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-2)"/></marker><marker id="fig-2-1-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-1)"/></marker></defs>
  <text class="tb" x="176" y="24" text-anchor="middle">Forward Pass（前向）</text>
  <text class="tb" x="482" y="24" text-anchor="middle">Backward Pass（反向）</text>
  <line x1="320" y1="36" x2="320" y2="364" stroke="currentColor" stroke-opacity=".25" stroke-dasharray="5 4"/>
  <rect x="101" y="51" width="150" height="50" rx="8" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="176" y="67" text-anchor="middle">输入</text>
  <text class="t" x="176" y="89" text-anchor="middle"><tspan font-style="italic">X</tspan><tspan dy="4" font-size="10">L−1</tspan><tspan dy="-4"> </tspan></text>
  <rect x="115" y="155" width="150" height="54" rx="8" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="190" y="173" text-anchor="middle">计算</text>
  <text class="t" x="190" y="195" text-anchor="middle"><tspan font-style="italic">X</tspan><tspan dy="4" font-size="10">L</tspan><tspan dy="-4"> </tspan>= <tspan font-style="italic">X</tspan><tspan dy="4" font-size="10">L−1</tspan><tspan dy="-4"> </tspan>· <tspan font-style="italic">W</tspan><tspan dy="4" font-size="10">L</tspan><tspan dy="-4"> </tspan></text>
  <rect x="10" y="157" width="68" height="50" rx="8" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text class="s" x="44" y="173" text-anchor="middle">权重</text>
  <text class="t" x="44" y="195" text-anchor="middle"><tspan font-style="italic">W</tspan><tspan dy="4" font-size="10">L</tspan><tspan dy="-4"> </tspan></text>
  <rect x="101" y="263" width="150" height="50" rx="8" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="s" x="176" y="279" text-anchor="middle">输出</text>
  <text class="t" x="176" y="301" text-anchor="middle"><tspan font-style="italic">X</tspan><tspan dy="4" font-size="10">L</tspan><tspan dy="-4"> </tspan></text>
  <line x1="176.0" y1="101.0" x2="176.0" y2="152.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-2-1-m0)"/>
  <line x1="78.0" y1="182.0" x2="112.0" y2="182.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-2-1-m0)"/>
  <line x1="176.0" y1="209.0" x2="176.0" y2="260.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-2-1-m0)"/>
  <rect x="210" y="376" width="220" height="44" rx="8" class="op"/>
  <text class="t" x="320" y="402" text-anchor="middle">进入深层 Layer L+1，等误差传回</text>
  <path d="M176,313 L176,398 L207,398" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-2-1-m0)"/>
  <path d="M430,398 L482,398 L482,316" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-2-1-m0)"/>
  <rect x="407" y="263" width="150" height="50" rx="8" style="fill: var(--fig-3); stroke: var(--fig-2)" stroke-width="1"/>
  <text class="s" x="482" y="279" text-anchor="middle">接收误差</text>
  <text class="t" x="482" y="301" text-anchor="middle">∇<tspan font-style="italic">X</tspan><tspan dy="4" font-size="10">L</tspan><tspan dy="-4"> </tspan></text>
  <rect x="328" y="155" width="152" height="54" rx="8" style="fill: color-mix(in srgb, var(--fig-hi) 14%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text class="s" x="404" y="173" text-anchor="middle">计算参数梯度</text>
  <text class="t" x="404" y="195" text-anchor="middle">∇<tspan font-style="italic">W</tspan><tspan dy="4" font-size="10">L</tspan><tspan dy="-4"> </tspan>= <tspan font-style="italic">X</tspan><tspan dy="4" font-size="10">L−1</tspan><tspan dy="-10" font-size="10">T</tspan><tspan dy="6"> </tspan>· ∇<tspan font-style="italic">X</tspan><tspan dy="4" font-size="10">L</tspan><tspan dy="-4"> </tspan></text>
  <rect x="486" y="155" width="152" height="54" rx="8" style="fill: var(--fig-3); stroke: var(--fig-2)" stroke-width="1"/>
  <text class="s" x="562" y="173" text-anchor="middle">计算激活梯度</text>
  <text class="t" x="562" y="195" text-anchor="middle">∇<tspan font-style="italic">X</tspan><tspan dy="4" font-size="10">L−1</tspan><tspan dy="-4"> </tspan>= ∇<tspan font-style="italic">X</tspan><tspan dy="4" font-size="10">L</tspan><tspan dy="-4"> </tspan>· <tspan font-style="italic">W</tspan><tspan dy="4" font-size="10">L</tspan><tspan dy="-10" font-size="10">T</tspan><tspan dy="6"> </tspan></text>
  <rect x="486" y="51" width="152" height="50" rx="8" style="fill: var(--fig-3); stroke: var(--fig-2)" stroke-width="1"/>
  <text class="t" x="562" y="80" text-anchor="middle">进入浅层 Layer L−1</text>
  <line x1="462.0" y1="263.0" x2="420.0" y2="212.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-2-1-m0)"/>
  <line x1="502.0" y1="263.0" x2="540.0" y2="212.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-2-1-m0)"/>
  <line x1="562.0" y1="155.0" x2="562.0" y2="104.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-2-1-m0)"/>
  <path d="M251,76 L404,76 L404,151" fill="none" style="stroke: var(--fig-2)" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#fig-2-1-m1)"/>
  <text class="s halo" x="262" y="66"><tspan font-style="italic">X</tspan><tspan dy="4" font-size="10">L−1</tspan><tspan dy="-4"> </tspan> 留到反向（第 3 节的 A）</text>
  <text class="s" x="262" y="66"><tspan font-style="italic">X</tspan><tspan dy="4" font-size="10">L−1</tspan><tspan dy="-4"> </tspan> 留到反向（第 3 节的 A）</text>
  <path d="M44,207 L44,236 L618,236 L618,212" fill="none" style="stroke: var(--fig-1)" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#fig-2-1-m2)"/>
  <text class="s halo" x="320" y="253" text-anchor="middle"><tspan font-style="italic">W</tspan><tspan dy="4" font-size="10">L</tspan><tspan dy="-4"> </tspan> 是参数，本来就在显存里</text>
  <text class="s" x="320" y="253" text-anchor="middle"><tspan font-style="italic">W</tspan><tspan dy="4" font-size="10">L</tspan><tspan dy="-4"> </tspan> 是参数，本来就在显存里</text>
</svg>
<figcaption><strong>图 2-1</strong> 一个 Linear 的前向与反向：前向从左边往下，误差从右边传回；虚线是反向要从前向拿的东西。</figcaption>
</figure>

每个参数对每个 token 做一次乘加，也就是 2 FLOPs，所以前向每个 token 约 2N FLOPs（N 是参数量），反向 4N，一步一共 6N × token 数，这里是 batch 4 × seq 512 = 2048 个 token。attention 的 QKᵀ 和 PV 没有参数，不在 6N 里，seq 512 时只占 2–4%，可以忽略。实测也是这样，反向耗时差不多是前向的两倍（[图 2-2](#fig-2-2)），因为反向要把 weight grad 和 activation grad 各算一遍。

<figure id="fig-2-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 172" width="100%" role="img" aria-label="三档模型一步训练的耗时构成：前向约 31%，反向约 62%，optimizer 约 7%；右侧是每步总耗时和 MFU">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <rect x="112" y="11" width="10" height="10" rx="2" style="fill: var(--fig-1)"/>
  <text class="lab" x="128" y="20">前向</text>
  <rect x="174" y="11" width="10" height="10" rx="2" style="fill: var(--fig-2)"/>
  <text class="lab" x="190" y="20">反向</text>
  <rect x="236" y="11" width="10" height="10" rx="2" style="fill: var(--fig-3); stroke: var(--fig-2)"/>
  <text class="lab" x="252" y="20">optimizer</text>
  <text class="lab" x="102" y="53" text-anchor="end">small</text>
  <text class="lab2" x="102" y="67" text-anchor="end">0.13B</text>
  <g class="m"><title>small 前向：17.2 ms（31%）</title><rect x="112.0" y="44.0" width="106.7" height="26.0" rx="3" style="fill: var(--fig-1)"/></g>
  <text x="165.3" y="61" text-anchor="middle" font-size="11" style="fill: var(--fig-on-1)">31%</text>
  <g class="m"><title>small 反向：34.8 ms（62%）</title><rect x="220.7" y="44.0" width="217.9" height="26.0" rx="3" style="fill: var(--fig-2)"/></g>
  <text x="329.7" y="61" text-anchor="middle" font-size="11" style="fill: var(--fig-on-2)">62%</text>
  <g class="m"><title>small optimizer：3.7 ms（7%）</title><rect x="440.6" y="44.0" width="21.4" height="26.0" rx="3" style="fill: var(--fig-3); stroke: var(--fig-2)"/></g>
  <text class="val" x="474" y="56">55.7 ms / 步</text>
  <text class="lab2" x="474" y="70">MFU 27%</text>
  <text class="lab" x="102" y="95" text-anchor="end">medium</text>
  <text class="lab2" x="102" y="109" text-anchor="end">0.42B</text>
  <g class="m"><title>medium 前向：51.1 ms（31%）</title><rect x="112.0" y="86.0" width="105.5" height="26.0" rx="3" style="fill: var(--fig-1)"/></g>
  <text x="164.8" y="103" text-anchor="middle" font-size="11" style="fill: var(--fig-on-1)">31%</text>
  <g class="m"><title>medium 反向：103.3 ms（62%）</title><rect x="219.5" y="86.0" width="215.3" height="26.0" rx="3" style="fill: var(--fig-2)"/></g>
  <text x="327.2" y="103" text-anchor="middle" font-size="11" style="fill: var(--fig-on-2)">62%</text>
  <g class="m"><title>medium optimizer：12.9 ms（8%）</title><rect x="436.9" y="86.0" width="25.1" height="26.0" rx="3" style="fill: var(--fig-3); stroke: var(--fig-2)"/></g>
  <text class="val" x="474" y="98">167.4 ms / 步</text>
  <text class="lab2" x="474" y="112">MFU 31%</text>
  <text class="lab" x="102" y="137" text-anchor="end">large</text>
  <text class="lab2" x="102" y="151" text-anchor="end">0.97B</text>
  <g class="m"><title>large 前向：118.1 ms（32%）</title><rect x="112.0" y="128.0" width="109.5" height="26.0" rx="3" style="fill: var(--fig-1)"/></g>
  <text x="166.8" y="145" text-anchor="middle" font-size="11" style="fill: var(--fig-on-1)">32%</text>
  <g class="m"><title>large 反向：227.8 ms（61%）</title><rect x="223.5" y="128.0" width="213.1" height="26.0" rx="3" style="fill: var(--fig-2)"/></g>
  <text x="330.1" y="145" text-anchor="middle" font-size="11" style="fill: var(--fig-on-2)">61%</text>
  <g class="m"><title>large optimizer：26.9 ms（7%）</title><rect x="438.6" y="128.0" width="23.4" height="26.0" rx="3" style="fill: var(--fig-3); stroke: var(--fig-2)"/></g>
  <text class="val" x="474" y="140">372.8 ms / 步</text>
  <text class="lab2" x="474" y="154">MFU 31%</text>
</svg>
<figcaption><strong>图 2-2</strong> 一步训练里前向、反向、optimizer 的耗时占比（fp32，batch 4，seq 512），右侧是每步耗时和 MFU。</figcaption>
</figure>

按 6N 算，medium 和 large 的 MFU 只有 31%（实际 3.2e13 FLOPS，fp32 峰值 1.05e14），small 是 27%。矩阵乘本身不慢，单个大矩阵乘能跑到峰值的 64%；问题是所有矩阵乘加起来只占一步 GPU 时间的 60%，剩下 40% 花在几乎不做计算的逐元素 kernel 上，比如 norm、softmax、mask、激活函数。

### 2.2 显存：权重只是小头 {#memory}

fp32 + AdamW 训练时，每个参数要占 16 B：权重 4 B、梯度 4 B、Adam 的 m 和 v 各 4 B，此外还有前向为反向留下的 activation。xl 光这部分就要 50.8 GiB，5090 放不下；10B 在建模型时就 OOM 了。下面用这几个记号：

- <span class="sw" style="background: var(--fig-1)"></span>**W** 全部权重；<span class="sw" style="background: var(--fig-hi)"></span>**G** 全部梯度 `.grad`，大小等于 W；<span class="sw" style="background: var(--fig-mute)"></span>Adam 的 m、v 合计 2W，第一步之后常驻；
- <span class="sw" style="background: var(--fig-2)"></span>**A** 前向为反向存下的张量（saved tensors）；<span class="sw" style="background: var(--fig-3)"></span>**T** 当前层的临时量，算完即释放。

实测 full step 的峰值里只有权重、Adam 状态和 A，没有梯度（[图 2-3](#fig-2-3)）。

<figure id="fig-2-3" class="fg-fig">
<svg class="fg" viewBox="0 0 640 300" width="100%" role="img" aria-label="三档模型 full step 的实测峰值显存，由权重、Adam 状态和 activation 组成，里面没有梯度">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <rect x="70" y="11" width="10" height="10" rx="2" style="fill: var(--fig-1)"/>
  <text class="lab" x="86" y="20">W 权重</text>
  <rect x="139.2" y="11" width="10" height="10" rx="2" style="fill: var(--fig-mute)"/>
  <text class="lab" x="155.2" y="20">Adam m、v</text>
  <rect x="229.39999999999998" y="11" width="10" height="10" rx="2" style="fill: var(--fig-2)"/>
  <text class="lab" x="245.39999999999998" y="20">A activation</text>
  <rect x="340.59999999999997" y="11" width="10" height="10" rx="2" style="fill: var(--fig-3); stroke: var(--fig-2)"/>
  <text class="lab" x="356.59999999999997" y="20">T 临时量</text>
  <line class="grid" x1="70" y1="262.0" x2="630" y2="262.0"/><text class="tick" x="62" y="266.0" text-anchor="end">0</text>
  <line class="grid" x1="70" y1="196.0" x2="630" y2="196.0"/><text class="tick" x="62" y="200.0" text-anchor="end">10</text>
  <line class="grid" x1="70" y1="130.0" x2="630" y2="130.0"/><text class="tick" x="62" y="134.0" text-anchor="end">20</text>
  <line class="grid" x1="70" y1="64.0" x2="630" y2="64.0"/><text class="tick" x="62" y="68.0" text-anchor="end">30</text>
  <text class="lab2" x="70" y="40">GiB</text>
  <g class="m"><title>small：W 0.48 GiB</title><rect x="135.0" y="259.8" width="70.0" height="1.2" rx="2" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>small：Adam 0.96 GiB</title><rect x="135.0" y="253.5" width="70.0" height="4.3" rx="2" style="fill: var(--fig-mute)"/></g>
  <g class="m"><title>small：A 3.50 GiB</title><rect x="135.0" y="230.4" width="70.0" height="21.1" rx="2" style="fill: var(--fig-2)"/></g>
  <text x="170" y="244.9" text-anchor="middle" font-size="11" style="fill: var(--fig-on-2)">A 3.5</text>
  <g class="m"><title>small：T 0.10 GiB</title><rect x="135.0" y="229.7" width="70.0" height="0.7" rx="2" style="fill: var(--fig-3); stroke: var(--fig-2)"/></g>
  <text class="val" x="170" y="222.7" text-anchor="middle">5.04 GiB</text>
  <text class="lab" x="170" y="280" text-anchor="middle">small 0.13B</text>
  <g class="m"><title>medium：W 1.58 GiB</title><rect x="305.0" y="252.6" width="70.0" height="8.4" rx="2" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>medium：Adam 3.16 GiB</title><rect x="305.0" y="231.7" width="70.0" height="18.9" rx="2" style="fill: var(--fig-mute)"/></g>
  <text x="340" y="245.1" text-anchor="middle" font-size="11" style="fill: var(--fig-on-mute)">Adam 3.2</text>
  <g class="m"><title>medium：A 8.91 GiB</title><rect x="305.0" y="172.9" width="70.0" height="56.8" rx="2" style="fill: var(--fig-2)"/></g>
  <text x="340" y="205.3" text-anchor="middle" font-size="11" style="fill: var(--fig-on-2)">A 8.9</text>
  <g class="m"><title>medium：T 0.09 GiB</title><rect x="305.0" y="172.3" width="70.0" height="0.6" rx="2" style="fill: var(--fig-3); stroke: var(--fig-2)"/></g>
  <text class="val" x="340" y="165.3" text-anchor="middle">13.74 GiB</text>
  <text class="lab" x="340" y="280" text-anchor="middle">medium 0.42B</text>
  <g class="m"><title>large：W 3.61 GiB</title><rect x="475.0" y="239.2" width="70.0" height="21.8" rx="2" style="fill: var(--fig-1)"/></g>
  <text x="510" y="254.1" text-anchor="middle" font-size="11" style="fill: var(--fig-on-1)">W 3.6</text>
  <g class="m"><title>large：Adam 7.22 GiB</title><rect x="475.0" y="191.5" width="70.0" height="45.7" rx="2" style="fill: var(--fig-mute)"/></g>
  <text x="510" y="218.3" text-anchor="middle" font-size="11" style="fill: var(--fig-on-mute)">Adam 7.2</text>
  <g class="m"><title>large：A 16.58 GiB</title><rect x="475.0" y="82.1" width="70.0" height="107.4" rx="2" style="fill: var(--fig-2)"/></g>
  <text x="510" y="139.8" text-anchor="middle" font-size="11" style="fill: var(--fig-on-2)">A 16.6</text>
  <g class="m"><title>large：T 0.10 GiB</title><rect x="475.0" y="81.4" width="70.0" height="0.7" rx="2" style="fill: var(--fig-3); stroke: var(--fig-2)"/></g>
  <text class="val" x="510" y="74.4" text-anchor="middle">27.51 GiB</text>
  <text class="lab" x="510" y="280" text-anchor="middle">large 0.97B</text>
  <line class="axis" x1="70" y1="262" x2="630" y2="262"/>
</svg>
<figcaption><strong>图 2-3</strong> full step 的实测峰值显存（batch 4，seq 512）。A = 带梯度的前向峰值 − W，顶上 ~0.1 GiB 是 T。</figcaption>
</figure>

梯度不在峰值里，是因为 G 和 A 此消彼长。反向走完 j 层（共 L 层）时，显存里是：

<p align="center">$M(j) = W + G \cdot \dfrac{j}{L} + A \cdot \dfrac{L-j}{L} + T$</p>

A 在前向一层层攒起来，反向再一层层释放；G 正好相反，前向时还不存在，`.grad` 要等反向算到那个参数才分配，optimizer step 结束后又被 `zero_grad(set_to_none=True)` 释放。一个涨一个降，M(j) 是一条直线，最高点只可能在两端。full step 里还有一直占着的 Adam 状态，m 和 v 各和权重一样大，共 2W，和权重加起来常驻 3W。所以 full step 的峰值是

<p align="center">$\mathrm{peak}_{\mathrm{full}} \approx 3W + \max(A,\ G)$</p>

拿 large 验算：3 × 3.61 + 16.58 = 27.4 GiB，实测 27.51 GiB，差的 0.1 GiB 就是 T。正常训练 token 多，A 比 G 大，峰值出现在前向刚结束时；token 很少时才反过来，比如 xl 在 seq 128（[图 2-4](#fig-2-4) 左）。

<figure id="fig-2-4" class="fg-fig">
<svg class="fg" viewBox="0 0 640 300" width="100%" role="img" aria-label="一步前向 + 反向的显存：按 M(j) 用实测的 W、A、G、T 堆叠；xl seq 128 时 G 大于 A，峰值在反向结束；small seq 512 时 A 大于 G，峰值在前向结束">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <rect x="64" y="9" width="10" height="10" rx="2" style="fill: var(--fig-1)"/>
  <text class="lab" x="80" y="18">W</text>
  <rect x="100.6" y="9" width="10" height="10" rx="2" style="fill: var(--fig-2)"/>
  <text class="lab" x="116.6" y="18">A</text>
  <rect x="137.2" y="9" width="10" height="10" rx="2" style="fill: var(--fig-hi)"/>
  <text class="lab" x="153.2" y="18">G</text>
  <rect x="173.79999999999998" y="9" width="10" height="10" rx="2" style="fill: var(--fig-3); stroke: var(--fig-2)"/>
  <text class="lab" x="189.79999999999998" y="18">T</text>
  <text x="212.39999999999998" y="18" class="val" >×</text>
  <text class="lab" x="226.39999999999998" y="18">逐层实测</text>
  <text class="ttl" x="64" y="44">xl，seq 128（G > A）</text>
  <line class="grid" x1="64" y1="250.0" x2="314" y2="250.0"/><text class="tick" x="58" y="254.0" text-anchor="end">0</text>
  <line class="grid" x1="64" y1="216.4" x2="314" y2="216.4"/><text class="tick" x="58" y="220.4" text-anchor="end">5</text>
  <line class="grid" x1="64" y1="182.9" x2="314" y2="182.9"/><text class="tick" x="58" y="186.9" text-anchor="end">10</text>
  <line class="grid" x1="64" y1="149.3" x2="314" y2="149.3"/><text class="tick" x="58" y="153.3" text-anchor="end">15</text>
  <line class="grid" x1="64" y1="115.7" x2="314" y2="115.7"/><text class="tick" x="58" y="119.7" text-anchor="end">20</text>
  <line class="grid" x1="64" y1="82.1" x2="314" y2="82.1"/><text class="tick" x="58" y="86.1" text-anchor="end">25</text>
  <g class="m"><title>W</title><polygon points="64.0,164.7 67.9,164.7 71.8,164.7 75.7,164.7 79.6,164.7 83.5,164.7 87.4,164.7 91.3,164.7 95.2,164.7 99.2,164.7 103.1,164.7 107.0,164.7 110.9,164.7 114.8,164.7 118.7,164.7 122.6,164.7 126.5,164.7 130.4,164.7 134.3,164.7 138.2,164.7 142.1,164.7 146.0,164.7 149.9,164.7 153.8,164.7 157.8,164.7 161.7,164.7 165.6,164.7 169.5,164.7 173.4,164.7 177.3,164.7 181.2,164.7 185.1,164.7 189.0,164.7 192.9,164.7 196.8,164.7 200.7,164.7 204.6,164.7 208.5,164.7 212.4,164.7 216.3,164.7 220.2,164.7 224.2,164.7 228.1,164.7 232.0,164.7 235.9,164.7 239.8,164.7 243.7,164.7 247.6,164.7 251.5,164.7 255.4,164.7 259.3,164.7 263.2,164.7 267.1,164.7 271.0,164.7 274.9,164.7 278.8,164.7 282.8,164.7 286.7,164.7 290.6,164.7 294.5,164.7 298.4,164.7 302.3,164.7 306.2,164.7 310.1,164.7 314.0,164.7 314.0,250.0 310.1,250.0 306.2,250.0 302.3,250.0 298.4,250.0 294.5,250.0 290.6,250.0 286.7,250.0 282.8,250.0 278.8,250.0 274.9,250.0 271.0,250.0 267.1,250.0 263.2,250.0 259.3,250.0 255.4,250.0 251.5,250.0 247.6,250.0 243.7,250.0 239.8,250.0 235.9,250.0 232.0,250.0 228.1,250.0 224.2,250.0 220.2,250.0 216.3,250.0 212.4,250.0 208.5,250.0 204.6,250.0 200.7,250.0 196.8,250.0 192.9,250.0 189.0,250.0 185.1,250.0 181.2,250.0 177.3,250.0 173.4,250.0 169.5,250.0 165.6,250.0 161.7,250.0 157.8,250.0 153.8,250.0 149.9,250.0 146.0,250.0 142.1,250.0 138.2,250.0 134.3,250.0 130.4,250.0 126.5,250.0 122.6,250.0 118.7,250.0 114.8,250.0 110.9,250.0 107.0,250.0 103.1,250.0 99.2,250.0 95.2,250.0 91.3,250.0 87.4,250.0 83.5,250.0 79.6,250.0 75.7,250.0 71.8,250.0 67.9,250.0 64.0,250.0" style="fill: var(--fig-1)" stroke-width=".8"/></g>
  <g class="m"><title>A</title><polygon points="64.0,164.7 67.9,163.6 71.8,162.5 75.7,161.4 79.6,160.3 83.5,159.2 87.4,158.1 91.3,156.9 95.2,155.8 99.2,154.7 103.1,153.6 107.0,152.5 110.9,151.4 114.8,150.3 118.7,149.2 122.6,148.0 126.5,146.9 130.4,145.8 134.3,144.7 138.2,143.6 142.1,142.5 146.0,141.4 149.9,140.3 153.8,139.2 157.8,138.0 161.7,136.9 165.6,135.8 169.5,134.7 173.4,133.6 177.3,132.5 181.2,131.4 185.1,130.3 189.0,129.1 192.9,130.3 196.8,131.4 200.7,132.5 204.6,133.6 208.5,134.7 212.4,135.8 216.3,136.9 220.2,138.0 224.2,139.2 228.1,140.3 232.0,141.4 235.9,142.5 239.8,143.6 243.7,144.7 247.6,145.8 251.5,146.9 255.4,148.0 259.3,149.2 263.2,150.3 267.1,151.4 271.0,152.5 274.9,153.6 278.8,154.7 282.8,155.8 286.7,156.9 290.6,158.1 294.5,159.2 298.4,160.3 302.3,161.4 306.2,162.5 310.1,163.6 314.0,164.7 314.0,164.7 310.1,164.7 306.2,164.7 302.3,164.7 298.4,164.7 294.5,164.7 290.6,164.7 286.7,164.7 282.8,164.7 278.8,164.7 274.9,164.7 271.0,164.7 267.1,164.7 263.2,164.7 259.3,164.7 255.4,164.7 251.5,164.7 247.6,164.7 243.7,164.7 239.8,164.7 235.9,164.7 232.0,164.7 228.1,164.7 224.2,164.7 220.2,164.7 216.3,164.7 212.4,164.7 208.5,164.7 204.6,164.7 200.7,164.7 196.8,164.7 192.9,164.7 189.0,164.7 185.1,164.7 181.2,164.7 177.3,164.7 173.4,164.7 169.5,164.7 165.6,164.7 161.7,164.7 157.8,164.7 153.8,164.7 149.9,164.7 146.0,164.7 142.1,164.7 138.2,164.7 134.3,164.7 130.4,164.7 126.5,164.7 122.6,164.7 118.7,164.7 114.8,164.7 110.9,164.7 107.0,164.7 103.1,164.7 99.2,164.7 95.2,164.7 91.3,164.7 87.4,164.7 83.5,164.7 79.6,164.7 75.7,164.7 71.8,164.7 67.9,164.7 64.0,164.7" style="fill: var(--fig-2)" stroke-width=".8"/></g>
  <g class="m"><title>G</title><polygon points="64.0,164.7 67.9,163.6 71.8,162.5 75.7,161.4 79.6,160.3 83.5,159.2 87.4,158.1 91.3,156.9 95.2,155.8 99.2,154.7 103.1,153.6 107.0,152.5 110.9,151.4 114.8,150.3 118.7,149.2 122.6,148.0 126.5,146.9 130.4,145.8 134.3,144.7 138.2,143.6 142.1,142.5 146.0,141.4 149.9,140.3 153.8,139.2 157.8,138.0 161.7,136.9 165.6,135.8 169.5,134.7 173.4,133.6 177.3,132.5 181.2,131.4 185.1,130.3 189.0,129.1 192.9,127.6 196.8,126.0 200.7,124.5 204.6,122.9 208.5,121.4 212.4,119.8 216.3,118.3 220.2,116.7 224.2,115.2 228.1,113.6 232.0,112.1 235.9,110.5 239.8,109.0 243.7,107.4 247.6,105.9 251.5,104.3 255.4,102.7 259.3,101.2 263.2,99.6 267.1,98.1 271.0,96.5 274.9,95.0 278.8,93.4 282.8,91.9 286.7,90.3 290.6,88.8 294.5,87.2 298.4,85.7 302.3,84.1 306.2,82.6 310.1,81.0 314.0,79.5 314.0,164.7 310.1,163.6 306.2,162.5 302.3,161.4 298.4,160.3 294.5,159.2 290.6,158.1 286.7,156.9 282.8,155.8 278.8,154.7 274.9,153.6 271.0,152.5 267.1,151.4 263.2,150.3 259.3,149.2 255.4,148.0 251.5,146.9 247.6,145.8 243.7,144.7 239.8,143.6 235.9,142.5 232.0,141.4 228.1,140.3 224.2,139.2 220.2,138.0 216.3,136.9 212.4,135.8 208.5,134.7 204.6,133.6 200.7,132.5 196.8,131.4 192.9,130.3 189.0,129.1 185.1,130.3 181.2,131.4 177.3,132.5 173.4,133.6 169.5,134.7 165.6,135.8 161.7,136.9 157.8,138.0 153.8,139.2 149.9,140.3 146.0,141.4 142.1,142.5 138.2,143.6 134.3,144.7 130.4,145.8 126.5,146.9 122.6,148.0 118.7,149.2 114.8,150.3 110.9,151.4 107.0,152.5 103.1,153.6 99.2,154.7 95.2,155.8 91.3,156.9 87.4,158.1 83.5,159.2 79.6,160.3 75.7,161.4 71.8,162.5 67.9,163.6 64.0,164.7" style="fill: var(--fig-hi)" stroke-width=".8"/></g>
  <g class="m"><title>T</title><polygon points="64.0,164.7 67.9,163.6 71.8,162.5 75.7,161.4 79.6,160.3 83.5,159.2 87.4,158.1 91.3,156.9 95.2,155.8 99.2,154.7 103.1,153.6 107.0,152.5 110.9,151.4 114.8,150.3 118.7,149.2 122.6,148.0 126.5,146.9 130.4,145.8 134.3,144.7 138.2,143.6 142.1,142.5 146.0,141.4 149.9,140.3 153.8,139.2 157.8,138.0 161.7,136.9 165.6,135.8 169.5,134.7 173.4,133.6 177.3,132.5 181.2,131.4 185.1,130.3 189.0,127.8 192.9,126.2 196.8,124.7 200.7,123.1 204.6,121.6 208.5,120.0 212.4,118.5 216.3,116.9 220.2,115.4 224.2,113.8 228.1,112.3 232.0,110.7 235.9,109.2 239.8,107.6 243.7,106.1 247.6,104.5 251.5,103.0 255.4,101.4 259.3,99.9 263.2,98.3 267.1,96.7 271.0,95.2 274.9,93.6 278.8,92.1 282.8,90.5 286.7,89.0 290.6,87.4 294.5,85.9 298.4,84.3 302.3,82.8 306.2,81.2 310.1,79.7 314.0,78.1 314.0,79.5 310.1,81.0 306.2,82.6 302.3,84.1 298.4,85.7 294.5,87.2 290.6,88.8 286.7,90.3 282.8,91.9 278.8,93.4 274.9,95.0 271.0,96.5 267.1,98.1 263.2,99.6 259.3,101.2 255.4,102.7 251.5,104.3 247.6,105.9 243.7,107.4 239.8,109.0 235.9,110.5 232.0,112.1 228.1,113.6 224.2,115.2 220.2,116.7 216.3,118.3 212.4,119.8 208.5,121.4 204.6,122.9 200.7,124.5 196.8,126.0 192.9,127.6 189.0,129.1 185.1,130.3 181.2,131.4 177.3,132.5 173.4,133.6 169.5,134.7 165.6,135.8 161.7,136.9 157.8,138.0 153.8,139.2 149.9,140.3 146.0,141.4 142.1,142.5 138.2,143.6 134.3,144.7 130.4,145.8 126.5,146.9 122.6,148.0 118.7,149.2 114.8,150.3 110.9,151.4 107.0,152.5 103.1,153.6 99.2,154.7 95.2,155.8 91.3,156.9 87.4,158.1 83.5,159.2 79.6,160.3 75.7,161.4 71.8,162.5 67.9,163.6 64.0,164.7" style="fill: var(--fig-3); stroke: var(--fig-2)" stroke-width=".8"/></g>
  <text x="67.9" y="166.2" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="71.8" y="165.1" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="75.7" y="164.0" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="79.6" y="162.9" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="83.5" y="161.8" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="87.4" y="160.7" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="91.3" y="159.7" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="95.2" y="158.6" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="99.2" y="157.5" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="103.1" y="156.4" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="107.0" y="155.3" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="110.9" y="154.2" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="114.8" y="153.1" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="118.7" y="152.0" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="122.6" y="150.9" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="126.5" y="149.8" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="130.4" y="148.7" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="134.3" y="147.6" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="138.2" y="146.5" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="142.1" y="145.4" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="146.0" y="144.3" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="149.9" y="143.2" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="153.8" y="142.1" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="157.8" y="141.0" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="161.7" y="139.9" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="165.6" y="138.8" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="169.5" y="137.7" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="173.4" y="136.6" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="177.3" y="135.5" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="181.2" y="134.4" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="185.1" y="133.3" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="189.0" y="132.2" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="192.9" y="130.0" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="196.8" y="128.5" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="200.7" y="127.0" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="204.6" y="125.5" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="208.5" y="123.9" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="212.4" y="122.4" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="216.3" y="120.9" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="220.2" y="119.3" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="224.2" y="117.8" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="228.1" y="116.3" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="232.0" y="114.7" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="235.9" y="113.2" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="239.8" y="111.7" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="243.7" y="110.2" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="247.6" y="108.6" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="251.5" y="107.1" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="255.4" y="105.6" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="259.3" y="104.0" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="263.2" y="102.5" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="267.1" y="101.0" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="271.0" y="99.5" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="274.9" y="97.9" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="278.8" y="96.4" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="282.8" y="94.9" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="286.7" y="93.3" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="290.6" y="91.8" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="294.5" y="90.3" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="298.4" y="88.7" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="302.3" y="87.2" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="306.2" y="85.7" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="310.1" y="84.1" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="314.0" y="82.0" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text class="val" x="308.0" y="70.1" text-anchor="end">峰值 25.56 GiB</text>
  <line x1="189.0" y1="62" x2="189.0" y2="250" stroke="currentColor" stroke-opacity=".3" stroke-dasharray="2 3"/>
  <line class="axis" x1="64" y1="250" x2="314" y2="250"/>
  <text class="tick" x="64.0" y="266" text-anchor="start">开始</text>
  <text class="tick" x="189.0" y="266" text-anchor="middle">前向结束</text>
  <text class="tick" x="314.0" y="266" text-anchor="end">反向结束</text>
  <text class="lab2" x="64" y="286">W = G = 12.7 GiB，A = 5.3 GiB，L = 32</text>
  <text class="ttl" x="364" y="44">small，seq 512（A > G）</text>
  <line class="grid" x1="364" y1="250.0" x2="614" y2="250.0"/><text class="tick" x="358" y="254.0" text-anchor="end">0</text>
  <line class="grid" x1="364" y1="208.2" x2="614" y2="208.2"/><text class="tick" x="358" y="212.2" text-anchor="end">1</text>
  <line class="grid" x1="364" y1="166.4" x2="614" y2="166.4"/><text class="tick" x="358" y="170.4" text-anchor="end">2</text>
  <line class="grid" x1="364" y1="124.7" x2="614" y2="124.7"/><text class="tick" x="358" y="128.7" text-anchor="end">3</text>
  <line class="grid" x1="364" y1="82.9" x2="614" y2="82.9"/><text class="tick" x="358" y="86.9" text-anchor="end">4</text>
  <g class="m"><title>W</title><polygon points="364.0,229.9 374.4,229.9 384.8,229.9 395.2,229.9 405.7,229.9 416.1,229.9 426.5,229.9 436.9,229.9 447.3,229.9 457.8,229.9 468.2,229.9 478.6,229.9 489.0,229.9 499.4,229.9 509.8,229.9 520.2,229.9 530.7,229.9 541.1,229.9 551.5,229.9 561.9,229.9 572.3,229.9 582.8,229.9 593.2,229.9 603.6,229.9 614.0,229.9 614.0,250.0 603.6,250.0 593.2,250.0 582.8,250.0 572.3,250.0 561.9,250.0 551.5,250.0 541.1,250.0 530.7,250.0 520.2,250.0 509.8,250.0 499.4,250.0 489.0,250.0 478.6,250.0 468.2,250.0 457.8,250.0 447.3,250.0 436.9,250.0 426.5,250.0 416.1,250.0 405.7,250.0 395.2,250.0 384.8,250.0 374.4,250.0 364.0,250.0" style="fill: var(--fig-1)" stroke-width=".8"/></g>
  <g class="m"><title>A</title><polygon points="364.0,229.9 374.4,218.1 384.8,206.2 395.2,194.3 405.7,182.5 416.1,170.6 426.5,158.7 436.9,146.8 447.3,135.0 457.8,123.1 468.2,111.2 478.6,99.4 489.0,87.5 499.4,99.4 509.8,111.2 520.2,123.1 530.7,135.0 541.1,146.8 551.5,158.7 561.9,170.6 572.3,182.5 582.8,194.3 593.2,206.2 603.6,218.1 614.0,229.9 614.0,229.9 603.6,229.9 593.2,229.9 582.8,229.9 572.3,229.9 561.9,229.9 551.5,229.9 541.1,229.9 530.7,229.9 520.2,229.9 509.8,229.9 499.4,229.9 489.0,229.9 478.6,229.9 468.2,229.9 457.8,229.9 447.3,229.9 436.9,229.9 426.5,229.9 416.1,229.9 405.7,229.9 395.2,229.9 384.8,229.9 374.4,229.9 364.0,229.9" style="fill: var(--fig-2)" stroke-width=".8"/></g>
  <g class="m"><title>G</title><polygon points="364.0,229.9 374.4,218.1 384.8,206.2 395.2,194.3 405.7,182.5 416.1,170.6 426.5,158.7 436.9,146.8 447.3,135.0 457.8,123.1 468.2,111.2 478.6,99.4 489.0,87.5 499.4,97.7 509.8,107.9 520.2,118.1 530.7,128.3 541.1,138.5 551.5,148.7 561.9,158.9 572.3,169.1 582.8,179.3 593.2,189.5 603.6,199.7 614.0,209.9 614.0,229.9 603.6,218.1 593.2,206.2 582.8,194.3 572.3,182.5 561.9,170.6 551.5,158.7 541.1,146.8 530.7,135.0 520.2,123.1 509.8,111.2 499.4,99.4 489.0,87.5 478.6,99.4 468.2,111.2 457.8,123.1 447.3,135.0 436.9,146.8 426.5,158.7 416.1,170.6 405.7,182.5 395.2,194.3 384.8,206.2 374.4,218.1 364.0,229.9" style="fill: var(--fig-hi)" stroke-width=".8"/></g>
  <g class="m"><title>T</title><polygon points="364.0,229.9 374.4,218.1 384.8,206.2 395.2,194.3 405.7,182.5 416.1,170.6 426.5,158.7 436.9,146.8 447.3,135.0 457.8,123.1 468.2,111.2 478.6,99.4 489.0,80.4 499.4,90.6 509.8,100.8 520.2,111.0 530.7,121.2 541.1,131.4 551.5,141.6 561.9,151.8 572.3,162.0 582.8,172.2 593.2,182.4 603.6,192.6 614.0,202.8 614.0,209.9 603.6,199.7 593.2,189.5 582.8,179.3 572.3,169.1 561.9,158.9 551.5,148.7 541.1,138.5 530.7,128.3 520.2,118.1 509.8,107.9 499.4,97.7 489.0,87.5 478.6,99.4 468.2,111.2 457.8,123.1 447.3,135.0 436.9,146.8 426.5,158.7 416.1,170.6 405.7,182.5 395.2,194.3 384.8,206.2 374.4,218.1 364.0,229.9" style="fill: var(--fig-3); stroke: var(--fig-2)" stroke-width=".8"/></g>
  <text x="374.4" y="221.2" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="384.8" y="209.9" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="395.2" y="198.7" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="405.7" y="187.4" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="416.1" y="176.1" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="426.5" y="164.8" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="436.9" y="153.5" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="447.3" y="142.2" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="457.8" y="131.0" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="468.2" y="119.7" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="478.6" y="108.4" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="489.0" y="97.1" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="499.4" y="105.7" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="509.8" y="115.5" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="520.2" y="125.3" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="530.7" y="135.0" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="541.1" y="144.9" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="551.5" y="154.6" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="561.9" y="164.4" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="572.3" y="174.3" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="582.8" y="184.0" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="593.2" y="193.8" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="603.6" y="203.6" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text x="614.0" y="212.4" text-anchor="middle" font-size="10" fill="currentColor">×</text>
  <text class="val" x="495.0" y="72.4" text-anchor="start">峰值 4.08 GiB</text>
  <line x1="489.0" y1="62" x2="489.0" y2="250" stroke="currentColor" stroke-opacity=".3" stroke-dasharray="2 3"/>
  <line class="axis" x1="364" y1="250" x2="614" y2="250"/>
  <text class="tick" x="364.0" y="266" text-anchor="start">开始</text>
  <text class="tick" x="489.0" y="266" text-anchor="middle">前向结束</text>
  <text class="tick" x="614.0" y="266" text-anchor="end">反向结束</text>
  <text class="lab2" x="364" y="286">W = G = 0.48 GiB，A = 3.41 GiB，L = 12</text>
  <text class="lab2" x="14" y="160" transform="rotate(-90 14 160)" text-anchor="middle">GiB</text>
</svg>
<figcaption><strong>图 2-4</strong> 一步前向 + 反向的显存（fp32）：色带按 M(j) 用实测的 W、A、G、T 堆叠，× 是逐层实测值。</figcaption>
</figure>

总账算下来有两条线索：时间上，40% 花在逐元素 kernel 上；显存上，峰值主要由 <span class="sw" style="background: var(--fig-2)"></span>A 决定，而且四项里只有 A 随 seq 变大。下面把单个 op 拆开，同时看这两件事：每一步算了多少、读写了多少显存、为反向存了什么。先拿最简单的 RMSNorm 把方法走一遍，再用到 attention 上。

---

## 3 拆开 RMSNorm {#rmsnorm}

RMSNorm 算的是 $y = w \odot (x \cdot r)$，其中 $r = (\tfrac{1}{d}\sum_j x_j^2 + \epsilon)^{-1/2}$ 每行一个数。eager 模式下它拆成 5 个 op（fp32，`x: [4, 512, 2560]`，一份 x 是 20 MiB）：

```python
rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # ① pow ② mean ③ rsqrt
x_hat = x * rms                                            # ④
y = weight * x_hat                                         # ⑤
```

PyTorch 的 `torch.autograd.graph.saved_tensors_hooks` 可以在 autograd 每存下（pack）或取出（unpack）一个张量时调用一个自定义函数，在里面打印就知道存了什么：

```
Saving  1  [4,512,2560]  grad_fn=None            ptr=…9040   # x
Saving  2  [4,512,1]     grad_fn=RsqrtBackward0  ptr=…6b00   # r
Saving  3  [4,512,1]     grad_fn=RsqrtBackward0  ptr=…6b00   # r
Saving  4  [4,512,2560]  grad_fn=None            ptr=…9040   # x
Saving  5  [4,512,2560]  grad_fn=MulBackward0    ptr=…3c00   # x̂
Saving  6  [2560]        grad_fn=None            ptr=…1000   # w
Loading    5 → 6 → 2 → 4 → 3 → 1
```

从打印能看出 autograd 存张量的规则：**一个 op 的局部偏导里用到谁，前向就存谁；偏导是常数就什么都不存**。存的是引用，不是拷贝，所以本来就在显存里的输入 `x` 和参数 `w` 不额外占显存。6 次 Saving 只落在 4 块内存上（ptr 相同就是同一块），新占显存的只有 $r$（8 KiB）和 $\hat{x}$（20 MiB），见[表 3-1](#tab-3-1) 和[图 3-1](#fig-3-1)。

| op | 反向要的偏导 | 存下 | 新占显存 |
|:--|:--|:--|--:|
| ① $x^2$ | $\partial x^2/\partial x = 2x$ | $x$ | 0 |
| ② $v=\tfrac1d\sum x^2$ | $\partial v/\partial x^2 = \tfrac1d$，常数 | 不存 | 0 |
| ③ $r=(v+\epsilon)^{-1/2}$ | $\partial r/\partial v = -\tfrac12 r^3$ | $r$ | **8 KiB** |
| ④ $\hat{x}=x\cdot r$ | $\partial\hat{x}/\partial x = r$，$\partial\hat{x}/\partial r = x$ | $r$、$x$ | 0 |
| ⑤ $y=w\odot\hat{x}$ | $\partial y/\partial w = \hat{x}$，$\partial y/\partial\hat{x} = w$ | $\hat{x}$、$w$ | **20 MiB** |
{#tab-3-1 caption="**表 3-1** RMSNorm 五个 op 为反向存的张量（eager）" note="④ 存的 $r$、$x$ 和 ③、① 存的是同一块内存，所以不再占显存。新占显存按 fp32 算：$r$ 是 [4, 512, 1]，$\hat{x}$ 是 [4, 512, 2560]。"}

<figure id="fig-3-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 330" width="100%" role="img" aria-label="RMSNorm eager：前向 5 个算子每元素约 1 次运算；为反向存了 6 次张量，落在 x、r、x̂、w 四块内存上，新占的只有 r 8 KiB 和 x̂ 20 MiB；反向各步读回这些张量">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <defs><marker id="fig-3-1-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-3-1-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-mute)"/></marker><marker id="fig-3-1-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-1)"/></marker><marker id="fig-3-1-m3" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker></defs>
  <rect class="band" x="4" y="6" width="632" height="98" rx="8"/>
  <text class="ttl" x="16" y="24">前向</text><text class="lab2" x="50" y="24">花 FLOPs</text>
  <rect class="band" x="4" y="114" width="632" height="98" rx="8"/>
  <text class="ttl" x="16" y="132">显存</text><text class="lab2" x="50" y="132">为反向存下：6 次 Saving，落在 4 块内存上</text>
  <rect class="band" x="4" y="222" width="632" height="100" rx="8"/>
  <text class="ttl" x="16" y="240">反向</text><text class="lab2" x="50" y="240">再花 FLOPs</text>
  <rect x="50.0" y="43.0" width="84" height="46" rx="6" class="op"/>
  <text class="tb" x="92" y="63" text-anchor="middle">① x²</text>
  <text class="s" x="92" y="79" text-anchor="middle">1 FLOP/元素</text>
  <rect x="164.0" y="43.0" width="84" height="46" rx="6" class="op"/>
  <text class="tb" x="206" y="63" text-anchor="middle">② mean</text>
  <text class="s" x="206" y="79" text-anchor="middle">1 FLOP/元素</text>
  <rect x="278.0" y="43.0" width="84" height="46" rx="6" class="op"/>
  <text class="tb" x="320" y="63" text-anchor="middle">③ rsqrt</text>
  <text class="s" x="320" y="79" text-anchor="middle">每行 2 FLOPs</text>
  <rect x="392.0" y="43.0" width="84" height="46" rx="6" class="op"/>
  <text class="tb" x="434" y="63" text-anchor="middle">④ x · r</text>
  <text class="s" x="434" y="79" text-anchor="middle">1 FLOP/元素</text>
  <rect x="506.0" y="43.0" width="84" height="46" rx="6" class="op"/>
  <text class="tb" x="548" y="63" text-anchor="middle">⑤ w ⊙ x̂</text>
  <text class="s" x="548" y="79" text-anchor="middle">1 FLOP/元素</text>
  <line x1="134.0" y1="62.0" x2="162.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <line x1="248.0" y1="62.0" x2="276.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <line x1="362.0" y1="62.0" x2="390.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <line x1="476.0" y1="62.0" x2="504.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <text class="t" x="12" y="66">x</text>
  <line x1="24.0" y1="62.0" x2="48.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <line x1="590.0" y1="62.0" x2="616.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <text class="t" x="622" y="66">y</text>
  <line x1="92.0" y1="89.0" x2="92.0" y2="154.0" style="stroke: var(--fig-mute)" stroke-width="1.1"/>
  <rect x="54.0" y="154" width="76" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="92" y="170" text-anchor="middle" font-size="11.5" fill="currentColor">x …9040</text>
  <text x="92" y="186" text-anchor="middle" font-size="10.5" opacity=".7" fill="currentColor">输入，不新占</text>
  <line x1="92.0" y1="194.0" x2="92.0" y2="264.0" style="stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-3-1-m1)"/>
  <line x1="320.0" y1="89.0" x2="320.0" y2="154.0" style="stroke: var(--fig-1)" stroke-width="1.1"/>
  <rect x="282.0" y="154" width="76" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text x="320" y="170" text-anchor="middle" font-size="11.5" fill="currentColor">r …6b00</text>
  <text x="320" y="186" text-anchor="middle" font-size="10.5" font-weight="600" fill="currentColor">+8 KiB</text>
  <line x1="320.0" y1="194.0" x2="320.0" y2="264.0" style="stroke: var(--fig-1)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-3-1-m2)"/>
  <line x1="407.0" y1="89.0" x2="407.0" y2="154.0" style="stroke: var(--fig-1)" stroke-width="1.1"/>
  <rect x="383.0" y="154" width="48" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 8%, transparent); stroke: var(--fig-1)" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="407" y="170" text-anchor="middle" font-size="10.5" fill="currentColor">r …6b00</text>
  <text x="407" y="186" text-anchor="middle" font-size="10.5" opacity=".7" fill="currentColor">同 ③</text>
  <line x1="407.0" y1="194.0" x2="407.0" y2="264.0" style="stroke: var(--fig-1)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-3-1-m2)"/>
  <line x1="461.0" y1="89.0" x2="461.0" y2="154.0" style="stroke: var(--fig-mute)" stroke-width="1.1"/>
  <rect x="437.0" y="154" width="48" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="461" y="170" text-anchor="middle" font-size="10.5" fill="currentColor">x …9040</text>
  <text x="461" y="186" text-anchor="middle" font-size="10.5" opacity=".7" fill="currentColor">同 ①</text>
  <line x1="461.0" y1="194.0" x2="461.0" y2="264.0" style="stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-3-1-m1)"/>
  <line x1="521.0" y1="89.0" x2="521.0" y2="154.0" style="stroke: var(--fig-hi)" stroke-width="1.1"/>
  <rect x="497.0" y="154" width="48" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="2"/>
  <text x="521" y="170" text-anchor="middle" font-size="10.5" fill="currentColor">x̂ …3c00</text>
  <text x="521" y="186" text-anchor="middle" font-size="10.5" font-weight="600" fill="currentColor">+20 MiB</text>
  <line x1="521.0" y1="194.0" x2="521.0" y2="264.0" style="stroke: var(--fig-hi)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-3-1-m3)"/>
  <line x1="575.0" y1="89.0" x2="575.0" y2="154.0" style="stroke: var(--fig-mute)" stroke-width="1.1"/>
  <rect x="551.0" y="154" width="48" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="575" y="170" text-anchor="middle" font-size="10.5" fill="currentColor">w …1000</text>
  <text x="575" y="186" text-anchor="middle" font-size="10.5" opacity=".7" fill="currentColor">参数</text>
  <line x1="575.0" y1="194.0" x2="575.0" y2="264.0" style="stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-3-1-m1)"/>
  <text class="lab2" x="206" y="172" text-anchor="middle">偏导是常数 1/d</text><text class="lab2" x="206" y="187" text-anchor="middle">什么都不存</text>
  <rect x="50.0" y="266.0" width="84" height="36" rx="6" class="op"/>
  <text class="t" x="92" y="288.5" text-anchor="middle">① 反向</text>
  <rect x="164.0" y="266.0" width="84" height="36" rx="6" class="op"/>
  <text class="t" x="206" y="288.5" text-anchor="middle">② 反向</text>
  <rect x="278.0" y="266.0" width="84" height="36" rx="6" class="op"/>
  <text class="t" x="320" y="288.5" text-anchor="middle">③ 反向</text>
  <rect x="392.0" y="266.0" width="84" height="36" rx="6" class="op"/>
  <text class="t" x="434" y="288.5" text-anchor="middle">④ 反向</text>
  <rect x="506.0" y="266.0" width="84" height="36" rx="6" class="op"/>
  <text class="t" x="548" y="288.5" text-anchor="middle">⑤ 反向</text>
  <line x1="164.0" y1="284.0" x2="136.0" y2="284.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <line x1="278.0" y1="284.0" x2="250.0" y2="284.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <line x1="392.0" y1="284.0" x2="364.0" y2="284.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <line x1="506.0" y1="284.0" x2="478.0" y2="284.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <text class="t" x="620" y="288">dy</text>
  <line x1="616.0" y1="284.0" x2="592.0" y2="284.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <line x1="50.0" y1="284.0" x2="28.0" y2="284.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-1-m0)"/>
  <text class="t" x="8" y="288">dx</text>
</svg>
<figcaption><strong>图 3-1</strong> RMSNorm（eager）：中排是打印里的 6 次 Saving，ptr 相同就是同一块内存，实线框才新占显存；反向沿虚线箭头读回它们。</figcaption>
</figure>

时间上，RMSNorm 每个元素只算约 4 次，5 个 op 却要读写约 140 MiB 显存：①、④、⑤ 各读 20 MiB、写 20 MiB，② 读 20 MiB。$I \approx 0.14$，是 memory-bound。显存上，它为反向多存了一份 $\hat{x}$。

$\hat{x}$ 其实不用存：它就是 $x \cdot r$，反向时用 $x$ 和 $r$ 重算一次就有。eager 模式做不到，因为每个 op 单独执行，⑤ 的反向只知道自己需要 $\hat{x}$，不知道它能由 $x$ 和 $r$ 算出来。用 `torch.compile` 把 ①–⑤ 编译成一个前向 kernel 和一个反向 kernel 后，只存 $x$、$w$、$r$：

```
Saving  1  [4,512,2560]  grad_fn=None  ptr=…3c80   # x
Saving  2  [2560]        grad_fn=None  ptr=…b3c0   # w
Saving  3  [4,512,1]     grad_fn=None  ptr=…f9c0   # r
Loading    1 → 2 → 3（与 Saving 同序）
```

反向要算 $\nabla x = r\,\big(g - \hat{x} \cdot \mathrm{mean}(g \odot \hat{x})\big)$，其中 $g = w \odot \nabla y$。式子里的 $\hat{x}$ 都在反向 kernel 里用 $x \cdot r$ 现算（[图 3-2](#fig-3-2)），前后对比见[表 3-2](#tab-3-2)。

<figure id="fig-3-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 330" width="100%" role="img" aria-label="torch.compile 融合后的 RMSNorm：前向一个 kernel；显存只多留 r 8 KiB，x̂ 不存；反向一个 kernel，用 x 和 r 现场重算 x̂">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <defs><marker id="fig-3-2-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-3-2-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-mute)"/></marker><marker id="fig-3-2-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-1)"/></marker></defs>
  <rect class="band" x="4" y="6" width="632" height="98" rx="8"/>
  <text class="ttl" x="16" y="24">前向</text><text class="lab2" x="50" y="24">花 FLOPs</text>
  <rect class="band" x="4" y="114" width="632" height="98" rx="8"/>
  <text class="ttl" x="16" y="132">显存</text><text class="lab2" x="50" y="132">为反向存下：3 次 Saving</text>
  <rect class="band" x="4" y="222" width="632" height="100" rx="8"/>
  <text class="ttl" x="16" y="240">反向</text><text class="lab2" x="50" y="240">再花 FLOPs</text>
  <rect x="120.0" y="43.0" width="470" height="46" rx="6" class="op"/>
  <text class="tb" x="355.0" y="63" text-anchor="middle">fused forward：①–⑤ 一个 kernel</text>
  <text class="s" x="355.0" y="79" text-anchor="middle">~4 FLOPs/元素</text>
  <text class="t" x="12" y="66">x</text>
  <line x1="24.0" y1="62.0" x2="118.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-2-m0)"/>
  <line x1="590.0" y1="62.0" x2="616.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-2-m0)"/>
  <text class="t" x="622" y="66">y</text>
  <rect x="142.0" y="154" width="96" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="190" y="170" text-anchor="middle" font-size="11.5" fill="currentColor">x …3c80</text>
  <text x="190" y="186" text-anchor="middle" font-size="10.5" opacity=".7" fill="currentColor">输入，不新占</text>
  <line x1="190.0" y1="89.0" x2="190.0" y2="154.0" style="stroke: var(--fig-mute)" stroke-width="1.1"/>
  <line x1="190.0" y1="194.0" x2="190.0" y2="260.0" style="stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-3-2-m1)"/>
  <rect x="275.0" y="154" width="90" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text x="320" y="170" text-anchor="middle" font-size="11.5" fill="currentColor">r …f9c0</text>
  <text x="320" y="186" text-anchor="middle" font-size="10.5" font-weight="600" fill="currentColor">+8 KiB</text>
  <line x1="320.0" y1="89.0" x2="320.0" y2="154.0" style="stroke: var(--fig-1)" stroke-width="1.1"/>
  <line x1="320.0" y1="194.0" x2="320.0" y2="260.0" style="stroke: var(--fig-1)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-3-2-m2)"/>
  <rect x="400.0" y="154" width="104" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 8%, transparent); stroke: var(--fig-hi)" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="452" y="170" text-anchor="middle" font-size="11.5" fill="currentColor">x̂ 不存</text>
  <text x="452" y="186" text-anchor="middle" font-size="10.5" opacity=".7" fill="currentColor">省下 20 MiB</text>
  <line x1="406.0" y1="188" x2="498.0" y2="160" style="stroke: var(--fig-hi)" stroke-width="1.4" stroke-opacity=".8"/>
  <rect x="516.0" y="154" width="64" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="548" y="170" text-anchor="middle" font-size="11.5" fill="currentColor">w …b3c0</text>
  <text x="548" y="186" text-anchor="middle" font-size="10.5" opacity=".7" fill="currentColor">参数</text>
  <line x1="548.0" y1="89.0" x2="548.0" y2="154.0" style="stroke: var(--fig-mute)" stroke-width="1.1"/>
  <line x1="548.0" y1="194.0" x2="548.0" y2="260.0" style="stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-3-2-m1)"/>
  <rect x="120.0" y="266.0" width="470" height="44" rx="6" class="op"/>
  <text class="tb" x="355.0" y="285" text-anchor="middle">fused backward：∂y/∂w、∂y/∂x 一个 kernel</text>
  <text class="s" x="355.0" y="301" text-anchor="middle">x̂ = x · r 现场重算，每元素多 1 FLOP</text>
  <text class="t" x="620" y="288">dy</text>
  <line x1="616.0" y1="284.0" x2="592.0" y2="284.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-2-m0)"/>
  <line x1="120.0" y1="284.0" x2="28.0" y2="284.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-3-2-m0)"/>
  <text class="t" x="8" y="288">dx</text>
</svg>
<figcaption><strong>图 3-2</strong> 融合后的 RMSNorm（泳道同<a href="#fig-3-1">图 3-1</a>）：只多存 $r$（8 KiB），$\hat{x}$ 在反向用 $x$、$r$ 重算。</figcaption>
</figure>

| | eager | 融合后 |
|:--|:--|:--|
| kernel 数 | 前向 5 个 | 前向 1 个 + 反向 1 个 |
| 为反向存的张量 | $x$、$w$、$r$、$\hat{x}$ | $x$、$w$、$r$ |
| 新占显存 | $r + \hat{x}$ ≈ 20 MiB | $r$ ≈ 8 KiB |
| 前向读写显存 | ~140 MiB | ~40 MiB（只读 $x$、写 $y$） |
| FLOPs / 元素 | 前向 ~4 | 前向 ~4，反向多 1（重算 $\hat{x} = x\cdot r$） |
{#tab-3-2 caption="**表 3-2** RMSNorm：eager 与 `torch.compile` 融合" note="存的张量是实测，FLOPs 和读写是纸面计数。"}

> 融合解决了两件事：几个 op 合成一个 kernel，中间结果不再进出显存；能重算的张量不存，反向时再算。[第 4 节](#attention)的 attention 和 [5.2 节](#checkpoint)的 checkpoint 都会再遇到这两件事。

---

## 4 拆开 attention {#attention}

一层 attention 的完整公式是

<p align="center">$O = \mathrm{softmax}\!\left(\dfrac{QK^{\top}}{\sqrt{d}} + M\right) V$</p>

$Q$、$K$、$V$ 的形状都是 `[b, h, seq, d]`（d 是 d_head），$M$ 是 causal mask（未来位置为 $-\infty$）。eager 模式下它拆成 5 步：① 算分数 $S = QK^{\top}$；② 除以 $\sqrt{d}$；③ 加 mask；④ 按行 softmax 得到 $P$；⑤ $O = PV$。S、P 的形状都是 `[b, h, seq, seq]`，O 和 Q 一样大。

和 RMSNorm 一样，逐步看它算了多少、读写了多少显存、为反向存了什么（[图 4-1](#fig-4-1)，medium、seq 1024：b = 4，h = 16，d = 64）。一份 S 或 P 是 4 × 16 × 1024 × 1024 × 4 B = 256 MiB，而 Q、K、V、O 各只有 16 MiB。

<figure id="fig-4-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 392" width="100%" role="img" aria-label="eager attention 一层：前向 5 步，矩阵乘的 FLOPs 最多，softmax 读写显存最多（2048 MiB）；为反向新存了 exp(S−m) 和 P 两个 256 MiB 的 seq × seq 张量，Q、K、V、mask 只是引用">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <defs><marker id="fig-4-1-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-4-1-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-mute)"/></marker><marker id="fig-4-1-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker></defs>
  <rect class="band" x="4" y="6" width="632" height="98" rx="8"/>
  <text class="ttl" x="16" y="24">前向</text><text class="lab2" x="50" y="24">花 FLOPs</text>
  <rect class="band" x="4" y="114" width="632" height="62" rx="8"/>
  <text class="ttl" x="16" y="132">显存读写</text><text class="lab2" x="76" y="132">每步读写多少（MiB）</text>
  <rect class="band" x="4" y="186" width="632" height="98" rx="8"/>
  <text class="ttl" x="16" y="204">为反向存下</text><text class="lab2" x="89" y="204">实测 saved tensors</text>
  <rect class="band" x="4" y="294" width="632" height="90" rx="8"/>
  <text class="ttl" x="16" y="312">反向</text><text class="lab2" x="50" y="312"></text>
  <rect x="50.0" y="43.0" width="84" height="46" rx="6" class="op"/>
  <text class="tb" x="92" y="63" text-anchor="middle">① S = QKᵀ</text>
  <text class="s" x="92" y="79" text-anchor="middle">8.6e9 FLOPs</text>
  <g class="m"><title>① S = QKᵀ：读写显存 288 MiB</title><rect x="84.7" y="134" width="14.6" height="12" rx="2" style="fill: var(--fig-hi)"/></g>
  <text class="val" x="92" y="160" text-anchor="middle">288</text>
  <rect x="164.0" y="43.0" width="84" height="46" rx="6" class="op"/>
  <text class="tb" x="206" y="63" text-anchor="middle">② ÷√d</text>
  <text class="s" x="206" y="79" text-anchor="middle">6.7e7 FLOPs</text>
  <g class="m"><title>② ÷√d：读写显存 512 MiB</title><rect x="193.0" y="134" width="26.0" height="12" rx="2" style="fill: var(--fig-hi)"/></g>
  <text class="val" x="206" y="160" text-anchor="middle">512</text>
  <rect x="278.0" y="43.0" width="84" height="46" rx="6" class="op"/>
  <text class="tb" x="320" y="63" text-anchor="middle">③ + mask</text>
  <text class="s" x="320" y="79" text-anchor="middle">0 FLOPs</text>
  <g class="m"><title>③ + mask：读写显存 513 MiB</title><rect x="307.0" y="134" width="26.1" height="12" rx="2" style="fill: var(--fig-hi)"/></g>
  <text class="val" x="320" y="160" text-anchor="middle">513</text>
  <rect x="387.0" y="43.0" width="94" height="46" rx="6" class="op"/>
  <text class="tb" x="434" y="63" text-anchor="middle">④ softmax</text>
  <text class="s" x="434" y="79" text-anchor="middle">1.8e9 FLOPs</text>
  <g class="m"><title>④ softmax：读写显存 2048 MiB</title><rect x="382.0" y="134" width="104.0" height="12" rx="2" style="fill: var(--fig-hi)"/></g>
  <text class="val" x="434" y="160" text-anchor="middle">2048</text>
  <rect x="506.0" y="43.0" width="84" height="46" rx="6" class="op"/>
  <text class="tb" x="548" y="63" text-anchor="middle">⑤ O = PV</text>
  <text class="s" x="548" y="79" text-anchor="middle">8.6e9 FLOPs</text>
  <g class="m"><title>⑤ O = PV：读写显存 288 MiB</title><rect x="540.7" y="134" width="14.6" height="12" rx="2" style="fill: var(--fig-hi)"/></g>
  <text class="val" x="548" y="160" text-anchor="middle">288</text>
  <line x1="134.0" y1="62.0" x2="162.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <line x1="248.0" y1="62.0" x2="276.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <line x1="362.0" y1="62.0" x2="385.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <line x1="481.0" y1="62.0" x2="504.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <text class="t" x="8" y="66">Q K</text>
  <line x1="32.0" y1="62.0" x2="48.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <line x1="590.0" y1="62.0" x2="616.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <text class="t" x="620" y="66">O</text>
  <rect x="41.0" y="226" width="48" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="65" y="242" text-anchor="middle" font-size="10.5" fill="currentColor">Q</text>
  <text x="65" y="258" text-anchor="middle" font-size="10.5" opacity=".7" fill="currentColor">引用</text>
  <line x1="65.0" y1="266.0" x2="65.0" y2="324.0" style="stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-4-1-m1)"/>
  <rect x="95.0" y="226" width="48" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="119" y="242" text-anchor="middle" font-size="10.5" fill="currentColor">K</text>
  <text x="119" y="258" text-anchor="middle" font-size="10.5" opacity=".7" fill="currentColor">引用</text>
  <line x1="119.0" y1="266.0" x2="119.0" y2="324.0" style="stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-4-1-m1)"/>
  <rect x="285.0" y="226" width="70" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="320" y="242" text-anchor="middle" font-size="11.5" fill="currentColor">mask</text>
  <text x="320" y="258" text-anchor="middle" font-size="10.5" opacity=".7" fill="currentColor">1 MiB</text>
  <line x1="320.0" y1="266.0" x2="320.0" y2="324.0" style="stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-4-1-m1)"/>
  <rect x="392.0" y="226" width="84" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="2"/>
  <text x="434" y="242" text-anchor="middle" font-size="11.5" fill="currentColor">exp(S − m)</text>
  <text x="434" y="258" text-anchor="middle" font-size="10.5" font-weight="600" fill="currentColor">+256 MiB</text>
  <line x1="434.0" y1="266.0" x2="434.0" y2="324.0" style="stroke: var(--fig-hi)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-4-1-m2)"/>
  <rect x="494.0" y="226" width="60" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 16%, transparent); stroke: var(--fig-hi)" stroke-width="2"/>
  <text x="524" y="242" text-anchor="middle" font-size="10.5" fill="currentColor">P</text>
  <text x="524" y="258" text-anchor="middle" font-size="10.5" font-weight="600" fill="currentColor">+256 MiB</text>
  <line x1="524.0" y1="266.0" x2="524.0" y2="324.0" style="stroke: var(--fig-hi)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-4-1-m2)"/>
  <rect x="562.0" y="226" width="44" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="584" y="242" text-anchor="middle" font-size="10.5" fill="currentColor">V</text>
  <text x="584" y="258" text-anchor="middle" font-size="10.5" opacity=".7" fill="currentColor">引用</text>
  <line x1="584.0" y1="266.0" x2="584.0" y2="324.0" style="stroke: var(--fig-mute)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#fig-4-1-m1)"/>
  <text class="lab2" x="206" y="250" text-anchor="middle">不存</text>
  <rect x="50.0" y="326.0" width="84" height="36" rx="6" class="op"/>
  <text class="t" x="92" y="348.5" text-anchor="middle">① 反向</text>
  <rect x="164.0" y="326.0" width="84" height="36" rx="6" class="op"/>
  <text class="t" x="206" y="348.5" text-anchor="middle">② 反向</text>
  <rect x="278.0" y="326.0" width="84" height="36" rx="6" class="op"/>
  <text class="t" x="320" y="348.5" text-anchor="middle">③ 反向</text>
  <rect x="392.0" y="326.0" width="84" height="36" rx="6" class="op"/>
  <text class="t" x="434" y="348.5" text-anchor="middle">④ 反向</text>
  <rect x="506.0" y="326.0" width="84" height="36" rx="6" class="op"/>
  <text class="t" x="548" y="348.5" text-anchor="middle">⑤ 反向</text>
  <line x1="164.0" y1="344.0" x2="136.0" y2="344.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <line x1="278.0" y1="344.0" x2="250.0" y2="344.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <line x1="392.0" y1="344.0" x2="364.0" y2="344.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <line x1="506.0" y1="344.0" x2="478.0" y2="344.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <text class="t" x="620" y="348">dO</text>
  <line x1="616.0" y1="344.0" x2="592.0" y2="344.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <line x1="50.0" y1="344.0" x2="40.0" y2="344.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-4-1-m0)"/>
  <text class="t" x="4" y="348">dQ dK</text>
</svg>
<figcaption><strong>图 4-1</strong> eager attention 一层（medium，seq 1024）：中排红条是每步读写显存的量，下面是实测为反向存下的张量；实线框新占显存，虚线框只是引用。max 的下标和行和每行一个数，不到 1 MiB，没有画。</figcaption>
</figure>

### 4.1 时间：都在搬 S 和 P {#attn-time}

FLOPs 集中在 ① 和 ⑤ 两个矩阵乘上，读写却集中在 ② ③ ④ 上，它们每一步都要把整个 256 MiB 的 S 读一遍、写一遍。放到 roofline 上（[图 1-1](#fig-1-1)），attention 的 op 全在斜坡上，包括 QKᵀ 和 PV 这两个矩阵乘。它们的 MBU 已经有 57–86%，kernel 本身没多少优化空间，要更快只能少搬数据。

搬得最多的是 ④ softmax。它对 S 的每一行算 $P_{ij} = e^{S_{ij} - m_i} / \sum_k e^{S_{ik} - m_i}$，其中 $m_i = \max_k S_{ik}$，减掉行最大值是为了防止 exp 溢出。eager 模式下这个公式拆成 5 个 kernel：求 max、减 max、exp、求和、除。每个 kernel 都要读或写和 S 一样大的张量，5 个加起来一共读写 8 次，2048 MiB（[图 4-2](#fig-4-2)）。

<figure id="fig-4-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 262" width="100%" role="img" aria-label="eager softmax 的 5 个 kernel 共读写显存里 S 大小的张量 8 次；融合成一个 kernel 后只读 S、写 P 两次">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <defs><marker id="fig-4-2-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker><marker id="fig-4-2-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <rect class="band" x="70" y="24" width="562" height="56" rx="8"/>
  <text class="lab" x="8" y="50">显存</text><text class="lab2" x="8" y="66">(HBM)</text>
  <rect class="band" x="70" y="122" width="562" height="56" rx="8"/>
  <text class="lab" x="8" y="148">kernel</text><text class="lab2" x="8" y="164">(片上算)</text>
  <rect x="133" y="37" width="64" height="30" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.5"/>
  <text class="t" x="165" y="56" text-anchor="middle">S</text>
  <rect x="237" y="37" width="76" height="30" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.5"/>
  <text class="t" x="275" y="56" text-anchor="middle">S − m</text>
  <rect x="388" y="37" width="104" height="30" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.5"/>
  <text class="t" x="440" y="56" text-anchor="middle">exp(S − m)</text>
  <rect x="562" y="37" width="56" height="30" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.5"/>
  <text class="t" x="590" y="56" text-anchor="middle">P</text>
  <rect x="81" y="135" width="62" height="30" rx="6" class="op"/>
  <text class="t" x="112" y="154" text-anchor="middle">max</text>
  <rect x="183" y="135" width="70" height="30" rx="6" class="op"/>
  <text class="t" x="218" y="154" text-anchor="middle">减 max</text>
  <rect x="301" y="135" width="58" height="30" rx="6" class="op"/>
  <text class="t" x="330" y="154" text-anchor="middle">exp</text>
  <rect x="409" y="135" width="62" height="30" rx="6" class="op"/>
  <text class="t" x="440" y="154" text-anchor="middle">求和</text>
  <rect x="519" y="135" width="58" height="30" rx="6" class="op"/>
  <text class="t" x="548" y="154" text-anchor="middle">除</text>
  <line x1="152.0" y1="67.0" x2="122.0" y2="133.0" style="stroke: var(--fig-hi)" stroke-width="1.8" marker-end="url(#fig-4-2-m0)"/>
  <text class="lab" x="99" y="104">① 读</text>
  <line x1="178.0" y1="67.0" x2="208.0" y2="133.0" style="stroke: var(--fig-hi)" stroke-width="1.8" marker-end="url(#fig-4-2-m0)"/>
  <text class="lab" x="199" y="104">② 读</text>
  <line x1="232.0" y1="135.0" x2="262.0" y2="69.0" style="stroke: var(--fig-hi)" stroke-width="1.8" marker-end="url(#fig-4-2-m0)"/>
  <text class="lab" x="255" y="108">③ 写</text>
  <line x1="288.0" y1="67.0" x2="320.0" y2="133.0" style="stroke: var(--fig-hi)" stroke-width="1.8" marker-end="url(#fig-4-2-m0)"/>
  <text class="lab" x="311" y="104">④ 读</text>
  <line x1="342.0" y1="135.0" x2="412.0" y2="69.0" style="stroke: var(--fig-hi)" stroke-width="1.8" marker-end="url(#fig-4-2-m0)"/>
  <text class="lab" x="383" y="112">⑤ 写</text>
  <line x1="440.0" y1="67.0" x2="440.0" y2="133.0" style="stroke: var(--fig-hi)" stroke-width="1.8" marker-end="url(#fig-4-2-m0)"/>
  <text class="lab" x="446" y="104">⑥ 读</text>
  <line x1="468.0" y1="67.0" x2="536.0" y2="133.0" style="stroke: var(--fig-hi)" stroke-width="1.8" marker-end="url(#fig-4-2-m0)"/>
  <text class="lab" x="510" y="104">⑦ 读</text>
  <line x1="560.0" y1="135.0" x2="584.0" y2="69.0" style="stroke: var(--fig-hi)" stroke-width="1.8" marker-end="url(#fig-4-2-m0)"/>
  <text class="lab" x="578" y="108">⑧ 写</text>
  <line x1="143.0" y1="150.0" x2="181.0" y2="150.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-4-2-m1)"/>
  <text class="lab2" x="162" y="143" text-anchor="middle">m</text>
  <line x1="471.0" y1="150.0" x2="517.0" y2="150.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-4-2-m1)"/>
  <text class="lab2" x="494" y="143" text-anchor="middle">Σ</text>
  <text class="lab2" x="351" y="192" text-anchor="middle">eager：5 个 kernel，S 大小的张量进出显存 8 次；m、Σ 每行一个数，可忽略</text>
  <text class="lab" x="8" y="240">融合后</text>
  <rect x="133" y="221" width="64" height="30" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.5"/>
  <text class="t" x="165" y="240" text-anchor="middle">S</text>
  <rect x="255" y="221" width="150" height="30" rx="6" class="op"/>
  <text class="t" x="330" y="240" text-anchor="middle">fused softmax</text>
  <rect x="467" y="221" width="56" height="30" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 14%, transparent); stroke: var(--fig-1)" stroke-width="1.5"/>
  <text class="t" x="495" y="240" text-anchor="middle">P</text>
  <line x1="197.0" y1="236.0" x2="253.0" y2="236.0" style="stroke: var(--fig-hi)" stroke-width="1.8" marker-end="url(#fig-4-2-m0)"/>
  <text class="lab" x="209" y="228">① 读</text>
  <line x1="405.0" y1="236.0" x2="465.0" y2="236.0" style="stroke: var(--fig-hi)" stroke-width="1.8" marker-end="url(#fig-4-2-m0)"/>
  <text class="lab" x="419" y="228">② 写</text>
</svg>
<figcaption><strong>图 4-2</strong> eager softmax 的显存读写：每条编号箭头是一次完整的读或写，共 8 次；融合后只剩 2 次。</figcaption>
</figure>

FLOPs 和耗时因此对不上：softmax 的运算量只有 PV 的 1/5，耗时却是 PV 的 6 倍（[图 4-3](#fig-4-3)）。

<figure id="fig-4-3" class="fg-fig">
<svg class="fg" viewBox="0 0 640 186" width="100%" role="img" aria-label="medium、seq 1024 一层 attention 里各 op 的 FLOPs 与实测 GPU 时间：softmax 和 ÷√d、mask 的 FLOPs 很少，时间却最多">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <rect x="112" y="11" width="10" height="10" rx="2" style="fill: var(--fig-1)"/>
  <text class="lab" x="128" y="20">矩阵乘</text>
  <rect x="186" y="11" width="10" height="10" rx="2" style="fill: var(--fig-hi)"/>
  <text class="lab" x="202" y="20">逐元素</text>
  <text class="ttl" x="112" y="48">FLOPs</text><text class="ttl" x="392" y="48">GPU 时间（ms）</text>
  <text class="lab" x="102" y="75" text-anchor="end">QKᵀ</text>
  <g class="m"><title>QKᵀ：8.6e9 FLOPs（一层）</title><rect x="112.0" y="62.0" width="191.1" height="18.0" rx="3" style="fill: var(--fig-1)"/></g>
  <text class="val" x="309.1" y="75">8.6e9</text>
  <g class="m"><title>QKᵀ：0.3 ms（一层）</title><rect x="392.0" y="62.0" width="40.0" height="18.0" rx="3" style="fill: var(--fig-1)"/></g>
  <text class="val" x="438.0" y="75">0.30</text>
  <text class="lab" x="102" y="105" text-anchor="end">÷√d + mask</text>
  <g class="m"><title>÷√d + mask：6.7e7 FLOPs（一层）</title><rect x="112.0" y="92.0" width="1.5" height="18.0" rx="3" style="fill: var(--fig-hi)"/></g>
  <text class="val" x="119.5" y="105">6.7e7</text>
  <g class="m"><title>÷√d + mask：0.75 ms（一层）</title><rect x="392.0" y="92.0" width="100.0" height="18.0" rx="3" style="fill: var(--fig-hi)"/></g>
  <text class="val" x="498.0" y="105">0.75</text>
  <text class="lab" x="102" y="135" text-anchor="end">softmax</text>
  <g class="m"><title>softmax：1.8e9 FLOPs（一层）</title><rect x="112.0" y="122.0" width="40.0" height="18.0" rx="3" style="fill: var(--fig-hi)"/></g>
  <text class="val" x="158.0" y="135">1.8e9</text>
  <g class="m"><title>softmax：1.43 ms（一层）</title><rect x="392.0" y="122.0" width="190.7" height="18.0" rx="3" style="fill: var(--fig-hi)"/></g>
  <text class="val" x="588.7" y="135">1.43</text>
  <text class="lab" x="102" y="165" text-anchor="end">PV</text>
  <g class="m"><title>PV：8.6e9 FLOPs（一层）</title><rect x="112.0" y="152.0" width="191.1" height="18.0" rx="3" style="fill: var(--fig-1)"/></g>
  <text class="val" x="309.1" y="165">8.6e9</text>
  <g class="m"><title>PV：0.23 ms（一层）</title><rect x="392.0" y="152.0" width="30.7" height="18.0" rx="3" style="fill: var(--fig-1)"/></g>
  <text class="val" x="428.7" y="165">0.23</text>
  <line class="axis" x1="112" y1="58" x2="112" y2="174"/>
  <line class="axis" x1="392" y1="58" x2="392" y2="174"/>
</svg>
<figcaption><strong>图 4-3</strong> 一层 attention 里各 op 的 FLOPs 与实测 GPU 时间（medium，seq 1024）。</figcaption>
</figure>

S、P 的读写量随 seq² 增长，Linear 只随 seq 线性增长。seq 从 256 增加到 1024，attention 占前向时间的比例从 10% 涨到 46%，多出来的几乎全是 softmax、除以 √d、mask 这类只搬数据的 op（[图 4-4](#fig-4-4)）。这就是第 2 节那 40% 里随 seq 涨得最快的部分。

<figure id="fig-4-4" class="fg-fig">
<svg class="fg" viewBox="0 0 640 262" width="100%" role="img" aria-label="attention 三段占 forward 时间随 seq 的变化：softmax 从 4% 涨到 24%，scores 从 4% 涨到 18%，PV 只从 2% 到 4%，合计从 10% 到 46%">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <line x1="70" y1="16" x2="82" y2="16" style="stroke: var(--fig-hi)" stroke-width="2"/>
  <text class="lab" x="86" y="20">softmax</text>
  <line x1="154.2" y1="16" x2="166.2" y2="16" style="stroke: var(--fig-1)" stroke-width="2"/>
  <text class="lab" x="170.2" y="20">scores（QKᵀ、÷√d、mask）</text>
  <line x1="345.79999999999995" y1="16" x2="357.79999999999995" y2="16" style="stroke: var(--fig-2)" stroke-width="2"/>
  <text class="lab" x="361.79999999999995" y="20">PV</text>
  <line x1="396.99999999999994" y1="16" x2="408.99999999999994" y2="16" stroke="currentColor" stroke-width="1.6" stroke-dasharray="4 3"/>
  <text class="lab" x="412.99999999999994" y="20">attention 合计</text>
  <line class="grid" x1="70" y1="236.0" x2="520" y2="236.0"/><text class="tick" x="62" y="240.0" text-anchor="end">0%</text>
  <line class="grid" x1="70" y1="200.0" x2="520" y2="200.0"/><text class="tick" x="62" y="204.0" text-anchor="end">10%</text>
  <line class="grid" x1="70" y1="164.0" x2="520" y2="164.0"/><text class="tick" x="62" y="168.0" text-anchor="end">20%</text>
  <line class="grid" x1="70" y1="128.0" x2="520" y2="128.0"/><text class="tick" x="62" y="132.0" text-anchor="end">30%</text>
  <line class="grid" x1="70" y1="92.0" x2="520" y2="92.0"/><text class="tick" x="62" y="96.0" text-anchor="end">40%</text>
  <line class="grid" x1="70" y1="56.0" x2="520" y2="56.0"/><text class="tick" x="62" y="60.0" text-anchor="end">50%</text>
  <text class="tick" x="110" y="254" text-anchor="middle">seq 256</text>
  <text class="tick" x="290" y="254" text-anchor="middle">seq 512</text>
  <text class="tick" x="470" y="254" text-anchor="middle">seq 1024</text>
  <text class="lab2" x="70" y="46">占 forward GPU 时间（medium，24 层）</text>
  <polyline points="110,200.0 290,156.8 470,70.4" fill="none" stroke="currentColor" stroke-opacity=".55" stroke-dasharray="6 4" stroke-width="2" stroke-linejoin="round"/>
  <g class="m"><title>seq 256：合计 2.3 ms，占 10%</title><circle cx="110" cy="200.0" r="4.5" class="ring" fill="currentColor" fill-opacity=".55"/></g>
  <g class="m"><title>seq 512：合计 10.2 ms，占 22%</title><circle cx="290" cy="156.8" r="4.5" class="ring" fill="currentColor" fill-opacity=".55"/></g>
  <g class="m"><title>seq 1024：合计 64.9 ms，占 46%</title><circle cx="470" cy="70.4" r="4.5" class="ring" fill="currentColor" fill-opacity=".55"/></g>
  <text class="val" x="482" y="74.4">合计 46%</text>
  <polyline points="110,221.6 290,196.4 470,149.6" fill="none" style="stroke: var(--fig-hi)" stroke-width="2" stroke-linejoin="round"/>
  <g class="m"><title>seq 256：softmax 1.0 ms，占 4%</title><circle cx="110" cy="221.6" r="4.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>seq 512：softmax 5.0 ms，占 11%</title><circle cx="290" cy="196.4" r="4.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>seq 1024：softmax 34.3 ms，占 24%</title><circle cx="470" cy="149.6" r="4.5" class="ring" style="fill: var(--fig-hi)"/></g>
  <text class="val" x="482" y="153.6">softmax 24%</text>
  <polyline points="110,221.6 290,207.2 470,171.2" fill="none" style="stroke: var(--fig-1)" stroke-width="2" stroke-linejoin="round"/>
  <g class="m"><title>seq 256：scores 0.9 ms，占 4%</title><circle cx="110" cy="221.6" r="4.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>seq 512：scores 3.6 ms，占 8%</title><circle cx="290" cy="207.2" r="4.5" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>seq 1024：scores 25.1 ms，占 18%</title><circle cx="470" cy="171.2" r="4.5" class="ring" style="fill: var(--fig-1)"/></g>
  <text class="val" x="482" y="175.2">scores 18%</text>
  <polyline points="110,228.8 290,225.2 470,221.6" fill="none" style="stroke: var(--fig-2)" stroke-width="2" stroke-linejoin="round"/>
  <g class="m"><title>seq 256：PV 0.4 ms，占 2%</title><circle cx="110" cy="228.8" r="4.5" class="ring" style="fill: var(--fig-2)"/></g>
  <g class="m"><title>seq 512：PV 1.6 ms，占 3%</title><circle cx="290" cy="225.2" r="4.5" class="ring" style="fill: var(--fig-2)"/></g>
  <g class="m"><title>seq 1024：PV 5.5 ms，占 4%</title><circle cx="470" cy="221.6" r="4.5" class="ring" style="fill: var(--fig-2)"/></g>
  <text class="val" x="482" y="225.6">PV 4%</text>
</svg>
<figcaption><strong>图 4-4</strong> attention 三段占 forward GPU 时间的比例随 seq 变化（medium）。</figcaption>
</figure>

要少搬，就得把几步合进一个 kernel，中间结果留在片上：融合的 softmax 只读一次 S、写一次 P（[图 4-2](#fig-4-2) 下）；FlashAttention 更进一步，S、P 根本不写回显存。

### 4.2 显存：S、P 占了一半以上 {#attn-memory}

用同样的 `saved_tensors_hooks` 打印 eager attention 存下的张量（medium、seq 1024，b·h = 64 合成一维）：

```
Saving  1  [64,64,1024]        float32  # K（转置后的 view）
Saving  2  [64,1024,64]        float32  # Q
Saving  3  [1024,1024]         bool     # mask
Saving  4  [4,16,1024,1]       int64    # 每行 max 的下标
Saving  5  [4,16,1024,1024]    float32  # e = exp(S − m)
Saving  6  [4,16,1024,1]       float32  # 行和 Σ
Saving  7  [4,16,1024,1024]    float32  # e（和第 5 条同一块内存）
Saving  8  [64,1024,64]        float32  # V
Saving  9  [64,1024,1024]      float32  # P
```

按 RMSNorm 那条规则逐个 op 对一遍（[表 4-1](#tab-4-1)）：9 次 Saving 里，Q、K、V、mask 本来就在显存里，只是引用；softmax 的 5 个 kernel 里，exp 的导数就是它自己的输出，除法要用分子和分母，所以存下 e 和 Σ；⑤ 要用 P 和 V。新占显存的是 e 和 P 两个 seq × seq 张量，各 256 MiB，加起来比 Q、K、V、O 的总和还大 8 倍。

| op | 反向要的偏导 | 存下 | 新占显存 |
|:--|:--|:--|--:|
| ① $S = QK^{\top}$ | $\partial S/\partial Q = K$，$\partial S/\partial K = Q$ | $Q$、$K$ | 0 |
| ② $\div\sqrt{d}$ | $1/\sqrt{d}$，常数 | 不存 | 0 |
| ③ $+M$ | 被 mask 的位置梯度为 0 | mask | 0 |
| ④ max | 只有最大值的位置有梯度 | 下标 | 0.5 MiB |
| ④ $e = \exp(S - m)$ | $\partial e/\partial S = e$ | $e$ | **256 MiB** |
| ④ $P = e / \Sigma$ | $\partial P/\partial e = 1/\Sigma$，$\partial P/\partial \Sigma = -e/\Sigma^2$ | $e$、$\Sigma$ | 0.2 MiB |
| ⑤ $O = PV$ | $\partial O/\partial P = V$，$\partial O/\partial V = P$ | $P$、$V$ | **256 MiB** |
{#tab-4-1 caption="**表 4-1** attention 各 op 为反向存的张量（eager，medium，seq 1024）" note="存下的张量是实测。④ 的减 max 和求和偏导是常数，不存；除法存的 e 和 exp 存的是同一块内存。"}

放到一整层上也是这样。xl 的一层（`torch.compile` 已经省掉了 RMSNorm 这类中间量）一共要为反向存 3655 MiB，其中一半以上是 S 和 P（[图 4-5](#fig-4-5)）。这组测量用的是 16 头，S、P 各 1 GiB；标准 xl 是 32 头，S、P 还要再大一倍。

<figure id="fig-4-5" class="fg-fig">
<svg class="fg" viewBox="0 0 640 246" width="100%" role="img" aria-label="xl 一层为反向存的 3655 MiB：S、P 占 56%，FFN 中间量 26%，[b, s, d] 级张量 17.5%，其他 0.2%">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <g class="m"><title>S、P（[b, h, s, s]）：2048 MiB，56.0%</title><path d="M150.00,30.00 A92,92 0 1 1 115.96,207.47 L128.54,175.88 A58,58 0 1 0 150.00,64.00 Z" class="ring" style="fill: var(--fig-hi)"/></g>
  <g class="m"><title>FFN 中间量（[b, s, d_ff]）：960 MiB，26.3%</title><path d="M115.96,207.47 A92,92 0 0 1 67.50,81.28 L97.99,96.33 A58,58 0 0 0 128.54,175.88 Z" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>[b, s, d] 级张量（x、norm 输出、Q、K、V 等）：640 MiB，17.5%</title><path d="M67.50,81.28 A92,92 0 0 1 148.89,30.01 L149.30,64.00 A58,58 0 0 0 97.99,96.33 Z" class="ring" style="fill: var(--fig-2)"/></g>
  <g class="m"><title>其他（mask、RoPE、softmax 统计量）：7 MiB，0.2%</title><path d="M148.89,30.01 A92,92 0 0 1 150.00,30.00 L150.00,64.00 A58,58 0 0 0 149.30,64.00 Z" class="ring" style="fill: var(--fig-3)"/></g>
  <text class="ttl" x="150" y="120" text-anchor="middle">3655 MiB</text><text class="lab2" x="150" y="137" text-anchor="middle">xl 一层</text>
  <rect x="290" y="52" width="12" height="12" rx="2" style="fill: var(--fig-hi)"/>
  <text class="lab" x="310" y="62">S、P</text><text class="lab2" x="310" y="78">[b, h, s, s]</text>
  <text class="val" x="630" y="62" text-anchor="end">2048 MiB · 56.0%</text>
  <rect x="290" y="92" width="12" height="12" rx="2" style="fill: var(--fig-1)"/>
  <text class="lab" x="310" y="102">FFN 中间量</text><text class="lab2" x="310" y="118">[b, s, d_ff]</text>
  <text class="val" x="630" y="102" text-anchor="end">960 MiB · 26.3%</text>
  <rect x="290" y="132" width="12" height="12" rx="2" style="fill: var(--fig-2)"/>
  <text class="lab" x="310" y="142">[b, s, d] 级张量</text><text class="lab2" x="310" y="158">x、norm 输出、Q、K、V 等</text>
  <text class="val" x="630" y="142" text-anchor="end">640 MiB · 17.5%</text>
  <rect x="290" y="172" width="12" height="12" rx="2" style="fill: var(--fig-3); stroke: var(--fig-2)"/>
  <text class="lab" x="310" y="182">其他</text><text class="lab2" x="310" y="198">mask、RoPE、softmax 统计量</text>
  <text class="val" x="630" y="182" text-anchor="end">7 MiB · 0.2%</text>
</svg>
<figcaption><strong>图 4-5</strong> xl 一层为反向存的张量（batch 4，seq 2048，16 头，<code>torch.compile</code> 后用 <code>saved_tensors_hooks</code> 实测）。</figcaption>
</figure>

xl 有 32 层，按 16 头算加起来也有 114 GiB，是 5090 显存的三倍多。而且只有 S、P 随 seq² 增长：32 头时，同一个 `[b, h, s, s]` 张量在 seq 128 时是 8 MiB，seq 2048 时是 2 GiB，是残差流上一个 `[b, s, d]` 张量的 25 倍。

[图 4-6](#fig-4-6) 是 xl 一步的显存时间线。seq 2048 只跑前向时，每层 attention 都让显存冲高约 8 GiB，算完再落回去，32 层都能跑完；加上反向后，每层的 S、P 都得留下，第 1 层就多占约 4.7 GiB，到第 2 层就 OOM 了。

<figure id="fig-4-6" class="fg-fig">
<svg class="fg" viewBox="0 0 640 420" width="100%" role="img" aria-label="xl 一步的显存时间线：seq 128 纯前向是平的；seq 2048 纯前向每层冲出一个尖峰；seq 128 full step 前向和反向一路上升，到 optimizer 时 OOM；seq 2048 带反向时第 2 层 OOM">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <text class="ttl" x="52" y="34">seq 128 · 纯前向</text>
  <line class="grid" x1="52" y1="178.0" x2="314" y2="178.0"/><text class="tick" x="46" y="182.0" text-anchor="end">0</text>
  <line class="grid" x1="52" y1="134.0" x2="314" y2="134.0"/><text class="tick" x="46" y="138.0" text-anchor="end">10</text>
  <line class="grid" x1="52" y1="90.0" x2="314" y2="90.0"/><text class="tick" x="46" y="94.0" text-anchor="end">20</text>
  <line class="grid" x1="52" y1="46.0" x2="314" y2="46.0"/><text class="tick" x="46" y="50.0" text-anchor="end">30</text>
  <polygon points="52.0,178.0 52.0,121.6 52.8,121.4 53.2,121.4 53.5,121.4 53.7,121.4 54.3,121.3 54.6,121.3 55.2,121.5 55.4,121.5 55.9,121.2 56.3,121.5 57.1,121.4 57.3,121.4 57.9,121.3 58.3,121.3 58.8,121.4 58.8,121.4 59.2,121.5 60.0,121.2 60.4,121.5 60.6,121.5 61.1,121.4 61.4,121.4 61.9,121.3 62.4,121.3 63.1,121.4 63.3,121.5 63.8,121.2 64.0,121.2 64.4,121.5 64.8,121.5 65.6,121.4 65.9,121.4 66.4,121.3 66.7,121.3 67.3,121.5 67.4,121.5 68.1,121.2 68.5,121.5 69.1,121.4 69.4,121.4 69.7,121.4 70.0,121.4 70.5,121.3 70.8,121.3 71.4,121.5 72.1,121.2 72.5,121.5 72.6,121.5 73.2,121.4 73.5,121.4 74.1,121.3 74.5,121.3 74.9,121.4 75.1,121.4 75.4,121.5 76.2,121.2 76.5,121.5 76.8,121.5 77.3,121.4 78.0,121.4 78.1,121.3 78.6,121.3 79.3,121.5 79.4,121.5 80.2,121.2 80.2,121.2 80.6,121.5 81.1,121.5 81.8,121.4 82.1,121.4 82.6,121.3 82.9,121.3 83.5,121.5 83.7,121.5 84.2,121.2 84.6,121.5 85.3,121.4 85.6,121.4 86.2,121.3 86.3,121.4 86.7,121.3 87.1,121.4 87.5,121.5 88.3,121.2 88.7,121.5 88.8,121.5 89.4,121.4 89.7,121.4 90.2,121.3 90.7,121.3 91.3,121.4 91.6,121.5 92.1,121.2 92.3,121.2 92.7,121.5 93.1,121.5 93.4,121.4 94.2,121.4 94.7,121.3 94.8,121.3 95.6,121.5 95.6,121.5 96.4,121.2 96.5,121.4 96.8,121.5 97.7,121.4 98.0,121.4 98.2,121.4 98.8,121.3 99.0,121.3 99.7,121.5 99.9,121.5 100.4,121.2 100.8,121.5 101.5,121.4 101.8,121.4 102.4,121.3 102.8,121.3 103.2,121.4 103.3,121.4 103.7,121.5 104.5,121.2 104.8,121.5 105.0,121.5 105.6,121.4 105.9,121.4 106.4,121.3 106.9,121.3 107.5,121.4 107.7,121.5 108.3,121.2 108.5,121.2 108.9,121.5 109.3,121.5 110.1,121.4 110.4,121.4 110.9,121.3 111.2,121.3 111.8,121.5 111.9,121.5 112.5,121.2 112.9,121.5 113.5,121.4 113.9,121.4 114.2,121.4 114.4,121.4 115.0,121.3 115.3,121.3 115.8,121.5 116.6,121.2 117.0,121.5 117.1,121.5 117.7,121.4 118.0,121.4 118.5,121.3 119.0,121.3 119.4,121.4 119.6,121.4 119.9,121.5 120.6,121.2 121.0,121.5 121.3,121.5 121.7,121.4 122.5,121.4 122.6,121.3 123.0,121.3 123.8,121.5 123.9,121.5 124.7,121.2 124.7,121.2 125.1,121.5 125.6,121.5 126.3,121.4 126.5,121.4 127.1,121.3 127.3,121.3 128.0,121.5 128.1,121.5 128.7,121.2 129.1,121.5 129.8,121.4 130.1,121.4 130.7,121.3 130.8,121.4 131.1,121.3 131.6,121.4 132.0,121.5 132.8,121.2 133.1,121.5 133.3,121.5 133.9,121.4 134.1,121.4 134.7,121.3 135.2,121.3 135.8,121.4 136.1,121.5 136.6,121.2 136.8,121.2 137.2,121.5 137.6,121.5 137.9,121.4 138.7,121.4 139.2,121.3 139.2,121.3 140.0,121.5 140.1,121.5 140.8,121.2 140.9,121.4 141.2,121.5 142.2,121.4 142.5,121.4 142.7,121.4 143.3,121.3 143.5,121.3 144.1,121.5 144.4,121.5 144.9,121.2 145.3,121.5 146.0,121.4 146.3,121.4 146.8,121.3 147.3,121.3 147.7,121.4 147.8,121.4 148.2,121.5 148.9,121.2 149.3,121.5 149.5,121.5 150.0,121.4 150.4,121.4 150.9,121.3 151.3,121.3 152.0,121.4 152.2,121.5 152.8,121.2 153.0,121.2 153.4,121.5 153.8,121.5 154.6,121.4 154.8,121.4 155.4,121.3 155.6,121.3 156.3,121.5 156.3,121.5 157.0,121.2 157.4,121.5 158.0,121.4 158.4,121.4 158.6,121.4 158.9,121.4 159.4,121.3 159.8,121.3 160.3,121.5 161.1,121.2 161.4,121.5 161.6,121.5 162.2,121.4 162.4,121.4 163.0,121.3 163.5,121.3 163.9,121.4 164.0,121.4 164.4,121.5 165.1,121.2 165.5,121.5 165.8,121.5 166.2,121.4 167.0,121.4 167.0,121.3 167.5,121.3 168.3,121.5 168.4,121.5 169.1,121.2 169.2,121.2 169.5,121.5 170.0,121.5 170.8,121.4 171.0,121.4 171.6,121.3 171.8,121.3 172.4,121.5 172.6,121.5 173.2,121.2 173.6,121.5 174.3,121.4 174.6,121.4 175.1,121.3 175.2,121.4 175.6,121.3 176.1,121.4 176.5,121.5 177.2,121.2 177.6,121.5 177.7,121.5 178.3,121.4 178.6,121.4 179.2,121.3 179.6,121.3 180.3,121.4 180.5,121.5 181.1,121.2 181.3,121.2 181.7,121.5 182.1,121.5 182.4,121.3 183.1,121.6 183.7,121.5 183.7,121.5 184.4,121.4 184.7,121.4 185.2,121.3 185.5,121.3 186.1,121.5 186.3,121.5 186.9,121.2 187.3,121.5 188.0,121.4 188.2,121.4 188.8,121.3 188.9,121.4 189.3,121.3 189.8,121.4 190.2,121.5 190.9,121.2 191.3,121.5 191.4,121.5 192.0,121.4 192.3,121.4 192.9,121.3 193.3,121.3 194.0,121.4 194.2,121.5 194.8,121.2 195.0,121.2 195.3,121.5 195.8,121.5 196.1,121.4 196.8,121.4 197.4,121.3 197.4,121.3 198.2,121.5 198.3,121.5 199.0,121.2 199.1,121.4 199.4,121.5 200.4,121.4 200.6,121.4 200.9,121.4 201.4,121.3 201.7,121.3 202.3,121.5 202.6,121.5 203.0,121.2 203.4,121.5 204.2,121.4 204.4,121.4 205.0,121.3 205.5,121.3 205.9,121.4 205.9,121.4 206.3,121.5 207.1,121.2 207.5,121.5 207.7,121.5 208.2,121.4 208.5,121.4 209.0,121.3 209.5,121.3 210.2,121.4 210.4,121.5 211.0,121.2 211.1,121.2 211.5,121.5 211.9,121.5 212.7,121.4 213.0,121.4 213.5,121.3 213.8,121.3 214.4,121.5 214.5,121.5 215.2,121.2 215.6,121.5 216.2,121.4 216.6,121.4 216.8,121.4 217.1,121.4 217.6,121.3 217.9,121.3 218.5,121.5 219.2,121.2 219.6,121.5 219.7,121.5 220.3,121.4 220.6,121.4 221.2,121.3 221.6,121.3 222.0,121.4 222.2,121.4 222.5,121.5 223.3,121.2 223.7,121.5 223.9,121.5 224.4,121.4 225.1,121.4 225.2,121.3 225.7,121.3 226.5,121.5 226.6,121.5 227.3,121.2 227.3,121.2 227.7,121.5 228.2,121.5 228.9,121.4 229.2,121.4 229.7,121.3 230.0,121.3 230.6,121.5 230.8,121.5 231.3,121.2 231.7,121.5 232.4,121.4 232.7,121.4 233.3,121.3 233.4,121.4 233.8,121.3 234.2,121.4 234.6,121.5 235.4,121.2 235.8,121.5 235.9,121.5 236.5,121.4 236.8,121.4 237.3,121.3 237.8,121.3 238.4,121.4 238.7,121.5 239.3,121.2 239.4,121.2 239.8,121.5 240.2,121.5 240.5,121.4 241.3,121.4 241.8,121.3 241.9,121.3 242.7,121.5 242.7,121.5 243.5,121.2 243.6,121.4 243.9,121.5 244.9,121.4 245.1,121.4 245.3,121.4 245.9,121.3 246.1,121.3 246.8,121.5 247.0,121.5 247.5,121.2 247.9,121.5 248.6,121.4 248.9,121.4 249.5,121.3 249.9,121.3 250.3,121.4 250.4,121.4 250.8,121.5 251.6,121.2 252.0,121.5 252.1,121.5 252.7,121.4 253.0,121.4 253.5,121.3 254.0,121.3 254.6,121.4 254.9,121.5 255.4,121.2 255.6,121.2 256.0,121.5 256.4,121.5 257.2,121.4 257.5,121.4 258.0,121.3 258.3,121.3 258.9,121.5 259.0,121.5 259.7,121.2 260.0,121.5 260.6,121.4 261.0,121.4 261.3,121.4 261.5,121.4 262.1,121.3 262.4,121.3 262.9,121.5 263.7,121.2 264.1,121.5 264.2,121.5 264.8,121.4 265.1,121.4 265.6,121.3 266.1,121.3 266.5,121.4 266.7,121.4 267.0,121.5 267.7,121.2 268.1,121.5 268.4,121.5 268.9,121.4 269.6,121.4 269.7,121.3 270.1,121.3 270.9,121.5 271.0,121.5 271.8,121.2 271.8,121.2 272.2,121.5 272.7,121.5 273.4,121.4 273.6,121.4 274.2,121.3 274.4,121.3 275.1,121.5 275.3,121.5 275.8,121.2 276.2,121.5 276.9,121.4 277.2,121.4 277.8,121.3 277.9,121.4 278.2,121.3 278.7,121.4 279.1,121.5 279.9,121.2 280.3,121.5 280.4,121.5 281.0,121.4 281.2,121.4 281.8,121.3 282.3,121.3 282.9,121.4 283.2,121.5 283.7,121.2 283.9,121.2 284.3,121.5 284.7,121.5 285.0,121.4 285.8,121.4 286.3,121.3 286.3,121.3 287.1,121.5 287.2,121.5 288.0,121.2 288.1,121.4 288.3,121.5 289.3,121.4 289.6,121.4 289.8,121.4 290.4,121.3 290.6,121.3 291.2,121.5 291.5,121.5 292.0,121.2 292.4,121.5 293.1,121.4 293.4,121.4 293.9,121.3 294.4,121.3 294.8,121.4 294.9,121.4 295.3,121.5 296.0,121.2 296.4,121.5 296.6,121.5 297.2,121.4 297.5,121.4 298.0,121.3 298.4,121.3 299.1,121.4 299.3,121.5 299.9,121.2 300.1,121.2 300.5,121.5 300.9,121.5 301.7,121.4 301.9,121.4 302.5,121.3 302.8,121.3 303.4,121.5 303.5,121.5 304.1,121.2 304.5,121.5 305.1,121.4 305.5,121.4 305.7,121.4 306.0,121.4 306.5,121.3 306.9,121.3 307.4,121.5 308.2,121.2 308.6,121.5 308.7,121.5 309.3,121.4 309.5,121.4 310.1,121.3 310.6,121.3 311.0,121.4 311.1,121.4 311.5,121.5 312.2,121.2 312.6,121.5 312.9,121.5 313.4,121.3 313.7,121.4 314.0,121.6 314.0,178.0" style="fill: var(--fig-3)" fill-opacity=".7"/>
  <polyline points="52.0,121.6 52.8,121.4 53.2,121.4 53.5,121.4 53.7,121.4 54.3,121.3 54.6,121.3 55.2,121.5 55.4,121.5 55.9,121.2 56.3,121.5 57.1,121.4 57.3,121.4 57.9,121.3 58.3,121.3 58.8,121.4 58.8,121.4 59.2,121.5 60.0,121.2 60.4,121.5 60.6,121.5 61.1,121.4 61.4,121.4 61.9,121.3 62.4,121.3 63.1,121.4 63.3,121.5 63.8,121.2 64.0,121.2 64.4,121.5 64.8,121.5 65.6,121.4 65.9,121.4 66.4,121.3 66.7,121.3 67.3,121.5 67.4,121.5 68.1,121.2 68.5,121.5 69.1,121.4 69.4,121.4 69.7,121.4 70.0,121.4 70.5,121.3 70.8,121.3 71.4,121.5 72.1,121.2 72.5,121.5 72.6,121.5 73.2,121.4 73.5,121.4 74.1,121.3 74.5,121.3 74.9,121.4 75.1,121.4 75.4,121.5 76.2,121.2 76.5,121.5 76.8,121.5 77.3,121.4 78.0,121.4 78.1,121.3 78.6,121.3 79.3,121.5 79.4,121.5 80.2,121.2 80.2,121.2 80.6,121.5 81.1,121.5 81.8,121.4 82.1,121.4 82.6,121.3 82.9,121.3 83.5,121.5 83.7,121.5 84.2,121.2 84.6,121.5 85.3,121.4 85.6,121.4 86.2,121.3 86.3,121.4 86.7,121.3 87.1,121.4 87.5,121.5 88.3,121.2 88.7,121.5 88.8,121.5 89.4,121.4 89.7,121.4 90.2,121.3 90.7,121.3 91.3,121.4 91.6,121.5 92.1,121.2 92.3,121.2 92.7,121.5 93.1,121.5 93.4,121.4 94.2,121.4 94.7,121.3 94.8,121.3 95.6,121.5 95.6,121.5 96.4,121.2 96.5,121.4 96.8,121.5 97.7,121.4 98.0,121.4 98.2,121.4 98.8,121.3 99.0,121.3 99.7,121.5 99.9,121.5 100.4,121.2 100.8,121.5 101.5,121.4 101.8,121.4 102.4,121.3 102.8,121.3 103.2,121.4 103.3,121.4 103.7,121.5 104.5,121.2 104.8,121.5 105.0,121.5 105.6,121.4 105.9,121.4 106.4,121.3 106.9,121.3 107.5,121.4 107.7,121.5 108.3,121.2 108.5,121.2 108.9,121.5 109.3,121.5 110.1,121.4 110.4,121.4 110.9,121.3 111.2,121.3 111.8,121.5 111.9,121.5 112.5,121.2 112.9,121.5 113.5,121.4 113.9,121.4 114.2,121.4 114.4,121.4 115.0,121.3 115.3,121.3 115.8,121.5 116.6,121.2 117.0,121.5 117.1,121.5 117.7,121.4 118.0,121.4 118.5,121.3 119.0,121.3 119.4,121.4 119.6,121.4 119.9,121.5 120.6,121.2 121.0,121.5 121.3,121.5 121.7,121.4 122.5,121.4 122.6,121.3 123.0,121.3 123.8,121.5 123.9,121.5 124.7,121.2 124.7,121.2 125.1,121.5 125.6,121.5 126.3,121.4 126.5,121.4 127.1,121.3 127.3,121.3 128.0,121.5 128.1,121.5 128.7,121.2 129.1,121.5 129.8,121.4 130.1,121.4 130.7,121.3 130.8,121.4 131.1,121.3 131.6,121.4 132.0,121.5 132.8,121.2 133.1,121.5 133.3,121.5 133.9,121.4 134.1,121.4 134.7,121.3 135.2,121.3 135.8,121.4 136.1,121.5 136.6,121.2 136.8,121.2 137.2,121.5 137.6,121.5 137.9,121.4 138.7,121.4 139.2,121.3 139.2,121.3 140.0,121.5 140.1,121.5 140.8,121.2 140.9,121.4 141.2,121.5 142.2,121.4 142.5,121.4 142.7,121.4 143.3,121.3 143.5,121.3 144.1,121.5 144.4,121.5 144.9,121.2 145.3,121.5 146.0,121.4 146.3,121.4 146.8,121.3 147.3,121.3 147.7,121.4 147.8,121.4 148.2,121.5 148.9,121.2 149.3,121.5 149.5,121.5 150.0,121.4 150.4,121.4 150.9,121.3 151.3,121.3 152.0,121.4 152.2,121.5 152.8,121.2 153.0,121.2 153.4,121.5 153.8,121.5 154.6,121.4 154.8,121.4 155.4,121.3 155.6,121.3 156.3,121.5 156.3,121.5 157.0,121.2 157.4,121.5 158.0,121.4 158.4,121.4 158.6,121.4 158.9,121.4 159.4,121.3 159.8,121.3 160.3,121.5 161.1,121.2 161.4,121.5 161.6,121.5 162.2,121.4 162.4,121.4 163.0,121.3 163.5,121.3 163.9,121.4 164.0,121.4 164.4,121.5 165.1,121.2 165.5,121.5 165.8,121.5 166.2,121.4 167.0,121.4 167.0,121.3 167.5,121.3 168.3,121.5 168.4,121.5 169.1,121.2 169.2,121.2 169.5,121.5 170.0,121.5 170.8,121.4 171.0,121.4 171.6,121.3 171.8,121.3 172.4,121.5 172.6,121.5 173.2,121.2 173.6,121.5 174.3,121.4 174.6,121.4 175.1,121.3 175.2,121.4 175.6,121.3 176.1,121.4 176.5,121.5 177.2,121.2 177.6,121.5 177.7,121.5 178.3,121.4 178.6,121.4 179.2,121.3 179.6,121.3 180.3,121.4 180.5,121.5 181.1,121.2 181.3,121.2 181.7,121.5 182.1,121.5 182.4,121.3 183.1,121.6 183.7,121.5 183.7,121.5 184.4,121.4 184.7,121.4 185.2,121.3 185.5,121.3 186.1,121.5 186.3,121.5 186.9,121.2 187.3,121.5 188.0,121.4 188.2,121.4 188.8,121.3 188.9,121.4 189.3,121.3 189.8,121.4 190.2,121.5 190.9,121.2 191.3,121.5 191.4,121.5 192.0,121.4 192.3,121.4 192.9,121.3 193.3,121.3 194.0,121.4 194.2,121.5 194.8,121.2 195.0,121.2 195.3,121.5 195.8,121.5 196.1,121.4 196.8,121.4 197.4,121.3 197.4,121.3 198.2,121.5 198.3,121.5 199.0,121.2 199.1,121.4 199.4,121.5 200.4,121.4 200.6,121.4 200.9,121.4 201.4,121.3 201.7,121.3 202.3,121.5 202.6,121.5 203.0,121.2 203.4,121.5 204.2,121.4 204.4,121.4 205.0,121.3 205.5,121.3 205.9,121.4 205.9,121.4 206.3,121.5 207.1,121.2 207.5,121.5 207.7,121.5 208.2,121.4 208.5,121.4 209.0,121.3 209.5,121.3 210.2,121.4 210.4,121.5 211.0,121.2 211.1,121.2 211.5,121.5 211.9,121.5 212.7,121.4 213.0,121.4 213.5,121.3 213.8,121.3 214.4,121.5 214.5,121.5 215.2,121.2 215.6,121.5 216.2,121.4 216.6,121.4 216.8,121.4 217.1,121.4 217.6,121.3 217.9,121.3 218.5,121.5 219.2,121.2 219.6,121.5 219.7,121.5 220.3,121.4 220.6,121.4 221.2,121.3 221.6,121.3 222.0,121.4 222.2,121.4 222.5,121.5 223.3,121.2 223.7,121.5 223.9,121.5 224.4,121.4 225.1,121.4 225.2,121.3 225.7,121.3 226.5,121.5 226.6,121.5 227.3,121.2 227.3,121.2 227.7,121.5 228.2,121.5 228.9,121.4 229.2,121.4 229.7,121.3 230.0,121.3 230.6,121.5 230.8,121.5 231.3,121.2 231.7,121.5 232.4,121.4 232.7,121.4 233.3,121.3 233.4,121.4 233.8,121.3 234.2,121.4 234.6,121.5 235.4,121.2 235.8,121.5 235.9,121.5 236.5,121.4 236.8,121.4 237.3,121.3 237.8,121.3 238.4,121.4 238.7,121.5 239.3,121.2 239.4,121.2 239.8,121.5 240.2,121.5 240.5,121.4 241.3,121.4 241.8,121.3 241.9,121.3 242.7,121.5 242.7,121.5 243.5,121.2 243.6,121.4 243.9,121.5 244.9,121.4 245.1,121.4 245.3,121.4 245.9,121.3 246.1,121.3 246.8,121.5 247.0,121.5 247.5,121.2 247.9,121.5 248.6,121.4 248.9,121.4 249.5,121.3 249.9,121.3 250.3,121.4 250.4,121.4 250.8,121.5 251.6,121.2 252.0,121.5 252.1,121.5 252.7,121.4 253.0,121.4 253.5,121.3 254.0,121.3 254.6,121.4 254.9,121.5 255.4,121.2 255.6,121.2 256.0,121.5 256.4,121.5 257.2,121.4 257.5,121.4 258.0,121.3 258.3,121.3 258.9,121.5 259.0,121.5 259.7,121.2 260.0,121.5 260.6,121.4 261.0,121.4 261.3,121.4 261.5,121.4 262.1,121.3 262.4,121.3 262.9,121.5 263.7,121.2 264.1,121.5 264.2,121.5 264.8,121.4 265.1,121.4 265.6,121.3 266.1,121.3 266.5,121.4 266.7,121.4 267.0,121.5 267.7,121.2 268.1,121.5 268.4,121.5 268.9,121.4 269.6,121.4 269.7,121.3 270.1,121.3 270.9,121.5 271.0,121.5 271.8,121.2 271.8,121.2 272.2,121.5 272.7,121.5 273.4,121.4 273.6,121.4 274.2,121.3 274.4,121.3 275.1,121.5 275.3,121.5 275.8,121.2 276.2,121.5 276.9,121.4 277.2,121.4 277.8,121.3 277.9,121.4 278.2,121.3 278.7,121.4 279.1,121.5 279.9,121.2 280.3,121.5 280.4,121.5 281.0,121.4 281.2,121.4 281.8,121.3 282.3,121.3 282.9,121.4 283.2,121.5 283.7,121.2 283.9,121.2 284.3,121.5 284.7,121.5 285.0,121.4 285.8,121.4 286.3,121.3 286.3,121.3 287.1,121.5 287.2,121.5 288.0,121.2 288.1,121.4 288.3,121.5 289.3,121.4 289.6,121.4 289.8,121.4 290.4,121.3 290.6,121.3 291.2,121.5 291.5,121.5 292.0,121.2 292.4,121.5 293.1,121.4 293.4,121.4 293.9,121.3 294.4,121.3 294.8,121.4 294.9,121.4 295.3,121.5 296.0,121.2 296.4,121.5 296.6,121.5 297.2,121.4 297.5,121.4 298.0,121.3 298.4,121.3 299.1,121.4 299.3,121.5 299.9,121.2 300.1,121.2 300.5,121.5 300.9,121.5 301.7,121.4 301.9,121.4 302.5,121.3 302.8,121.3 303.4,121.5 303.5,121.5 304.1,121.2 304.5,121.5 305.1,121.4 305.5,121.4 305.7,121.4 306.0,121.4 306.5,121.3 306.9,121.3 307.4,121.5 308.2,121.2 308.6,121.5 308.7,121.5 309.3,121.4 309.5,121.4 310.1,121.3 310.6,121.3 311.0,121.4 311.1,121.4 311.5,121.5 312.2,121.2 312.6,121.5 312.9,121.5 313.4,121.3 313.7,121.4 314.0,121.6" fill="none" style="stroke: var(--fig-1)" stroke-width="1.2" stroke-linejoin="round"/>
  <line x1="52" y1="121.6" x2="314" y2="121.6" stroke="currentColor" stroke-opacity=".5" stroke-dasharray="2 3"/>
  <text class="lab2" x="314" y="134.6" text-anchor="end">权重 12.8 GiB</text>
  <g class="m"><title>峰值 12.90 GiB</title><text class="val" x="56" y="56">峰值 12.90 GiB</text></g>
  <line class="axis" x1="52" y1="178" x2="314" y2="178"/>
  <text class="tick" x="314" y="193" text-anchor="end">10110 次分配 / 释放</text>
  <text class="ttl" x="352" y="34">seq 2048 · 纯前向</text>
  <line class="grid" x1="352" y1="178.0" x2="614" y2="178.0"/><text class="tick" x="346" y="182.0" text-anchor="end">0</text>
  <line class="grid" x1="352" y1="134.0" x2="614" y2="134.0"/><text class="tick" x="346" y="138.0" text-anchor="end">10</text>
  <line class="grid" x1="352" y1="90.0" x2="614" y2="90.0"/><text class="tick" x="346" y="94.0" text-anchor="end">20</text>
  <line class="grid" x1="352" y1="46.0" x2="614" y2="46.0"/><text class="tick" x="346" y="50.0" text-anchor="end">30</text>
  <polygon points="352.0,178.0 352.0,121.6 352.8,119.8 352.9,119.8 353.7,118.1 353.9,119.1 354.5,92.7 354.6,83.9 355.4,120.2 355.5,120.9 356.2,116.0 356.3,117.4 356.6,121.2 357.1,119.8 357.8,118.1 358.1,119.1 358.6,83.9 358.9,100.9 359.5,120.9 359.7,120.9 360.3,116.0 360.6,121.2 361.4,118.5 361.6,119.5 362.2,109.7 362.7,83.9 363.1,118.8 363.1,118.1 363.5,120.9 364.3,116.0 364.7,121.2 364.9,121.2 365.4,118.5 365.7,119.5 366.4,101.5 366.7,83.9 367.3,119.1 367.6,120.9 368.2,116.4 368.3,116.0 368.7,121.2 369.1,120.9 369.4,118.5 370.2,119.1 370.7,83.9 370.8,83.9 371.6,120.9 371.7,120.9 372.4,116.0 372.5,118.8 372.8,121.2 373.7,119.5 374.0,118.1 374.2,119.1 374.8,83.9 375.1,100.9 375.7,120.9 376.4,116.0 376.8,120.9 376.8,121.2 377.5,118.5 377.8,119.5 378.4,109.7 378.8,83.9 379.2,118.8 379.3,118.5 379.7,120.9 380.5,116.0 380.8,121.2 381.1,121.2 381.6,118.5 382.3,119.1 382.5,101.5 382.9,83.9 383.6,119.8 383.7,120.9 384.3,116.4 384.5,116.0 384.9,121.2 385.3,120.5 386.1,118.1 386.4,119.1 386.9,83.9 387.0,92.7 387.8,120.9 388.0,120.9 388.5,116.0 388.9,121.2 389.5,119.1 389.9,119.5 390.1,118.1 390.4,118.5 390.9,83.9 391.3,110.0 391.8,120.9 392.6,116.0 393.0,121.2 393.1,121.2 393.7,118.5 393.9,119.5 394.6,101.5 395.0,83.9 395.4,118.8 395.9,120.9 396.4,117.8 396.6,116.0 397.0,121.2 397.4,120.9 397.7,118.5 398.5,119.1 399.0,92.7 399.0,83.9 399.8,120.2 399.9,120.9 400.6,116.0 400.7,117.4 401.0,121.2 401.6,119.8 402.3,118.1 402.5,119.1 403.1,83.9 403.3,100.9 403.9,120.9 404.1,120.9 404.7,116.0 405.1,121.2 405.8,118.5 406.1,119.5 406.6,109.7 407.1,83.9 407.5,118.8 407.6,118.1 408.0,120.9 408.7,116.0 409.1,121.2 409.3,121.2 409.8,118.5 410.1,119.5 410.8,101.5 411.1,83.9 411.8,119.1 412.0,120.9 412.6,116.4 412.8,116.0 413.2,121.2 413.6,120.9 413.9,118.5 414.6,119.1 415.2,83.9 415.2,83.9 416.1,120.9 416.1,120.9 416.8,116.0 416.9,118.8 417.2,121.2 418.2,119.5 418.4,118.1 418.7,119.1 419.2,83.9 419.5,100.9 420.1,120.9 420.8,116.0 421.2,120.9 421.2,121.2 422.0,118.5 422.2,119.5 422.8,109.7 423.2,83.9 423.7,118.8 423.8,118.5 424.1,120.9 424.9,116.0 425.3,121.2 425.5,121.2 426.0,118.5 426.7,119.1 426.9,101.5 427.3,83.9 428.0,119.8 428.2,120.9 428.7,116.4 428.9,116.0 429.3,121.2 429.7,120.5 430.5,118.1 430.8,119.1 431.3,83.9 431.5,92.7 432.2,120.9 432.4,120.9 433.0,116.0 433.3,121.2 433.9,119.1 434.3,119.5 434.6,118.1 434.9,118.5 435.4,83.9 435.7,110.0 436.2,120.9 437.0,116.0 437.4,121.2 437.5,121.2 438.1,118.5 438.4,119.5 439.1,101.5 439.4,83.9 439.8,118.8 440.3,120.9 440.8,117.8 441.0,116.0 441.4,121.2 441.8,120.9 442.1,118.5 442.9,119.1 443.4,92.7 443.4,83.9 444.2,120.2 444.3,120.9 445.1,116.0 445.1,117.4 445.5,121.2 446.0,119.8 446.7,118.1 446.9,119.1 447.5,83.9 447.7,100.9 448.4,120.9 448.5,120.9 449.1,116.0 449.5,121.2 450.2,118.5 450.5,119.5 451.1,109.7 451.5,83.9 451.9,118.8 452.0,118.1 452.4,120.9 453.1,116.0 453.5,121.2 453.7,121.2 454.3,118.5 454.5,119.5 455.2,101.5 455.6,83.9 456.2,119.1 456.4,120.9 457.0,116.4 457.2,116.0 457.6,121.2 458.0,120.9 458.3,118.5 459.1,119.1 459.6,83.9 459.6,83.9 460.5,120.9 460.6,120.9 461.2,116.0 461.4,118.8 461.6,121.2 462.6,119.5 462.8,118.1 463.1,119.1 463.6,83.9 463.9,100.9 464.5,120.9 465.3,116.0 465.6,120.9 465.7,121.2 466.4,118.5 466.6,119.5 467.2,109.7 467.7,83.9 468.1,118.8 468.2,118.5 468.6,120.9 469.3,116.0 469.7,121.2 470.0,121.2 470.4,118.5 471.2,119.1 471.4,101.5 471.7,83.9 472.4,119.8 472.6,120.9 473.2,116.4 473.3,116.0 473.7,121.2 474.2,120.5 474.9,118.1 475.2,119.1 475.8,83.9 475.9,92.7 476.6,120.9 476.8,120.9 477.4,116.0 477.8,121.2 478.4,119.1 478.8,119.5 479.0,118.1 479.3,118.5 479.8,83.9 480.2,110.0 480.7,120.9 481.4,116.0 481.8,121.2 481.9,121.2 482.6,117.5 482.8,118.9 483.2,121.6 483.6,121.2 484.1,118.5 484.8,119.1 485.0,101.5 485.4,83.9 486.1,119.8 486.3,120.9 486.8,116.4 487.0,116.0 487.4,121.2 487.8,120.5 488.6,118.1 488.9,119.1 489.4,83.9 489.6,92.7 490.3,120.9 490.5,120.9 491.1,116.0 491.4,121.2 492.0,119.1 492.4,119.5 492.7,118.1 493.0,118.5 493.5,83.9 493.8,110.0 494.3,120.9 495.1,116.0 495.5,121.2 495.6,121.2 496.2,118.5 496.5,119.5 497.2,101.5 497.5,83.9 497.9,118.8 498.4,120.9 498.9,117.8 499.1,116.0 499.5,121.2 499.9,120.9 500.2,118.5 501.0,119.1 501.5,92.7 501.5,83.9 502.3,120.2 502.4,120.9 503.2,116.0 503.2,117.4 503.6,121.2 504.1,119.8 504.8,118.1 505.0,119.1 505.6,83.9 505.8,100.9 506.5,120.9 506.6,120.9 507.2,116.0 507.6,121.2 508.3,118.5 508.6,119.5 509.1,109.7 509.6,83.9 510.0,118.8 510.1,118.1 510.5,120.9 511.2,116.0 511.6,121.2 511.8,121.2 512.4,118.5 512.6,119.5 513.3,101.5 513.7,83.9 514.3,119.1 514.5,120.9 515.1,116.4 515.3,116.0 515.7,121.2 516.1,120.9 516.4,118.5 517.1,119.1 517.7,83.9 517.7,83.9 518.6,120.9 518.6,120.9 519.3,116.0 519.5,118.8 519.7,121.2 520.7,119.5 520.9,118.1 521.2,119.1 521.7,83.9 522.0,100.9 522.6,120.9 523.4,116.0 523.7,120.9 523.7,121.2 524.5,118.5 524.7,119.5 525.3,109.7 525.8,83.9 526.2,118.8 526.3,118.5 526.6,120.9 527.4,116.0 527.8,121.2 528.0,121.2 528.5,118.5 529.3,119.1 529.5,101.5 529.8,83.9 530.5,119.8 530.7,120.9 531.3,116.4 531.4,116.0 531.8,121.2 532.3,120.5 533.0,118.1 533.3,119.1 533.8,83.9 534.0,92.7 534.7,120.9 534.9,120.9 535.5,116.0 535.9,121.2 536.5,119.1 536.8,119.5 537.1,118.1 537.4,118.5 537.9,83.9 538.2,110.0 538.8,120.9 539.5,116.0 539.9,121.2 540.0,121.2 540.6,118.5 540.9,119.5 541.6,101.5 541.9,83.9 542.3,118.8 542.8,120.9 543.3,117.8 543.6,116.0 543.9,121.2 544.4,120.9 544.7,118.5 545.4,119.1 545.9,92.7 546.0,83.9 546.7,120.2 546.8,120.9 547.6,116.0 547.6,117.4 548.0,121.2 548.5,119.8 549.2,118.1 549.5,119.1 550.0,83.9 550.3,100.9 550.9,120.9 551.1,120.9 551.6,116.0 552.0,121.2 552.7,118.5 553.0,119.5 553.6,109.7 554.0,83.9 554.5,118.8 554.5,118.1 554.9,120.9 555.7,116.0 556.1,121.2 556.2,121.2 556.8,118.5 557.0,119.5 557.7,101.5 558.1,83.9 558.7,119.1 559.0,120.9 559.5,116.4 559.7,116.0 560.1,121.2 560.5,120.9 560.8,118.5 561.6,119.1 562.1,83.9 562.2,83.9 563.0,120.9 563.1,120.9 563.7,116.0 563.9,118.8 564.1,121.2 565.1,119.5 565.4,118.1 565.6,119.1 566.2,83.9 566.4,100.9 567.0,120.9 567.8,116.0 568.1,120.9 568.2,121.2 568.9,118.5 569.2,119.5 569.7,109.7 570.2,83.9 570.6,118.8 570.7,118.5 571.1,120.9 571.8,116.0 572.2,121.2 572.5,121.2 572.9,118.5 573.7,119.1 573.9,101.5 574.2,83.9 575.0,119.8 575.1,120.9 575.7,116.4 575.9,116.0 576.3,121.2 576.7,120.5 577.5,118.1 577.7,119.1 578.3,83.9 578.4,92.7 579.2,120.9 579.3,120.9 579.9,116.0 580.3,121.2 580.9,119.1 581.3,119.5 581.5,118.1 581.8,118.5 582.3,83.9 582.7,110.0 583.2,120.9 583.9,116.0 584.3,121.2 584.4,121.2 585.1,118.5 585.3,119.5 586.0,101.5 586.4,83.9 586.8,118.8 587.2,120.9 587.8,117.8 588.0,116.0 588.4,121.2 588.8,120.9 589.1,118.5 589.8,119.1 590.3,92.7 590.4,83.9 591.2,120.2 591.3,120.9 592.0,116.0 592.1,117.4 592.4,121.2 592.9,119.8 593.6,118.1 593.9,119.1 594.4,83.9 594.7,100.9 595.3,120.9 595.5,120.9 596.1,116.0 596.4,121.2 597.2,118.5 597.4,119.5 598.0,109.7 598.5,83.9 598.9,118.8 598.9,118.1 599.3,120.9 600.1,116.0 600.5,121.2 600.7,121.2 601.2,118.5 601.5,119.5 602.2,101.5 602.5,83.9 603.1,119.1 603.4,120.9 604.0,116.4 604.1,116.0 604.5,121.2 604.9,120.9 605.2,118.5 606.0,119.1 606.5,83.9 606.6,83.9 607.4,120.9 607.5,120.9 608.2,116.0 608.3,118.8 608.6,121.2 609.5,119.5 609.8,118.1 610.0,119.1 610.6,83.9 610.9,100.9 611.5,120.9 612.2,116.0 612.6,120.9 612.6,121.2 613.4,117.5 613.6,118.9 614.0,121.6 614.0,178.0" style="fill: var(--fig-3)" fill-opacity=".7"/>
  <polyline points="352.0,121.6 352.8,119.8 352.9,119.8 353.7,118.1 353.9,119.1 354.5,92.7 354.6,83.9 355.4,120.2 355.5,120.9 356.2,116.0 356.3,117.4 356.6,121.2 357.1,119.8 357.8,118.1 358.1,119.1 358.6,83.9 358.9,100.9 359.5,120.9 359.7,120.9 360.3,116.0 360.6,121.2 361.4,118.5 361.6,119.5 362.2,109.7 362.7,83.9 363.1,118.8 363.1,118.1 363.5,120.9 364.3,116.0 364.7,121.2 364.9,121.2 365.4,118.5 365.7,119.5 366.4,101.5 366.7,83.9 367.3,119.1 367.6,120.9 368.2,116.4 368.3,116.0 368.7,121.2 369.1,120.9 369.4,118.5 370.2,119.1 370.7,83.9 370.8,83.9 371.6,120.9 371.7,120.9 372.4,116.0 372.5,118.8 372.8,121.2 373.7,119.5 374.0,118.1 374.2,119.1 374.8,83.9 375.1,100.9 375.7,120.9 376.4,116.0 376.8,120.9 376.8,121.2 377.5,118.5 377.8,119.5 378.4,109.7 378.8,83.9 379.2,118.8 379.3,118.5 379.7,120.9 380.5,116.0 380.8,121.2 381.1,121.2 381.6,118.5 382.3,119.1 382.5,101.5 382.9,83.9 383.6,119.8 383.7,120.9 384.3,116.4 384.5,116.0 384.9,121.2 385.3,120.5 386.1,118.1 386.4,119.1 386.9,83.9 387.0,92.7 387.8,120.9 388.0,120.9 388.5,116.0 388.9,121.2 389.5,119.1 389.9,119.5 390.1,118.1 390.4,118.5 390.9,83.9 391.3,110.0 391.8,120.9 392.6,116.0 393.0,121.2 393.1,121.2 393.7,118.5 393.9,119.5 394.6,101.5 395.0,83.9 395.4,118.8 395.9,120.9 396.4,117.8 396.6,116.0 397.0,121.2 397.4,120.9 397.7,118.5 398.5,119.1 399.0,92.7 399.0,83.9 399.8,120.2 399.9,120.9 400.6,116.0 400.7,117.4 401.0,121.2 401.6,119.8 402.3,118.1 402.5,119.1 403.1,83.9 403.3,100.9 403.9,120.9 404.1,120.9 404.7,116.0 405.1,121.2 405.8,118.5 406.1,119.5 406.6,109.7 407.1,83.9 407.5,118.8 407.6,118.1 408.0,120.9 408.7,116.0 409.1,121.2 409.3,121.2 409.8,118.5 410.1,119.5 410.8,101.5 411.1,83.9 411.8,119.1 412.0,120.9 412.6,116.4 412.8,116.0 413.2,121.2 413.6,120.9 413.9,118.5 414.6,119.1 415.2,83.9 415.2,83.9 416.1,120.9 416.1,120.9 416.8,116.0 416.9,118.8 417.2,121.2 418.2,119.5 418.4,118.1 418.7,119.1 419.2,83.9 419.5,100.9 420.1,120.9 420.8,116.0 421.2,120.9 421.2,121.2 422.0,118.5 422.2,119.5 422.8,109.7 423.2,83.9 423.7,118.8 423.8,118.5 424.1,120.9 424.9,116.0 425.3,121.2 425.5,121.2 426.0,118.5 426.7,119.1 426.9,101.5 427.3,83.9 428.0,119.8 428.2,120.9 428.7,116.4 428.9,116.0 429.3,121.2 429.7,120.5 430.5,118.1 430.8,119.1 431.3,83.9 431.5,92.7 432.2,120.9 432.4,120.9 433.0,116.0 433.3,121.2 433.9,119.1 434.3,119.5 434.6,118.1 434.9,118.5 435.4,83.9 435.7,110.0 436.2,120.9 437.0,116.0 437.4,121.2 437.5,121.2 438.1,118.5 438.4,119.5 439.1,101.5 439.4,83.9 439.8,118.8 440.3,120.9 440.8,117.8 441.0,116.0 441.4,121.2 441.8,120.9 442.1,118.5 442.9,119.1 443.4,92.7 443.4,83.9 444.2,120.2 444.3,120.9 445.1,116.0 445.1,117.4 445.5,121.2 446.0,119.8 446.7,118.1 446.9,119.1 447.5,83.9 447.7,100.9 448.4,120.9 448.5,120.9 449.1,116.0 449.5,121.2 450.2,118.5 450.5,119.5 451.1,109.7 451.5,83.9 451.9,118.8 452.0,118.1 452.4,120.9 453.1,116.0 453.5,121.2 453.7,121.2 454.3,118.5 454.5,119.5 455.2,101.5 455.6,83.9 456.2,119.1 456.4,120.9 457.0,116.4 457.2,116.0 457.6,121.2 458.0,120.9 458.3,118.5 459.1,119.1 459.6,83.9 459.6,83.9 460.5,120.9 460.6,120.9 461.2,116.0 461.4,118.8 461.6,121.2 462.6,119.5 462.8,118.1 463.1,119.1 463.6,83.9 463.9,100.9 464.5,120.9 465.3,116.0 465.6,120.9 465.7,121.2 466.4,118.5 466.6,119.5 467.2,109.7 467.7,83.9 468.1,118.8 468.2,118.5 468.6,120.9 469.3,116.0 469.7,121.2 470.0,121.2 470.4,118.5 471.2,119.1 471.4,101.5 471.7,83.9 472.4,119.8 472.6,120.9 473.2,116.4 473.3,116.0 473.7,121.2 474.2,120.5 474.9,118.1 475.2,119.1 475.8,83.9 475.9,92.7 476.6,120.9 476.8,120.9 477.4,116.0 477.8,121.2 478.4,119.1 478.8,119.5 479.0,118.1 479.3,118.5 479.8,83.9 480.2,110.0 480.7,120.9 481.4,116.0 481.8,121.2 481.9,121.2 482.6,117.5 482.8,118.9 483.2,121.6 483.6,121.2 484.1,118.5 484.8,119.1 485.0,101.5 485.4,83.9 486.1,119.8 486.3,120.9 486.8,116.4 487.0,116.0 487.4,121.2 487.8,120.5 488.6,118.1 488.9,119.1 489.4,83.9 489.6,92.7 490.3,120.9 490.5,120.9 491.1,116.0 491.4,121.2 492.0,119.1 492.4,119.5 492.7,118.1 493.0,118.5 493.5,83.9 493.8,110.0 494.3,120.9 495.1,116.0 495.5,121.2 495.6,121.2 496.2,118.5 496.5,119.5 497.2,101.5 497.5,83.9 497.9,118.8 498.4,120.9 498.9,117.8 499.1,116.0 499.5,121.2 499.9,120.9 500.2,118.5 501.0,119.1 501.5,92.7 501.5,83.9 502.3,120.2 502.4,120.9 503.2,116.0 503.2,117.4 503.6,121.2 504.1,119.8 504.8,118.1 505.0,119.1 505.6,83.9 505.8,100.9 506.5,120.9 506.6,120.9 507.2,116.0 507.6,121.2 508.3,118.5 508.6,119.5 509.1,109.7 509.6,83.9 510.0,118.8 510.1,118.1 510.5,120.9 511.2,116.0 511.6,121.2 511.8,121.2 512.4,118.5 512.6,119.5 513.3,101.5 513.7,83.9 514.3,119.1 514.5,120.9 515.1,116.4 515.3,116.0 515.7,121.2 516.1,120.9 516.4,118.5 517.1,119.1 517.7,83.9 517.7,83.9 518.6,120.9 518.6,120.9 519.3,116.0 519.5,118.8 519.7,121.2 520.7,119.5 520.9,118.1 521.2,119.1 521.7,83.9 522.0,100.9 522.6,120.9 523.4,116.0 523.7,120.9 523.7,121.2 524.5,118.5 524.7,119.5 525.3,109.7 525.8,83.9 526.2,118.8 526.3,118.5 526.6,120.9 527.4,116.0 527.8,121.2 528.0,121.2 528.5,118.5 529.3,119.1 529.5,101.5 529.8,83.9 530.5,119.8 530.7,120.9 531.3,116.4 531.4,116.0 531.8,121.2 532.3,120.5 533.0,118.1 533.3,119.1 533.8,83.9 534.0,92.7 534.7,120.9 534.9,120.9 535.5,116.0 535.9,121.2 536.5,119.1 536.8,119.5 537.1,118.1 537.4,118.5 537.9,83.9 538.2,110.0 538.8,120.9 539.5,116.0 539.9,121.2 540.0,121.2 540.6,118.5 540.9,119.5 541.6,101.5 541.9,83.9 542.3,118.8 542.8,120.9 543.3,117.8 543.6,116.0 543.9,121.2 544.4,120.9 544.7,118.5 545.4,119.1 545.9,92.7 546.0,83.9 546.7,120.2 546.8,120.9 547.6,116.0 547.6,117.4 548.0,121.2 548.5,119.8 549.2,118.1 549.5,119.1 550.0,83.9 550.3,100.9 550.9,120.9 551.1,120.9 551.6,116.0 552.0,121.2 552.7,118.5 553.0,119.5 553.6,109.7 554.0,83.9 554.5,118.8 554.5,118.1 554.9,120.9 555.7,116.0 556.1,121.2 556.2,121.2 556.8,118.5 557.0,119.5 557.7,101.5 558.1,83.9 558.7,119.1 559.0,120.9 559.5,116.4 559.7,116.0 560.1,121.2 560.5,120.9 560.8,118.5 561.6,119.1 562.1,83.9 562.2,83.9 563.0,120.9 563.1,120.9 563.7,116.0 563.9,118.8 564.1,121.2 565.1,119.5 565.4,118.1 565.6,119.1 566.2,83.9 566.4,100.9 567.0,120.9 567.8,116.0 568.1,120.9 568.2,121.2 568.9,118.5 569.2,119.5 569.7,109.7 570.2,83.9 570.6,118.8 570.7,118.5 571.1,120.9 571.8,116.0 572.2,121.2 572.5,121.2 572.9,118.5 573.7,119.1 573.9,101.5 574.2,83.9 575.0,119.8 575.1,120.9 575.7,116.4 575.9,116.0 576.3,121.2 576.7,120.5 577.5,118.1 577.7,119.1 578.3,83.9 578.4,92.7 579.2,120.9 579.3,120.9 579.9,116.0 580.3,121.2 580.9,119.1 581.3,119.5 581.5,118.1 581.8,118.5 582.3,83.9 582.7,110.0 583.2,120.9 583.9,116.0 584.3,121.2 584.4,121.2 585.1,118.5 585.3,119.5 586.0,101.5 586.4,83.9 586.8,118.8 587.2,120.9 587.8,117.8 588.0,116.0 588.4,121.2 588.8,120.9 589.1,118.5 589.8,119.1 590.3,92.7 590.4,83.9 591.2,120.2 591.3,120.9 592.0,116.0 592.1,117.4 592.4,121.2 592.9,119.8 593.6,118.1 593.9,119.1 594.4,83.9 594.7,100.9 595.3,120.9 595.5,120.9 596.1,116.0 596.4,121.2 597.2,118.5 597.4,119.5 598.0,109.7 598.5,83.9 598.9,118.8 598.9,118.1 599.3,120.9 600.1,116.0 600.5,121.2 600.7,121.2 601.2,118.5 601.5,119.5 602.2,101.5 602.5,83.9 603.1,119.1 603.4,120.9 604.0,116.4 604.1,116.0 604.5,121.2 604.9,120.9 605.2,118.5 606.0,119.1 606.5,83.9 606.6,83.9 607.4,120.9 607.5,120.9 608.2,116.0 608.3,118.8 608.6,121.2 609.5,119.5 609.8,118.1 610.0,119.1 610.6,83.9 610.9,100.9 611.5,120.9 612.2,116.0 612.6,120.9 612.6,121.2 613.4,117.5 613.6,118.9 614.0,121.6" fill="none" style="stroke: var(--fig-1)" stroke-width="1.2" stroke-linejoin="round"/>
  <line x1="352" y1="121.6" x2="614" y2="121.6" stroke="currentColor" stroke-opacity=".5" stroke-dasharray="2 3"/>
  <text class="lab2" x="614" y="134.6" text-anchor="end">权重 12.8 GiB</text>
  <g class="m"><title>峰值 21.38 GiB</title><text class="val" x="356" y="56">峰值 21.38 GiB</text></g>
  <line class="axis" x1="352" y1="178" x2="614" y2="178"/>
  <text class="tick" x="614" y="193" text-anchor="end">10121 次分配 / 释放</text>
  <text class="ttl" x="52" y="230">seq 128 · full step</text>
  <line class="grid" x1="52" y1="374.0" x2="314" y2="374.0"/><text class="tick" x="46" y="378.0" text-anchor="end">0</text>
  <line class="grid" x1="52" y1="330.0" x2="314" y2="330.0"/><text class="tick" x="46" y="334.0" text-anchor="end">10</text>
  <line class="grid" x1="52" y1="286.0" x2="314" y2="286.0"/><text class="tick" x="46" y="290.0" text-anchor="end">20</text>
  <line class="grid" x1="52" y1="242.0" x2="314" y2="242.0"/><text class="tick" x="46" y="246.0" text-anchor="end">30</text>
  <polygon points="52.0,374.0 52.0,317.6 52.6,317.3 52.9,317.4 53.6,317.2 53.8,317.3 54.4,316.8 54.6,316.8 55.4,316.5 55.8,316.5 56.1,316.6 56.3,316.6 57.2,315.9 57.3,316.0 58.0,315.8 58.3,315.9 58.8,315.4 59.0,315.4 59.7,315.2 59.8,315.2 60.3,315.0 60.6,315.2 61.5,314.6 61.5,314.6 62.3,314.4 62.9,314.5 63.2,314.3 63.2,314.2 63.9,313.8 64.1,313.8 64.9,313.6 65.1,313.7 65.7,313.2 65.9,313.2 66.7,313.0 67.1,312.9 67.3,313.0 67.5,313.0 68.3,312.4 68.5,312.4 69.2,312.2 69.6,312.3 70.1,311.8 70.2,311.8 71.0,311.6 71.1,311.7 71.6,311.5 71.9,311.6 72.7,311.0 72.7,311.0 73.4,310.8 74.1,310.9 74.4,310.8 74.4,310.8 75.2,310.2 75.3,310.3 76.1,310.0 76.3,310.2 77.0,309.6 77.1,309.7 77.7,309.5 78.4,309.3 78.6,309.4 78.8,309.4 79.6,308.8 79.6,308.8 80.4,308.6 80.9,308.7 81.3,308.3 81.4,308.3 81.9,308.0 82.4,308.1 82.9,307.9 83.1,308.0 83.9,307.5 83.9,307.5 84.7,307.3 85.1,307.2 85.4,307.3 85.7,307.3 86.4,306.6 86.6,306.7 87.4,306.5 87.6,306.6 88.2,306.1 88.3,306.1 89.0,305.9 89.1,305.9 89.6,305.7 90.0,305.9 90.8,305.2 90.9,305.3 91.7,305.1 92.1,305.2 92.6,304.8 92.6,304.8 93.2,304.5 93.6,304.5 94.1,304.3 94.4,304.4 95.0,303.9 95.2,303.9 96.0,303.7 96.4,303.6 96.6,303.7 96.9,303.7 97.7,303.0 97.8,303.1 98.6,302.9 98.9,303.0 99.4,302.5 99.5,302.5 100.2,302.3 100.4,302.4 100.9,302.2 101.3,302.3 102.1,301.7 102.1,301.7 102.9,301.5 103.4,301.6 103.8,301.4 103.8,301.3 104.5,300.9 104.7,300.9 105.4,300.7 105.6,300.9 106.2,300.4 106.4,300.4 107.2,300.1 107.6,300.0 107.9,300.2 108.1,300.2 109.0,299.5 109.1,299.5 109.8,299.3 110.1,299.4 110.7,298.9 110.7,299.0 111.5,298.7 111.6,298.8 112.2,298.6 112.5,298.7 113.3,298.1 113.3,298.1 114.1,298.0 114.7,298.0 115.0,297.9 115.0,297.8 115.7,297.3 115.9,297.4 116.7,297.2 116.9,297.3 117.5,296.8 117.6,296.8 118.5,296.6 118.9,296.5 119.2,296.6 119.4,296.6 120.1,295.9 120.3,296.0 121.0,295.8 121.4,295.9 121.9,295.4 122.0,295.4 122.8,295.2 122.9,295.2 123.4,295.0 123.7,295.2 124.5,294.5 124.5,294.5 125.3,294.3 125.8,294.5 126.0,293.7 126.9,294.1 127.0,293.3 127.2,293.8 127.8,292.8 128.0,293.2 128.8,293.3 129.2,293.1 129.6,293.3 130.0,293.2 130.4,293.3 130.8,293.3 131.2,293.0 131.4,292.9 132.1,293.1 132.4,293.1 132.9,292.0 133.2,292.6 133.3,291.8 134.3,292.3 134.8,292.1 134.9,292.2 135.7,292.3 136.3,292.3 136.5,292.1 136.6,292.2 137.0,291.9 137.9,292.1 138.1,291.3 138.7,291.6 138.8,290.8 139.9,291.3 140.0,291.1 140.3,291.1 140.7,291.3 141.1,291.2 141.5,291.3 141.8,291.3 142.5,290.9 142.6,291.0 143.5,291.1 143.8,290.8 143.9,290.0 144.4,289.8 145.1,290.3 145.4,290.3 145.8,290.1 146.2,290.2 146.7,290.3 147.4,290.3 147.8,290.0 147.9,290.1 148.0,289.9 149.0,290.1 149.5,289.0 149.8,289.6 149.9,288.8 150.9,289.3 151.0,289.1 151.4,289.1 152.0,289.3 152.1,289.2 152.9,289.3 153.0,289.3 153.6,288.9 154.5,289.1 154.7,288.2 154.9,288.8 155.4,287.8 155.6,288.2 156.4,288.3 156.5,288.3 156.9,288.1 157.7,288.2 158.1,288.3 158.4,288.3 158.8,288.0 159.1,287.9 159.8,288.1 160.1,288.1 160.5,287.0 160.8,287.6 161.0,286.8 162.0,287.3 162.4,287.1 162.6,287.2 163.3,287.3 164.0,287.3 164.2,287.1 164.3,287.2 164.6,286.9 165.6,287.1 165.7,286.2 166.4,286.6 166.5,285.8 167.5,286.3 167.6,286.1 168.0,286.1 168.4,286.3 168.7,286.2 169.2,286.3 169.5,286.3 170.2,285.9 170.3,286.0 171.1,286.1 171.1,286.1 171.6,285.0 172.0,284.8 172.7,285.3 173.1,285.3 173.5,285.1 173.8,285.2 174.4,285.3 175.0,285.3 175.4,285.0 175.6,285.1 175.7,284.9 176.7,285.1 177.1,283.9 177.4,284.6 177.6,283.8 178.6,284.3 178.7,284.1 179.0,284.1 179.7,284.2 179.8,284.2 180.6,284.3 180.7,284.3 181.2,283.9 182.2,284.1 182.3,283.2 182.5,283.7 183.1,282.8 183.3,283.2 183.8,283.3 184.1,283.3 184.6,283.1 185.3,283.2 185.8,283.3 186.1,283.3 186.5,283.0 186.8,282.9 187.4,283.1 187.7,283.1 188.2,281.9 188.5,282.5 188.6,281.8 189.7,282.3 190.1,282.1 190.1,282.1 191.0,282.2 191.6,282.3 191.8,282.1 191.9,282.2 192.3,281.9 193.3,282.1 193.4,281.2 193.6,281.7 194.2,280.8 195.2,281.3 195.3,281.1 195.6,281.1 196.0,281.2 196.4,281.2 196.8,281.3 197.2,281.3 197.8,280.9 198.0,281.0 198.5,281.1 198.8,281.1 199.3,279.9 199.7,279.8 200.4,280.3 200.7,280.3 201.2,280.1 201.4,280.2 202.1,280.3 202.7,280.3 203.0,280.0 203.2,280.1 203.4,279.9 204.3,280.1 204.8,278.9 205.1,279.5 205.2,278.8 206.3,279.3 206.4,279.1 206.7,279.0 207.3,279.2 207.5,279.2 208.2,279.3 208.3,279.3 208.9,278.9 209.9,279.1 209.9,278.5 210.2,278.7 210.8,277.8 211.0,278.2 211.5,278.2 211.8,278.3 212.2,278.0 212.6,278.1 213.1,278.3 213.8,278.3 214.2,277.9 214.3,278.1 214.4,277.9 215.4,278.1 215.9,276.9 216.1,277.5 216.3,276.7 217.3,277.3 217.4,277.0 217.8,277.0 218.4,277.2 219.3,277.2 219.4,277.1 219.5,277.1 220.0,276.9 220.9,277.1 221.1,276.2 221.3,276.7 221.8,275.7 222.9,276.3 222.9,276.1 223.3,276.0 223.7,276.2 224.1,276.2 224.5,276.2 224.8,276.2 225.5,275.9 225.5,275.9 226.2,276.0 226.5,276.0 226.9,274.9 227.4,274.7 228.1,275.2 228.4,275.3 228.8,275.0 229.0,275.1 229.7,275.2 230.4,275.2 230.6,275.0 230.9,275.0 231.0,274.9 232.0,275.0 232.1,274.2 232.7,274.5 232.9,273.7 233.9,274.2 234.0,274.0 234.4,274.0 235.0,274.2 235.1,274.2 235.6,274.2 235.9,274.2 236.6,273.8 237.5,274.0 237.6,273.5 237.9,273.7 238.4,272.7 238.6,273.1 239.1,273.2 239.5,273.2 239.9,273.0 240.2,273.1 240.8,273.2 241.4,273.2 241.8,272.9 242.0,273.0 242.1,272.8 243.1,273.0 243.5,271.9 243.8,272.5 244.0,271.7 245.0,272.2 245.1,272.0 245.4,272.0 246.1,272.2 247.0,272.2 247.1,272.1 247.2,272.1 247.6,271.8 248.6,272.0 248.7,271.2 248.9,271.7 249.5,270.7 249.7,271.1 250.5,271.2 251.0,271.0 251.4,271.2 251.7,271.1 252.2,271.2 252.5,271.2 252.9,270.9 253.2,270.8 253.8,271.0 254.1,271.0 254.6,269.9 254.9,270.5 255.0,269.7 256.1,270.2 256.5,270.0 256.6,270.1 257.4,270.2 258.0,270.2 258.2,270.0 258.3,270.1 258.7,269.8 259.7,270.0 259.8,269.2 260.4,269.5 260.6,268.7 261.6,269.2 261.7,269.0 262.0,269.0 262.4,269.2 262.8,269.1 263.2,269.2 263.6,269.2 264.2,268.8 264.4,268.9 265.2,269.0 265.5,268.7 265.7,267.9 266.1,267.7 266.8,268.2 267.1,268.2 267.6,268.0 267.9,268.1 268.5,268.2 269.1,268.2 269.5,267.9 269.6,268.0 269.8,267.8 270.5,267.9 270.7,268.0 271.3,268.0 272.1,268.0 273.0,268.0 273.9,268.0 274.7,268.0 275.3,267.2 275.9,267.5 276.1,266.7 277.1,267.2 277.2,267.0 277.5,267.0 278.2,267.2 278.3,267.1 278.8,267.2 279.1,267.2 279.8,266.8 280.7,267.0 280.8,266.6 281.1,266.7 281.6,265.7 281.6,265.7 282.3,266.2 282.6,266.2 283.1,266.0 283.4,266.1 284.0,266.2 284.6,266.2 285.0,265.9 285.2,266.0 285.3,265.8 286.3,266.0 286.7,264.9 287.0,265.5 287.2,264.7 288.2,265.2 288.3,265.0 288.6,265.0 289.2,265.2 289.4,265.1 290.1,265.2 290.3,265.1 290.8,264.8 291.8,265.0 291.9,264.1 292.1,264.7 292.7,263.7 292.9,264.1 293.7,264.2 294.1,264.0 294.6,264.2 294.9,264.1 295.4,264.2 295.7,264.2 296.1,263.9 296.4,263.8 297.0,264.0 297.3,264.0 297.8,262.9 298.1,263.5 298.2,262.7 299.2,263.2 299.7,263.0 299.8,263.0 300.6,263.2 301.2,263.2 301.4,263.0 301.5,263.1 301.9,262.8 302.9,263.0 303.0,262.1 303.6,262.5 303.8,261.7 304.8,262.2 304.9,262.0 305.2,262.0 305.6,262.1 306.0,262.1 306.5,262.2 306.8,262.2 307.5,261.8 307.6,261.9 308.4,262.0 308.4,262.0 309.3,260.2 309.3,260.2 310.1,256.5 310.2,257.3 311.0,253.9 311.0,254.3 311.6,252.8 311.9,253.0 312.7,253.0 313.2,249.6 313.6,249.7 313.9,248.3 313.9,374.0" style="fill: var(--fig-3)" fill-opacity=".7"/>
  <polyline points="52.0,317.6 52.6,317.3 52.9,317.4 53.6,317.2 53.8,317.3 54.4,316.8 54.6,316.8 55.4,316.5 55.8,316.5 56.1,316.6 56.3,316.6 57.2,315.9 57.3,316.0 58.0,315.8 58.3,315.9 58.8,315.4 59.0,315.4 59.7,315.2 59.8,315.2 60.3,315.0 60.6,315.2 61.5,314.6 61.5,314.6 62.3,314.4 62.9,314.5 63.2,314.3 63.2,314.2 63.9,313.8 64.1,313.8 64.9,313.6 65.1,313.7 65.7,313.2 65.9,313.2 66.7,313.0 67.1,312.9 67.3,313.0 67.5,313.0 68.3,312.4 68.5,312.4 69.2,312.2 69.6,312.3 70.1,311.8 70.2,311.8 71.0,311.6 71.1,311.7 71.6,311.5 71.9,311.6 72.7,311.0 72.7,311.0 73.4,310.8 74.1,310.9 74.4,310.8 74.4,310.8 75.2,310.2 75.3,310.3 76.1,310.0 76.3,310.2 77.0,309.6 77.1,309.7 77.7,309.5 78.4,309.3 78.6,309.4 78.8,309.4 79.6,308.8 79.6,308.8 80.4,308.6 80.9,308.7 81.3,308.3 81.4,308.3 81.9,308.0 82.4,308.1 82.9,307.9 83.1,308.0 83.9,307.5 83.9,307.5 84.7,307.3 85.1,307.2 85.4,307.3 85.7,307.3 86.4,306.6 86.6,306.7 87.4,306.5 87.6,306.6 88.2,306.1 88.3,306.1 89.0,305.9 89.1,305.9 89.6,305.7 90.0,305.9 90.8,305.2 90.9,305.3 91.7,305.1 92.1,305.2 92.6,304.8 92.6,304.8 93.2,304.5 93.6,304.5 94.1,304.3 94.4,304.4 95.0,303.9 95.2,303.9 96.0,303.7 96.4,303.6 96.6,303.7 96.9,303.7 97.7,303.0 97.8,303.1 98.6,302.9 98.9,303.0 99.4,302.5 99.5,302.5 100.2,302.3 100.4,302.4 100.9,302.2 101.3,302.3 102.1,301.7 102.1,301.7 102.9,301.5 103.4,301.6 103.8,301.4 103.8,301.3 104.5,300.9 104.7,300.9 105.4,300.7 105.6,300.9 106.2,300.4 106.4,300.4 107.2,300.1 107.6,300.0 107.9,300.2 108.1,300.2 109.0,299.5 109.1,299.5 109.8,299.3 110.1,299.4 110.7,298.9 110.7,299.0 111.5,298.7 111.6,298.8 112.2,298.6 112.5,298.7 113.3,298.1 113.3,298.1 114.1,298.0 114.7,298.0 115.0,297.9 115.0,297.8 115.7,297.3 115.9,297.4 116.7,297.2 116.9,297.3 117.5,296.8 117.6,296.8 118.5,296.6 118.9,296.5 119.2,296.6 119.4,296.6 120.1,295.9 120.3,296.0 121.0,295.8 121.4,295.9 121.9,295.4 122.0,295.4 122.8,295.2 122.9,295.2 123.4,295.0 123.7,295.2 124.5,294.5 124.5,294.5 125.3,294.3 125.8,294.5 126.0,293.7 126.9,294.1 127.0,293.3 127.2,293.8 127.8,292.8 128.0,293.2 128.8,293.3 129.2,293.1 129.6,293.3 130.0,293.2 130.4,293.3 130.8,293.3 131.2,293.0 131.4,292.9 132.1,293.1 132.4,293.1 132.9,292.0 133.2,292.6 133.3,291.8 134.3,292.3 134.8,292.1 134.9,292.2 135.7,292.3 136.3,292.3 136.5,292.1 136.6,292.2 137.0,291.9 137.9,292.1 138.1,291.3 138.7,291.6 138.8,290.8 139.9,291.3 140.0,291.1 140.3,291.1 140.7,291.3 141.1,291.2 141.5,291.3 141.8,291.3 142.5,290.9 142.6,291.0 143.5,291.1 143.8,290.8 143.9,290.0 144.4,289.8 145.1,290.3 145.4,290.3 145.8,290.1 146.2,290.2 146.7,290.3 147.4,290.3 147.8,290.0 147.9,290.1 148.0,289.9 149.0,290.1 149.5,289.0 149.8,289.6 149.9,288.8 150.9,289.3 151.0,289.1 151.4,289.1 152.0,289.3 152.1,289.2 152.9,289.3 153.0,289.3 153.6,288.9 154.5,289.1 154.7,288.2 154.9,288.8 155.4,287.8 155.6,288.2 156.4,288.3 156.5,288.3 156.9,288.1 157.7,288.2 158.1,288.3 158.4,288.3 158.8,288.0 159.1,287.9 159.8,288.1 160.1,288.1 160.5,287.0 160.8,287.6 161.0,286.8 162.0,287.3 162.4,287.1 162.6,287.2 163.3,287.3 164.0,287.3 164.2,287.1 164.3,287.2 164.6,286.9 165.6,287.1 165.7,286.2 166.4,286.6 166.5,285.8 167.5,286.3 167.6,286.1 168.0,286.1 168.4,286.3 168.7,286.2 169.2,286.3 169.5,286.3 170.2,285.9 170.3,286.0 171.1,286.1 171.1,286.1 171.6,285.0 172.0,284.8 172.7,285.3 173.1,285.3 173.5,285.1 173.8,285.2 174.4,285.3 175.0,285.3 175.4,285.0 175.6,285.1 175.7,284.9 176.7,285.1 177.1,283.9 177.4,284.6 177.6,283.8 178.6,284.3 178.7,284.1 179.0,284.1 179.7,284.2 179.8,284.2 180.6,284.3 180.7,284.3 181.2,283.9 182.2,284.1 182.3,283.2 182.5,283.7 183.1,282.8 183.3,283.2 183.8,283.3 184.1,283.3 184.6,283.1 185.3,283.2 185.8,283.3 186.1,283.3 186.5,283.0 186.8,282.9 187.4,283.1 187.7,283.1 188.2,281.9 188.5,282.5 188.6,281.8 189.7,282.3 190.1,282.1 190.1,282.1 191.0,282.2 191.6,282.3 191.8,282.1 191.9,282.2 192.3,281.9 193.3,282.1 193.4,281.2 193.6,281.7 194.2,280.8 195.2,281.3 195.3,281.1 195.6,281.1 196.0,281.2 196.4,281.2 196.8,281.3 197.2,281.3 197.8,280.9 198.0,281.0 198.5,281.1 198.8,281.1 199.3,279.9 199.7,279.8 200.4,280.3 200.7,280.3 201.2,280.1 201.4,280.2 202.1,280.3 202.7,280.3 203.0,280.0 203.2,280.1 203.4,279.9 204.3,280.1 204.8,278.9 205.1,279.5 205.2,278.8 206.3,279.3 206.4,279.1 206.7,279.0 207.3,279.2 207.5,279.2 208.2,279.3 208.3,279.3 208.9,278.9 209.9,279.1 209.9,278.5 210.2,278.7 210.8,277.8 211.0,278.2 211.5,278.2 211.8,278.3 212.2,278.0 212.6,278.1 213.1,278.3 213.8,278.3 214.2,277.9 214.3,278.1 214.4,277.9 215.4,278.1 215.9,276.9 216.1,277.5 216.3,276.7 217.3,277.3 217.4,277.0 217.8,277.0 218.4,277.2 219.3,277.2 219.4,277.1 219.5,277.1 220.0,276.9 220.9,277.1 221.1,276.2 221.3,276.7 221.8,275.7 222.9,276.3 222.9,276.1 223.3,276.0 223.7,276.2 224.1,276.2 224.5,276.2 224.8,276.2 225.5,275.9 225.5,275.9 226.2,276.0 226.5,276.0 226.9,274.9 227.4,274.7 228.1,275.2 228.4,275.3 228.8,275.0 229.0,275.1 229.7,275.2 230.4,275.2 230.6,275.0 230.9,275.0 231.0,274.9 232.0,275.0 232.1,274.2 232.7,274.5 232.9,273.7 233.9,274.2 234.0,274.0 234.4,274.0 235.0,274.2 235.1,274.2 235.6,274.2 235.9,274.2 236.6,273.8 237.5,274.0 237.6,273.5 237.9,273.7 238.4,272.7 238.6,273.1 239.1,273.2 239.5,273.2 239.9,273.0 240.2,273.1 240.8,273.2 241.4,273.2 241.8,272.9 242.0,273.0 242.1,272.8 243.1,273.0 243.5,271.9 243.8,272.5 244.0,271.7 245.0,272.2 245.1,272.0 245.4,272.0 246.1,272.2 247.0,272.2 247.1,272.1 247.2,272.1 247.6,271.8 248.6,272.0 248.7,271.2 248.9,271.7 249.5,270.7 249.7,271.1 250.5,271.2 251.0,271.0 251.4,271.2 251.7,271.1 252.2,271.2 252.5,271.2 252.9,270.9 253.2,270.8 253.8,271.0 254.1,271.0 254.6,269.9 254.9,270.5 255.0,269.7 256.1,270.2 256.5,270.0 256.6,270.1 257.4,270.2 258.0,270.2 258.2,270.0 258.3,270.1 258.7,269.8 259.7,270.0 259.8,269.2 260.4,269.5 260.6,268.7 261.6,269.2 261.7,269.0 262.0,269.0 262.4,269.2 262.8,269.1 263.2,269.2 263.6,269.2 264.2,268.8 264.4,268.9 265.2,269.0 265.5,268.7 265.7,267.9 266.1,267.7 266.8,268.2 267.1,268.2 267.6,268.0 267.9,268.1 268.5,268.2 269.1,268.2 269.5,267.9 269.6,268.0 269.8,267.8 270.5,267.9 270.7,268.0 271.3,268.0 272.1,268.0 273.0,268.0 273.9,268.0 274.7,268.0 275.3,267.2 275.9,267.5 276.1,266.7 277.1,267.2 277.2,267.0 277.5,267.0 278.2,267.2 278.3,267.1 278.8,267.2 279.1,267.2 279.8,266.8 280.7,267.0 280.8,266.6 281.1,266.7 281.6,265.7 281.6,265.7 282.3,266.2 282.6,266.2 283.1,266.0 283.4,266.1 284.0,266.2 284.6,266.2 285.0,265.9 285.2,266.0 285.3,265.8 286.3,266.0 286.7,264.9 287.0,265.5 287.2,264.7 288.2,265.2 288.3,265.0 288.6,265.0 289.2,265.2 289.4,265.1 290.1,265.2 290.3,265.1 290.8,264.8 291.8,265.0 291.9,264.1 292.1,264.7 292.7,263.7 292.9,264.1 293.7,264.2 294.1,264.0 294.6,264.2 294.9,264.1 295.4,264.2 295.7,264.2 296.1,263.9 296.4,263.8 297.0,264.0 297.3,264.0 297.8,262.9 298.1,263.5 298.2,262.7 299.2,263.2 299.7,263.0 299.8,263.0 300.6,263.2 301.2,263.2 301.4,263.0 301.5,263.1 301.9,262.8 302.9,263.0 303.0,262.1 303.6,262.5 303.8,261.7 304.8,262.2 304.9,262.0 305.2,262.0 305.6,262.1 306.0,262.1 306.5,262.2 306.8,262.2 307.5,261.8 307.6,261.9 308.4,262.0 308.4,262.0 309.3,260.2 309.3,260.2 310.1,256.5 310.2,257.3 311.0,253.9 311.0,254.3 311.6,252.8 311.9,253.0 312.7,253.0 313.2,249.6 313.6,249.7 313.9,248.3" fill="none" style="stroke: var(--fig-1)" stroke-width="1.2" stroke-linejoin="round"/>
  <line x1="52" y1="317.6" x2="314" y2="317.6" stroke="currentColor" stroke-opacity=".5" stroke-dasharray="2 3"/>
  <text class="lab2" x="314" y="330.6" text-anchor="end">权重 12.8 GiB</text>
  <g class="m"><title>峰值 28.57 GiB</title><text class="val" x="56" y="252">峰值 28.57 GiB</text></g>
  <line x1="125.0" y1="242" x2="125.0" y2="374" style="stroke: var(--fig-hi)" stroke-width="1.2" stroke-dasharray="4 3"/>
  <text class="lab2" x="128.0" y="368" text-anchor="start">反向</text>
  <line x1="308.9" y1="242" x2="308.9" y2="374" style="stroke: var(--fig-hi)" stroke-width="1.2" stroke-dasharray="4 3"/>
  <text class="lab2" x="305.9" y="368" text-anchor="end">optimizer</text>
  <text x="310.9" y="243.3" text-anchor="end" font-size="11" font-weight="600" style="fill: var(--fig-hi)">OOM</text>
  <line class="axis" x1="52" y1="374" x2="314" y2="374"/>
  <text class="tick" x="314" y="389" text-anchor="end">13354 次分配 / 释放</text>
  <text class="ttl" x="352" y="230">seq 2048 · 前向 + 反向</text>
  <line class="grid" x1="352" y1="374.0" x2="614" y2="374.0"/><text class="tick" x="346" y="378.0" text-anchor="end">0</text>
  <line class="grid" x1="352" y1="330.0" x2="614" y2="330.0"/><text class="tick" x="346" y="334.0" text-anchor="end">10</text>
  <line class="grid" x1="352" y1="286.0" x2="614" y2="286.0"/><text class="tick" x="346" y="290.0" text-anchor="end">20</text>
  <line class="grid" x1="352" y1="242.0" x2="614" y2="242.0"/><text class="tick" x="346" y="246.0" text-anchor="end">30</text>
  <polygon points="352.0,374.0 352.0,317.6 353.4,317.6 354.7,317.6 356.1,317.6 357.4,317.6 358.8,317.6 360.1,317.6 361.5,317.2 362.8,317.2 364.2,316.9 365.5,316.9 366.9,316.9 368.2,317.2 369.6,317.2 370.9,317.2 372.3,317.2 373.6,317.2 375.0,317.2 376.3,317.2 377.7,316.9 379.0,316.9 380.4,316.6 381.7,316.6 383.1,316.2 384.4,316.2 385.8,316.2 387.1,315.8 388.5,315.8 389.8,315.5 391.2,315.5 392.5,315.5 393.9,315.5 395.2,315.5 396.6,315.1 397.9,315.1 399.3,314.8 400.6,314.8 402.0,315.1 403.3,314.8 404.7,314.8 406.0,314.5 407.4,314.5 408.7,314.1 410.1,314.1 411.4,314.5 412.8,314.5 414.1,314.8 415.5,314.8 416.8,315.1 418.2,315.1 419.5,315.1 420.9,314.8 422.2,314.5 423.6,314.5 424.9,314.8 426.3,314.5 427.6,314.1 429.0,314.1 430.3,313.8 431.7,313.8 433.0,314.1 434.4,314.1 435.7,314.5 437.1,314.5 438.4,314.8 439.8,314.5 441.1,314.1 442.5,314.1 443.8,305.3 445.2,305.3 446.5,296.5 447.9,296.5 449.2,305.3 450.6,296.5 451.9,296.5 453.3,305.3 454.6,305.3 456.0,305.3 457.3,296.5 458.7,296.5 460.0,287.7 461.4,287.7 462.7,287.7 464.1,287.7 465.4,278.9 466.8,278.9 468.1,278.9 469.5,278.9 470.8,287.7 472.2,287.4 473.5,287.0 474.9,287.0 476.2,295.8 477.6,295.5 478.9,295.1 480.3,295.1 481.6,295.5 483.0,295.5 484.4,295.8 485.7,295.8 487.1,296.2 488.4,296.2 489.8,296.5 491.1,296.5 492.5,296.8 493.8,296.8 495.2,297.2 496.5,296.8 497.9,296.8 499.2,297.2 500.6,296.8 501.9,296.8 503.3,296.8 504.6,297.2 506.0,297.2 507.3,297.2 508.7,297.2 510.0,297.2 511.4,297.2 512.7,297.2 514.1,296.8 515.4,296.5 516.8,295.1 518.1,293.7 519.5,292.4 520.8,291.0 522.2,289.6 523.5,289.3 524.9,288.9 526.2,288.9 527.6,289.3 528.9,288.9 530.3,288.9 531.6,288.9 533.0,289.3 534.3,289.3 535.7,289.3 537.0,289.3 538.4,289.3 539.7,289.3 541.1,289.3 542.4,288.9 543.8,288.6 545.1,288.2 546.5,287.9 547.8,287.6 549.2,287.6 550.5,287.6 551.9,287.2 553.2,286.9 554.6,286.9 555.9,287.2 557.3,286.9 558.6,286.5 560.0,286.2 561.3,286.2 562.7,286.5 564.0,286.5 565.4,286.9 566.7,286.9 568.1,287.2 569.4,287.2 570.8,287.2 572.1,286.9 573.5,286.5 574.8,286.5 576.2,286.9 577.5,286.5 578.9,286.2 580.2,285.8 581.6,285.8 582.9,286.2 584.3,286.2 585.6,286.5 587.0,286.5 588.3,286.9 589.7,286.5 591.0,286.2 592.4,286.2 593.7,277.4 595.1,277.4 596.4,268.6 597.8,268.6 599.1,277.4 600.5,268.6 601.8,268.6 603.2,277.4 604.5,277.4 605.9,277.4 607.2,277.4 608.6,268.6 609.9,268.6 611.3,259.8 612.6,259.8 614.0,259.8 614.0,374.0" style="fill: var(--fig-3)" fill-opacity=".7"/>
  <polyline points="352.0,317.6 353.4,317.6 354.7,317.6 356.1,317.6 357.4,317.6 358.8,317.6 360.1,317.6 361.5,317.2 362.8,317.2 364.2,316.9 365.5,316.9 366.9,316.9 368.2,317.2 369.6,317.2 370.9,317.2 372.3,317.2 373.6,317.2 375.0,317.2 376.3,317.2 377.7,316.9 379.0,316.9 380.4,316.6 381.7,316.6 383.1,316.2 384.4,316.2 385.8,316.2 387.1,315.8 388.5,315.8 389.8,315.5 391.2,315.5 392.5,315.5 393.9,315.5 395.2,315.5 396.6,315.1 397.9,315.1 399.3,314.8 400.6,314.8 402.0,315.1 403.3,314.8 404.7,314.8 406.0,314.5 407.4,314.5 408.7,314.1 410.1,314.1 411.4,314.5 412.8,314.5 414.1,314.8 415.5,314.8 416.8,315.1 418.2,315.1 419.5,315.1 420.9,314.8 422.2,314.5 423.6,314.5 424.9,314.8 426.3,314.5 427.6,314.1 429.0,314.1 430.3,313.8 431.7,313.8 433.0,314.1 434.4,314.1 435.7,314.5 437.1,314.5 438.4,314.8 439.8,314.5 441.1,314.1 442.5,314.1 443.8,305.3 445.2,305.3 446.5,296.5 447.9,296.5 449.2,305.3 450.6,296.5 451.9,296.5 453.3,305.3 454.6,305.3 456.0,305.3 457.3,296.5 458.7,296.5 460.0,287.7 461.4,287.7 462.7,287.7 464.1,287.7 465.4,278.9 466.8,278.9 468.1,278.9 469.5,278.9 470.8,287.7 472.2,287.4 473.5,287.0 474.9,287.0 476.2,295.8 477.6,295.5 478.9,295.1 480.3,295.1 481.6,295.5 483.0,295.5 484.4,295.8 485.7,295.8 487.1,296.2 488.4,296.2 489.8,296.5 491.1,296.5 492.5,296.8 493.8,296.8 495.2,297.2 496.5,296.8 497.9,296.8 499.2,297.2 500.6,296.8 501.9,296.8 503.3,296.8 504.6,297.2 506.0,297.2 507.3,297.2 508.7,297.2 510.0,297.2 511.4,297.2 512.7,297.2 514.1,296.8 515.4,296.5 516.8,295.1 518.1,293.7 519.5,292.4 520.8,291.0 522.2,289.6 523.5,289.3 524.9,288.9 526.2,288.9 527.6,289.3 528.9,288.9 530.3,288.9 531.6,288.9 533.0,289.3 534.3,289.3 535.7,289.3 537.0,289.3 538.4,289.3 539.7,289.3 541.1,289.3 542.4,288.9 543.8,288.6 545.1,288.2 546.5,287.9 547.8,287.6 549.2,287.6 550.5,287.6 551.9,287.2 553.2,286.9 554.6,286.9 555.9,287.2 557.3,286.9 558.6,286.5 560.0,286.2 561.3,286.2 562.7,286.5 564.0,286.5 565.4,286.9 566.7,286.9 568.1,287.2 569.4,287.2 570.8,287.2 572.1,286.9 573.5,286.5 574.8,286.5 576.2,286.9 577.5,286.5 578.9,286.2 580.2,285.8 581.6,285.8 582.9,286.2 584.3,286.2 585.6,286.5 587.0,286.5 588.3,286.9 589.7,286.5 591.0,286.2 592.4,286.2 593.7,277.4 595.1,277.4 596.4,268.6 597.8,268.6 599.1,277.4 600.5,268.6 601.8,268.6 603.2,277.4 604.5,277.4 605.9,277.4 607.2,277.4 608.6,268.6 609.9,268.6 611.3,259.8 612.6,259.8 614.0,259.8" fill="none" style="stroke: var(--fig-1)" stroke-width="1.2" stroke-linejoin="round"/>
  <line x1="352" y1="317.6" x2="614" y2="317.6" stroke="currentColor" stroke-opacity=".5" stroke-dasharray="2 3"/>
  <text class="lab2" x="614" y="330.6" text-anchor="end">权重 12.8 GiB</text>
  <g class="m"><title>峰值 25.96 GiB</title><text class="val" x="356" y="252">峰值 25.96 GiB</text></g>
  <text x="611.0" y="254.8" text-anchor="end" font-size="11" font-weight="600" style="fill: var(--fig-hi)">OOM</text>
  <line class="axis" x1="352" y1="374" x2="614" y2="374"/>
  <text class="tick" x="614" y="389" text-anchor="end">195 次分配 / 释放</text>
  <text class="lab2" x="12" y="210" transform="rotate(-90 12 210)" text-anchor="middle">显存（GiB）</text>
</svg>
<figcaption><strong>图 4-6</strong> xl（batch 4，32 头）一步的显存时间线，横轴是分配 / 释放的次序。</figcaption>
</figure>

---

## 5 能省吗：bf16 和 checkpoint 都差一口气 {#savings}

要减小 A 有两种办法：把每个张量存得小一点（bf16），或者少存一些、反向时重算（checkpoint）。

### 5.1 bf16 autocast {#bf16}

autocast 只把矩阵乘的输入换成 bf16，权重、梯度和 Adam 状态还是 fp32。速度提升很明显，前向快了 1.9–2.3 倍（[图 5-1](#fig-5-1)）：矩阵乘换到 bf16 的 Tensor core 上，峰值从 1.05e14 翻倍到 2.1e14，要搬的字节也少了一半。显存只省了 18–21%：W、G 和 Adam 状态大小不变；A 也没有减半，因为 norm、softmax、残差和 loss 还在 fp32 下算（这些累加在 bf16 下不准，bf16 只有 7 位尾数，把 0.01 累加 1000 次只能得到 4.0），反向还要多存一份 bf16 的权重副本。存下的张量还是那些，只是一部分从 4 字节变成了 2 字节。

<figure id="fig-5-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 252" width="100%" role="img" aria-label="bf16 autocast 相对 fp32：前向快 1.87 到 2.30 倍，反向快 1.69 到 1.87 倍；前向 + 反向的峰值显存少 18% 到 21%">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <rect x="64" y="11" width="10" height="10" rx="2" style="fill: var(--fig-1)"/>
  <text class="lab" x="80" y="20">前向</text>
  <rect x="126" y="11" width="10" height="10" rx="2" style="fill: var(--fig-2)"/>
  <text class="lab" x="142" y="20">反向</text>
  <text class="ttl" x="64" y="46">加速比（bf16 相对 fp32）</text>
  <line class="grid" x1="64" y1="226" x2="300" y2="226"/><text class="tick" x="56" y="230" text-anchor="end">0×</text>
  <line class="ref" x1="64" y1="162" x2="300" y2="162"/><text class="tick" x="56" y="166" text-anchor="end">1×</text>
  <line class="grid" x1="64" y1="98" x2="300" y2="98"/><text class="tick" x="56" y="102" text-anchor="end">2×</text>
  <g class="m"><title>small 前向：1.87×</title><rect x="78.0" y="106.3" width="24.0" height="119.7" rx="3" style="fill: var(--fig-1)"/></g>
  <text class="val" x="90" y="101.3" text-anchor="middle">1.87</text>
  <g class="m"><title>small 反向：1.69×</title><rect x="106.0" y="117.8" width="24.0" height="108.2" rx="3" style="fill: var(--fig-2)"/></g>
  <text class="val" x="118" y="112.8" text-anchor="middle">1.69</text>
  <text class="lab" x="104" y="243" text-anchor="middle">small</text>
  <g class="m"><title>medium 前向：2.05×</title><rect x="156.0" y="94.8" width="24.0" height="131.2" rx="3" style="fill: var(--fig-1)"/></g>
  <text class="val" x="168" y="89.8" text-anchor="middle">2.05</text>
  <g class="m"><title>medium 反向：1.79×</title><rect x="184.0" y="111.4" width="24.0" height="114.6" rx="3" style="fill: var(--fig-2)"/></g>
  <text class="val" x="196" y="106.4" text-anchor="middle">1.79</text>
  <text class="lab" x="182" y="243" text-anchor="middle">medium</text>
  <g class="m"><title>large 前向：2.30×</title><rect x="234.0" y="78.8" width="24.0" height="147.2" rx="3" style="fill: var(--fig-1)"/></g>
  <text class="val" x="246" y="73.8" text-anchor="middle">2.30</text>
  <g class="m"><title>large 反向：1.87×</title><rect x="262.0" y="106.3" width="24.0" height="119.7" rx="3" style="fill: var(--fig-2)"/></g>
  <text class="val" x="274" y="101.3" text-anchor="middle">1.87</text>
  <text class="lab" x="260" y="243" text-anchor="middle">large</text>
  <rect x="384" y="11" width="10" height="10" rx="2" style="fill: var(--fig-3); stroke: var(--fig-2)"/>
  <text class="lab" x="400" y="20">fp32</text>
  <rect x="448.4" y="11" width="10" height="10" rx="2" style="fill: var(--fig-1)"/>
  <text class="lab" x="464.4" y="20">bf16</text>
  <text class="ttl" x="384" y="46">前向 + 反向的峰值显存（GiB）</text>
  <line class="grid" x1="384" y1="226" x2="630" y2="226"/><text class="tick" x="376" y="230" text-anchor="end">0</text>
  <line class="grid" x1="384" y1="156" x2="630" y2="156"/><text class="tick" x="376" y="160" text-anchor="end">10</text>
  <line class="grid" x1="384" y1="86" x2="630" y2="86"/><text class="tick" x="376" y="90" text-anchor="end">20</text>
  <g class="m"><title>small fp32：4.08 GiB</title><rect x="400.0" y="197.4" width="24.0" height="28.6" rx="3" style="fill: var(--fig-3); stroke: var(--fig-2)"/></g>
  <g class="m"><title>small bf16：3.18 GiB</title><rect x="428.0" y="203.7" width="24.0" height="22.3" rx="3" style="fill: var(--fig-1)"/></g>
  <text class="val" x="440" y="198.7" text-anchor="middle">−21%</text><text class="lab" x="426" y="243" text-anchor="middle">small</text>
  <g class="m"><title>medium fp32：10.58 GiB</title><rect x="480.0" y="151.9" width="24.0" height="74.1" rx="3" style="fill: var(--fig-3); stroke: var(--fig-2)"/></g>
  <g class="m"><title>medium bf16：8.36 GiB</title><rect x="508.0" y="167.5" width="24.0" height="58.5" rx="3" style="fill: var(--fig-1)"/></g>
  <text class="val" x="520" y="162.5" text-anchor="middle">−21%</text><text class="lab" x="506" y="243" text-anchor="middle">medium</text>
  <g class="m"><title>large fp32：20.28 GiB</title><rect x="560.0" y="84.0" width="24.0" height="142.0" rx="3" style="fill: var(--fig-3); stroke: var(--fig-2)"/></g>
  <g class="m"><title>large bf16：16.61 GiB</title><rect x="588.0" y="109.7" width="24.0" height="116.3" rx="3" style="fill: var(--fig-1)"/></g>
  <text class="val" x="600" y="104.7" text-anchor="middle">−18%</text><text class="lab" x="586" y="243" text-anchor="middle">large</text>
</svg>
<figcaption><strong>图 5-1</strong> bf16 autocast 相对 fp32（前向 + 反向，batch 4，seq 512）：左边是加速比，右边是峰值显存。</figcaption>
</figure>

### 5.2 activation checkpoint {#checkpoint}

checkpoint 和[第 3 节](#rmsnorm)里融合 RMSNorm 的做法一样，只是从一个 op 扩大到几层：前向只存每段的入口（entry，xl、seq 2048 时 80 MiB），反向走到这一段时，用入口把这段的前向重跑一遍，用完就释放。4 层 xl block 每 2 层设一个 checkpoint，峰值就从 4 × 3655 MiB = 14.6 GiB 降到「2 个 entry 加一段」的 7.5 GiB（[图 5-2](#fig-5-2)）。

<figure id="fig-5-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 340" width="100%" role="img" aria-label="4 层 xl block：不 checkpoint 时每层留 3655 MiB 一起活到反向，峰值 14.6 GiB；每 2 层一个 checkpoint 时只留 entry x0、x2，反向时逐段重算，峰值 7.5 GiB">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <defs><marker id="fig-5-2-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <rect class="band" x="4" y="6" width="632" height="150" rx="8"/>
  <text class="ttl" x="16" y="24">全部存下</text><text class="lab2" x="76" y="24">峰值 14.6 GiB</text>
  <rect class="band" x="4" y="168" width="632" height="166" rx="8"/>
  <text class="ttl" x="16" y="186">每 2 层一段</text><text class="lab2" x="115" y="186">峰值 7.5 GiB</text>
  <text class="t" x="16" y="66">x0</text>
  <line x1="36.0" y1="62.0" x2="106.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-5-2-m0)"/>
  <rect x="108.0" y="51.0" width="64" height="30" rx="6" class="op"/>
  <text class="tb" x="140" y="63" text-anchor="middle">L1</text>
  <rect x="90.0" y="94.0" width="100" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="t" x="140" y="111" text-anchor="middle">3655 MiB</text>
  <text class="s" x="140" y="127" text-anchor="middle">含 x0</text>
  <line x1="140.0" y1="81.0" x2="140.0" y2="94.0" style="stroke: var(--fig-2)" stroke-width="1.2"/>
  <rect x="230.0" y="51.0" width="64" height="30" rx="6" class="op"/>
  <text class="tb" x="262" y="63" text-anchor="middle">L2</text>
  <line x1="172.0" y1="62.0" x2="228.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-5-2-m0)"/>
  <rect x="212.0" y="94.0" width="100" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="t" x="262" y="111" text-anchor="middle">3655 MiB</text>
  <text class="s" x="262" y="127" text-anchor="middle">含 x1</text>
  <line x1="262.0" y1="81.0" x2="262.0" y2="94.0" style="stroke: var(--fig-2)" stroke-width="1.2"/>
  <rect x="352.0" y="51.0" width="64" height="30" rx="6" class="op"/>
  <text class="tb" x="384" y="63" text-anchor="middle">L3</text>
  <line x1="294.0" y1="62.0" x2="350.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-5-2-m0)"/>
  <rect x="334.0" y="94.0" width="100" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="t" x="384" y="111" text-anchor="middle">3655 MiB</text>
  <text class="s" x="384" y="127" text-anchor="middle">含 x2</text>
  <line x1="384.0" y1="81.0" x2="384.0" y2="94.0" style="stroke: var(--fig-2)" stroke-width="1.2"/>
  <rect x="474.0" y="51.0" width="64" height="30" rx="6" class="op"/>
  <text class="tb" x="506" y="63" text-anchor="middle">L4</text>
  <line x1="416.0" y1="62.0" x2="472.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-5-2-m0)"/>
  <rect x="456.0" y="94.0" width="100" height="40" rx="6" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="t" x="506" y="111" text-anchor="middle">3655 MiB</text>
  <text class="s" x="506" y="127" text-anchor="middle">含 x3</text>
  <line x1="506.0" y1="81.0" x2="506.0" y2="94.0" style="stroke: var(--fig-2)" stroke-width="1.2"/>
  <line x1="538.0" y1="62.0" x2="600.0" y2="62.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-5-2-m0)"/>
  <text class="t" x="606" y="66">y</text>
  <text class="lab2" x="626" y="148" text-anchor="end">4 份同时留到反向</text>
  <rect x="24.0" y="211.0" width="56" height="30" rx="6" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="t" x="52" y="223" text-anchor="middle">x0</text>
  <text class="lab2" x="52" y="256" text-anchor="middle">entry 80 MiB</text>
  <rect x="308.0" y="211.0" width="56" height="30" rx="6" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text class="t" x="336" y="223" text-anchor="middle">x2</text>
  <text class="lab2" x="336" y="256" text-anchor="middle">entry 80 MiB</text>
  <rect x="120.0" y="209.0" width="152" height="34" rx="6" class="op"/>
  <text class="tb" x="196" y="223" text-anchor="middle">checkpoint［L1 L2］</text>
  <rect x="108.0" y="270.0" width="176" height="40" rx="6" style="fill: var(--fig-3); stroke: var(--fig-1)" stroke-width="1.2" stroke-dasharray="5 3"/>
  <text class="t" x="196" y="287" text-anchor="middle">2 × 3655 MiB</text>
  <text class="s" x="196" y="303" text-anchor="middle">反向时用 x0 重算，用完即丢</text>
  <line x1="196.0" y1="243.0" x2="196.0" y2="270.0" style="stroke: var(--fig-1)" stroke-width="1.2"/>
  <rect x="400.0" y="209.0" width="152" height="34" rx="6" class="op"/>
  <text class="tb" x="476" y="223" text-anchor="middle">checkpoint［L3 L4］</text>
  <rect x="388.0" y="270.0" width="176" height="40" rx="6" style="fill: var(--fig-3); stroke: var(--fig-1)" stroke-width="1.2" stroke-dasharray="5 3"/>
  <text class="t" x="476" y="287" text-anchor="middle">2 × 3655 MiB</text>
  <text class="s" x="476" y="303" text-anchor="middle">反向时用 x2 重算，用完即丢</text>
  <line x1="476.0" y1="243.0" x2="476.0" y2="270.0" style="stroke: var(--fig-1)" stroke-width="1.2"/>
  <line x1="80.0" y1="222.0" x2="118.0" y2="222.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-5-2-m0)"/>
  <line x1="272.0" y1="222.0" x2="306.0" y2="222.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-5-2-m0)"/>
  <line x1="364.0" y1="222.0" x2="398.0" y2="222.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-5-2-m0)"/>
  <line x1="552.0" y1="222.0" x2="600.0" y2="222.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.6" marker-end="url(#fig-5-2-m0)"/>
  <text class="t" x="606" y="226">y</text>
  <text class="lab2" x="626" y="326" text-anchor="end">反向时一次只重算一段</text>
</svg>
<figcaption><strong>图 5-2</strong> 4 层 xl block 有无 checkpoint：钢蓝框一直占到反向，浅蓝虚线框在反向时用 entry 重算、用完即丢。</figcaption>
</figure>

代价是整个网络要多跑一遍前向。xl 在 seq 2048 下光参数加梯度就要 25.4 GiB，放不下 activation，所以我在 large 上扫了每段放几层（[图 5-3](#fig-5-3)）。不管怎么切，step 都是 302–313 ms，比不用 checkpoint 的 236 ms 慢 28–33%，正好对应一步从 3F 变成 4F（F 是一次前向，反向约 2F）。显存则是切得越细越省，每层一个 checkpoint 时最低，7.8 GiB，不用时是 15.0 GiB。

<figure id="fig-5-3" class="fg-fig">
<svg class="fg" viewBox="0 0 640 250" width="100%" role="img" aria-label="checkpoint 段长扫描：step 时间都在 302 到 313 ms，高于不 checkpoint 的 236 ms；峰值显存随每段层数增加，每层一个 checkpoint 时最低 7.8 GiB">
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
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 5px; stroke-linejoin: round; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: 540px; } }
  </style>
  <line x1="64" y1="14" x2="76" y2="14" style="stroke: var(--fig-1)" stroke-width="2"/>
  <text class="lab" x="80" y="18">checkpoint</text>
  <line x1="168.0" y1="14" x2="180.0" y2="14" style="stroke: var(--fig-hi)" stroke-width="2"/>
  <text class="lab" x="184.0" y="18">不 checkpoint</text>
  <text class="ttl" x="64" y="42">step 时间（ms）</text>
  <line class="grid" x1="64" y1="206.0" x2="314" y2="206.0"/><text class="tick" x="58" y="210.0" text-anchor="end">0</text>
  <line class="grid" x1="64" y1="162.0" x2="314" y2="162.0"/><text class="tick" x="58" y="166.0" text-anchor="end">100</text>
  <line class="grid" x1="64" y1="118.0" x2="314" y2="118.0"/><text class="tick" x="58" y="122.0" text-anchor="end">200</text>
  <line class="grid" x1="64" y1="74.0" x2="314" y2="74.0"/><text class="tick" x="58" y="78.0" text-anchor="end">300</text>
  <line x1="64" y1="102.2" x2="314" y2="102.2" style="stroke: var(--fig-hi)" stroke-width="1.6" stroke-dasharray="5 4"/>
  <text class="val" x="314" y="97.2" text-anchor="end">不 checkpoint 236 ms</text>
  <polyline points="64.0,73.1 112.4,71.5 140.6,69.3 160.7,69.4 189.0,68.6 217.3,69.9 237.4,68.6 265.6,68.2 314.0,68.1" fill="none" style="stroke: var(--fig-1)" stroke-width="2"/>
  <g class="m"><title>每段 1 层：302 ms</title><circle cx="64.0" cy="73.1" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 2 层：306 ms</title><circle cx="112.4" cy="71.5" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 3 层：311 ms</title><circle cx="140.6" cy="69.3" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 4 层：311 ms</title><circle cx="160.7" cy="69.4" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 6 层：312 ms</title><circle cx="189.0" cy="68.6" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 9 层：309 ms</title><circle cx="217.3" cy="69.9" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 12 层：312 ms</title><circle cx="237.4" cy="68.6" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 18 层：313 ms</title><circle cx="265.6" cy="68.2" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 36 层：313 ms</title><circle cx="314.0" cy="68.1" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <text class="tick" x="64.0" y="221" text-anchor="middle">1</text>
  <text class="tick" x="112.4" y="221" text-anchor="middle">2</text>
  <text class="tick" x="140.6" y="221" text-anchor="middle">3</text>
  <text class="tick" x="160.7" y="221" text-anchor="middle">4</text>
  <text class="tick" x="189.0" y="221" text-anchor="middle">6</text>
  <text class="tick" x="217.3" y="221" text-anchor="middle">9</text>
  <text class="tick" x="237.4" y="221" text-anchor="middle">12</text>
  <text class="tick" x="265.6" y="221" text-anchor="middle">18</text>
  <text class="tick" x="314.0" y="221" text-anchor="middle">36</text>
  <line class="axis" x1="64" y1="206" x2="314" y2="206"/>
  <text class="lab2" x="189" y="238" text-anchor="middle">每段几层（对数轴）</text>
  <text class="ttl" x="364" y="42">峰值显存（GiB）</text>
  <line class="grid" x1="364" y1="206.0" x2="614" y2="206.0"/><text class="tick" x="358" y="210.0" text-anchor="end">0</text>
  <line class="grid" x1="364" y1="167.5" x2="614" y2="167.5"/><text class="tick" x="358" y="171.5" text-anchor="end">4</text>
  <line class="grid" x1="364" y1="129.0" x2="614" y2="129.0"/><text class="tick" x="358" y="133.0" text-anchor="end">8</text>
  <line class="grid" x1="364" y1="90.5" x2="614" y2="90.5"/><text class="tick" x="358" y="94.5" text-anchor="end">12</text>
  <line class="grid" x1="364" y1="52.0" x2="614" y2="52.0"/><text class="tick" x="358" y="56.0" text-anchor="end">16</text>
  <line x1="364" y1="61.4" x2="614" y2="61.4" style="stroke: var(--fig-hi)" stroke-width="1.6" stroke-dasharray="5 4"/>
  <text class="val" x="614" y="56.4" text-anchor="end">不 checkpoint 15.0 GiB</text>
  <polyline points="364.0,130.7 412.4,128.7 440.6,126.8 460.7,124.8 489.0,120.8 517.3,114.9 537.4,108.9 565.6,97.0 614.0,61.3" fill="none" style="stroke: var(--fig-1)" stroke-width="2"/>
  <g class="m"><title>每段 1 层：7.8 GiB</title><circle cx="364.0" cy="130.7" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 2 层：8.0 GiB</title><circle cx="412.4" cy="128.7" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 3 层：8.2 GiB</title><circle cx="440.6" cy="126.8" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 4 层：8.4 GiB</title><circle cx="460.7" cy="124.8" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 6 层：8.8 GiB</title><circle cx="489.0" cy="120.8" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 9 层：9.5 GiB</title><circle cx="517.3" cy="114.9" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 12 层：10.1 GiB</title><circle cx="537.4" cy="108.9" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 18 层：11.3 GiB</title><circle cx="565.6" cy="97.0" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <g class="m"><title>每段 36 层：15.0 GiB</title><circle cx="614.0" cy="61.3" r="4" class="ring" style="fill: var(--fig-1)"/></g>
  <text class="tick" x="364.0" y="221" text-anchor="middle">1</text>
  <text class="tick" x="412.4" y="221" text-anchor="middle">2</text>
  <text class="tick" x="440.6" y="221" text-anchor="middle">3</text>
  <text class="tick" x="460.7" y="221" text-anchor="middle">4</text>
  <text class="tick" x="489.0" y="221" text-anchor="middle">6</text>
  <text class="tick" x="517.3" y="221" text-anchor="middle">9</text>
  <text class="tick" x="537.4" y="221" text-anchor="middle">12</text>
  <text class="tick" x="565.6" y="221" text-anchor="middle">18</text>
  <text class="tick" x="614.0" y="221" text-anchor="middle">36</text>
  <line class="axis" x1="364" y1="206" x2="614" y2="206"/>
  <text class="lab2" x="489" y="238" text-anchor="middle">每段几层（对数轴）</text>
</svg>
<figcaption><strong>图 5-3</strong> checkpoint 段长扫描（large，batch 1，seq 1024，前向 + 反向，fp32 eager）。</figcaption>
</figure>

切得越细越省，是因为入口很小。设共 $L$ 层、每段 $e$ 层、入口大小 $a$、一层的 A 为 $A_1$，峰值约为 $\frac{L}{e}a + e\,A_1$。只要所有入口加起来比一层的 A 小（这里 36 × 5 MiB = 180 MiB < 220 MiB），$e = 1$ 就是最优。

但 checkpoint 解决不了 S、P。重算到某一层时，这一层的 S、P 照样要完整写进显存再读出来，seq 2048 时每层约 8 GiB 的峰值还在。

---

## 6 小结 {#conclusion}

时间和显存的问题最后都落在 S、P 上。时间上，它们让 attention 成了 memory-bound，seq 1024 时占前向将近一半的时间；显存上，它们占一层 saved tensors 的一半以上，xl 在 seq 2048 时第 2 层就 OOM。bf16 只能把它们存小一点，checkpoint 只能推迟它们出现，都没能让它们离开显存。

要解决，得把 RMSNorm 的做法用到 attention 上：把 QKᵀ、softmax、PV 合进一个 kernel，分块在片上算完，S、P 不写回显存，反向需要时再重算。下一篇 [FlashAttention 1–4](/blog/flashattention-1-to-4/) 讲的就是这个。
