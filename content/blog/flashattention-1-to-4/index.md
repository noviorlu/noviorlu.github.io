---
title: "FlashAttention 1–4：从 IO 出发重写 attention kernel"
date: 2026-10-05
draft: false
math: true
description: "FlashAttention 四代的技术梳理：IO-aware 的 tiling，更好的工作划分，Hopper 的异步执行，再到针对 Blackwell 各硬件单元扩展不对称的协同设计。GPU training optimization 系列的第四篇。"
tags: ["FlashAttention", "Transformer", "Attention", "GPU", "LLM", "AI"]
categories: ["AI笔记"]
series: ["GPU training optimization"]
series_order: 4
---

<style>
.fa-intuit{margin:1.4em 0;padding:.85em 1.1em;border-radius:12px;background:color-mix(in srgb,var(--fig-3) 28%,transparent);border-left:3px solid var(--fig-1);line-height:1.65}
.fa-intuit>b:first-child{color:var(--fig-1)}
.fa-road{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:1.6em 0 .6em}
@media (max-width:760px){.fa-road{grid-template-columns:repeat(2,minmax(0,1fr))}}
.fa-step{border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:.7em .85em;font-size:.82em;line-height:1.5}
.fa-step b{display:block;font-size:1.1em}
.fa-step i{display:block;font-style:normal;opacity:.65;margin-bottom:.5em}
.fa-step span{display:block}
.fa-step span+span{margin-top:.45em;color:var(--fig-1)}
@media (max-width:640px){.article-content figure{overflow-x:auto}.article-content figure>svg{min-width:520px}}
</style>

> **前置阅读：**[GPU 与 Triton 入门](/blog/gpu-triton-intro/)讲 GPU 的存储层级和 Triton 基础；[单卡 GPU 优化方法](/blog/gpu-single-gpu-optimization/)讲单卡上常用的优化方法：divergence、低精度、fusion、recomputation、coalescing、bank conflict、tiling；[谁偷走了 5090 的算力和显存](/blog/gpu-training-analysis/)讲 FLOPs、算术强度、roofline 模型和峰值显存，并在 RTX 5090 上实测了标准 attention 的运行时间和激活显存为什么都由 $N\times N$ 的 score 矩阵主导。这篇就从这两个问题出发。

Transformer 的每一层 self-attention 都要付两份代价：一份是 FLOPs，另一份是访存，后者的代价大得多。FlashAttention 是一族计算精确 attention 的 kernel，专门解决访存这个问题。它的每一代针对的瓶颈都不一样，而这个瓶颈要等上一代的问题解决之后才显露出来。[第 1 节](#fa1)到[第 4 节](#fa4)逐代讲每一代改了什么、为什么在当时的硬件上重要，[第 5 节](#constant)讲四代不变的东西，[第 6 节](#summary)是总结；四代的瓶颈和做法先列在下面。

<div class="fa-road">
<div class="fa-step"><b>v1 · 2022</b><i>A100（Ampere）</i><span>瓶颈：N×N 的 score 矩阵要在慢速的 HBM 里来回读写。</span><span>做法：tiling，把所有步骤融合进一个 kernel，矩阵留在片上。</span></div>
<div class="fa-step"><b>v2 · 2023</b><i>A100（Ampere）</i><span>瓶颈：GPU 大部分时间闲着，thread block 太少，非 matmul 的计算太多。</span><span>做法：沿序列方向并行，最后只做一次归一化。</span></div>
<div class="fa-step"><b>v3 · 2024</b><i>H100（Hopper）</i><span>瓶颈：读 tile 和做计算仍然轮流进行。</span><span>做法：异步的 producer/consumer pipeline，softmax 和 matmul 重叠，FP8。</span></div>
<div class="fa-step"><b>v4 · 2026</b><i>B200（Blackwell）</i><span>瓶颈：Tensor core 的算力翻倍了，exp 单元和 shared memory 没有。</span><span>做法：一部分 exp 用软件算，跳过大部分 rescale，用 TMEM。</span></div>
</div>

原始论文的链接列在文末的 [References](#references)，建议至少直接读一遍第一篇论文，不要只看二手总结（包括这一篇）。

## 1 FlashAttention：不让 N × N 矩阵进 HBM {#fa1}

标准 attention 慢在 $N\times N$ 的 $S$ 和 $P$ 要在 HBM 里来回读写。Dao、Fu、Ermon、Rudra、Ré 2022 年的 FlashAttention 把整个 attention 融合成一个 kernel，用 **tiling** 让这两个矩阵完全不进 HBM，用 **recomputation** 省掉反向要存的 $P$；难点在于让 softmax 也能分块算。

<p align="center">$S = \dfrac{QK^\top}{\sqrt d}, \qquad P = \mathrm{softmax}(S), \qquad O = PV, \qquad Q \in \mathbb{R}^{N_q \times d},\ K, V \in \mathbb{R}^{N_k \times d}$</p>

### 1.1 Tiling {#fa1-tiling}

把 $Q$ 按 $B_q$ 行切成 $T_q$ 个 block $Q_i$，$K$、$V$ 按 $B_k$ 行切成 $T_k$ 个 block $K_j$、$V_j$（$d$ 方向不切），[图 1-1](#fig-1-1) 的格子换成 block，$S_{ij} = Q_iK_j^\top$ 就是 (a) 的外积，$O_i = \sum_j P_{ij}V_j$ 就是 (b) 的列组合。

<figure id="fig-1-1">
<svg viewBox="0 0 680 252" width="100%" role="img" aria-label="(a) 列向量 q 乘行向量 k 转置等于 3 乘 4 的矩阵 S，箭头从 k 的每个元素 k_j 指向 q，S 的第 j 列等于 k_j 乘 q。(b) 3 乘 4 的矩阵 P 乘 4 乘 1 的列向量 v 等于 3 乘 1 的列向量 o，箭头从 v 的每个元素 v_j 指向 P 的第 j 列，o 等于 v_j 乘 P 第 j 列之和。格子换成 block 就是图 1-2：(a) 是 S_ij = Q_i K_j 转置，(b) 是 O_i 等于 P_ij V_j 之和。颜色和图 1-2 一样。">
  <style>
    .os2 { font-size: 11px; fill: currentColor; }
    .os2-b { font-size: 12px; font-weight: 600; fill: currentColor; }
    .os2-s { font-size: 10.5px; fill: currentColor; opacity: .72; }
  </style>
<defs><marker id="fig-1-1-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor" fill-opacity=".6"/></marker><pattern id="fig-1-1-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-1)" stroke-opacity=".55" stroke-width="1.3"/></pattern><pattern id="fig-1-1-hatch2" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-2)" stroke-opacity=".6" stroke-width="1.3"/></pattern><pattern id="fig-1-1-hatch3" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-hi)" stroke-opacity=".6" stroke-width="1.3"/></pattern></defs>
<text class="os2-b" x="8" y="20" text-anchor="start">(a) 向量乘向量 S = q kᵀ：S 的列是 q 的倍数</text>
<rect x="16" y="96" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
<text class="os2" x="28" y="112" text-anchor="middle">q<tspan dy="3.5" font-size="72%">1</tspan></text>
<rect x="16" y="120" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
<text class="os2" x="28" y="136" text-anchor="middle">q<tspan dy="3.5" font-size="72%">2</tspan></text>
<rect x="16" y="144" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
<text class="os2" x="28" y="160" text-anchor="middle">q<tspan dy="3.5" font-size="72%">3</tspan></text>
<rect x="80" y="120" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
<text class="os2" x="92" y="136" text-anchor="middle">k<tspan dy="3.5" font-size="72%">1</tspan></text>
<path d="M92,120 C92,80 19,80 19,95" fill="none" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#fig-1-1-ar)"/>
<rect x="212" y="96" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="224" y="112" text-anchor="middle">s<tspan dy="3.5" font-size="72%">11</tspan></text>
<rect x="212" y="120" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="224" y="136" text-anchor="middle">s<tspan dy="3.5" font-size="72%">21</tspan></text>
<rect x="212" y="144" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="224" y="160" text-anchor="middle">s<tspan dy="3.5" font-size="72%">31</tspan></text>
<text class="os2" x="224" y="88" text-anchor="middle">k<tspan dy="3.5" font-size="72%">1</tspan><tspan dy="-3.5">q</tspan></text>
<rect x="104" y="120" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch); stroke: var(--fig-1)" stroke-width="1"/>
<text class="os2" x="116" y="136" text-anchor="middle">k<tspan dy="3.5" font-size="72%">2</tspan></text>
<path d="M116,120 C116,69 25,69 25,95" fill="none" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#fig-1-1-ar)"/>
<rect x="236" y="96" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="248" y="112" text-anchor="middle">s<tspan dy="3.5" font-size="72%">12</tspan></text>
<rect x="236" y="120" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="248" y="136" text-anchor="middle">s<tspan dy="3.5" font-size="72%">22</tspan></text>
<rect x="236" y="144" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="248" y="160" text-anchor="middle">s<tspan dy="3.5" font-size="72%">32</tspan></text>
<text class="os2" x="248" y="88" text-anchor="middle">k<tspan dy="3.5" font-size="72%">2</tspan><tspan dy="-3.5">q</tspan></text>
<rect x="128" y="120" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch); stroke: var(--fig-1)" stroke-width="1"/>
<text class="os2" x="140" y="136" text-anchor="middle">k<tspan dy="3.5" font-size="72%">3</tspan></text>
<path d="M140,120 C140,58 31,58 31,95" fill="none" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#fig-1-1-ar)"/>
<rect x="260" y="96" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="272" y="112" text-anchor="middle">s<tspan dy="3.5" font-size="72%">13</tspan></text>
<rect x="260" y="120" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="272" y="136" text-anchor="middle">s<tspan dy="3.5" font-size="72%">23</tspan></text>
<rect x="260" y="144" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="272" y="160" text-anchor="middle">s<tspan dy="3.5" font-size="72%">33</tspan></text>
<text class="os2" x="272" y="88" text-anchor="middle">k<tspan dy="3.5" font-size="72%">3</tspan><tspan dy="-3.5">q</tspan></text>
<rect x="152" y="120" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch); stroke: var(--fig-1)" stroke-width="1"/>
<text class="os2" x="164" y="136" text-anchor="middle">k<tspan dy="3.5" font-size="72%">4</tspan></text>
<path d="M164,120 C164,47 37,47 37,95" fill="none" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#fig-1-1-ar)"/>
<rect x="284" y="96" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="296" y="112" text-anchor="middle">s<tspan dy="3.5" font-size="72%">14</tspan></text>
<rect x="284" y="120" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="296" y="136" text-anchor="middle">s<tspan dy="3.5" font-size="72%">24</tspan></text>
<rect x="284" y="144" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="296" y="160" text-anchor="middle">s<tspan dy="3.5" font-size="72%">34</tspan></text>
<text class="os2" x="296" y="88" text-anchor="middle">k<tspan dy="3.5" font-size="72%">4</tspan><tspan dy="-3.5">q</tspan></text>
<text class="os2-b" x="194" y="137" text-anchor="middle">=</text>
<text class="os2-b" x="28" y="184" text-anchor="middle">q</text>
<text class="os2-s" x="28" y="198" text-anchor="middle">3 × 1</text>
<text class="os2-b" x="128" y="160" text-anchor="middle">kᵀ</text>
<text class="os2-s" x="128" y="174" text-anchor="middle">1 × 4</text>
<text class="os2-b" x="260" y="184" text-anchor="middle">S</text>
<text class="os2-s" x="260" y="198" text-anchor="middle">3 × 4</text>
<text class="os2-b" x="16" y="240" text-anchor="start">S 的第 j 列 = k<tspan dy="3.5" font-size="72%">j</tspan><tspan dy="-3.5"> · q</tspan></text>
<text class="os2-b" x="330" y="20" text-anchor="start">(b) 矩阵乘向量 o = P v：o 是 P 各列的组合</text>
<rect x="340" y="96" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="352" y="112" text-anchor="middle">P<tspan dy="3.5" font-size="72%">11</tspan></text>
<rect x="340" y="120" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="352" y="136" text-anchor="middle">P<tspan dy="3.5" font-size="72%">21</tspan></text>
<rect x="340" y="144" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="352" y="160" text-anchor="middle">P<tspan dy="3.5" font-size="72%">31</tspan></text>
<rect x="500" y="84" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
<text class="os2" x="512" y="100" text-anchor="middle">v<tspan dy="3.5" font-size="72%">1</tspan></text>
<path d="M500,96 H490 Q486,96 486,92 V60 Q486,56 482,56 H356 Q352,56 352,60 V95" fill="none" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#fig-1-1-ar)"/>
<rect x="364" y="96" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="376" y="112" text-anchor="middle">P<tspan dy="3.5" font-size="72%">12</tspan></text>
<rect x="364" y="120" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="376" y="136" text-anchor="middle">P<tspan dy="3.5" font-size="72%">22</tspan></text>
<rect x="364" y="144" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="376" y="160" text-anchor="middle">P<tspan dy="3.5" font-size="72%">32</tspan></text>
<rect x="500" y="108" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
<text class="os2" x="512" y="124" text-anchor="middle">v<tspan dy="3.5" font-size="72%">2</tspan></text>
<path d="M500,120 H484 Q480,120 480,116 V68 Q480,64 476,64 H380 Q376,64 376,68 V95" fill="none" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#fig-1-1-ar)"/>
<rect x="388" y="96" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="400" y="112" text-anchor="middle">P<tspan dy="3.5" font-size="72%">13</tspan></text>
<rect x="388" y="120" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="400" y="136" text-anchor="middle">P<tspan dy="3.5" font-size="72%">23</tspan></text>
<rect x="388" y="144" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="400" y="160" text-anchor="middle">P<tspan dy="3.5" font-size="72%">33</tspan></text>
<rect x="500" y="132" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
<text class="os2" x="512" y="148" text-anchor="middle">v<tspan dy="3.5" font-size="72%">3</tspan></text>
<path d="M500,144 H478 Q474,144 474,140 V76 Q474,72 470,72 H404 Q400,72 400,76 V95" fill="none" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#fig-1-1-ar)"/>
<rect x="412" y="96" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="424" y="112" text-anchor="middle">P<tspan dy="3.5" font-size="72%">14</tspan></text>
<rect x="412" y="120" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="424" y="136" text-anchor="middle">P<tspan dy="3.5" font-size="72%">24</tspan></text>
<rect x="412" y="144" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch2); stroke: var(--fig-2)" stroke-width="1"/>
<text class="os2" x="424" y="160" text-anchor="middle">P<tspan dy="3.5" font-size="72%">34</tspan></text>
<rect x="500" y="156" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
<text class="os2" x="512" y="172" text-anchor="middle">v<tspan dy="3.5" font-size="72%">4</tspan></text>
<path d="M500,168 H472 Q468,168 468,164 V84 Q468,80 464,80 H428 Q424,80 424,84 V95" fill="none" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#fig-1-1-ar)"/>
<rect x="572" y="96" width="24" height="24" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
<text class="os2" x="584" y="112" text-anchor="middle">o<tspan dy="3.5" font-size="72%">1</tspan></text>
<rect x="572" y="120" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch3); stroke: var(--fig-hi)" stroke-width="1"/>
<text class="os2" x="584" y="136" text-anchor="middle">o<tspan dy="3.5" font-size="72%">2</tspan></text>
<rect x="572" y="144" width="24" height="24" rx="1.5" style="fill: url(#fig-1-1-hatch3); stroke: var(--fig-hi)" stroke-width="1"/>
<text class="os2" x="584" y="160" text-anchor="middle">o<tspan dy="3.5" font-size="72%">3</tspan></text>
<text class="os2-b" x="452" y="137" text-anchor="middle">×</text>
<text class="os2-b" x="548" y="137" text-anchor="middle">=</text>
<text class="os2-b" x="388" y="190" text-anchor="middle">P</text>
<text class="os2-s" x="388" y="204" text-anchor="middle">3 × 4</text>
<text class="os2-b" x="512" y="198" text-anchor="middle">v</text>
<text class="os2-s" x="512" y="212" text-anchor="middle">4 × 1</text>
<text class="os2-b" x="584" y="190" text-anchor="middle">o</text>
<text class="os2-s" x="584" y="204" text-anchor="middle">3 × 1</text>
<text class="os2-b" x="330" y="240" text-anchor="start">o = v<tspan dy="3.5" font-size="72%">1</tspan><tspan dy="-3.5"> · P</tspan><tspan dy="3.5" font-size="72%">:,1</tspan><tspan dy="-3.5"> + v</tspan><tspan dy="3.5" font-size="72%">2</tspan><tspan dy="-3.5"> · P</tspan><tspan dy="3.5" font-size="72%">:,2</tspan><tspan dy="-3.5"> + v</tspan><tspan dy="3.5" font-size="72%">3</tspan><tspan dy="-3.5"> · P</tspan><tspan dy="3.5" font-size="72%">:,3</tspan><tspan dy="-3.5"> + v</tspan><tspan dy="3.5" font-size="72%">4</tspan><tspan dy="-3.5"> · P</tspan><tspan dy="3.5" font-size="72%">:,4</tspan></text>
</svg>
<figcaption><strong>图 1-1</strong> 两种看矩阵乘法的方式；格子换成 block 就是<a href="#fig-1-2">图 1-2</a>的两次乘法，配色也相同。</figcaption>
</figure>

如果 attention 只有两次乘法，$O = SV$，tiling 直接就成立：$O_i = \sum_{j=1}^{T_k} S_{ij}V_j$。负责第 $i$ 个 query block 的 program 先把 $Q_i$ 读进片上，然后沿 $j$ 每次读一对 $K_j$、$V_j$，算出 $S_{ij}$，乘 $V_j$ 加进 $O_i$，$S_{ij}$ 用完就丢（[图 1-2](#fig-1-2)）。片上只放 $Q_i$、$O_i$ 和当前这一对 $K_j$、$V_j$，$N_q\times N_k$ 的 $S$ 从不写进 HBM，$K$、$V$ 对每个 $i$ 只读一遍。

<figure id="fig-1-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 334" width="100%" role="img" aria-label="FlashAttention 的 tiling，按 Triton program 看：i = 2 的 program 读 Q 的第 2 个 block，沿 j 依次读 K、V 的 block，S 只在片上算，乘 V 以后累加进 O 的第 2 个 block，最后写回；causal 时对角线右上的 block 跳过。">
  <style>
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-1-2-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-1-2-mb" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-1-2-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker><pattern id="fig-1-2-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-1)" stroke-opacity=".55" stroke-width="1.3"/></pattern></defs>
  <rect class="band" x="4" y="4" width="632" height="326" rx="10"/>
  <text class="s" x="624" y="26" text-anchor="end" style="fill: var(--fig-hi)">causal</text>
  <rect x="150" y="58" width="20" height="44" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="172" y="58" width="20" height="44" rx="1.5" style="fill: url(#fig-1-2-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="194" y="58" width="20" height="44" rx="1.5" style="fill: url(#fig-1-2-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="216" y="58" width="20" height="44" rx="1.5" style="fill: url(#fig-1-2-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="238" y="58" width="20" height="44" rx="1.5" style="fill: url(#fig-1-2-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="260" y="58" width="20" height="44" rx="1.5" style="fill: url(#fig-1-2-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="282" y="58" width="20" height="44" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="304" y="58" width="20" height="44" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="tb" x="142" y="70" text-anchor="end">K<tspan dy="-5" font-size="72%">ᵀ</tspan></text>
  <text class="s" x="142" y="83" text-anchor="end">d × N<tspan dy="3.5" font-size="72%">k</tspan></text>
  <line x1="150" y1="50" x2="280" y2="50" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-2-m0)"/>
  <text class="s" x="216" y="44" text-anchor="middle">j = 0 … 5</text>
  <rect x="82" y="116" width="44" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="82" y="160" width="44" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="82" y="204" width="44" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="82" y="248" width="44" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <text class="tb" x="104" y="96" text-anchor="middle">Q</text>
  <line x1="24" y1="116" x2="24" y2="290" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-2-m0)"/>
  <text class="s" x="0" y="0" text-anchor="middle" transform="translate(16 204.0) rotate(-90)">i = 0 … 3</text>
  <text class="s" x="104" y="109" text-anchor="middle">N<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> × d</tspan></text>
  <rect x="147" y="113" width="180" height="180" rx="4" fill="none" stroke="currentColor" stroke-opacity=".4" stroke-dasharray="4 3"/>
  <rect x="150" y="116" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="172" y="116" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="194" y="116" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="216" y="116" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="238" y="116" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="260" y="116" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="282" y="116" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="304" y="116" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="150" y="160" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="172" y="160" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="194" y="160" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="216" y="160" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="238" y="160" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="260" y="160" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="282" y="160" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="304" y="160" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="150" y="204" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="172" y="204" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="194" y="204" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="216" y="204" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="238" y="204" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="260" y="204" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="282" y="204" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="304" y="204" width="20" height="42" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="150" y="248" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="172" y="248" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="194" y="248" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="216" y="248" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="238" y="248" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="260" y="248" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="282" y="248" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="304" y="248" width="20" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <line x1="150" y1="116" x2="326" y2="292" stroke="currentColor" stroke-opacity=".45" stroke-dasharray="4 3"/>
  <line x1="130" y1="225" x2="146" y2="225" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-2-m0)"/>
  <rect x="370" y="116" width="44" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="370" y="138" width="44" height="20" rx="1.5" style="fill: url(#fig-1-2-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="370" y="160" width="44" height="20" rx="1.5" style="fill: url(#fig-1-2-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="370" y="182" width="44" height="20" rx="1.5" style="fill: url(#fig-1-2-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="370" y="204" width="44" height="20" rx="1.5" style="fill: url(#fig-1-2-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="370" y="226" width="44" height="20" rx="1.5" style="fill: url(#fig-1-2-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="370" y="248" width="44" height="20" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="370" y="270" width="44" height="20" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="tb" x="392" y="96" text-anchor="middle">V</text>
  <text class="s" x="392" y="109" text-anchor="middle">N<tspan dy="3.5" font-size="72%">k</tspan><tspan dy="-3.5"> × d</tspan></text>
  <rect x="454" y="116" width="44" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="454" y="160" width="44" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="454" y="204" width="44" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="454" y="248" width="44" height="42" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <text class="tb" x="476" y="96" text-anchor="middle">O</text>
  <text class="s" x="476" y="109" text-anchor="middle">N<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> × d</tspan></text>
  <line x1="332" y1="225" x2="366" y2="225" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-2-m0)"/>
  <text class="s" x="348" y="220" text-anchor="middle">×</text>
  <line x1="418" y1="225" x2="450" y2="225" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-2-m0)"/>
  <text class="s" x="434" y="220" text-anchor="middle">=</text>
  <rect x="516" y="120" width="12" height="12" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="534" y="130">这一轮读的（j = 0）</text>
  <rect x="516" y="140" width="12" height="12" rx="1.5" style="fill: url(#fig-1-2-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="534" y="150">之后几轮读的</text>
  <rect x="516" y="160" width="12" height="12" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <text class="s" x="534" y="170">写回的输出</text>
  <rect x="516" y="180" width="12" height="12" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <text class="s" x="534" y="190">片上算的 S 块</text>
  <rect x="516" y="200" width="12" height="12" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="534" y="210">causal 跳过</text>
  <rect x="516" y="220" width="12" height="12" rx="1.5" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <text class="s" x="534" y="230">别的 program 的</text>
</svg>
<figcaption><strong>图 1-2</strong> 按 Triton program 看的 tiling，画法同 GPU 与 Triton 入门的<a href="/blog/gpu-triton-intro/#fig-o-1">图 2-1</a>，循环顺序按 FA2，下标按 Triton 从 0 数。$i = 2$ 的 program 读 $Q$ 的第 2 个 block，沿 $j$ 依次读 $K$、$V$ 的 block，$S$ 只在片上算，乘 $V$ 以后累加进 $O$ 的第 2 个 block。causal 时对角线右上方的 block 整块跳过（<a href="#fa-algorithm">1.3</a>）。</figcaption>
</figure>

但 $S$ 和 $V$ 之间隔着按行做的 softmax。取 $S$ 的一行，记作 $s_1, \dots, s_{N_k}$，$s_k$ 是这个 query 和第 $k$ 个 key 的 score：

<p align="center">$p_k = \dfrac{e^{s_k - m}}{\ell}, \qquad m = \max_{k} s_k, \qquad \ell = \sum_{k=1}^{N_k} e^{s_k - m}$</p>

减去 $m$ 不改变结果，只是让指数都不大于 0，免得溢出（见 1.2 的 notes）。$m$ 和 $\ell$ 都要看完这一行的 $N_k$ 个 score，也就是全部 $T_k$ 个 block。读到第 $j$ 个 block 时后面的 block 还没算，$P_{ij}$ 就算不出来，$P_{ij}V_j$ 也没法加进 $O_i$。按定义算，要先扫一遍 $K$ 求 $m$，再扫一遍求 $\ell$，第三遍才能算 $PV$。要回到只扫一遍，就得把 $m$、$\ell$ 和 $O$ 都改成一个 block 一个 block 地算：1.2 先做 $m$、$\ell$，再做 $O$；1.3 把它们放进两层循环，就是 FlashAttention-2 论文的 Algorithm 1。



### 1.2 Online softmax {#online-softmax}

online softmax（Milakov 和 Gimelshein 2018）的核心是把一行的两段合并。一段 A 的最大值和分母是

<p align="center">$m_A = \max_{k \in A} s_k, \qquad \ell_A = \sum_{k \in A} e^{s_k - m_A}$</p>

两段 A、B 合起来是

<p align="center">$m = \max(m_A, m_B), \qquad \ell = \ell_A\, e^{m_A - m} + \ell_B\, e^{m_B - m}$</p>

放到 block 上，固定第 $i$ 个 query block。每读进一个 $K_j$，A 是前 $j-1$ 个 block，记作 $(m_{j-1}, \ell_{j-1})$；B 是第 $j$ 个 block，$m_B = \mathrm{rowmax}(S_{ij})$，$\ell_B = \mathrm{rowsum}(e^{S_{ij} - m_B})$；合并的结果记作 $(m_j, \ell_j)$，$m_0 = -\infty$，$\ell_0 = 0$。这里的 $m$、$\ell$ 都是长度 $B_q$ 的向量，一行一个数。$\ell_B$ 乘上 $e^{m_B - m_j}$ 正好是 $\mathrm{rowsum}(e^{S_{ij} - m_j})$，记 $\tilde P_{ij} = e^{S_{ij} - m_j}$，合并公式就变成

<p align="center">$m_j = \max\big(m_{j-1},\, \mathrm{rowmax}(S_{ij})\big), \qquad \tilde P_{ij} = e^{S_{ij} - m_j}, \qquad \ell_j = e^{m_{j-1} - m_j}\, \ell_{j-1} + \mathrm{rowsum}(\tilde P_{ij})$</p>

扫完 $j = T_k$，$m_{T_k}$、$\ell_{T_k}$ 就是整行的 $m$、$\ell$。这三个式子是 Algorithm 1 的第 9 行，推导和证明见下面的 notes。

<details class="fold doc aside">
<summary>第 1 到第 4 步：从 naive softmax 到 online softmax 和并行</summary>

下面按 Milakov 和 Gimelshein 论文的顺序推一遍。一行 score 记作 $s_1, \dots, s_N$，分母记作 $\ell$（论文里记作 $d$）。

**第 1 步：naive softmax。** 按定义

<p align="center">$y_i = \dfrac{e^{s_i}}{\sum_{j=1}^{N} e^{s_j}}$</p>

要扫两遍：第 1 遍求分母，第 2 遍算输出。每个元素访存 3 次（读两次、写一次）。

```
ℓ_0 ← 0
for j ← 1 … N:                      # 第 1 遍：读 s_j
    ℓ_j ← ℓ_{j−1} + e^(s_j)
for i ← 1 … N:                      # 第 2 遍：读 s_i，写 y_i
    y_i ← e^(s_i) / ℓ_N
```

问题是 $e^{s_j}$ 会溢出：fp32 里 $s$ 超过 88.7 时 $e^{s}$ 就是 inf。

**第 2 步：safe softmax。** 分子分母同乘 $e^{-m_N}$，$m_N$ 是整行的最大值。结果不变，指数都不大于 0，不会溢出：

<p align="center">$y_i = \dfrac{e^{s_i - m_N}}{\sum_{j=1}^{N} e^{s_j - m_N}}, \qquad m_N = \max_{k=1}^{N} s_k$</p>

PyTorch 等框架用的就是这个版本，代价是多扫一遍，每个元素访存 4 次：

```
m_0 ← −∞
for k ← 1 … N:                      # 第 1 遍：读 s_k
    m_k ← max(m_{k−1}, s_k)
ℓ_0 ← 0
for j ← 1 … N:                      # 第 2 遍：读 s_j
    ℓ_j ← ℓ_{j−1} + e^(s_j − m_N)
for i ← 1 … N:                      # 第 3 遍：读 s_i，写 y_i
    y_i ← e^(s_i − m_N) / ℓ_N
```

第 2 遍的每一项都要用 $m_N$，必须等第 1 遍扫完整行。

**第 3 步：online softmax。** 把前两遍合成一遍，边扫边维护到目前为止的最大值 $m_j$ 和分母 $\ell_j$。最大值变大时，之前加进 $\ell$ 的项都是按旧的最大值算的，先乘 $e^{m_{j-1}-m_j}$ 换到新的最大值，再加上新的一项：

```
m_0 ← −∞,  ℓ_0 ← 0
for j ← 1 … N:                      # 第 1 遍：读 s_j，m 和 ℓ 一起更新
    m_j ← max(m_{j−1}, s_j)
    ℓ_j ← ℓ_{j−1} · e^(m_{j−1} − m_j) + e^(s_j − m_j)
for i ← 1 … N:                      # 第 2 遍：读 s_i，写 y_i
    y_i ← e^(s_i − m_N) / ℓ_N
```

每个元素访存回到 3 次，和 naive 一样，但不会溢出。

这样算为什么对：要证的是扫完第 $j$ 个元素后 $\ell_j = \sum_{k=1}^{j} e^{s_k - m_j}$，用归纳法。$j = 1$ 时 $m_1 = s_1$，$\ell_1 = e^{s_1 - m_1}$，成立。假设 $j-1$ 时成立，代入第 $j$ 步的更新，再用 $e^a \cdot e^b = e^{a+b}$：

<p align="center">$\begin{aligned} \ell_j &= \ell_{j-1}\, e^{m_{j-1}-m_j} + e^{s_j-m_j} \\ &= \Big(\sum_{k=1}^{j-1} e^{s_k-m_{j-1}}\Big)\, e^{m_{j-1}-m_j} + e^{s_j-m_j} \\ &= \sum_{k=1}^{j-1} e^{s_k-m_j} + e^{s_j-m_j} \;=\; \sum_{k=1}^{j} e^{s_k-m_j} \end{aligned}$</p>

$j = N$ 时就是 safe softmax 的分母。

**第 4 步：两段合并，并行。** 第 3 步每读一个元素，是把「前 $j-1$ 个元素」的结果和「第 $j$ 个元素」合起来。同样的办法能合并任意两段。A 段和 B 段各自按第 3 步扫完，得到 $(m_A, \ell_A)$ 和 $(m_B, \ell_B)$，其中 $\ell_A = \sum_{k\in A} e^{s_k - m_A}$。合并后的最大值是 $m = \max(m_A, m_B)$，分母和第 3 步的证明一样用 $e^a \cdot e^b = e^{a+b}$：

<p align="center">$\begin{aligned} \ell_A\, e^{m_A-m} + \ell_B\, e^{m_B-m} &= \Big(\sum_{k\in A} e^{s_k-m_A}\Big) e^{m_A-m} + \Big(\sum_{k\in B} e^{s_k-m_B}\Big) e^{m_B-m} \\ &= \sum_{k\in A} e^{s_k-m} + \sum_{k\in B} e^{s_k-m} \;=\; \sum_{k\in A\cup B} e^{s_k-m} \end{aligned}$</p>

第 3 步那一行就是 B 段只有一个元素的情况：$m_B = s_j$，$\ell_B = e^{s_j - s_j} = 1$。论文把这个合并记作 $\oplus$。

能合并任意两段，一行就可以切成几段，交给不同的 thread 同时按第 3 步的第 1 个循环扫，再合并。合并以后每个 thread 都拿到整行的 $m_N$、$\ell_N$，写 $y_i$ 的第 2 个循环也各写各的那一段（[图 1-3](#fig-1-3) 最下面一栏）。合并的顺序不影响结果（$\oplus$ 满足结合律和交换律），所以先合并哪两段都行。

<figure id="fig-1-3">
<svg viewBox="0 0 680 346" width="100%" role="img" aria-label="一行 8 个元素。safe softmax 顺序扫三遍：求 m、求 ℓ、写 y，共 24 步。online softmax 顺序扫两遍：同时更新 m 和 ℓ、写 y，共 16 步。两个 thread 并行：各扫 4 个元素更新 m 和 ℓ，合并一次，再各写 4 个 y，共 9 步">
  <style>
    .os2 { font-size: 11px; fill: currentColor; }
    .os2-b { font-size: 12px; font-weight: 600; fill: currentColor; }
    .os2-s { font-size: 10.5px; fill: currentColor; opacity: .72; }
    .os2-c { font-size: 12px; font-weight: 600; text-anchor: middle; fill: currentColor; }
    .os2-hi { fill: var(--fig-hi); font-weight: 600; }
  </style>
<rect x="150" y="7" width="14" height="14" rx="2" style="fill: var(--fig-2)"/>
<text class="os2-s" x="169" y="18" text-anchor="start">求 m</text>
<rect x="225" y="7" width="14" height="14" rx="2" style="fill: var(--fig-1)"/>
<text class="os2-s" x="244" y="18" text-anchor="start">求 ℓ</text>
<rect x="300" y="7" width="14" height="14" rx="2" style="fill: var(--fig-1)"/><rect x="300" y="7" width="14" height="4.2" style="fill: var(--fig-2)"/>
<text class="os2-s" x="319" y="18" text-anchor="start">同时更新 m、ℓ</text>
<rect x="425" y="7" width="14" height="14" rx="2" style="fill: var(--fig-3)"/>
<text class="os2-s" x="444" y="18" text-anchor="start">写 y<tspan dy="3.5" font-size="72%">i</tspan></text>
<rect x="520" y="7" width="14" height="14" rx="2" style="fill: var(--fig-hi)"/>
<text class="os2-s" x="539" y="18" text-anchor="start">合并</text>
<line x1="150.0" y1="40" x2="150.0" y2="256" stroke="currentColor" stroke-opacity=".1"/>
<text class="os2-s" x="150.0" y="270" text-anchor="middle">0</text>
<line x1="232.0" y1="40" x2="232.0" y2="256" stroke="currentColor" stroke-opacity=".1"/>
<text class="os2-s" x="232.0" y="270" text-anchor="middle">4</text>
<line x1="314.0" y1="40" x2="314.0" y2="256" stroke="currentColor" stroke-opacity=".1"/>
<text class="os2-s" x="314.0" y="270" text-anchor="middle">8</text>
<line x1="396.0" y1="40" x2="396.0" y2="256" stroke="currentColor" stroke-opacity=".1"/>
<text class="os2-s" x="396.0" y="270" text-anchor="middle">12</text>
<line x1="478.0" y1="40" x2="478.0" y2="256" stroke="currentColor" stroke-opacity=".1"/>
<text class="os2-s" x="478.0" y="270" text-anchor="middle">16</text>
<line x1="560.0" y1="40" x2="560.0" y2="256" stroke="currentColor" stroke-opacity=".1"/>
<text class="os2-s" x="560.0" y="270" text-anchor="middle">20</text>
<line x1="642.0" y1="40" x2="642.0" y2="256" stroke="currentColor" stroke-opacity=".1"/>
<text class="os2-s" x="642.0" y="270" text-anchor="middle">24</text>
<text class="os2-s" x="642.0" y="284" text-anchor="end">步 →</text>
<text class="os2-b" x="8" y="52" text-anchor="start">safe softmax（第 2 步）</text>
<text class="os2-s" x="8" y="74" text-anchor="start">1 个 thread，3 遍，24 步</text>
<rect x="150.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-2)"/><text class="os2-s" x="160.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-2); font-weight: 600" opacity="1">1</text>
<rect x="171.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-2)"/><text class="os2-s" x="180.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-2); font-weight: 600" opacity="1">2</text>
<rect x="191.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-2)"/><text class="os2-s" x="201.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-2); font-weight: 600" opacity="1">3</text>
<rect x="212.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-2)"/><text class="os2-s" x="221.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-2); font-weight: 600" opacity="1">4</text>
<rect x="232.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-2)"/><text class="os2-s" x="242.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-2); font-weight: 600" opacity="1">5</text>
<rect x="253.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-2)"/><text class="os2-s" x="262.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-2); font-weight: 600" opacity="1">6</text>
<rect x="273.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-2)"/><text class="os2-s" x="283.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-2); font-weight: 600" opacity="1">7</text>
<rect x="294.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-2)"/><text class="os2-s" x="303.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-2); font-weight: 600" opacity="1">8</text>
<rect x="314.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><text class="os2-s" x="324.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">1</text>
<rect x="335.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><text class="os2-s" x="344.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">2</text>
<rect x="355.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><text class="os2-s" x="365.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">3</text>
<rect x="376.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><text class="os2-s" x="385.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">4</text>
<rect x="396.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><text class="os2-s" x="406.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">5</text>
<rect x="417.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><text class="os2-s" x="426.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">6</text>
<rect x="437.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><text class="os2-s" x="447.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">7</text>
<rect x="458.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><text class="os2-s" x="467.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">8</text>
<rect x="478.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="488.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">1</text>
<rect x="499.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="508.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">2</text>
<rect x="519.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="529.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">3</text>
<rect x="540.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="549.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">4</text>
<rect x="560.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="570.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">5</text>
<rect x="581.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="590.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">6</text>
<rect x="601.75" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="611.25" y="74.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">7</text>
<rect x="622.25" y="58" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="631.75" y="74.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">8</text>
<line x1="642.0" y1="55" x2="642.0" y2="85" style="stroke: var(--fig-hi)" stroke-width="2"/>
<text class="os2-b" x="8" y="112" text-anchor="start">online softmax（第 3 步）</text>
<text class="os2-s" x="8" y="134" text-anchor="start">1 个 thread，2 遍，16 步</text>
<rect x="150.75" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="150.75" y="118" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="160.25" y="137.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">1</text>
<rect x="171.25" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="171.25" y="118" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="180.75" y="137.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">2</text>
<rect x="191.75" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="191.75" y="118" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="201.25" y="137.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">3</text>
<rect x="212.25" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="212.25" y="118" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="221.75" y="137.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">4</text>
<rect x="232.75" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="232.75" y="118" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="242.25" y="137.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">5</text>
<rect x="253.25" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="253.25" y="118" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="262.75" y="137.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">6</text>
<rect x="273.75" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="273.75" y="118" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="283.25" y="137.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">7</text>
<rect x="294.25" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="294.25" y="118" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="303.75" y="137.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">8</text>
<rect x="314.75" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="324.25" y="134.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">1</text>
<rect x="335.25" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="344.75" y="134.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">2</text>
<rect x="355.75" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="365.25" y="134.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">3</text>
<rect x="376.25" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="385.75" y="134.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">4</text>
<rect x="396.75" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="406.25" y="134.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">5</text>
<rect x="417.25" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="426.75" y="134.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">6</text>
<rect x="437.75" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="447.25" y="134.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">7</text>
<rect x="458.25" y="118" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="467.75" y="134.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">8</text>
<line x1="478.0" y1="115" x2="478.0" y2="145" style="stroke: var(--fig-hi)" stroke-width="2"/>
<text class="os2-b" x="8" y="172" text-anchor="start">online softmax 并行（第 4 步）</text>
<text class="os2-s" x="8" y="202" text-anchor="start">thread 0：s<tspan dy="3.5" font-size="72%">1</tspan><tspan dy="-3.5"> … s</tspan><tspan dy="3.5" font-size="72%">4</tspan></text>
<text class="os2-s" x="8" y="230" text-anchor="start">thread 1：s<tspan dy="3.5" font-size="72%">5</tspan><tspan dy="-3.5"> … s</tspan><tspan dy="3.5" font-size="72%">8</tspan></text>
<rect x="150.75" y="186" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="150.75" y="186" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="160.25" y="205.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">1</text>
<rect x="150.75" y="214" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="150.75" y="214" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="160.25" y="233.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">5</text>
<rect x="253.25" y="186" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="262.75" y="202.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">1</text>
<rect x="253.25" y="214" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="262.75" y="230.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">5</text>
<rect x="171.25" y="186" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="171.25" y="186" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="180.75" y="205.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">2</text>
<rect x="171.25" y="214" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="171.25" y="214" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="180.75" y="233.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">6</text>
<rect x="273.75" y="186" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="283.25" y="202.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">2</text>
<rect x="273.75" y="214" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="283.25" y="230.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">6</text>
<rect x="191.75" y="186" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="191.75" y="186" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="201.25" y="205.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">3</text>
<rect x="191.75" y="214" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="191.75" y="214" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="201.25" y="233.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">7</text>
<rect x="294.25" y="186" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="303.75" y="202.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">3</text>
<rect x="294.25" y="214" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="303.75" y="230.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">7</text>
<rect x="212.25" y="186" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="212.25" y="186" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="221.75" y="205.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">4</text>
<rect x="212.25" y="214" width="19.0" height="24" rx="2" style="fill: var(--fig-1)"/><rect x="212.25" y="214" width="19.0" height="7.199999999999999" style="fill: var(--fig-2)"/><text class="os2-s" x="221.75" y="233.6" text-anchor="middle" style="fill: var(--fig-on-1); font-weight: 600" opacity="1">8</text>
<rect x="314.75" y="186" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="324.25" y="202.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">4</text>
<rect x="314.75" y="214" width="19.0" height="24" rx="2" style="fill: var(--fig-3)"/><text class="os2-s" x="324.25" y="230.0" text-anchor="middle" style="fill: var(--fig-on-3); font-weight: 600" opacity="1">8</text>
<rect x="232.75" y="186" width="19.0" height="52" rx="2" style="fill: var(--fig-hi)"/><text class="os2" text-anchor="middle" style="fill: var(--fig-on-hi); font-weight: 600"><tspan x="242.25" y="210.0">合</tspan><tspan x="242.25" y="224.0">并</tspan></text>
<line x1="334.5" y1="183" x2="334.5" y2="241" style="stroke: var(--fig-hi)" stroke-width="2"/>
<text class="os2 os2-hi" x="340.5" y="216" text-anchor="start">9 步</text>
<text class="os2" x="484.0" y="134.0" text-anchor="middle"></text>
<text class="os2" x="150" y="304" text-anchor="start">合并（第 4 步）：m = max(m<tspan dy="3.5" font-size="72%">A</tspan><tspan dy="-3.5">, m</tspan><tspan dy="3.5" font-size="72%">B</tspan><tspan dy="-3.5">)，ℓ = ℓ</tspan><tspan dy="3.5" font-size="72%">A</tspan><tspan dy="-3.5"> · e</tspan><tspan dy="-5" font-size="72%">m</tspan><tspan dy="3" font-size="58%">A</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="5"> + ℓ</tspan><tspan dy="3.5" font-size="72%">B</tspan><tspan dy="-3.5"> · e</tspan><tspan dy="-5" font-size="72%">m</tspan><tspan dy="3" font-size="58%">B</tspan><tspan dy="-3" font-size="72%">−m</tspan></text>
<text class="os2-s" x="150" y="322" text-anchor="start">合并以后两个 thread 都拿到整行的 m、ℓ；每个 y<tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5"> 只用自己的 x</tspan><tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5">，两段 y 各写各的</tspan></text>
<text class="os2-s" x="150" y="340" text-anchor="start">格子里的数字是元素的下标 k</text>
</svg>
<figcaption><strong>图 1-3</strong> 一行 8 个元素，三种算法放在同一条时间轴上，一格是处理一个元素。</figcaption>
</figure>

</details>

到这里 $m$、$\ell$ 扫一遍就能求出来，但 softmax 的结果 $P_{ij} = e^{S_{ij} - m_{T_k}}/\ell_{T_k}$ 要用扫完以后的 $m_{T_k}$、$\ell_{T_k}$，单算 softmax 还得再扫一遍才能写出 $P$。attention 要的不是 $P$，而是 $O = PV$，这一遍可以省掉。

先把分母提出来。$O$ 的一行是 $V$ 各行的加权和，权重是 $P$ 的那一行（[图 1-1](#fig-1-1) (b)），这些权重的分母都是同一个 $\ell$，可以提到求和外面：

<p align="center">$o = \sum_{k=1}^{N_k} \dfrac{e^{s_k - m}}{\ell}\, v_k = \dfrac{1}{\ell} \sum_{k=1}^{N_k} e^{s_k - m}\, v_k$</p>

$v_k$ 是 $V$ 的第 $k$ 行。右边的求和记作 $\tilde o$，是还没除以 $\ell$ 的输出，所以 $\tilde o$ 和 $\ell$ 可以分开算，最后再相除。

$\tilde o = \sum_k e^{s_k - m} v_k$ 和 $\ell = \sum_k e^{s_k - m}$ 是同一种求和，只是每一项多乘了一个 $v_k$，所以 $\tilde o$ 也能照 $\ell$ 的办法一个 block 一个 block 地累加：先按当前的最大值 $m_j$ 算，最大值变大时把旧的和整体乘 $e^{m_{j-1} - m_j}$。整体乘是对的：旧的和里每一项的权重都是按 $m_{j-1}$ 算的 $e^{s_k - m_{j-1}}$，都要乘同一个 $e^{m_{j-1} - m_j}$ 才变成按 $m_j$ 算的值，权重全乘同一个数，线性组合的结果也就乘这个数。第 $i$ 个 query block 的累加结果记作 $\tilde O_j$，是 $B_q \times d$ 的矩阵，和 $\ell_j$ 并排写：

<p align="center">$\ell_j = e^{m_{j-1} - m_j}\, \ell_{j-1} + \mathrm{rowsum}(\tilde P_{ij}), \qquad \tilde O_j = \mathrm{diag}\big(e^{m_{j-1} - m_j}\big)\, \tilde O_{j-1} + \tilde P_{ij} V_j$</p>

$\tilde O_0 = 0$。两式只差最后一项：$\ell$ 把 $\tilde P_{ij}$ 每一行的元素加起来，$\tilde O$ 拿 $\tilde P_{ij}$ 的每一行当权重去组合 $V_j$ 的各行。每一行是一个 query，有自己的修正因子，所以 $\tilde O$ 前面写成 $\mathrm{diag}$，意思是第 $r$ 行乘第 $r$ 个数。这是 Algorithm 1 的第 10 行。

只有两个 block 时展开看一下。记 $m = m_2$ 为整行的最大值，$e^{\cdots}$ 都是每行一个数：

<p align="center">$\tilde O_2 = e^{m_1 - m}\, \tilde O_1 + e^{S_{i2} - m}\, V_2 = e^{m_1 - m}\, e^{S_{i1} - m_1}\, V_1 + e^{S_{i2} - m}\, V_2 = e^{S_{i1} - m}\, V_1 + e^{S_{i2} - m}\, V_2$</p>

第一个 block 先按 $m_1$ 算，乘上修正因子以后就和直接按整行的 $m$ 算一样；block 更多时逐个这样推下去。扫完 $j = T_k$，每行除以自己的 $\ell_{T_k}$，$O_i = \mathrm{diag}(\ell_{T_k})^{-1}\, \tilde O_{T_k}$，是第 12 行。$P$ 从头到尾没有写出来，结果和按定义算的完全一样，只差浮点加法的顺序。

### 1.3 FlashAttention-2 的 Algorithm 1 {#fa-algorithm}

把 1.2 的三个递推放进两层循环，外层走 query block $i$，内层走 key/value block $j$，就是 FlashAttention-2 论文（Dao 2023）的 Algorithm 1。下面照抄，记号换成本文的：论文的 $B_r$、$B_c$、$T_r$、$T_c$ 在这里是 $B_q$、$B_k$、$T_q$、$T_k$；$m_i^{(j)}$ 就是 1.2 里固定 $i$ 时的 $m_j$，$\ell_i^{(j)}$、$\tilde O_i^{(j)}$ 同理。论文省略了 $1/\sqrt d$，当作已经乘进 $Q$ 里。

```
Algorithm 1  FlashAttention-2 forward pass
Require: HBM 里的 Q, K, V ∈ ℝ^(N×d)，block 大小 B_q, B_k
 1: 把 Q 切成 T_q = ⌈N/B_q⌉ 个 block Q_1 … Q_Tq，每个 B_q × d；
    把 K、V 切成 T_k = ⌈N/B_k⌉ 个 block K_1 … K_Tk 和 V_1 … V_Tk，每个 B_k × d
 2: 把输出 O ∈ ℝ^(N×d) 切成 T_q 个 block O_1 … O_Tq，每个 B_q × d；
    把 logsumexp L 切成 T_q 个 block L_1 … L_Tq，每个长 B_q
 3: for 1 ≤ i ≤ T_q do
 4:     从 HBM 读 Q_i 到片上 SRAM
 5:     片上初始化 Õ_i^(0) = 0 ∈ ℝ^(B_q×d)，ℓ_i^(0) = 0 ∈ ℝ^(B_q)，m_i^(0) = −∞ ∈ ℝ^(B_q)
 6:     for 1 ≤ j ≤ T_k do
 7:         从 HBM 读 K_j、V_j 到片上 SRAM
 8:         片上算 S_i^(j) = Q_i K_jᵀ ∈ ℝ^(B_q×B_k)
 9:         片上算 m_i^(j) = max(m_i^(j−1), rowmax(S_i^(j)))，
                   P̃_i^(j) = exp(S_i^(j) − m_i^(j))（逐元素），
                   ℓ_i^(j) = e^(m_i^(j−1) − m_i^(j)) · ℓ_i^(j−1) + rowsum(P̃_i^(j))
10:         片上算 Õ_i^(j) = diag(e^(m_i^(j−1) − m_i^(j))) · Õ_i^(j−1) + P̃_i^(j) V_j
11:     end for
12:     片上算 O_i = diag(ℓ_i^(T_k))⁻¹ · Õ_i^(T_k)
13:     片上算 L_i = m_i^(T_k) + log(ℓ_i^(T_k))
14:     把 O_i 写回 HBM，作为 O 的第 i 个 block
15:     把 L_i 写回 HBM，作为 L 的第 i 个 block
16: end for
17: 返回 O 和 L
```

论文第 10 行写的是 $O_i^{(j)}$，这里加上波浪号，表示还没除以 $\ell$；论文在 $\mathrm{diag}(\cdot)$ 后面还多写了一个 $^{-1}$，按论文 3.1.1 节的推导应该没有。

逐行对回前面：第 8 行是 [图 1-1](#fig-1-1) (a) 的外积，第 9 行是 1.2 的 $m$、$\ell$ 递推，第 10 行是 $\tilde O$ 的递推，其中 $\tilde P_{ij}V_j$ 是图 1-1 (b) 的线性组合，第 12 行最后除一次 $\ell$。第 7 行是唯一读 $K$、$V$ 的地方，对每个 $i$，每个 $K_j$、$V_j$ 只读一次；$S$ 和 $\tilde P$ 只在第 8 到 10 行的片上出现，从不写回 HBM。第 13 行的 $L_i$ 留给反向：反向要重算 $P$，有 $L$ 就够了，$P_{ij} = e^{S_{ij} - L_i}$。额外的显存只有每行一个 $L$，是 $O(N)$。

causal attention 里，key 下标大于 query 下标的位置要设成 $-\infty$。按 block 算，列下标全大于行下标的 block 整块跳过，第 6 行的 $j$ 只走到对角线；只有对角线穿过的 block 要在块内 mask。序列长时差不多一半的 block 被跳过，论文测得比不带 mask 快 1.7–1.8 倍。[图 1-4](#fig-1-4) 是 Triton 里一个 program 怎么执行第 6 到 12 行：下栏的 ①、②、③、⑤ 是第 9 行，④ 是第 10 行，循环后的 $\div\,\ell$ 是第 12 行；$Q_i$、$\tilde O_i$、$m$、$\ell$ 一直在 register，$K_j$、$V_j$ 经 shared memory 轮流搬进来。

<figure id="fig-1-4" class="fg-fig">
<svg class="fg" viewBox="0 0 640 654" width="100%" role="img" aria-label="上：grid 里每个 Q block 一个 program，causal 时每个 program 算的 block 数不同，对角线上的 block 部分 mask。下：一个 program 的一轮，dot 算 S，online softmax 更新 m、ℓ，再 dot 乘 V 累加进 O_i；K、V 经 shared memory 轮流搬，Q_i、O_i、m、ℓ 一直在 register，扫完除以 ℓ 写回 O。">
  <style>
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-1-4-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-1-4-mb" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-1-4-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker><pattern id="fig-1-4-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-1)" stroke-opacity=".55" stroke-width="1.3"/></pattern></defs>
  <g transform="translate(0 -334)">
  <rect class="band" x="4" y="338" width="632" height="182" rx="10"/>
  <text class="tb" x="16" y="360">每个 program 干什么（causal，B<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> = 2B</tspan><tspan dy="3.5" font-size="72%">k</tspan><tspan dy="-3.5">）</tspan></text>
  <text class="s" x="16" y="391">pid_i = 0</text>
  <rect x="74" y="378" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <path d="M74,378 L90,378 L90,394 Z" style="fill: color-mix(in srgb, var(--fig-mute) 45%, transparent)"/>
  <rect x="92" y="378" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <path d="M92,378 L108,378 L108,394 Z" style="fill: color-mix(in srgb, var(--fig-mute) 45%, transparent)"/>
  <rect x="110" y="378" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="128" y="378" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="146" y="378" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="164" y="378" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="182" y="378" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="200" y="378" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="16" y="419">pid_i = 1</text>
  <rect x="74" y="406" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="92" y="406" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="110" y="406" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <path d="M110,406 L126,406 L126,422 Z" style="fill: color-mix(in srgb, var(--fig-mute) 45%, transparent)"/>
  <rect x="128" y="406" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <path d="M128,406 L144,406 L144,422 Z" style="fill: color-mix(in srgb, var(--fig-mute) 45%, transparent)"/>
  <rect x="146" y="406" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="164" y="406" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="182" y="406" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="200" y="406" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="16" y="447" style="opacity: 1; font-weight: 600">pid_i = 2</text>
  <rect x="74" y="434" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="92" y="434" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="110" y="434" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="128" y="434" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="146" y="434" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <path d="M146,434 L162,434 L162,450 Z" style="fill: color-mix(in srgb, var(--fig-mute) 45%, transparent)"/>
  <rect x="164" y="434" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <path d="M164,434 L180,434 L180,450 Z" style="fill: color-mix(in srgb, var(--fig-mute) 45%, transparent)"/>
  <rect x="182" y="434" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="200" y="434" width="16" height="16" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="16" y="475">pid_i = 3</text>
  <rect x="74" y="462" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="92" y="462" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="110" y="462" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="128" y="462" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="146" y="462" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="164" y="462" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <rect x="182" y="462" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <path d="M182,462 L198,462 L198,478 Z" style="fill: color-mix(in srgb, var(--fig-mute) 45%, transparent)"/>
  <rect x="200" y="462" width="16" height="16" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <path d="M200,462 L216,462 L216,478 Z" style="fill: color-mix(in srgb, var(--fig-mute) 45%, transparent)"/>
  <rect x="74" y="494" width="12" height="12" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <text class="s" x="92" y="504">整块都算</text>
  <rect x="164" y="494" width="12" height="12" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width="1"/>
  <path d="M164,494 L176,494 L176,506 Z" style="fill: color-mix(in srgb, var(--fig-mute) 45%, transparent)"/>
  <text class="s" x="182" y="504">部分 mask</text>
  <rect x="254" y="494" width="12" height="12" rx="1.5" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="272" y="504">整块跳过</text>
  <text class="s" x="474" y="380" text-anchor="middle" style="opacity: 1; font-weight: 600">S<tspan dy="3.5" font-size="72%">2,4</tspan></text>
  <rect x="454" y="388" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="464" y="388" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="474" y="388" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="484" y="388" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="454" y="398" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="464" y="398" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="474" y="398" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="484" y="398" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="454" y="408" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="464" y="408" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="474" y="408" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="484" y="408" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="454" y="418" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="464" y="418" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="474" y="418" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="484" y="418" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="454" y="428" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="464" y="428" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="474" y="428" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="484" y="428" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="454" y="438" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="464" y="438" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="474" y="438" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="484" y="438" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="454" y="448" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="464" y="448" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="474" y="448" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="484" y="448" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="454" y="458" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="464" y="458" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="474" y="458" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="484" y="458" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <text class="s" x="536" y="380" text-anchor="middle" style="opacity: 1; font-weight: 600">S<tspan dy="3.5" font-size="72%">2,5</tspan></text>
  <rect x="516" y="388" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="526" y="388" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="536" y="388" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="546" y="388" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="516" y="398" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="526" y="398" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="536" y="398" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="546" y="398" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="516" y="408" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="526" y="408" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="536" y="408" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="546" y="408" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="516" y="418" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="526" y="418" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="536" y="418" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="546" y="418" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="516" y="428" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="526" y="428" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="536" y="428" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="546" y="428" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="516" y="438" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="526" y="438" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="536" y="438" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="546" y="438" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="516" y="448" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="526" y="448" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="536" y="448" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="546" y="448" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)" stroke-width=".8"/>
  <rect x="516" y="458" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="526" y="458" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="536" y="458" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <rect x="546" y="458" width="9" height="9" rx="1" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)" stroke-width=".8"/>
  <path d="M558,388 h4 v39 h-4" fill="none" stroke="currentColor" stroke-opacity=".6"/>
  <text class="s" x="566" y="405">整行</text>
  <text class="s" x="566" y="418">都是 −∞</text>
  <text class="s" x="448" y="392" text-anchor="end">query</text>
  <text class="s" x="448" y="404" text-anchor="end">↓</text>
  <text class="s" x="474" y="481" text-anchor="middle">key →</text>
  <text class="ttl" x="10" y="552">pid_i = 2 这个 program，j = 4 这一轮</text>
  <rect class="band" x="10" y="562" width="620" height="82" rx="8"/>
  <text class="lab2" x="18" y="580">计算</text>
  <text class="lab2" x="18" y="593">单元</text>
  <rect class="band" x="10" y="652" width="620" height="50" rx="8"/>
  <text class="lab2" x="18" y="670">register</text>
  <rect class="band" x="10" y="710" width="620" height="58" rx="8"/>
  <text class="lab2" x="18" y="728">shared</text>
  <text class="lab2" x="18" y="741">memory</text>
  <rect class="band" x="10" y="776" width="620" height="92" rx="8"/>
  <text class="lab2" x="18" y="794">显存</text>
  <rect class="op" x="92" y="594" width="92" height="40" rx="6"/>
  <text class="tb" x="138" y="611" text-anchor="middle">dot</text>
  <text class="s" x="138" y="626" text-anchor="middle">Q<tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5"> K</tspan><tspan dy="3.5" font-size="72%">j</tspan><tspan dy="-8.5" font-size="72%">ᵀ</tspan></text>
  <rect class="op" x="200" y="594" width="92" height="40" rx="6" style="stroke: var(--fig-hi); stroke-opacity: 1"/>
  <text class="tb" x="246" y="611" text-anchor="middle">mask</text>
  <text class="s" x="246" y="626" text-anchor="middle">tl.where</text>
  <rect class="op" x="308" y="594" width="92" height="40" rx="6" style="stroke: var(--fig-hi); stroke-opacity: 1"/>
  <text class="tb" x="354" y="611" text-anchor="middle">① ② ③</text>
  <text class="s" x="354" y="626" text-anchor="middle">max·exp2·sum</text>
  <rect class="op" x="416" y="594" width="92" height="40" rx="6"/>
  <text class="tb" x="462" y="611" text-anchor="middle">dot  ④</text>
  <text class="s" x="462" y="626" text-anchor="middle">P̃<tspan dy="3.5" font-size="72%">ij</tspan><tspan dy="-3.5"> V</tspan><tspan dy="3.5" font-size="72%">j</tspan></text>
  <rect class="op" x="524" y="594" width="92" height="40" rx="6" stroke-dasharray="4 3"/>
  <text class="tb" x="570" y="611" text-anchor="middle">÷ ℓ</text>
  <text class="s" x="570" y="626" text-anchor="middle">O<tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5"> / ℓ</tspan></text>
  <line x1="185" y1="614" x2="198" y2="614" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-4-m0)"/>
  <line x1="293" y1="614" x2="306" y2="614" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-4-m0)"/>
  <line x1="401" y1="614" x2="414" y2="614" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-4-m0)"/>
  <path d="M138.0,592 V580 H462.0 V592" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.5"/>
  <text class="s" x="300" y="576" text-anchor="middle" style="fill: var(--fig-hi); opacity: 1">第 j 轮</text>
  <rect x="96" y="665" width="52" height="24" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 18%, transparent); stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="122" y="681" text-anchor="middle" style="opacity: 1">Q<tspan dy="3.5" font-size="72%">i</tspan></text>
  <rect x="200" y="665" width="92" height="24" rx="5" style="fill: color-mix(in srgb, var(--fig-3) 45%, transparent); stroke: var(--fig-2)" stroke-width="1.2"/>
  <text class="s" x="246" y="681" text-anchor="middle" style="opacity: 1">S<tspan dy="3.5" font-size="72%">ij</tspan><tspan dy="-3.5">、P̃</tspan><tspan dy="3.5" font-size="72%">ij</tspan></text>
  <rect x="308" y="665" width="92" height="24" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 18%, transparent); stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="354" y="681" text-anchor="middle" style="opacity: 1">m、ℓ</text>
  <rect x="438" y="665" width="70" height="24" rx="5" style="fill: color-mix(in srgb, var(--fig-1) 18%, transparent); stroke: var(--fig-1)" stroke-width="1.2"/>
  <text class="s" x="473" y="681" text-anchor="middle" style="opacity: 1">O<tspan dy="3.5" font-size="72%">i</tspan></text>
  <line x1="122" y1="665" x2="122" y2="635" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-4-m0)"/>
  <line x1="246.0" y1="664" x2="246.0" y2="635" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-start="url(#fig-1-4-mb)" marker-end="url(#fig-1-4-mb)"/>
  <line x1="354.0" y1="664" x2="354.0" y2="635" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-start="url(#fig-1-4-mb)" marker-end="url(#fig-1-4-mb)"/>
  <line x1="473.0" y1="664" x2="473.0" y2="635" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-start="url(#fig-1-4-mb)" marker-end="url(#fig-1-4-mb)"/>
  <rect x="144" y="719" width="44" height="40" rx="5" style="fill: var(--fig-1); stroke: var(--fig-1)"/>
  <text class="tb" x="166" y="743" text-anchor="middle" style="fill: var(--fig-on-1)">K<tspan dy="3.5" font-size="72%">j</tspan></text>
  <rect x="404" y="719" width="44" height="40" rx="5" style="fill: var(--fig-1); stroke: var(--fig-1)"/>
  <text class="tb" x="426" y="743" text-anchor="middle" style="fill: var(--fig-on-1)">V<tspan dy="3.5" font-size="72%">j</tspan></text>
  <rect x="252" y="719" width="92" height="40" rx="5" fill="none" stroke="currentColor" stroke-opacity=".5" stroke-dasharray="4 3"/>
  <text class="s" x="298" y="743" text-anchor="middle">K<tspan dy="3.5" font-size="72%">j+1</tspan><tspan dy="-3.5">、V</tspan><tspan dy="3.5" font-size="72%">j+1</tspan></text>
  <line x1="166" y1="719" x2="166" y2="635" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-4-m0)"/>
  <line x1="426" y1="719" x2="426" y2="635" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-4-m0)"/>
  <rect x="109" y="788" width="26" height="12.5" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="109" y="802" width="26" height="12.5" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="109" y="816" width="26" height="12.5" rx="1" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="109" y="830" width="26" height="12.5" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <text class="s" x="122" y="857" text-anchor="middle" style="opacity: 1; font-weight: 600">Q</text>
  <rect x="160.5" y="788" width="10" height="26" rx="1" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="171.5" y="788" width="10" height="26" rx="1" style="fill: url(#fig-1-4-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="182.5" y="788" width="10" height="26" rx="1" style="fill: url(#fig-1-4-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="193.5" y="788" width="10" height="26" rx="1" style="fill: url(#fig-1-4-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="204.5" y="788" width="10" height="26" rx="1" style="fill: url(#fig-1-4-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="215.5" y="788" width="10" height="26" rx="1" style="fill: url(#fig-1-4-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="226.5" y="788" width="10" height="26" rx="1" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="237.5" y="788" width="10" height="26" rx="1" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="204.5" y="857" text-anchor="middle" style="opacity: 1; font-weight: 600">K<tspan dy="-5" font-size="72%">ᵀ</tspan></text>
  <rect x="413" y="788" width="26" height="5.5" rx="1" style="fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="413" y="795" width="26" height="5.5" rx="1" style="fill: url(#fig-1-4-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="413" y="802" width="26" height="5.5" rx="1" style="fill: url(#fig-1-4-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="413" y="809" width="26" height="5.5" rx="1" style="fill: url(#fig-1-4-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="413" y="816" width="26" height="5.5" rx="1" style="fill: url(#fig-1-4-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="413" y="823" width="26" height="5.5" rx="1" style="fill: url(#fig-1-4-hatch); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="413" y="830" width="26" height="5.5" rx="1" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <rect x="413" y="837" width="26" height="5.5" rx="1" style="fill: none; stroke: var(--fig-mute)" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="s" x="426" y="857" text-anchor="middle" style="opacity: 1; font-weight: 600">V</text>
  <rect x="557" y="788" width="26" height="12.5" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="557" y="802" width="26" height="12.5" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="557" y="816" width="26" height="12.5" rx="1" style="fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)" stroke-width="1"/>
  <rect x="557" y="830" width="26" height="12.5" rx="1" style="fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <text class="s" x="570" y="857" text-anchor="middle" style="opacity: 1; font-weight: 600">O</text>
  <line x1="122" y1="786" x2="122" y2="690" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-4-m0)"/>
  <line x1="166" y1="786" x2="166" y2="760" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-4-m0)"/>
  <line x1="426" y1="786" x2="426" y2="760" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-4-m0)"/>
  <path d="M570.0,635 V786" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.5" marker-end="url(#fig-1-4-m2)"/>
  <text class="s" x="576" y="743" style="fill: var(--fig-hi); opacity: 1">tl.store</text>
  <rect class="band" x="10" y="884" width="620" height="96" rx="8"/>
  <text class="lab2" x="18" y="902">第 j 轮</text>
  <text class="s" x="18" y="920">Algorithm 1</text>
  <text class="s" x="18" y="934">第 9–10 行</text>
  <text class="lab" x="92" y="914">① m<tspan dy="3.5" font-size="72%">new</tspan><tspan dy="-3.5"> = max(m, rowmax(S</tspan><tspan dy="3.5" font-size="72%">ij</tspan><tspan dy="-3.5">))</tspan></text>
  <text class="lab" x="92" y="938">② P̃<tspan dy="3.5" font-size="72%">ij</tspan><tspan dy="-3.5"> = exp(S</tspan><tspan dy="3.5" font-size="72%">ij</tspan><tspan dy="-3.5"> − m</tspan><tspan dy="3.5" font-size="72%">new</tspan><tspan dy="-3.5">)</tspan></text>
  <text class="lab" x="92" y="962">③ ℓ ← exp(m − m<tspan dy="3.5" font-size="72%">new</tspan><tspan dy="-3.5">)·ℓ + rowsum(P̃</tspan><tspan dy="3.5" font-size="72%">ij</tspan><tspan dy="-3.5">)</tspan></text>
  <text class="lab" x="364" y="914">④ O<tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5"> ← exp(m − m</tspan><tspan dy="3.5" font-size="72%">new</tspan><tspan dy="-3.5">)·O</tspan><tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5"> + P̃</tspan><tspan dy="3.5" font-size="72%">ij</tspan><tspan dy="-3.5">V</tspan><tspan dy="3.5" font-size="72%">j</tspan></text>
  <text class="lab" x="364" y="938">⑤ m ← m<tspan dy="3.5" font-size="72%">new</tspan></text>
  <text class="lab" x="364" y="962">循环后：O<tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5"> ← O</tspan><tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5"> / ℓ</tspan></text>
  </g>
</svg>
<figcaption><strong>图 1-4</strong> Algorithm 1 在 Triton 里的执行，画法同 GPU 与 Triton 入门的<a href="/blog/gpu-triton-intro/#fig-t-4">图 2-5</a>，下标按 Triton 从 0 数。上：causal 时每个 program 算哪些 block（$B_q = 2B_k$），右边放大了 pid_i = 2 部分 mask 的两个 block；下：pid_i = 2 这个 program 在 $j = 4$ 这一轮。第 0 个 block 没有整行被 mask 的行，所以 $m$ 不会停在 $-\infty$。</figcaption>
</figure>

FA1 论文的 Algorithm 1 循环是反过来套的：外层走 $K$/$V$ block，内层走 $Q$ block，所以内层每一步都要从 HBM 读出 $O_i$、$\ell_i$、$m_i$，更新后再写回去，每一步还要用 $\mathrm{diag}(\ell_i)^{-1}$ 把 $O_i$ 归一化一次。FA2 把 $i$ 放到外层，这几个量一直留在片上，归一化也只在第 12 行做一次，这是[第 2 节](#fa2)的内容；IO 复杂度对两种顺序都一样。

前向只存 $O$ 和 $L$，不存 $P$，反向却要用到 $P$，这是下一小节的事。

### 1.4 反向的 recomputation {#fa1-recompute}

*待补。*

## 2 FlashAttention-2：让 GPU 忙起来 {#fa2}

<div class="fa-intuit"><b>思路：</b>v1 搬的数据量已经对了，但 GPU 大部分时候闲着。v2 不改数学，只改调度：分出多得多的互相独立的工作，让每个 SM 都有活干；同时少花指令在 matmul 以外的簿记上。</div>

FlashAttention-2（Dao，2023）最要紧的改动有三处。

1. **减少非 matmul 的 FLOPs。** 非 matmul 的计算（exp、行最大值和行和、rescale 的乘法）跑在 FP32 ALU 和特殊函数单元上，不走 Tensor core。A100 的 FP32 算力是 1.95e13 FLOPS，FP16 matmul 是 3.12e14 FLOPS，所以每个非 matmul 的 FLOP 大约贵 16 倍。v1 每一步都让输出保持归一化，除了免不掉的 $\exp(m_{\text{old}}-m_{\text{new}})$ 修正，还要再除以当前的累计和。FA2 的累加器不做归一化，每个 block 只乘 $\exp(m_{\text{old}}-m_{\text{new}})$ 这一项修正，最后才除一次 $\ell$（即 [Algorithm 1](#fa-algorithm) 第 12 行）；给反向存的也只有每行一个 logsumexp $L=m+\log\ell$，不再把 $m$ 和 $\ell$ 都存下来。
2. **交换内外层循环，在序列长度上并行。** v1 只在 (batch × head) 上并行。现在的序列长、每张卡上的 batch 小，这样分出的 thread block 可能不够占满 GPU 的 SM（[图 2-1](#fig-2-1)）。FA2 把 query block 的循环放到外层，各个行 block 之间互相独立：前向里每个 thread block 负责一个 query 行 block。反向里每个 thread block 负责一个 key/value 列 block，在本地累加自己的 $dK_j$、$dV_j$，$dQ$ 则用 atomic add 累加到一块共享的缓冲区里。

<figure id="fig-2-1">
<svg viewBox="0 0 640 168" width="100%" role="img" aria-label="例子：一条 8K token 的序列、16 个 head，跑在有 108 个 SM 的 A100 上。FlashAttention-1 启动 16 个 thread block，108 个 SM 里只有 16 个在忙；FlashAttention-2 还把每个 head 的 query 切成 64 个 block，启动 1024 个 thread block，108 个 SM 全部在忙">
  <style>
    .sm-h { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .sm-s { font-size: 11px; fill: currentColor; opacity: .75; }
    .sm-f { font-size: 11.5px; font-weight: 600; fill: currentColor; }
  </style>
  <defs>
    <pattern id="sm-busy" x="26" y="46" width="15" height="15" patternUnits="userSpaceOnUse">
      <rect width="12" height="12" rx="2" fill="var(--fig-1)"/>
    </pattern>
    <pattern id="sm-idle" x="26" y="46" width="15" height="15" patternUnits="userSpaceOnUse">
      <rect x=".5" y=".5" width="11" height="11" rx="2" fill="none" stroke="currentColor" stroke-opacity=".3"/>
    </pattern>
  </defs>
  <text x="26" y="18" class="sm-h">v1：每个 (batch, head) 一个 thread block</text>
  <text x="26" y="34" class="sm-s">batch 1 × 16 个 head = 16 个 thread block</text>
  <rect x="26" y="46" width="270" height="90" fill="url(#sm-idle)"/>
  <rect x="26" y="46" width="240" height="15" fill="url(#sm-busy)"/>
  <text x="26" y="156" class="sm-f">108 个 SM 里 16 个在忙（15%）</text>
  <text x="341" y="18" class="sm-h">v2：每个 128 行的 query block 也各一个</text>
  <text x="341" y="34" class="sm-s">16 个 head × 64 个 query block = 1024 个 thread block</text>
  <rect x="341" y="46" width="270" height="90" fill="url(#sm-busy)"/>
  <text x="341" y="156" class="sm-f">108 个 SM 全部在忙（约 9.5 个 wave）</text>
</svg>
<figcaption><strong>图 2-1</strong> 在序列上并行的作用，以一个示例说明：A100 上一条 8K token 的序列、16 个 head，方块是 A100 的 108 个 SM。每个 (batch, head) 一个 thread block 时，只有 16 个 SM 分到活，其余空闲；把每个 head 的 query 按 128 行切成 block，得到 1024 个互相独立的 thread block，足够把整张卡填满好几遍。</figcaption>
</figure>

3. **更好的 warp 分工。** 在一个 thread block 内，v1 把 $K$/$V$ 分给各个 warp，每个 warp 都能看到全部 $Q$（split-K 方案）；于是每个 warp 都得把自己的部分结果写进 shared memory，同步之后再跨 warp 归约。FA2 反过来，把 $Q$ 分给各个 warp，每个 warp 都能看到全部 $K$/$V$（split-Q），这样每个 warp 算出的就是一段完整的输出，**不需要通过 shared memory 做跨 warp 归约**（[图 2-2](#fig-2-2)）。各个 warp 一起加载每个 $K$/$V$ tile 时，仍然要在整个 block 的 barrier 上会合，但它们从不交换部分输出。

<figure id="fig-2-2">
<svg viewBox="0 0 640 256" width="100%" role="img" aria-label="split-K 的 warp 分工要在 shared memory 里跨 warp 归约部分输出；split-Q 让每个 warp 得到一段完整的输出，不需要跨 warp 归约">
  <style>
    .wp-title { font-size: 13px; font-weight: 600; fill: currentColor; }
    .wp-lab   { font-size: 12px; fill: var(--fig-on-1); text-anchor: middle; }
    .wp-lab-dk { font-size: 12px; fill: var(--fig-on-3); text-anchor: middle; }
    .wp-sub   { font-size: 11px; fill: currentColor; opacity: .7; text-anchor: middle; }
    .wp-box   { font-size: 11px; fill: currentColor; text-anchor: middle; }
  </style>
  <text x="170" y="18" text-anchor="middle" class="wp-title">v1：split-K</text>
  <rect x="30" y="30" width="280" height="30" rx="4" fill="rgba(128,128,128,.12)" stroke="currentColor" stroke-opacity=".35"/>
  <text x="170" y="50" class="wp-box">Q：共享，每个 warp 都能看到</text>
  <g>
    <rect x="30"  y="78" width="60" height="34" rx="4" fill="var(--fig-2)"/>
    <text x="60"  y="99" class="wp-lab-dk">K/V 0</text>
    <rect x="100" y="78" width="60" height="34" rx="4" fill="var(--fig-2)"/>
    <text x="130" y="99" class="wp-lab-dk">K/V 1</text>
    <rect x="170" y="78" width="60" height="34" rx="4" fill="var(--fig-2)"/>
    <text x="200" y="99" class="wp-lab-dk">K/V 2</text>
    <rect x="240" y="78" width="60" height="34" rx="4" fill="var(--fig-2)"/>
    <text x="270" y="99" class="wp-lab-dk">K/V 3</text>
  </g>
  <text x="170" y="128" class="wp-sub">warp 0–3：各算一段 K/V 的部分和</text>
  <path d="M60,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <path d="M130,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <path d="M210,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <path d="M280,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <rect x="90" y="190" width="160" height="32" rx="4" fill="var(--fig-1)" fill-opacity=".16" stroke="var(--fig-1)"/>
  <text x="170" y="211" class="wp-box">在 shared memory 里归约</text>
  <text x="170" y="242" class="wp-sub">写部分和 · barrier 同步 · 相加</text>
  <line x1="330" y1="10" x2="330" y2="250" stroke="currentColor" stroke-opacity=".15"/>
  <text x="490" y="18" text-anchor="middle" class="wp-title">v2：split-Q</text>
  <rect x="350" y="30" width="280" height="30" rx="4" fill="rgba(128,128,128,.12)" stroke="currentColor" stroke-opacity=".35"/>
  <text x="490" y="50" class="wp-box">K、V：共享，每个 warp 都能看到</text>
  <g>
    <rect x="350" y="78" width="60" height="34" rx="4" fill="var(--fig-1)"/>
    <text x="380" y="99" class="wp-lab">Q 0</text>
    <rect x="420" y="78" width="60" height="34" rx="4" fill="var(--fig-1)"/>
    <text x="450" y="99" class="wp-lab">Q 1</text>
    <rect x="490" y="78" width="60" height="34" rx="4" fill="var(--fig-1)"/>
    <text x="520" y="99" class="wp-lab">Q 2</text>
    <rect x="560" y="78" width="60" height="34" rx="4" fill="var(--fig-1)"/>
    <text x="590" y="99" class="wp-lab">Q 3</text>
  </g>
  <text x="490" y="128" class="wp-sub">warp 0–3：各算出一段 Q 的完整输出</text>
  <line x1="380" y1="134" x2="380" y2="188" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <line x1="450" y1="134" x2="450" y2="188" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <line x1="520" y1="134" x2="520" y2="188" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <line x1="590" y1="134" x2="590" y2="188" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <defs>
    <marker id="wp-arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" opacity=".35"/>
    </marker>
  </defs>
  <rect x="350" y="190" width="60" height="32" rx="4" fill="var(--fig-1)" fill-opacity=".14" stroke="var(--fig-1)" stroke-opacity=".5"/>
  <text x="380" y="211" class="wp-box">O 0</text>
  <rect x="420" y="190" width="60" height="32" rx="4" fill="var(--fig-1)" fill-opacity=".14" stroke="var(--fig-1)" stroke-opacity=".5"/>
  <text x="450" y="211" class="wp-box">O 1</text>
  <rect x="490" y="190" width="60" height="32" rx="4" fill="var(--fig-1)" fill-opacity=".14" stroke="var(--fig-1)" stroke-opacity=".5"/>
  <text x="520" y="211" class="wp-box">O 2</text>
  <rect x="560" y="190" width="60" height="32" rx="4" fill="var(--fig-1)" fill-opacity=".14" stroke="var(--fig-1)" stroke-opacity=".5"/>
  <text x="590" y="211" class="wp-box">O 3</text>
  <text x="490" y="242" class="wp-sub">各段输出互相独立，不需要跨 warp 归约</text>
</svg>
<figcaption><strong>图 2-2</strong> 同样四个 warp，切法相反。v1 切 $K$/$V$，四份部分和要经 shared memory 合并；v2 切 $Q$，每个 warp 的 matmul 一算完，它那段输出就已经是完整的。</figcaption>
</figure>

v2 不需要任何新的硬件特性，就是一篇「把手里已有的 GPU 用得更好」的论文，这也是它在 Hopper 以外的 GPU（以及 AMD 的 ROCm 移植版）上部署最广的原因。

*RTX 5090 实测待补：v1 和 v2 的前向、反向吞吐和占 Tensor core 峰值的比例。*

Hopper 新增了几种异步硬件，v1 和 v2 都不是为它们设计的，[第 3 节](#fa3)的 FlashAttention-3 就是围绕这些硬件写的。

## 3 FlashAttention-3：用上 Hopper 新加的硬件 {#fa3}

<div class="fa-intuit"><b>思路：</b>在 Hopper 上，搬 tile 和做矩阵乘由不同的硬件单元负责，两者可以同时进行。v3 把 kernel 改成一条流水线：一个 warp 只负责取数据，其余的只负责计算，两组负责计算的 warpgroup 轮流工作，Tensor core 不用停下来等 softmax。</div>

H100 新加了两个原语。一是 **Tensor Memory Accelerator（TMA）**，一个在 HBM 和 shared memory 之间异步整块搬 tile 的硬件单元（TMA 和异步同步见入门篇的 [Appendix C](/blog/gpu-triton-intro/#tma)）。二是 **warpgroup 级的 MMA（WGMMA）**，由 4 个 warp 组成的 warpgroup 一起发出的 Tensor core 指令，比以前的更宽，而且是异步的。FlashAttention-3（Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao，2024）围绕这两个原语来设计，FP8 部分另外用了 block 量化和 incoherent processing。具体有三点。

1. **Warp specialization（producer/consumer）。** FA2 里每个 warp 既要发出自己那份 tile 的拷贝（用 Ampere 的 cp.async，已经是异步的），又要做计算，各阶段之间用整个 block 的 barrier 同步。FA3 改成由一个 producer warpgroup 发 TMA load（实际发指令的只是其中一个 warp，这个 warpgroup 用 `setmaxnreg` 把 register 让出来），把 tile 搬进 shared memory 里一个多级的环形缓冲，缓冲的状态由硬件 barrier 跟踪；register 大部分分给 consumer warpgroup，它们只对已经搬好的 tile 跑 WGMMA（[图 3-1](#fig-3-1)，这种分工在入门篇的 [C.3](/blog/gpu-triton-intro/#tma-ws) 讲过）。这样 kernel 就成了一条显式的软件 pipeline。

<figure id="fig-3-1">
<svg viewBox="0 0 640 186" width="100%" role="img" aria-label="Warp specialization：producer warp 用 TMA 把 K、V tile 从 HBM 搬进 shared memory 里有 4 个 slot 的环形缓冲，两个 consumer warpgroup 取走已经搬满的 tile，做 WGMMA 和 softmax，用完的 slot 通过硬件 barrier 还给 producer">
  <style>
    .pc-h { font-size: 12px; font-weight: 600; fill: currentColor; text-anchor: middle; }
    .pc-s { font-size: 11px; fill: currentColor; opacity: .75; text-anchor: middle; }
    .pc-t { font-size: 10.5px; fill: currentColor; opacity: .75; text-anchor: middle; }
  </style>
  <rect x="16" y="58" width="80" height="74" rx="6" fill="rgba(128,128,128,.10)" stroke="currentColor" stroke-opacity=".4"/>
  <text x="56" y="91" class="pc-h">HBM</text>
  <text x="56" y="108" class="pc-s">K、V tile</text>
  <line x1="98" y1="95" x2="127" y2="95" stroke="currentColor" stroke-opacity=".5" stroke-width="1.5"/>
  <polygon points="134,95 126,91 126,99" fill="currentColor" fill-opacity=".5"/>
  <rect x="136" y="66" width="100" height="58" rx="6" fill="var(--fig-2)" fill-opacity=".22" stroke="var(--fig-2)"/>
  <text x="186" y="90" class="pc-h">producer warp</text>
  <text x="186" y="107" class="pc-s">只发 TMA load</text>
  <line x1="238" y1="95" x2="261" y2="95" stroke="currentColor" stroke-opacity=".5" stroke-width="1.5"/>
  <polygon points="268,95 260,91 260,99" fill="currentColor" fill-opacity=".5"/>
  <text x="352" y="54" class="pc-s">shared memory 环形缓冲</text>
  <rect x="270" y="66" width="38" height="58" rx="4" fill="var(--fig-2)" fill-opacity=".55" stroke="var(--fig-2)"/>
  <rect x="312" y="66" width="38" height="58" rx="4" fill="var(--fig-2)" fill-opacity=".55" stroke="var(--fig-2)"/>
  <rect x="354" y="66" width="38" height="58" rx="4" fill="none" stroke="var(--fig-2)"/>
  <rect x="354" y="95" width="38" height="29" fill="var(--fig-2)" fill-opacity=".55"/>
  <rect x="396" y="66" width="38" height="58" rx="4" fill="none" stroke="var(--fig-2)" stroke-dasharray="3 2"/>
  <text x="289" y="140" class="pc-t">已满</text>
  <text x="331" y="140" class="pc-t">已满</text>
  <text x="373" y="140" class="pc-t">装入中</text>
  <text x="415" y="140" class="pc-t">空闲</text>
  <line x1="436" y1="88" x2="467" y2="72" stroke="currentColor" stroke-opacity=".5" stroke-width="1.5"/>
  <polygon points="474,69 465,68 469,76" fill="currentColor" fill-opacity=".5"/>
  <line x1="436" y1="102" x2="467" y2="118" stroke="currentColor" stroke-opacity=".5" stroke-width="1.5"/>
  <polygon points="474,121 469,114 465,122" fill="currentColor" fill-opacity=".5"/>
  <rect x="476" y="48" width="148" height="42" rx="6" fill="var(--fig-1)" fill-opacity=".12" stroke="var(--fig-1)"/>
  <text x="550" y="66" class="pc-h">consumer warpgroup 1</text>
  <text x="550" y="82" class="pc-s">WGMMA + softmax</text>
  <rect x="476" y="100" width="148" height="42" rx="6" fill="var(--fig-1)" fill-opacity=".12" stroke="var(--fig-1)"/>
  <text x="550" y="118" class="pc-h">consumer warpgroup 2</text>
  <text x="550" y="134" class="pc-s">WGMMA + softmax</text>
  <path d="M550,144 V168 H186 V133" fill="none" stroke="currentColor" stroke-opacity=".45" stroke-dasharray="4 3"/>
  <polygon points="186,126 182,134 190,134" fill="currentColor" fill-opacity=".45"/>
  <text x="368" y="161" class="pc-s">slot 用完 → 通过硬件 barrier 还给 producer</text>
</svg>
<figcaption><strong>图 3-1</strong> Warp specialization 的分工。producer warp 只发 TMA，把 tile 搬进 shared memory 的环形 slot；consumer warpgroup 只算已经搬满的 slot。只要环形缓冲一直领先，Tensor core 就不用等 load。</figcaption>
</figure>

2. **GEMM 和 softmax 重叠。** 在单个 tile 上，softmax 依赖这个 block 的 $QK^\top$，$PV$ 的 GEMM 又依赖 softmax，所以它们只能串行。FA3 从两个互补的方向拆开这条依赖链。第一个是 warpgroup 之间的 ping-pong（[图 3-2](#fig-3-2)）：两个 consumer warpgroup 各自负责一个不同的 query tile，用 barrier 强制它们交替执行，warpgroup A 做 GEMM 时 warpgroup B 做 softmax，反过来也一样。第二个是 warpgroup 内部的 pipeline：在同一个 warpgroup 里，下一个 block 的 $QK^\top$ 和当前 block 的 $PV$ 这两组异步 WGMMA 按一定的时机发出，让 softmax 在它们执行期间进行。

<figure id="fig-3-2">
<svg viewBox="0 0 640 210" width="100%" role="img" aria-label="Ping-pong 调度：两个 warpgroup 交替做 GEMM 和 softmax，一个做 softmax 的时段和另一个做 matmul 的时段重叠；在 H100 上 softmax 的耗时约为 GEMM 的一半">
  <style>
    .pp-lane { font-size: 12px; fill: currentColor; opacity: .8; }
    .pp-seg  { font-size: 11px; fill: var(--fig-on-1); text-anchor: middle; }
    .pp-seg-dk { font-size: 11px; fill: var(--fig-on-3); text-anchor: middle; }
    .pp-wait { font-size: 11px; fill: currentColor; opacity: .7; text-anchor: middle; }
    .pp-sub  { font-size: 11px; fill: currentColor; opacity: .7; }
    .pp-leg  { font-size: 11px; fill: currentColor; }
  </style>
  <text x="20" y="45" class="pp-lane">Warpgroup A</text>
  <text x="20" y="105" class="pp-lane">Warpgroup B</text>
  <g>
    <rect x="130" y="24" width="120" height="34" fill="var(--fig-1)"/>
    <text x="190" y="45" class="pp-seg">GEMM</text>
    <rect x="250" y="24" width="60" height="34" fill="var(--fig-2)"/>
    <text x="280" y="45" class="pp-seg-dk">softmax</text>
    <rect x="310" y="24" width="60" height="34" fill="rgba(128,128,128,.10)"/>
    <text x="340" y="45" class="pp-wait">等待</text>
    <rect x="370" y="24" width="120" height="34" fill="var(--fig-1)"/>
    <text x="430" y="45" class="pp-seg">GEMM</text>
    <rect x="490" y="24" width="60" height="34" fill="var(--fig-2)"/>
    <text x="520" y="45" class="pp-seg-dk">softmax</text>
    <rect x="550" y="24" width="60" height="34" fill="rgba(128,128,128,.10)"/>
    <text x="580" y="45" class="pp-wait">等待</text>
  </g>
  <g>
    <rect x="130" y="84" width="60" height="34" fill="var(--fig-2)"/>
    <text x="160" y="105" class="pp-seg-dk">softmax</text>
    <rect x="190" y="84" width="60" height="34" fill="rgba(128,128,128,.10)"/>
    <text x="220" y="105" class="pp-wait">等待</text>
    <rect x="250" y="84" width="120" height="34" fill="var(--fig-1)"/>
    <text x="310" y="105" class="pp-seg">GEMM</text>
    <rect x="370" y="84" width="60" height="34" fill="var(--fig-2)"/>
    <text x="400" y="105" class="pp-seg-dk">softmax</text>
    <rect x="430" y="84" width="60" height="34" fill="rgba(128,128,128,.10)"/>
    <text x="460" y="105" class="pp-wait">等待</text>
    <rect x="490" y="84" width="120" height="34" fill="var(--fig-1)"/>
    <text x="550" y="105" class="pp-seg">GEMM</text>
  </g>
  <line x1="130" y1="18" x2="130" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="250" y1="18" x2="250" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="370" y1="18" x2="370" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="490" y1="18" x2="490" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="610" y1="18" x2="610" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <text x="610" y="145" text-anchor="end" class="pp-sub">时间 →</text>
  <path d="M250,122 L250,134 L370,134 L370,122" fill="none" stroke="currentColor" stroke-opacity=".4"/>
  <text x="310" y="160" text-anchor="middle" class="pp-sub">A 做 softmax 时 B 在做 GEMM，两者用的是不同的计算单元</text>
  <g>
    <rect x="130" y="184" width="14" height="14" fill="var(--fig-1)"/>
    <text x="150" y="195" class="pp-leg">GEMM（Tensor core）</text>
    <rect x="300" y="184" width="14" height="14" fill="var(--fig-2)"/>
    <text x="320" y="195" class="pp-leg">softmax（exp/ALU 单元）</text>
  </g>
</svg>
<figcaption><strong>图 3-2</strong> 两个 warpgroup 的相位错开一步：一个在做 softmax 的标量运算时，另一个正在 Tensor core 上做 matmul，非 matmul 的开销藏进了 matmul 的时间里，不再和它串行。各段宽度是示意，按 H100、head dim 128 的比例画，exp 的耗时约为与它重叠的 GEMM 的一半。</figcaption>
</figure>

3. **FP8：block 量化和 incoherent processing。** 在 matmul 之前直接把 $Q$、$K$ 量化成 FP8，吞吐上很划算，但精度不够：少数离群值占满了量化范围，其余数值的分辨率被压得很低（FP8 和 block 量化见[单卡优化篇](/blog/gpu-single-gpu-optimization/#precision)）。FA3 用了两个互补的办法。第一是 block 量化：每个 tile 一个 FP8 scale，而不是整个张量一个，离群值的影响只限于它所在的 block；kernel 本来就按 tile 计算，所以这样做没有额外开销。第二是 incoherent processing：量化前把 $Q$ 和 $K$ 乘上同一个随机正交矩阵 $R$。因为 $R$ 正交，$(QR)(KR)^\top = QRR^\top K^\top = QK^\top$，attention 分数在数学上不变；但 $QR$ 的每个元素现在是许多原始元素的随机混合，任何一个坐标都不会被单个离群值主导（[图 3-3](#fig-3-3)）。实现上用的是带随机符号翻转的 Hadamard 变换，长度为 $d$ 的一行代价是 $O(d\log d)$，稠密的旋转则要 $O(d^2)$。

<figure id="fig-3-3">
<svg viewBox="0 0 640 222" width="100%" role="img" aria-label="一个 8 维向量，有一个 8.0 的离群值，其余 7 个元素的绝对值都小于 0.4；随机翻转符号并做 Hadamard 旋转后，每个元素的绝对值都在 2.3 到 3.1 之间，最大值从 8.0 降到 3.1，向量长度保持 8.03">
  <style>
    .ic-h { font-size: 12px; font-weight: 600; fill: currentColor; text-anchor: middle; }
    .ic-s { font-size: 11px; fill: currentColor; opacity: .75; }
    .ic-c { font-size: 11px; fill: currentColor; opacity: .75; text-anchor: middle; }
  </style>
  <text x="170" y="16" class="ic-h">旋转前：有一个离群值</text>
  <text x="490" y="16" class="ic-h">旋转后：随机翻转符号 + Hadamard 变换</text>
  <line x1="44" y1="30" x2="296" y2="30" stroke="currentColor" stroke-opacity=".4" stroke-dasharray="4 3"/>
  <line x1="44" y1="190" x2="296" y2="190" stroke="currentColor" stroke-opacity=".4" stroke-dasharray="4 3"/>
  <line x1="44" y1="110" x2="296" y2="110" stroke="currentColor" stroke-opacity=".35"/>
  <text x="46" y="44" class="ic-s">±8.0</text>
  <rect x="50" y="107" width="20" height="3" fill="var(--fig-1)"/>
  <rect x="80" y="110" width="20" height="2" fill="var(--fig-1)"/>
  <rect x="110" y="109" width="20" height="1" fill="var(--fig-1)"/>
  <rect x="140" y="30" width="20" height="80" fill="var(--fig-2)"/>
  <rect x="170" y="110" width="20" height="4" fill="var(--fig-1)"/>
  <rect x="200" y="108" width="20" height="2" fill="var(--fig-1)"/>
  <rect x="230" y="109" width="20" height="1" fill="var(--fig-1)"/>
  <rect x="260" y="110" width="20" height="3" fill="var(--fig-1)"/>
  <line x1="364" y1="78.9" x2="616" y2="78.9" stroke="currentColor" stroke-opacity=".4" stroke-dasharray="4 3"/>
  <line x1="364" y1="141.1" x2="616" y2="141.1" stroke="currentColor" stroke-opacity=".4" stroke-dasharray="4 3"/>
  <line x1="364" y1="110" x2="616" y2="110" stroke="currentColor" stroke-opacity=".35"/>
  <text x="366" y="72" class="ic-s">±3.1</text>
  <rect x="370" y="78.9" width="20" height="31.1" fill="var(--fig-1)"/>
  <rect x="400" y="110" width="20" height="26.2" fill="var(--fig-1)"/>
  <rect x="430" y="110" width="20" height="23.3" fill="var(--fig-1)"/>
  <rect x="460" y="81.7" width="20" height="28.3" fill="var(--fig-1)"/>
  <rect x="490" y="80.3" width="20" height="29.7" fill="var(--fig-1)"/>
  <rect x="520" y="110" width="20" height="29" fill="var(--fig-1)"/>
  <rect x="550" y="110" width="20" height="30.4" fill="var(--fig-1)"/>
  <rect x="580" y="81.7" width="20" height="28.3" fill="var(--fig-1)"/>
  <text x="170" y="212" class="ic-c">最大值 8.0 决定了量化的 scale</text>
  <text x="490" y="212" class="ic-c">最大值 3.1 · 长度不变（8.03）</text>
</svg>
<figcaption><strong>图 3-3</strong> 在一个 8 维的玩具向量上做 incoherent processing（数值是精确算出来的）。量化的 scale 由 block 里的最大值决定，一个 8.0 的离群值让其余 7 个小值只能挤在可表示范围里很窄的一段；随机翻转符号再做 Hadamard 变换，把离群值的能量分散到每个坐标上。最大值从 8.0 降到 3.1，向量长度不变；$Q$ 和 $K$ 做的是同一个旋转，所以 $QK^\top$ 里的每个点积也不变。</figcaption>
</figure>

*RTX 5090 实测待补：5090 没有 WGMMA，FA3 的 kernel 跑不了；这里改测同样思路在 5090 上能做到的部分，以及 FP8 路径的误差。*

[图 3-2](#fig-3-2) 里 softmax 只用 GEMM 一半左右的时间，到了 Blackwell 的 B200 上两者一样长，[第 4 节](#fa4)的 FA4 就从这里出发。

## 4 FlashAttention-4：瓶颈随硬件换代而转移 {#fa4}

<div class="fa-intuit"><b>思路：</b>Blackwell 把矩阵乘的速度提高了一倍，做 exp 的单元和 shared memory 却没变，原来藏在 matmul 后面的 softmax 现在和 matmul 用的时间一样长。FA4 直接削减这部分工作：一部分 exp 改到别的单元上算，跳过不影响结果的 rescale，把更多数据放进新增的 tensor memory。</div>

FlashAttention-4 的作者是 Zadouri、Hoehnerbach、Shah、Liu、Thakkar 和 Dao（Princeton、Meta、Colfax Research、NVIDIA、Georgia Tech 和 Together AI），2026 年 3 月发在 arXiv 上。论文针对 Blackwell 数据中心 GPU（B200/GB200）；开源实现（flash-attention 仓库里的 `flash_attn/cute`，安装包名是 `flash-attn-4`）也带了 Hopper 的 kernel。它是四代里最新的一代，在生产环境里经过的检验也最少。

论文的出发点是**硬件的非对称扩展**（asymmetric hardware scaling）。从 Hopper 到 Blackwell，dense BF16 的 Tensor core 吞吐翻了一倍（每个 SM 每个时钟 8192 和 4096 FLOPs；整卡约 2.25e15 和 1e15 FLOPS），shared memory 的读带宽（每个 SM 每个时钟 128 B）和做 exp 的 MUFU 单元（每个 SM 每个时钟 16 次）完全没变。以 $d=128$ 时一个 $128\times128$ 的 tile 为例（[图 4-1](#fig-4-1)），前向的两个 MMA 在 Hopper 上要 $4\cdot128^3/4096=2048$ 个周期，$128^2$ 个 exp 要 $128^2/16=1024$ 个周期，exp 只占 matmul 的一半，FA3 的 ping-pong 能把它们藏在 matmul 后面。B200 上 MMA 只要 1024 个周期，exp 仍然要 1024 个，只靠重叠已经藏不住了。这和最初的 FlashAttention 是同一类问题：各个硬件单元的吞吐配比失衡，最慢的那个决定 kernel 的速度。只是 FA1 卡在 HBM 带宽上，这里前向卡在 MUFU 的 exp 吞吐上，反向卡在 shared memory 带宽上。

<figure id="fig-4-1">
<svg viewBox="0 0 640 250" width="100%" role="img" aria-label="前向一个 tile（128×128，head 维度 128）每个 SM 的周期估算。H100：Tensor core 的 MMA 2048 个周期，exp 单元 1024 个周期。B200：MMA 1024 个周期，exp 单元 1024 个周期，shared memory 读 768 个周期。从 H100 到 B200，MMA 的时间减半，exp 的时间不变。">
  <style>
    .as-grp { font-size: 12px; font-weight: 600; fill: currentColor; }
    .as-lab { font-size: 12px; fill: currentColor; text-anchor: end; }
    .as-val { font-size: 11px; fill: currentColor; }
    .as-tick { font-size: 11px; fill: currentColor; opacity: .7; text-anchor: middle; }
    .as-note { font-size: 11px; fill: currentColor; opacity: .75; }
  </style>
  <line x1="160" y1="14" x2="160" y2="196" stroke="currentColor" stroke-opacity=".4"/>
  <line x1="265" y1="14" x2="265" y2="196" stroke="currentColor" stroke-opacity=".12" stroke-dasharray="2 3"/>
  <line x1="370" y1="14" x2="370" y2="196" stroke="currentColor" stroke-opacity=".3" stroke-dasharray="2 3"/>
  <line x1="475" y1="14" x2="475" y2="196" stroke="currentColor" stroke-opacity=".12" stroke-dasharray="2 3"/>
  <line x1="580" y1="14" x2="580" y2="196" stroke="currentColor" stroke-opacity=".12" stroke-dasharray="2 3"/>
  <text x="20" y="26" class="as-grp">H100</text>
  <text x="150" y="46" class="as-lab">MMA（Tensor core）</text>
  <rect x="160" y="33" width="420" height="18" fill="var(--fig-1)"/>
  <text x="586" y="46" class="as-val">2048</text>
  <text x="150" y="72" class="as-lab">exp（MUFU）</text>
  <rect x="160" y="59" width="210" height="18" fill="var(--fig-2)" fill-opacity=".25" stroke="var(--fig-2)" stroke-width="1.5"/>
  <text x="376" y="72" class="as-val">1024</text>
  <text x="420" y="72" class="as-note">MMA 的一半，藏得住</text>
  <text x="20" y="104" class="as-grp">B200</text>
  <text x="150" y="124" class="as-lab">MMA（Tensor core）</text>
  <rect x="160" y="111" width="210" height="18" fill="var(--fig-1)"/>
  <text x="376" y="124" class="as-val">1024</text>
  <text x="150" y="150" class="as-lab">exp（MUFU）</text>
  <rect x="160" y="137" width="210" height="18" fill="var(--fig-2)" fill-opacity=".25" stroke="var(--fig-2)" stroke-width="1.5"/>
  <text x="376" y="150" class="as-val">1024</text>
  <text x="420" y="150" class="as-note">和 MMA 一样长，藏不住</text>
  <text x="150" y="176" class="as-lab">shared memory 读</text>
  <rect x="160" y="163" width="157.5" height="18" fill="rgba(128,128,128,.12)" stroke="currentColor" stroke-opacity=".5" stroke-dasharray="3 2"/>
  <text x="323.5" y="176" class="as-val">768</text>
  <line x1="160" y1="196" x2="580" y2="196" stroke="currentColor" stroke-opacity=".4"/>
  <text x="160" y="211" class="as-tick">0</text>
  <text x="265" y="211" class="as-tick">512</text>
  <text x="370" y="211" class="as-tick">1024</text>
  <text x="475" y="211" class="as-tick">1536</text>
  <text x="580" y="211" class="as-tick">2048</text>
  <text x="370" y="232" class="as-tick">一个 128×128 tile、head 维度 128 时每个 SM 的周期数（越少越快）</text>
</svg>
<figcaption><strong>图 4-1</strong> 前向一个 tile（$M=N=d=128$）每个 SM 的周期数，是 roofline 估算，不是实测；MMA 按 $4MNd$ 除以每个 SM 每周期的 FLOPs 算（Hopper 是 $4MNd/4096=2048$），exp 按 $MN/16=1024$ 算。B200 的数取自 FA4 论文 Table 1，H100 用同样的公式套 Hopper 的 MMA 速率；Hopper 的 MMA tile 是 64 行，操作数重读的次数不同，所以 H100 没画 shared memory。</figcaption>
</figure>

FA4 的做法有四条：

1. **围绕完全异步的 MMA 和更大的 tile 重建 pipeline。** Blackwell 的 MMA 以 $128\times N$ 的 tile 为单位，累加结果异步写进 tensor memory，不经过 register。FA4 保留了 FA3 的 ping-pong，现在是在每个 thread block 的两个 128 行 $Q$ tile 之间交替；每行由一个 thread 负责，求行最大值和行和都不需要 warp shuffle。$P$ 经由 tensor memory 而不是 register 送进 $PV$ 的 MMA，所以输出的 rescale 可以挪到单独的 correction warpgroup 里做，不再占 softmax 的关键路径。tile 调度器按最长任务优先（longest-processing-time-first）分配，改善了 causal 和变长 batch 的负载均衡。
2. **软件模拟 exp，以及有条件的 softmax rescale。** 软件模拟不替代硬件 exp，而是多出一个算 exp 的地方。FA4 把每行 exp 里调好比例的一部分（10–25%）放到 FMA 单元上算，不走 MUFU：先拆成 $2^x = 2^{\lfloor x\rfloor}\cdot 2^{x-\lfloor x\rfloor}$（Cody–Waite range reduction），对 $f\in[0,1)$ 用 3 次多项式求 $2^f$，再把整数移位放进浮点数的指数位，得到 $2^{\lfloor x\rfloor}$。每次软件模拟的开销比一条硬件 `ex2` 大，但两边并行，总的 exp 吞吐提高了。有条件的 rescale 针对的是另一项非 matmul 开销，即 [1.2 节](#online-softmax)里 $O \leftarrow e^{m_{\text{old}}-m_{\text{new}}}O$ 的修正。这一步从来不是正确性所必需的，只是为了让数值不越界：$O$ 和 $\ell$ 对着同一个参考最大值累加，参考值取多少，$O/\ell$ 都是精确的。所以 FA4 沿用旧的最大值，只有新 block 把最大值抬高超过 $\log_2 256 = 8$ 时才更新，$P$ 里未归一化的元素因此最多放大 256 倍，FP32 累加器完全容得下（[图 4-2](#fig-4-2)）。

<figure id="fig-4-2">
<svg viewBox="0 0 640 240" width="100%" role="img" aria-label="十个 key/value block 上的有条件 rescale：分数的当前最大值在其中九个 block 上升，标准 online softmax 要 rescale 九次；FA4 保留旧的参考最大值，只在第 6 和第 9 个 block 更新，这两处真实最大值比参考值高出超过 8（log₂ 单位）">
  <style>
    .cr-s { font-size: 11px; fill: currentColor; opacity: .75; }
    .cr-t { font-size: 11px; fill: currentColor; }
    .cr-k { font-size: 10.5px; fill: currentColor; opacity: .65; text-anchor: middle; }
    .cr-hl { font-size: 11px; font-weight: 600; fill: currentColor; }
  </style>
  <line x1="60" y1="12" x2="82" y2="12" stroke="var(--fig-1)" stroke-width="2"/>
  <text x="88" y="16" class="cr-t">分数的当前最大值</text>
  <line x1="252" y1="12" x2="274" y2="12" stroke="var(--fig-2)" stroke-width="3.5"/>
  <text x="280" y="16" class="cr-t">FA4 实际用的最大值</text>
  <rect x="424" y="6" width="12" height="12" fill="var(--fig-2)" fill-opacity=".15"/>
  <text x="442" y="16" class="cr-t">允许的差距：最多 256×</text>
  <rect x="42" y="138.4" width="280" height="44.8" fill="var(--fig-2)" fill-opacity=".15"/>
  <rect x="322" y="85.2" width="168" height="44.8" fill="var(--fig-2)" fill-opacity=".15"/>
  <rect x="490" y="34.8" width="112" height="44.8" fill="var(--fig-2)" fill-opacity=".15"/>
  <line x1="40" y1="30" x2="40" y2="200" stroke="currentColor" stroke-opacity=".4"/>
  <line x1="40" y1="200" x2="604" y2="200" stroke="currentColor" stroke-opacity=".4"/>
  <text x="34" y="203" text-anchor="end" class="cr-k">0</text>
  <text x="34" y="147" text-anchor="end" class="cr-k">10</text>
  <text x="34" y="91" text-anchor="end" class="cr-k">20</text>
  <text x="34" y="35" text-anchor="end" class="cr-k">30</text>
  <text transform="translate(12 118) rotate(-90)" text-anchor="middle" class="cr-k">行最大值（log₂ 单位）</text>
  <polyline points="42,183.2 322,183.2 322,130 490,130 490,79.6 602,79.6" fill="none" stroke="var(--fig-2)" stroke-width="3.5"/>
  <polyline points="42,183.2 98,183.2 98,169.2 154,169.2 154,165.3 210,165.3 210,149.6 266,149.6 266,147.4 322,147.4 322,130 378,130 378,128.3 434,128.3 434,127.2 490,127.2 490,79.6 546,79.6 546,78.5 602,78.5" fill="none" stroke="var(--fig-1)" stroke-width="2"/>
  <circle cx="70" cy="183.2" r="3" fill="var(--fig-1)"/>
  <circle cx="126" cy="169.2" r="3" fill="var(--fig-1)"/>
  <circle cx="182" cy="165.3" r="3" fill="var(--fig-1)"/>
  <circle cx="238" cy="149.6" r="3" fill="var(--fig-1)"/>
  <circle cx="294" cy="147.4" r="3" fill="var(--fig-1)"/>
  <circle cx="350" cy="130" r="3" fill="var(--fig-1)"/>
  <circle cx="406" cy="128.3" r="3" fill="var(--fig-1)"/>
  <circle cx="462" cy="127.2" r="3" fill="var(--fig-1)"/>
  <circle cx="518" cy="79.6" r="3" fill="var(--fig-1)"/>
  <circle cx="574" cy="78.5" r="3" fill="var(--fig-1)"/>
  <circle cx="322" cy="130" r="6" fill="none" stroke="var(--fig-2)" stroke-width="2"/>
  <circle cx="490" cy="79.6" r="6" fill="none" stroke="var(--fig-2)" stroke-width="2"/>
  <text x="314" y="114" text-anchor="end" class="cr-hl">rescale</text>
  <text x="482" y="64" text-anchor="end" class="cr-hl">rescale</text>
  <text x="56" y="50" class="cr-s">标准 online softmax 在 9 次上升时都要 rescale，FA4 只做 2 次。</text>
  <text x="70" y="214" class="cr-k">1</text>
  <text x="126" y="214" class="cr-k">2</text>
  <text x="182" y="214" class="cr-k">3</text>
  <text x="238" y="214" class="cr-k">4</text>
  <text x="294" y="214" class="cr-k">5</text>
  <text x="350" y="214" class="cr-k">6</text>
  <text x="406" y="214" class="cr-k">7</text>
  <text x="462" y="214" class="cr-k">8</text>
  <text x="518" y="214" class="cr-k">9</text>
  <text x="574" y="214" class="cr-k">10</text>
  <text x="322" y="232" text-anchor="middle" class="cr-k">key/value block 序号</text>
</svg>
<figcaption><strong>图 4-2</strong> 一行上的有条件 rescale（示意数据）。当前最大值在 10 个 block 里有 9 个上升，标准 online softmax 每次都要 rescale $O$ 和 $\ell$；FA4 保留旧的参考最大值（粗线），只在真实最大值比它高出 8 以上（$\log_2$ 单位，阴影带）时才更新，这里只有两次。</figcaption>
</figure>

3. **tensor memory（TMEM）和 2-CTA MMA 模式，主要用在反向。** Blackwell 给每个 SM 加了 256KB 的 tensor memory，Tensor core 直接把累加结果写进去。2-CTA 模式下，同一个 cluster 里一对 SM 上的两个 thread block（CTA）一起执行一次 $M=256$ 的 MMA；每个 CTA 存 A tile 和累加器的一半，B 操作数也只暂存一半，B 的 shared memory 流量减半。B200 上反向受 shared memory 限制：按论文的模型，shared memory 流量约 3328 个周期，MMA 只有 2560 个周期。把更多中间结果放进 TMEM、再用 2-CTA 模式，可以降到约 2688 个周期，对 $dQ$ 的 atomic add 也减少了。
4. **用 CuTe-DSL 写，不用原生的 CUDA C++ 模板。** CuTe-DSL 是 CUTLASS 嵌在 Python 里的 kernel 语言。表达能力和 C++ 模板相当，编译快得多；attention 的变体（ALiBi、sliding window、soft-capping）可以写成普通的 Python 分数修改函数，JIT 编译进 kernel。

作者还和 cuDNN 团队合作，从 9.13/9.14 起把其中几项技术并进了 cuDNN。

*RTX 5090 实测待补：5090（sm_120）没有 TMEM 和 2-CTA MMA，FA4 的 Blackwell kernel 跑不了；这里改测软件模拟 exp 和有条件 rescale 这两项在 5090 上的效果。*

## 5 四代没变的东西 {#constant}

kernel 技巧一条条列下来，很容易忽略哪些东西始终没变：

- **四代算的是同一个数学函数。** 都是 $\mathrm{softmax}(QK^\top/\sqrt d)V$，差别只在浮点舍入；FP8 路径还有量化误差，论文没有证明它的上界，而是实测了 RMSE：加上 block 量化和 incoherent processing，误差比 baseline FP8 attention 低 2.6 倍。
- **没有一代降低了 FLOP 数的渐近量级。** 每个版本的计算量都是 $\Theta(N^2d)$；重计算会多出一点，有 causal mask 时跳过完全被 mask 掉的 tile，实际执行的计算比算完整矩阵再 mask 少一半左右。整条演进线一直在做同一件事：去掉 FLOP 数本身并不需要的数据搬运和非 matmul 开销。
- **每一代针对的都是上一代留下的新瓶颈。** v1 去掉了 $N^2$ 大小的矩阵在 HBM 上的往返。v2 解决了 v1 的调度留下的低 occupancy 和 warp 之间的通信开销。v3 用上了 v1、v2 出现时还没有的异步硬件（TMA/WGMMA），另外用 FP8 处理精度和吞吐之间的取舍。v4 应对的是 Blackwell 的非对称扩展：exp 的 FLOP 数很少，在前向里却和 matmul 占一样多的周期，反向里 shared memory 的流量还超过了 matmul。所以「attention 慢」不是一次能解决的问题：每一代解决的只是当时最紧的约束，下一代 GPU 又会带来新的约束。

## 6 总结 {#summary}

[表 6-1](#tab-6-1) 列出每一代的年份、目标硬件和核心做法。

| 版本 | 年份 | 目标硬件 | 核心做法 |
|---|---|---|---|
| FlashAttention | 2022 | Ampere 时代的 GPU | tiling + online softmax + 重计算，N×N 矩阵从不在 HBM 里实体化 |
| FlashAttention-2 | 2023 | Ampere/Ada/Hopper | 推迟归一化；交换内外层循环，并在序列长度上并行；warp 之间切分 Q（split-Q） |
| FlashAttention-3 | 2024 | Hopper（H100） | warp specialization 的异步 pipeline（TMA + WGMMA）；GEMM 与 softmax 重叠；FP8 加 block 量化和 incoherent processing |
| FlashAttention-4 | 2026 | Blackwell（B200/GB200），代码也能在 Hopper 上跑 | 异步 MMA + 更大的 tile；软件模拟 exp + 有条件的 rescale；TMEM + 2-CTA MMA；CuTe-DSL |
{#tab-6-1 caption="**表 6-1** FlashAttention 四代对比"}

## References {#references}

1. Dao, Fu, Ermon, Rudra, Ré. ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135), NeurIPS 2022. 图 1-2 的 tiling 示意图以这篇论文的 Figure 1 为基础。
2. Dao. ["FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"](https://arxiv.org/abs/2307.08691), ICLR 2024. 另见 [Stanford Hazy Research 的介绍文章](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)。
3. Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. ["FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"](https://arxiv.org/abs/2407.08608), NeurIPS 2024. 另见作者的[博客文章](https://tridao.me/blog/2024/flash3/)。
4. Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao. ["FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"](https://arxiv.org/abs/2603.05451), arXiv:2603.05451, 2026 年 3 月。
5. [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)。四个版本的参考实现。
6. Milakov, Gimelshein. ["Online normalizer calculation for softmax"](https://arxiv.org/abs/1805.02867), arXiv:1805.02867, 2018. online softmax 的出处，1.2 节用到。
7. [Transformer FLOPs 计算器](/projects/flops/)。按同样的思路估算整个 Transformer 的参数量、FLOPs、显存和训练时间。
