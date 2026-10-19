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
.fa-intuit{margin:1.4em 0;padding:.85em 1.1em;border-radius:12px;background:rgba(var(--color-primary-500),.08);border-left:3px solid rgb(var(--color-primary-500));line-height:1.65}
.fa-intuit>b:first-child{color:rgb(var(--color-primary-600))}
html.dark .fa-intuit>b:first-child{color:rgb(var(--color-primary-300))}
.fa-road{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:1.6em 0 .6em}
@media (max-width:760px){.fa-road{grid-template-columns:repeat(2,minmax(0,1fr))}}
.fa-step{border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:.7em .85em;font-size:.82em;line-height:1.5}
.fa-step b{display:block;font-size:1.1em}
.fa-step i{display:block;font-style:normal;opacity:.65;margin-bottom:.5em}
.fa-step span{display:block}
.fa-step span+span{margin-top:.45em;color:rgb(var(--color-primary-600))}
html.dark .fa-step span+span{color:rgb(var(--color-primary-300))}
@media (max-width:640px){.article-content figure{overflow-x:auto}.article-content figure>svg{min-width:520px}}
</style>

> **前置阅读：**[GPU 与 Triton 入门](/blog/gpu-triton-intro/)讲 GPU 的存储层级和 Triton 基础；[单卡 GPU 优化方法](/blog/gpu-single-gpu-optimization/)讲单卡上常用的优化方法：divergence、低精度、fusion、recomputation、coalescing、bank conflict、tiling；[谁偷走了 5090 的算力和显存](/blog/gpu-training-analysis/)讲 FLOPs、算术强度、roofline 模型和峰值显存，并在 RTX 5090 上实测了标准 attention 的运行时间和激活显存为什么都由 $N\times N$ 的 score 矩阵主导。这篇就从这两个问题出发。

Transformer 的每一层 self-attention 都要付两份代价：一份是 FLOPs，另一份是访存，后者的代价大得多。FlashAttention 是一族计算精确 attention 的 kernel，专门解决访存这个问题。它的每一代针对的瓶颈都不一样，而这个瓶颈要等上一代的问题解决之后才显露出来。这篇依次讲四代：每一代具体改了什么，这个改动在当时的硬件上为什么重要，以及四代之间有什么始终没变。

四代的瓶颈和做法列在下面。[第 1 节](#memory-bound)先说明 attention 为什么受限于访存，[第 2 节](#fa1)到[第 5 节](#fa4)逐代展开，[第 6 节](#constant)讲四代不变的东西，[第 7 节](#summary)是总结。

<div class="fa-road">
<div class="fa-step"><b>v1 · 2022</b><i>A100（Ampere）</i><span>瓶颈：N×N 的 score 矩阵要在慢速的 HBM 里来回读写。</span><span>做法：tiling，把所有步骤融合进一个 kernel，矩阵留在片上。</span></div>
<div class="fa-step"><b>v2 · 2023</b><i>A100（Ampere）</i><span>瓶颈：GPU 大部分时间闲着，thread block 太少，非 matmul 的计算太多。</span><span>做法：沿序列方向并行，最后只做一次归一化。</span></div>
<div class="fa-step"><b>v3 · 2024</b><i>H100（Hopper）</i><span>瓶颈：读 tile 和做计算仍然轮流进行。</span><span>做法：异步的 producer/consumer pipeline，softmax 和 matmul 重叠，FP8。</span></div>
<div class="fa-step"><b>v4 · 2026</b><i>B200（Blackwell）</i><span>瓶颈：Tensor core 的算力翻倍了，exp 单元和 shared memory 没有。</span><span>做法：一部分 exp 用软件算，跳过大部分 rescale，用 TMEM。</span></div>
</div>

原始论文的链接列在文末的 [References](#references)，建议至少直接读一遍第一篇论文，不要只看二手总结（包括这一篇）。

## 1 attention 受限于访存，而不是算力 {#memory-bound}

四代 FlashAttention 都建立在同一个判断上：标准 attention 慢在搬数据，不在计算。前面的[谁偷走了 5090 的算力和显存](/blog/gpu-training-analysis/)已经在 RTX 5090 上用 roofline 和实测讲过这一点，这里只补 FlashAttention 论文用的 A100 上的两层存储（[图 1-1](#fig-1-1)）。

<figure id="fig-1-1">
<svg viewBox="0 0 640 230" width="100%" role="img" aria-label="A100 的存储层级：上层是合计约 20 MB、速度快的片上 shared memory，下层是 40 到 80 GB、速度慢的 HBM">
  <style>
    .mh-label { font-size: 13px; fill: currentColor; }
    .mh-sub { font-size: 11px; fill: currentColor; opacity: .7; }
    .mh-box { stroke-width: 1.5; }
  </style>
  <rect x="165" y="20" width="310" height="56" rx="6" class="mh-box" fill="rgba(var(--color-primary-500), .16)" stroke="rgb(var(--color-primary-500))"/>
  <text x="320" y="44" text-anchor="middle" class="mh-label">片上 shared memory（SRAM）</text>
  <text x="320" y="62" text-anchor="middle" class="mh-sub">共约 20 MB（192 KB × 108 SM）· 总带宽约 1.9e13 B/s</text>
  <line x1="320" y1="80" x2="320" y2="106" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-start="url(#mh-arrow)" marker-end="url(#mh-arrow)"/>
  <defs>
    <marker id="mh-arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto-start-reverse">
      <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" opacity=".35"/>
    </marker>
  </defs>
  <rect x="90" y="110" width="460" height="100" rx="6" class="mh-box" fill="rgba(128,128,128,.10)" stroke="currentColor" stroke-opacity=".4"/>
  <text x="320" y="156" text-anchor="middle" class="mh-label">HBM（显存）</text>
  <text x="320" y="176" text-anchor="middle" class="mh-sub">40–80 GB · 约 1.5e12–2.0e12 B/s</text>
  <text x="320" y="224" text-anchor="middle" class="mh-sub">未按比例绘制</text>
</svg>
<figcaption><strong>图 1-1</strong> A100 的两层存储。HBM 的容量约是全部片上 shared memory 的 2000 倍（40 GB 对约 20 MB），shared memory 的带宽约是 HBM 的 10 倍。每一代 FlashAttention 的核心，都是用不同的办法让 $N\times N$ 的中间结果不落到下层。</figcaption>
</figure>

*RTX 5090 实测待补：标准 attention 各步（两个 matmul、mask、softmax、dropout）的耗时拆分。*

所以要让 attention 变快，关键是减少 $N\times N$ 矩阵进出 HBM 的次数。[第 2 节](#fa1)的 FlashAttention v1 更进一步，让这个矩阵完全不进 HBM。

## 2 FlashAttention：不让 N × N 矩阵进 HBM {#fa1}

第 1 节的结论是，标准 attention 慢在 $N \times N$ 的 score 矩阵要在 HBM 里来回读写，而不是算得多。FlashAttention 的第一版就从这里下手：不把这个矩阵完整地算出来，也就不用把它写进 HBM。

<div class="fa-intuit"><b>思路：</b>一次只在片上处理一小块 tile。每一行记两个随循环更新的数：到目前为止见过的最大 score，和指数的和。遇到更大的 score 时，把已经算好的部分结果修正一下。最后得到的输出和一次算完的完全相同。</div>

Dao、Fu、Ermon、Rudra、Ré 2022 年的论文把这当成 IO 问题来处理，用的是两个系统里常见的办法：**tiling** 和 **recomputation**，难点在于让 softmax 也能分块算。[图 2-1](#fig-2-1) 是两种实现的对比。

<figure id="fig-2-1">
<svg viewBox="0 0 640 300" width="100%" role="img" aria-label="标准 attention 跑三个 kernel，N × N 的 S 和 P 都要写到 HBM 再读回来；FlashAttention 只跑一个融合的 kernel，读 Q、K、V，写 O，经过 HBM 的只有 N × d 的数据">
  <style>
    .dm-h { font-size: 13px; font-weight: 600; fill: currentColor; }
    .dm-hs { font-size: 11px; fill: currentColor; opacity: .7; }
    .dm-k { font-size: 12.5px; font-weight: 600; fill: currentColor; text-anchor: middle; }
    .dm-ks { font-size: 11px; fill: currentColor; opacity: .75; text-anchor: middle; }
    .dm-m { font-size: 11px; fill: currentColor; text-anchor: middle; }
    .dm-a { font-size: 11px; font-weight: 600; fill: currentColor; }
  </style>
  <text x="166" y="18" text-anchor="middle" class="dm-h">标准 attention</text>
  <text x="166" y="35" text-anchor="middle" class="dm-hs">三个 kernel，每步一个</text>
  <text x="480" y="18" text-anchor="middle" class="dm-h">FlashAttention</text>
  <text x="480" y="35" text-anchor="middle" class="dm-hs">一个融合的 kernel，在片上逐个 tile 循环</text>
  <rect x="22" y="52" width="84" height="52" rx="6" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))"/>
  <text x="64" y="75" class="dm-k">QK<tspan dy="-5" font-size="9">T</tspan></text>
  <text x="64" y="93" class="dm-ks">Q, K → S</text>
  <rect x="124" y="52" width="84" height="52" rx="6" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))"/>
  <text x="166" y="75" class="dm-k">softmax</text>
  <text x="166" y="93" class="dm-ks">S → P</text>
  <rect x="226" y="52" width="84" height="52" rx="6" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))"/>
  <text x="268" y="75" class="dm-k">× V</text>
  <text x="268" y="93" class="dm-ks">P, V → O</text>
  <rect x="20" y="200" width="292" height="58" rx="6" fill="rgba(128,128,128,.10)" stroke="currentColor" stroke-opacity=".4"/>
  <text x="28" y="214" class="dm-a">HBM</text>
  <rect x="24" y="220" width="46" height="30" rx="4" fill="rgba(var(--color-primary-500), .15)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".6"/>
  <text x="47" y="239" class="dm-m">Q K V</text>
  <rect x="76" y="220" width="80" height="30" rx="4" fill="rgba(var(--color-secondary-500), .25)" stroke="rgb(var(--color-secondary-500))"/>
  <text x="116" y="239" class="dm-m">S (N×N)</text>
  <rect x="180" y="220" width="82" height="30" rx="4" fill="rgba(var(--color-secondary-500), .25)" stroke="rgb(var(--color-secondary-500))"/>
  <text x="221" y="239" class="dm-m">P (N×N)</text>
  <rect x="272" y="220" width="36" height="30" rx="4" fill="rgba(var(--color-primary-500), .15)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".6"/>
  <text x="290" y="239" class="dm-m">O</text>
  <line x1="44" y1="218" x2="44" y2="113" stroke="rgb(var(--color-primary-500))" stroke-width="2"/>
  <polygon points="44,105 39.5,114 48.5,114" fill="rgb(var(--color-primary-500))"/>
  <line x1="94" y1="105" x2="94" y2="207" stroke="rgb(var(--color-secondary-500))" stroke-width="7"/>
  <polygon points="94,219 86,206 102,206" fill="rgb(var(--color-secondary-500))"/>
  <line x1="140" y1="218" x2="140" y2="117" stroke="rgb(var(--color-secondary-500))" stroke-width="7"/>
  <polygon points="140,105 132,118 148,118" fill="rgb(var(--color-secondary-500))"/>
  <line x1="196" y1="105" x2="196" y2="207" stroke="rgb(var(--color-secondary-500))" stroke-width="7"/>
  <polygon points="196,219 188,206 204,206" fill="rgb(var(--color-secondary-500))"/>
  <line x1="244" y1="218" x2="244" y2="117" stroke="rgb(var(--color-secondary-500))" stroke-width="7"/>
  <polygon points="244,105 236,118 252,118" fill="rgb(var(--color-secondary-500))"/>
  <line x1="290" y1="105" x2="290" y2="211" stroke="rgb(var(--color-primary-500))" stroke-width="2"/>
  <polygon points="290,219 285.5,210 294.5,210" fill="rgb(var(--color-primary-500))"/>
  <text x="50" y="160" class="dm-a">Q, K</text>
  <text x="103" y="160" class="dm-a">S</text>
  <text x="149" y="160" class="dm-a">S</text>
  <text x="205" y="160" class="dm-a">P</text>
  <text x="253" y="160" class="dm-a">P, V</text>
  <text x="297" y="160" class="dm-a">O</text>
  <text x="166" y="276" text-anchor="middle" class="dm-hs">S、P（都是 N×N）写到 HBM 再读回来</text>
  <rect x="350" y="52" width="260" height="52" rx="6" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))"/>
  <rect x="362" y="64" width="62" height="28" rx="4" fill="rgba(var(--color-primary-500), .18)"/>
  <text x="393" y="83" class="dm-m">QK<tspan dy="-4" font-size="8">T</tspan></text>
  <text x="433" y="83" class="dm-m">→</text>
  <rect x="444" y="64" width="78" height="28" rx="4" fill="rgba(var(--color-secondary-500), .25)"/>
  <text x="483" y="83" class="dm-m">softmax</text>
  <text x="532" y="83" class="dm-m">→</text>
  <rect x="542" y="64" width="56" height="28" rx="4" fill="rgba(var(--color-primary-500), .18)"/>
  <text x="570" y="83" class="dm-m">× V</text>
  <rect x="348" y="200" width="264" height="58" rx="6" fill="rgba(128,128,128,.10)" stroke="currentColor" stroke-opacity=".4"/>
  <text x="356" y="214" class="dm-a">HBM</text>
  <rect x="366" y="220" width="46" height="30" rx="4" fill="rgba(var(--color-primary-500), .15)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".6"/>
  <text x="389" y="239" class="dm-m">Q K V</text>
  <rect x="562" y="220" width="36" height="30" rx="4" fill="rgba(var(--color-primary-500), .15)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".6"/>
  <text x="580" y="239" class="dm-m">O</text>
  <line x1="389" y1="218" x2="389" y2="113" stroke="rgb(var(--color-primary-500))" stroke-width="2"/>
  <polygon points="389,105 384.5,114 393.5,114" fill="rgb(var(--color-primary-500))"/>
  <line x1="580" y1="105" x2="580" y2="211" stroke="rgb(var(--color-primary-500))" stroke-width="2"/>
  <polygon points="580,219 575.5,210 584.5,210" fill="rgb(var(--color-primary-500))"/>
  <text x="395" y="160" class="dm-a">Q, K, V</text>
  <text x="587" y="160" class="dm-a">O</text>
  <text x="480" y="276" text-anchor="middle" class="dm-hs">经过 HBM 的只有 N×d 的张量</text>
  <line x1="150" y1="290" x2="176" y2="290" stroke="rgb(var(--color-secondary-500))" stroke-width="7"/>
  <text x="184" y="294" class="dm-hs">N×N 矩阵</text>
  <line x1="350" y1="290" x2="376" y2="290" stroke="rgb(var(--color-primary-500))" stroke-width="2"/>
  <text x="384" y="294" class="dm-hs">N×d 矩阵</text>
</svg>
<figcaption><strong>图 2-1</strong> 左边是标准实现（FlashAttention 论文的 Algorithm 0），每步一个 kernel，$N\times N$ 的 $S$ 和 $P$ 各要进出 HBM 一次，训练时 mask 和 dropout 还要再多几次。右边是 FlashAttention，三步合成一个在片上逐 tile 循环的 kernel，HBM 的读写只剩读 $Q$、$K$、$V$ 和写 $O$。</figcaption>
</figure>

### 2.1 Tiling {#fa1-tiling}

把 $Q$ 按行切成 $B_q$ 行一个的 block $Q_i$，$K$、$V$ 按行切成 $B_k$ 行一个的 block $K_j$、$V_j$，$d$ 方向不切。block 小到 $Q_i$、$K_j$、$V_j$、$O_i$ 能一起放进片上的 SRAM（即 shared memory）。每个局部的 score block $S_{ij}=Q_iK_j^\top/\sqrt d$ 都在片上算，完整的 $S$ 和 $P$ 从不写回 HBM（[图 2-2](#fig-2-2)）。tiling 为什么能少读显存，在[单卡 GPU 优化方法](/blog/gpu-single-gpu-optimization/#tiling)里按 matmul 算过一遍；这里多出来的一步是 softmax。

<figure id="fig-2-2">
<svg viewBox="0 0 680 612" width="100%" role="img" aria-label="Tiling：Q 按 B_q 行切成 block，K 和 V 按 B_k 行切成 block。Q_i 乘 K_j 转置得到 B_q 乘 B_k 的 score tile S_ij，修正后乘 B_k 乘 d 的 V_j，加进 B_q 乘 d 的 O_i。内层循环 j 走过每个 K/V block，外层循环 i 走过每个 Q block">
  <style>
    .os2 { font-size: 11px; fill: currentColor; }
    .os2-b { font-size: 12px; font-weight: 600; fill: currentColor; }
    .os2-s { font-size: 10.5px; fill: currentColor; opacity: .72; }
    .os2-c { font-size: 12px; font-weight: 600; text-anchor: middle; fill: currentColor; }
    .os2-hi { fill: var(--fig-hi); font-weight: 600; }
  </style>
<defs><marker id="tl-a" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".6"/></marker></defs>
<rect x="210" y="112" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="240" y="112" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="270" y="112" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="300" y="112" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="330" y="112" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="360" y="112" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="390" y="112" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="420" y="112" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="210" y="142" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="240" y="142" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="270" y="142" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="300" y="142" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="330" y="142" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="360" y="142" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="390" y="142" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="420" y="142" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="210" y="172" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".35" stroke-opacity=".3"/>
<rect x="240" y="172" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".35" stroke-opacity=".3"/>
<rect x="270" y="172" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".35" stroke-opacity=".3"/>
<rect x="330" y="172" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="360" y="172" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="390" y="172" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="420" y="172" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="210" y="202" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="240" y="202" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="270" y="202" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="300" y="202" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="330" y="202" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="360" y="202" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="390" y="202" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="420" y="202" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="210" y="232" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="240" y="232" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="270" y="232" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="300" y="232" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="330" y="232" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="360" y="232" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="390" y="232" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="420" y="232" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="210" y="262" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="240" y="262" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="270" y="262" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="300" y="262" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="330" y="262" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="360" y="262" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="390" y="262" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="420" y="262" width="30" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="300" y="172" width="30" height="30" style="fill: var(--fig-hi); stroke: var(--fig-1)" stroke-opacity=".5"/><text class="os2-b" x="315.0" y="191.0" text-anchor="middle" style="fill: var(--fig-on-3)">S<tspan dy="3.5" font-size="72%">ij</tspan></text>
<text class="os2-s" x="330.0" y="322" text-anchor="middle">S = QKᵀ 是 N<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> × N</tspan><tspan dy="3.5" font-size="72%">k</tspan><tspan dy="-3.5">，从不完整存下来</tspan></text>
<rect x="210" y="44" width="30" height="56" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="240" y="44" width="30" height="56" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="270" y="44" width="30" height="56" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="300" y="44" width="30" height="56" style="fill: var(--fig-3); stroke: var(--fig-1)" stroke-opacity=".5"/>
<rect x="330" y="44" width="30" height="56" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="360" y="44" width="30" height="56" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="390" y="44" width="30" height="56" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="420" y="44" width="30" height="56" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<line x1="210" y1="58.0" x2="450" y2="58.0" stroke="currentColor" stroke-opacity=".18"/>
<line x1="210" y1="72.0" x2="450" y2="72.0" stroke="currentColor" stroke-opacity=".18"/>
<line x1="210" y1="86.0" x2="450" y2="86.0" stroke="currentColor" stroke-opacity=".18"/>
<text class="os2-b" x="315.0" y="76.0" text-anchor="middle" style="fill: var(--fig-on-3)">K<tspan dy="3.5" font-size="72%">j</tspan><tspan dy="-3.5">ᵀ</tspan></text>
<text class="os2-b" x="200" y="62" text-anchor="end">Kᵀ</text>
<text class="os2-s" x="200" y="76" text-anchor="end">d × N<tspan dy="3.5" font-size="72%">k</tspan></text>
<rect x="132" y="112" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="584" y="112" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="132" y="142" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="584" y="142" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="132" y="172" width="56" height="30" style="fill: var(--fig-3); stroke: var(--fig-1)" stroke-opacity=".5"/>
<rect x="584" y="172" width="56" height="30" style="fill: var(--fig-3); stroke: var(--fig-1)" stroke-opacity=".5"/>
<rect x="132" y="202" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="584" y="202" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="132" y="232" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="584" y="232" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="132" y="262" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="584" y="262" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="490" y="112" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="490" y="142" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="490" y="172" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="490" y="202" width="56" height="30" style="fill: var(--fig-3); stroke: var(--fig-1)" stroke-opacity=".5"/>
<rect x="490" y="232" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="490" y="262" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="490" y="292" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<rect x="490" y="322" width="56" height="30" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity=".08" stroke-opacity=".3"/>
<line x1="146.0" y1="112" x2="146.0" y2="292" stroke="currentColor" stroke-opacity=".18"/>
<line x1="160.0" y1="112" x2="160.0" y2="292" stroke="currentColor" stroke-opacity=".18"/>
<line x1="174.0" y1="112" x2="174.0" y2="292" stroke="currentColor" stroke-opacity=".18"/>
<line x1="598.0" y1="112" x2="598.0" y2="292" stroke="currentColor" stroke-opacity=".18"/>
<line x1="612.0" y1="112" x2="612.0" y2="292" stroke="currentColor" stroke-opacity=".18"/>
<line x1="626.0" y1="112" x2="626.0" y2="292" stroke="currentColor" stroke-opacity=".18"/>
<line x1="504.0" y1="112" x2="504.0" y2="352" stroke="currentColor" stroke-opacity=".18"/>
<line x1="518.0" y1="112" x2="518.0" y2="352" stroke="currentColor" stroke-opacity=".18"/>
<line x1="532.0" y1="112" x2="532.0" y2="352" stroke="currentColor" stroke-opacity=".18"/>
<text class="os2-b" x="160.0" y="191.0" text-anchor="middle" style="fill: var(--fig-on-3)">Q<tspan dy="3.5" font-size="72%">i</tspan></text>
<text class="os2-b" x="612.0" y="191.0" text-anchor="middle" style="fill: var(--fig-on-3)">O<tspan dy="3.5" font-size="72%">i</tspan></text>
<text class="os2-b" x="518.0" y="221.0" text-anchor="middle" style="fill: var(--fig-on-3)">V<tspan dy="3.5" font-size="72%">j</tspan></text>
<text class="os2-b" x="160.0" y="308" text-anchor="middle">Q</text>
<text class="os2-s" x="160.0" y="322" text-anchor="middle">N<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> × d</tspan></text>
<text class="os2-b" x="518.0" y="368" text-anchor="middle">V</text>
<text class="os2-s" x="518.0" y="382" text-anchor="middle">N<tspan dy="3.5" font-size="72%">k</tspan><tspan dy="-3.5"> × d</tspan></text>
<text class="os2-b" x="612.0" y="308" text-anchor="middle">O</text>
<text class="os2-s" x="612.0" y="322" text-anchor="middle">N<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> × d</tspan></text>
<line x1="190" y1="187.0" x2="298" y2="187.0" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#tl-a)"/>
<line x1="315.0" y1="102" x2="315.0" y2="170" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#tl-a)"/>
<line x1="214" y1="34" x2="446" y2="34" stroke="currentColor" stroke-opacity=".45" stroke-width="1.4" marker-end="url(#tl-a)"/>
<text class="os2" x="330.0" y="26" text-anchor="middle">内层循环：j 走过每个 K/V block</text>
<line x1="116" y1="116" x2="116" y2="288" stroke="currentColor" stroke-opacity=".45" stroke-width="1.4" marker-end="url(#tl-a)"/>
<text class="os2" transform="translate(108 202.0) rotate(-90)" text-anchor="middle">外层循环：i 走过每个 Q block</text>
<line x1="8" y1="408" x2="672" y2="408" stroke="currentColor" stroke-opacity=".15"/>
<text class="os2-b" x="8" y="428" text-anchor="start">内层循环的一轮（都在片上）</text>
<rect x="40" y="452.0" width="56" height="40" style="fill: var(--fig-3); stroke: var(--fig-1)" stroke-opacity=".5"/>
<line x1="54.0" y1="452.0" x2="54.0" y2="492.0" stroke="currentColor" stroke-opacity=".18"/>
<line x1="68.0" y1="452.0" x2="68.0" y2="492.0" stroke="currentColor" stroke-opacity=".18"/>
<line x1="82.0" y1="452.0" x2="82.0" y2="492.0" stroke="currentColor" stroke-opacity=".18"/>
<text class="os2-b" x="68.0" y="476.0" text-anchor="middle" style="fill: var(--fig-on-3)">Q<tspan dy="3.5" font-size="72%">i</tspan></text>
<text class="os2" x="68.0" y="518" text-anchor="middle">B<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> × d</tspan></text>
<text class="os2-b" x="108" y="477.0" text-anchor="middle">×</text>
<rect x="120" y="444.0" width="40" height="56" style="fill: var(--fig-3); stroke: var(--fig-1)" stroke-opacity=".5"/>
<line x1="120" y1="458.0" x2="160" y2="458.0" stroke="currentColor" stroke-opacity=".18"/>
<line x1="120" y1="472.0" x2="160" y2="472.0" stroke="currentColor" stroke-opacity=".18"/>
<line x1="120" y1="486.0" x2="160" y2="486.0" stroke="currentColor" stroke-opacity=".18"/>
<text class="os2-b" x="140.0" y="476.0" text-anchor="middle" style="fill: var(--fig-on-3)">K<tspan dy="3.5" font-size="72%">j</tspan><tspan dy="-3.5">ᵀ</tspan></text>
<text class="os2" x="140.0" y="518" text-anchor="middle">d × B<tspan dy="3.5" font-size="72%">k</tspan></text>
<text class="os2-b" x="172" y="477.0" text-anchor="middle">=</text>
<rect x="184" y="452.0" width="40" height="40" style="fill: var(--fig-hi); stroke: var(--fig-1)" stroke-opacity=".5"/>
<text class="os2-b" x="204.0" y="476.0" text-anchor="middle" style="fill: var(--fig-on-3)">S<tspan dy="3.5" font-size="72%">ij</tspan></text>
<text class="os2" x="204.0" y="518" text-anchor="middle">B<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> × B</tspan><tspan dy="3.5" font-size="72%">k</tspan></text>
<line x1="228" y1="472.0" x2="324" y2="472.0" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#tl-a)"/>
<text class="os2-s" x="276" y="464.0" text-anchor="middle">exp(S − m<tspan dy="3.5" font-size="72%">new</tspan><tspan dy="-3.5">)</tspan></text>
<rect x="332" y="452.0" width="40" height="40" style="fill: var(--fig-hi); stroke: var(--fig-1)" stroke-opacity=".5"/>
<text class="os2-b" x="352.0" y="476.0" text-anchor="middle" style="fill: var(--fig-on-3)">P<tspan dy="3.5" font-size="72%">ij</tspan></text>
<text class="os2" x="352.0" y="518" text-anchor="middle">B<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> × B</tspan><tspan dy="3.5" font-size="72%">k</tspan></text>
<text class="os2-b" x="384" y="477.0" text-anchor="middle">×</text>
<rect x="396" y="452.0" width="56" height="40" style="fill: var(--fig-3); stroke: var(--fig-1)" stroke-opacity=".5"/>
<line x1="410.0" y1="452.0" x2="410.0" y2="492.0" stroke="currentColor" stroke-opacity=".18"/>
<line x1="424.0" y1="452.0" x2="424.0" y2="492.0" stroke="currentColor" stroke-opacity=".18"/>
<line x1="438.0" y1="452.0" x2="438.0" y2="492.0" stroke="currentColor" stroke-opacity=".18"/>
<text class="os2-b" x="424.0" y="476.0" text-anchor="middle" style="fill: var(--fig-on-3)">V<tspan dy="3.5" font-size="72%">j</tspan></text>
<text class="os2" x="424.0" y="518" text-anchor="middle">B<tspan dy="3.5" font-size="72%">k</tspan><tspan dy="-3.5"> × d</tspan></text>
<line x1="456" y1="472.0" x2="552" y2="472.0" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#tl-a)"/>
<text class="os2-s" x="504" y="464.0" text-anchor="middle">加进</text>
<rect x="560" y="452.0" width="56" height="40" style="fill: var(--fig-3); stroke: var(--fig-1)" stroke-opacity=".5"/>
<line x1="574.0" y1="452.0" x2="574.0" y2="492.0" stroke="currentColor" stroke-opacity=".18"/>
<line x1="588.0" y1="452.0" x2="588.0" y2="492.0" stroke="currentColor" stroke-opacity=".18"/>
<line x1="602.0" y1="452.0" x2="602.0" y2="492.0" stroke="currentColor" stroke-opacity=".18"/>
<text class="os2-b" x="588.0" y="476.0" text-anchor="middle" style="fill: var(--fig-on-3)">O<tspan dy="3.5" font-size="72%">i</tspan></text>
<text class="os2" x="588.0" y="518" text-anchor="middle">B<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> × d</tspan></text>
<path d="M40,526 v4 H160 v-4" fill="none" style="stroke: var(--fig-hi)" stroke-opacity=".8"/>
<text class="os2 os2-hi" x="100.0" y="544" text-anchor="middle">GEMM：Q<tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5"> K</tspan><tspan dy="3.5" font-size="72%">j</tspan><tspan dy="-3.5">ᵀ</tspan></text>
<path d="M332,526 v4 H452 v-4" fill="none" style="stroke: var(--fig-hi)" stroke-opacity=".8"/>
<text class="os2 os2-hi" x="392.0" y="544" text-anchor="middle">GEMM：P<tspan dy="3.5" font-size="72%">ij</tspan><tspan dy="-3.5"> V</tspan><tspan dy="3.5" font-size="72%">j</tspan></text>
<text class="os2-s" x="40" y="572" text-anchor="start">Q<tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5">、O</tspan><tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5"> 和每行的 m、ℓ 在整个内层循环里一直留在片上；K</tspan><tspan dy="3.5" font-size="72%">j</tspan><tspan dy="-3.5">、V</tspan><tspan dy="3.5" font-size="72%">j</tspan><tspan dy="-3.5"> 每一轮从 HBM 读一个新的</tspan></text>
<text class="os2-s" x="40" y="590" text-anchor="start">每个 batch、每个 head 各有一份 Q、K、V，各算一遍；FA2 里每个（Q block，batch × head）是一个 thread block</text>
</svg>
<figcaption><strong>图 2-2</strong> FlashAttention 的 tiling，按 FA2 的循环顺序画，和论文的 Figure 1 是同一件事。上半是各个 tile 在整个矩阵里的位置，下半是内层循环一轮里各个 tile 的形状，两次乘法都是 GEMM。</figcaption>
</figure>

<details class="fold doc aside">
<summary>kernel 里怎么找到 Q_i：stride</summary>

kernel 拿到的只是每个 tensor 的起始地址，tile 在哪要靠 stride 算。stride 是某一维往前走一格时地址要加多少。$Q$、$K$、$V$、$O$ 都是 batch × 行 × $d$ 三维，一共 12 个 stride（[图 2-2a](#fig-2-2a)）。tensor 不连续时（比如转置过，或者是从多个 head 里切出来的 view），stride 不等于括号里的值，按 stride 算地址才对。

<figure id="fig-2-2a">
<svg viewBox="0 0 680 324" width="100%" role="img" aria-label="Q 的形状是 batch × N_q × d。在第 b 个 batch 里，往下一行地址加 stride_qq，往右一列加 stride_qd，下一个 batch 加 stride_qb。Q_i 从第 i 乘 B_q 行开始，共 B_q 行">
  <style>
    .os2 { font-size: 11px; fill: currentColor; }
    .os2-b { font-size: 12px; font-weight: 600; fill: currentColor; }
    .os2-s { font-size: 10.5px; fill: currentColor; opacity: .72; }
    .os2-c { font-size: 12px; font-weight: 600; text-anchor: middle; fill: currentColor; }
    .os2-hi { fill: var(--fig-hi); font-weight: 600; }
  </style>
<defs><marker id="tl-a" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".6"/></marker></defs>
<text class="os2-b" x="134.0" y="24" text-anchor="middle">Q[b]：第 b 个 batch</text>
<text class="os2-s" x="134.0" y="38" text-anchor="middle">N<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> × d</tspan></text>
<rect x="70" y="50" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="86" y="50" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="102" y="50" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="118" y="50" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="134" y="50" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="150" y="50" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="166" y="50" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="182" y="50" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="70" y="64" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="86" y="64" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="102" y="64" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="118" y="64" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="134" y="64" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="150" y="64" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="166" y="64" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="182" y="64" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="70" y="78" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="86" y="78" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="102" y="78" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="118" y="78" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="134" y="78" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="150" y="78" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="166" y="78" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="182" y="78" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="70" y="92" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="86" y="92" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="102" y="92" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="118" y="92" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="134" y="92" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="150" y="92" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="166" y="92" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="182" y="92" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="70" y="106" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="86" y="106" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="102" y="106" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="118" y="106" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="134" y="106" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="150" y="106" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="166" y="106" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="182" y="106" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="70" y="120" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="86" y="120" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="102" y="120" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="118" y="120" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="134" y="120" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="150" y="120" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="166" y="120" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="182" y="120" width="16" height="14" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="1" stroke-opacity=".25"/>
<rect x="70" y="134" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="86" y="134" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="102" y="134" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="118" y="134" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="134" y="134" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="150" y="134" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="166" y="134" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="182" y="134" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="70" y="148" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="86" y="148" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="102" y="148" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="118" y="148" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="134" y="148" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="150" y="148" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="166" y="148" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="182" y="148" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="70" y="162" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="86" y="162" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="102" y="162" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="118" y="162" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="134" y="162" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="150" y="162" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="166" y="162" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="182" y="162" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="70" y="176" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="86" y="176" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="102" y="176" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="118" y="176" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="134" y="176" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="150" y="176" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="166" y="176" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<rect x="182" y="176" width="16" height="14" style="fill: var(--fig-2); stroke: var(--fig-1)" fill-opacity="0.1" stroke-opacity=".25"/>
<text class="os2-b" x="134.0" y="117.0" text-anchor="middle" style="fill: var(--fig-on-3)">Q<tspan dy="3.5" font-size="72%">i</tspan></text>
<path d="M64,92 h-4 v42 h4" fill="none" stroke="currentColor" stroke-opacity=".6"/>
<text class="os2" x="56" y="117.0" text-anchor="end">B<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> 行</tspan></text>
<text class="os2-s" x="56" y="86" text-anchor="end">第 i·B<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> 行</tspan></text>
<line x1="70" y1="204" x2="118" y2="204" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#tl-a)"/>
<text class="os2" x="70" y="222" text-anchor="start">往右一列：+ stride_qd（连续存时 = 1）</text>
<line x1="212" y1="50" x2="212" y2="92" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#tl-a)"/>
<text class="os2" x="220" y="75.0" text-anchor="start">往下一行：+ stride_qq（= d）</text>
<text class="os2" x="220" y="134" text-anchor="start">下一个 batch：+ stride_qb（= N<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> · d）</tspan></text>
<text class="os2-b" x="20" y="274" text-anchor="start">Q<tspan dy="3.5" font-size="72%">i</tspan><tspan dy="-3.5"> 第 r 行、第 c 列的地址</tspan></text>
<text class="os2" x="20" y="294" text-anchor="start">= Q_ptr + b · stride_qb + (i · B<tspan dy="3.5" font-size="72%">q</tspan><tspan dy="-3.5"> + r) · stride_qq + c · stride_qd</tspan></text>
<text class="os2-b" x="400" y="166" text-anchor="start">4 个 tensor × 3 维 = 12 个 stride</text>
<text class="os2-s" x="406" y="186" text-anchor="start"></text>
<text class="os2-s" x="461.0" y="186" text-anchor="middle">batch</text>
<text class="os2-s" x="543.0" y="186" text-anchor="middle">行</text>
<text class="os2-s" x="625.0" y="186" text-anchor="middle">列</text>
<text class="os2-b" x="406" y="206" text-anchor="start">Q</text>
<text class="os2" x="461.0" y="206" text-anchor="middle">stride_qb</text>
<text class="os2" x="543.0" y="206" text-anchor="middle">stride_qq</text>
<text class="os2" x="625.0" y="206" text-anchor="middle">stride_qd</text>
<text class="os2-b" x="406" y="226" text-anchor="start">K</text>
<text class="os2" x="461.0" y="226" text-anchor="middle">stride_kb</text>
<text class="os2" x="543.0" y="226" text-anchor="middle">stride_kk</text>
<text class="os2" x="625.0" y="226" text-anchor="middle">stride_kd</text>
<text class="os2-b" x="406" y="246" text-anchor="start">V</text>
<text class="os2" x="461.0" y="246" text-anchor="middle">stride_vb</text>
<text class="os2" x="543.0" y="246" text-anchor="middle">stride_vk</text>
<text class="os2" x="625.0" y="246" text-anchor="middle">stride_vd</text>
<text class="os2-b" x="406" y="266" text-anchor="start">O</text>
<text class="os2" x="461.0" y="266" text-anchor="middle">stride_ob</text>
<text class="os2" x="543.0" y="266" text-anchor="middle">stride_oq</text>
<text class="os2" x="625.0" y="266" text-anchor="middle">stride_od</text>
</svg>
<figcaption><strong>图 2-2a</strong> 用 stride 找到第 b 个 batch 的 $Q_i$。括号里是连续存时的值。</figcaption>
</figure>

</details>

### 2.2 Online softmax {#online-softmax}

online softmax 来自 Milakov 和 Gimelshein 2018 年的论文。下面按论文的顺序，从按定义写的 softmax 一步一步推到 online softmax，最后推广到 attention。一行 score 记作 $x_1, \dots, x_N$，分母记作 $\ell$（论文里记作 $d$）。[图 2-3](#fig-2-3) 是下面三种算法在一行 8 个元素上的时间线。

<figure id="fig-2-3">
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
<text class="os2-s" x="8" y="202" text-anchor="start">thread 0：x<tspan dy="3.5" font-size="72%">1</tspan><tspan dy="-3.5"> … x</tspan><tspan dy="3.5" font-size="72%">4</tspan></text>
<text class="os2-s" x="8" y="230" text-anchor="start">thread 1：x<tspan dy="3.5" font-size="72%">5</tspan><tspan dy="-3.5"> … x</tspan><tspan dy="3.5" font-size="72%">8</tspan></text>
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
<figcaption><strong>图 2-3</strong> 一行 8 个元素，三种算法放在同一条时间轴上，一格是处理一个元素。</figcaption>
</figure>

**第 1 步：naive softmax。** 按定义

<p align="center">$y_i = \dfrac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$</p>

要扫两遍：第 1 遍求分母，第 2 遍算输出。每个元素访存 3 次（读两次、写一次）。

```
ℓ_0 ← 0
for j ← 1 … N:                      # 第 1 遍：读 x_j
    ℓ_j ← ℓ_{j−1} + e^(x_j)
for i ← 1 … N:                      # 第 2 遍：读 x_i，写 y_i
    y_i ← e^(x_i) / ℓ_N
```

问题是 $e^{x_j}$ 会溢出，fp32 里 $x$ 超过 88.7 时 $e^x$ 就是 inf。

**第 2 步：safe softmax。** 分子分母同乘 $e^{-m_N}$，$m_N$ 是整行的最大值。结果不变，指数都不大于 0，不会溢出：

<p align="center">$y_i = \dfrac{e^{x_i - m_N}}{\sum_{j=1}^{N} e^{x_j - m_N}}, \qquad m_N = \max_{k=1}^{N} x_k$</p>

PyTorch 等框架用的就是这个版本，代价是多扫一遍，每个元素访存 4 次：

```
m_0 ← −∞
for k ← 1 … N:                      # 第 1 遍：读 x_k
    m_k ← max(m_{k−1}, x_k)
ℓ_0 ← 0
for j ← 1 … N:                      # 第 2 遍：读 x_j
    ℓ_j ← ℓ_{j−1} + e^(x_j − m_N)
for i ← 1 … N:                      # 第 3 遍：读 x_i，写 y_i
    y_i ← e^(x_i − m_N) / ℓ_N
```

第 2 遍的每一项都要用 $m_N$，必须等第 1 遍扫完整行。

**第 3 步：online softmax。** 把前两遍合成一遍，边扫边维护到目前为止的最大值 $m_j$ 和分母 $\ell_j$。最大值变大时，之前加进 $\ell$ 的项都是按旧的最大值算的，先乘 $e^{m_{j-1}-m_j}$ 换到新的最大值，再加上新的一项：

```
m_0 ← −∞,  ℓ_0 ← 0
for j ← 1 … N:                      # 第 1 遍：读 x_j，m 和 ℓ 一起更新
    m_j ← max(m_{j−1}, x_j)
    ℓ_j ← ℓ_{j−1} · e^(m_{j−1} − m_j) + e^(x_j − m_j)
for i ← 1 … N:                      # 第 2 遍：读 x_i，写 y_i
    y_i ← e^(x_i − m_N) / ℓ_N
```

每个元素访存回到 3 次，和 naive 一样，但不会溢出。

这样算为什么对：要证的是扫完第 $j$ 个元素后 $\ell_j = \sum_{k=1}^{j} e^{x_k - m_j}$，用归纳法。$j = 1$ 时 $m_1 = x_1$，$\ell_1 = e^{x_1 - m_1}$，成立。假设 $j-1$ 时成立，代入第 $j$ 步的更新，再用 $e^a \cdot e^b = e^{a+b}$：

<p align="center">$\begin{aligned} \ell_j &= \ell_{j-1}\, e^{m_{j-1}-m_j} + e^{x_j-m_j} \\ &= \Big(\sum_{k=1}^{j-1} e^{x_k-m_{j-1}}\Big)\, e^{m_{j-1}-m_j} + e^{x_j-m_j} \\ &= \sum_{k=1}^{j-1} e^{x_k-m_j} + e^{x_j-m_j} \;=\; \sum_{k=1}^{j} e^{x_k-m_j} \end{aligned}$</p>

$j = N$ 时就是 safe softmax 的分母。

**第 4 步：两段合并，并行。** 第 3 步每读一个元素，是把「前 $j-1$ 个元素」的结果和「第 $j$ 个元素」合起来。同样的办法能合并任意两段。A 段和 B 段各自按第 3 步扫完，得到 $(m_A, \ell_A)$ 和 $(m_B, \ell_B)$，其中 $\ell_A = \sum_{k\in A} e^{x_k - m_A}$。合并后的最大值是 $m = \max(m_A, m_B)$，分母和第 3 步的证明一样用 $e^a \cdot e^b = e^{a+b}$：

<p align="center">$\begin{aligned} \ell_A\, e^{m_A-m} + \ell_B\, e^{m_B-m} &= \Big(\sum_{k\in A} e^{x_k-m_A}\Big) e^{m_A-m} + \Big(\sum_{k\in B} e^{x_k-m_B}\Big) e^{m_B-m} \\ &= \sum_{k\in A} e^{x_k-m} + \sum_{k\in B} e^{x_k-m} \;=\; \sum_{k\in A\cup B} e^{x_k-m} \end{aligned}$</p>

第 3 步那一行就是 B 段只有一个元素的情况：$m_B = x_j$，$\ell_B = e^{x_j - x_j} = 1$。论文把这个合并记作 $\oplus$。

能合并任意两段，一行就可以切成几段，交给不同的 thread 同时按第 3 步的第 1 个循环扫，再合并。合并以后每个 thread 都拿到整行的 $m_N$、$\ell_N$，写 $y_i$ 的第 2 个循环也各写各的那一段（[图 2-3](#fig-2-3) 最下面一栏）。合并的顺序不影响结果（$\oplus$ 满足结合律和交换律），所以先合并哪两段都行。

**第 5 步：推广到 attention。** 论文只算 softmax，attention 还要乘 $V$，一行的输出是 $\text{out} = \sum_k p_k v_k$。分母 $\ell$ 对每一项都一样，提出来以后输出是分子 $O$ 除以 $\ell$，$O$ 也是按同一个最大值加起来的求和（[图 2-4](#fig-2-4)）。

<figure id="fig-2-4">
<svg viewBox="0 0 680 316" width="100%" role="img" aria-label="attention 一行的输出是 p_k 乘 v_k 之和。每一项 p_k v_k 等于 e 的 x_k 减 m 次方乘 v_k，再乘整行共用的 1 除以 ℓ。共用的 1/ℓ 提到求和外面，输出等于 O 乘 1/ℓ，O 是 e 的 x_k 减 m 次方乘 v_k 之和">
  <style>
    .os2 { font-size: 11px; fill: currentColor; }
    .os2-b { font-size: 12px; font-weight: 600; fill: currentColor; }
    .os2-s { font-size: 10.5px; fill: currentColor; opacity: .72; }
    .os2-c { font-size: 12px; font-weight: 600; text-anchor: middle; fill: currentColor; }
    .os2-hi { fill: var(--fig-hi); font-weight: 600; }
  </style>
<text class="os2-b" x="338.0" y="16" text-anchor="middle">一行 attention 的输出 out = Σ<tspan dy="3.5" font-size="72%">k</tspan><tspan dy="-3.5"> p</tspan><tspan dy="3.5" font-size="72%">k</tspan><tspan dy="-3.5"> v</tspan><tspan dy="3.5" font-size="72%">k</tspan></text>
<text class="os2-b" x="8" y="44.0" text-anchor="start">out 的每一项</text>
<text class="os2-s" x="8" y="58.0" text-anchor="start">softmax 权重 × v</text>
<text class="os2-b" x="8" y="100.0" text-anchor="start">每个元素自己的部分</text>
<text class="os2-b" x="8" y="154.0" text-anchor="start">整行同一个数</text>
<text class="os2-b" x="8" y="224.0" text-anchor="start">合并</text>
<text class="os2-s" x="8" y="238.0" text-anchor="start">共用的数提到求和外面</text>
<rect x="150" y="26" width="88" height="28" rx="3" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.2" stroke-opacity=".35"/><text class="os2" x="194.0" y="44.0" text-anchor="middle">p<tspan dy="3.5" font-size="72%">1</tspan><tspan dy="-3.5"> v</tspan><tspan dy="3.5" font-size="72%">1</tspan></text>
<text class="os2-b" x="194.0" y="72" text-anchor="middle">=</text>
<rect x="150" y="82" width="88" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.6" stroke-opacity=".35"/><text class="os2" x="194.0" y="100.0" text-anchor="middle">e<tspan dy="-5" font-size="72%">x</tspan><tspan dy="3" font-size="58%">1</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="5"> v</tspan><tspan dy="3.5" font-size="72%">1</tspan></text>
<rect x="246" y="26" width="88" height="28" rx="3" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.2" stroke-opacity=".35"/><text class="os2" x="290.0" y="44.0" text-anchor="middle">p<tspan dy="3.5" font-size="72%">2</tspan><tspan dy="-3.5"> v</tspan><tspan dy="3.5" font-size="72%">2</tspan></text>
<text class="os2-b" x="290.0" y="72" text-anchor="middle">=</text>
<rect x="246" y="82" width="88" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.6" stroke-opacity=".35"/><text class="os2" x="290.0" y="100.0" text-anchor="middle">e<tspan dy="-5" font-size="72%">x</tspan><tspan dy="3" font-size="58%">2</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="5"> v</tspan><tspan dy="3.5" font-size="72%">2</tspan></text>
<text class="os2-b" x="386.0" y="44.0" text-anchor="middle">⋯</text>
<text class="os2-b" x="386.0" y="100.0" text-anchor="middle">⋯</text>
<rect x="438" y="26" width="88" height="28" rx="3" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.2" stroke-opacity=".35"/><text class="os2" x="482.0" y="44.0" text-anchor="middle">p<tspan dy="3.5" font-size="72%">N</tspan><tspan dy="-3.5"> v</tspan><tspan dy="3.5" font-size="72%">N</tspan></text>
<text class="os2-b" x="482.0" y="72" text-anchor="middle">=</text>
<rect x="438" y="82" width="88" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.6" stroke-opacity=".35"/><text class="os2" x="482.0" y="100.0" text-anchor="middle">e<tspan dy="-5" font-size="72%">x</tspan><tspan dy="3" font-size="58%">N</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="5"> v</tspan><tspan dy="3.5" font-size="72%">N</tspan></text>
<text class="os2-b" x="338.0" y="128" text-anchor="middle">×</text>
<rect x="150" y="136" width="376" height="28" rx="3" style="fill: var(--fig-hi); stroke: var(--fig-1)" fill-opacity="0.18" stroke-opacity=".35"/>
<text class="os2 os2-hi" x="338.0" y="154.0" text-anchor="middle">1 / ℓ，ℓ = Σ<tspan dy="3.5" font-size="72%">j</tspan><tspan dy="-3.5"> e</tspan><tspan dy="-5" font-size="72%">x</tspan><tspan dy="3" font-size="58%">j</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="5">（softmax 的分母）</tspan></text>
<text class="os2-s" x="338.0" y="186" text-anchor="middle">O = 第二行全部加起来 = Σ<tspan dy="3.5" font-size="72%">k</tspan><tspan dy="-3.5"> e</tspan><tspan dy="-5" font-size="72%">x</tspan><tspan dy="3" font-size="58%">k</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="5"> v</tspan><tspan dy="3.5" font-size="72%">k</tspan></text>
<rect x="150" y="206" width="376" height="28" rx="3" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.35" stroke-opacity=".35"/><text class="os2" x="338.0" y="224.0" text-anchor="middle">O × 1 / ℓ</text>
<text class="os2-b" x="536" y="224.0" text-anchor="start">= out</text>
<text class="os2-s" x="150" y="260" text-anchor="start">p<tspan dy="3.5" font-size="72%">k</tspan><tspan dy="-3.5"> = e</tspan><tspan dy="-5" font-size="72%">x</tspan><tspan dy="3" font-size="58%">k</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="5"> / ℓ。O 是分子，ℓ 是分母，两个都是一项一项加起来的</tspan></text>
<text class="os2-s" x="150" y="278" text-anchor="start">所以只要能分段算出 O 和 ℓ，最后除一次就得到 out</text>
</svg>
<figcaption><strong>图 2-4</strong> 一行的输出 = 分子 $O$ / 分母 $\ell$。</figcaption>
</figure>

所以 $O$ 和 $\ell$ 用同一个因子修正，状态从 $(m, \ell)$ 变成 $(m, \ell, O)$。FlashAttention 在一个 thread block 里沿 K/V 一个 block 一个 block 地扫，就是第 3 步把一个元素换成一个 block：扫过的 block 合起来是 A 段，新读的 block 是 B 段，按第 4 步合并（[图 2-5](#fig-2-5)）。输出直接是 $O / \ell$，不需要第 3 步的第 2 遍，所以对 K/V 只扫一遍。

<figure id="fig-2-5">
<svg viewBox="0 0 680 252" width="100%" role="img" aria-label="顺序扫：之前扫过的 block 1 到 j−1 只留下 m_old、ℓ_old、O_old，当成一个 tile；新读的 block j 按 m_new 算每一项。ℓ_old 乘 e 的 m_old 减 m_new 次方，加上 block j 的项，得到 ℓ_new">
  <style>
    .os2 { font-size: 11px; fill: currentColor; }
    .os2-b { font-size: 12px; font-weight: 600; fill: currentColor; }
    .os2-s { font-size: 10.5px; fill: currentColor; opacity: .72; }
    .os2-c { font-size: 12px; font-weight: 600; text-anchor: middle; fill: currentColor; }
    .os2-hi { fill: var(--fig-hi); font-weight: 600; }
  </style>
<text class="os2-b" x="265.0" y="16" text-anchor="middle">之前扫过的 block 1 ⋯ j−1</text>
<text class="os2-b" x="472.0" y="16" text-anchor="middle">这一步读 block j</text>
<text class="os2-s" x="619.0" y="16" text-anchor="middle">还没读</text>
<text class="os2-b" x="8" y="44.0" text-anchor="start">合并前</text>
<text class="os2-b" x="8" y="94.0" text-anchor="start">对齐到 m<tspan dy="3.5" font-size="72%">new</tspan></text>
<text class="os2-b" x="8" y="148.0" text-anchor="start">合并</text>
<rect x="150" y="26" width="230" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.6" stroke-opacity=".35"/><text class="os2" x="265.0" y="44.0" text-anchor="middle">m<tspan dy="3.5" font-size="72%">old</tspan><tspan dy="-3.5">，ℓ</tspan><tspan dy="3.5" font-size="72%">old</tspan><tspan dy="-3.5">，O</tspan><tspan dy="3.5" font-size="72%">old</tspan><tspan dy="-3.5">（按 m</tspan><tspan dy="3.5" font-size="72%">old</tspan><tspan dy="-3.5"> 算的）</tspan></text>
<rect x="398.0" y="26" width="72.0" height="28" rx="3" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.2" stroke-opacity=".35"/><text class="os2" x="434.0" y="44.0" text-anchor="middle">e<tspan dy="-5" font-size="72%">x</tspan><tspan dy="3" font-size="58%">k</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="3" font-size="58%">new</tspan></text>
<rect x="474.0" y="26" width="72.0" height="28" rx="3" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.2" stroke-opacity=".35"/><text class="os2" x="510.0" y="44.0" text-anchor="middle">e<tspan dy="-5" font-size="72%">x</tspan><tspan dy="3" font-size="58%">k</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="3" font-size="58%">new</tspan></text>
<rect x="564" y="26" width="110" height="28" rx="3" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.05" stroke-opacity=".35" stroke-dasharray="4 3"/>
<text class="os2-s" x="619.0" y="44.0" text-anchor="middle">block j+1 ⋯</text>
<text class="os2-b" x="265.0" y="70" text-anchor="middle">×</text>
<rect x="150" y="76" width="230" height="28" rx="3" style="fill: var(--fig-hi); stroke: var(--fig-1)" fill-opacity="0.18" stroke-opacity=".35"/>
<text class="os2 os2-hi" x="265.0" y="94.0" text-anchor="middle">e<tspan dy="-5" font-size="72%">m</tspan><tspan dy="3" font-size="58%">old</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="3" font-size="58%">new</tspan></text>
<text class="os2-s" x="472.0" y="94.0" text-anchor="middle">已经按 m<tspan dy="3.5" font-size="72%">new</tspan><tspan dy="-3.5"> 算，不用乘</tspan></text>
<rect x="150" y="130" width="230" height="28" rx="3" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.35" stroke-opacity=".35"/><text class="os2" x="265.0" y="148.0" text-anchor="middle">e<tspan dy="-5" font-size="72%">m</tspan><tspan dy="3" font-size="58%">old</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="3" font-size="58%">new</tspan><tspan dy="2"> · ℓ</tspan><tspan dy="3.5" font-size="72%">old</tspan></text>
<text class="os2-b" x="389" y="148.0" text-anchor="middle">+</text>
<rect x="398" y="130" width="148" height="28" rx="3" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="0.35" stroke-opacity=".35"/><text class="os2" x="472.0" y="148.0" text-anchor="middle">Σ<tspan dy="3.5" font-size="72%">k∈j</tspan><tspan dy="-3.5"> e</tspan><tspan dy="-5" font-size="72%">x</tspan><tspan dy="3" font-size="58%">k</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="3" font-size="58%">new</tspan></text>
<text class="os2-b" x="556" y="148.0" text-anchor="start">= ℓ<tspan dy="3.5" font-size="72%">new</tspan></text>
<text class="os2-s" x="150" y="184" text-anchor="start">m<tspan dy="3.5" font-size="72%">new</tspan><tspan dy="-3.5"> = max(m</tspan><tspan dy="3.5" font-size="72%">old</tspan><tspan dy="-3.5">, block j 的最大值)</tspan></text>
<text class="os2-s" x="150" y="202" text-anchor="start">O 同理：e<tspan dy="-5" font-size="72%">m</tspan><tspan dy="3" font-size="58%">old</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="3" font-size="58%">new</tspan><tspan dy="2"> · O</tspan><tspan dy="3.5" font-size="72%">old</tspan><tspan dy="-3.5"> + Σ</tspan><tspan dy="3.5" font-size="72%">k∈j</tspan><tspan dy="-3.5"> e</tspan><tspan dy="-5" font-size="72%">x</tspan><tspan dy="3" font-size="58%">k</tspan><tspan dy="-3" font-size="72%">−m</tspan><tspan dy="3" font-size="58%">new</tspan><tspan dy="2"> v</tspan><tspan dy="3.5" font-size="72%">k</tspan><tspan dy="-3.5"> = O</tspan><tspan dy="3.5" font-size="72%">new</tspan></text>
<text class="os2-s" x="150" y="220" text-anchor="start">读下一个 block 时，m<tspan dy="3.5" font-size="72%">new</tspan><tspan dy="-3.5">、ℓ</tspan><tspan dy="3.5" font-size="72%">new</tspan><tspan dy="-3.5">、O</tspan><tspan dy="3.5" font-size="72%">new</tspan><tspan dy="-3.5"> 就是新的 m</tspan><tspan dy="3.5" font-size="72%">old</tspan><tspan dy="-3.5">、ℓ</tspan><tspan dy="3.5" font-size="72%">old</tspan><tspan dy="-3.5">、O</tspan><tspan dy="3.5" font-size="72%">old</tspan><tspan dy="-3.5">；扫完整行再算 O / ℓ</tspan></text>
</svg>
<figcaption><strong>图 2-5</strong> 顺序扫的一步。kernel 只存每行的 $m$、$\ell$、$O$。</figcaption>
</figure>

写成伪代码，一次处理一个 query block，最后归一化一次：

```
for 每个 query block i:
    O_i ← 0,  ℓ_i ← 0,  m_i ← -∞           # 只放在 shared memory / register 里
    for 每个 key/value block j:
        S_ij  ← Q_i K_jᵀ / sqrt(d)         # B_q × B_k，只在片上，不进 HBM
        m_new ← max(m_i, rowmax(S_ij))
        P_ij  ← exp(S_ij − m_new)
        ℓ_i   ← exp(m_i − m_new)·ℓ_i + rowsum(P_ij)
        O_i   ← exp(m_i − m_new)·O_i + P_ij·V_j
        m_i   ← m_new
    O_i ← O_i / ℓ_i                        # O_i（和每行的统计量）只写回 HBM 一次
```

严格说，这是 FlashAttention-2 采用的循环顺序。FA1 论文的 Algorithm 1 是反过来套的：外层循环走 $K$/$V$ block，内层循环走 $Q$ block，所以内层的每一步都要从 HBM 读出 $O_i$、$\ell_i$、$m_i$，更新后再写回去，而且每一步都用 $\mathrm{diag}(\ell_i)^{-1}$ 把 $O_i$ 归一化一次。后面的 IO 复杂度对两种顺序都成立；换循环顺序是 FA2 的改动之一（[第 3 节](#fa2)）。

修正以后每一项都等于按最终最大值算的值，所以结果是精确的，只差浮点加法的顺序；额外的显存只有每行的 $m$、$\ell$，是 $O(N)$。

前向不存 $P$，反向却要用到它，这是下一小节的事。

### 2.3 反向的 recomputation {#fa1-recompute}

attention 的反向一般要再用一次 $P$。为此存下完整的 $N\times N$ 矩阵，就把刚省下的显存又花回去了。FlashAttention 只存 $O$ 和每行的统计量 $(m, \ell)$，反向时在片上把需要的 $S_{ij}$、$P_{ij}$ block 重新算出来。这是用少量额外的 matmul 计算换显存和显存读写的大幅减少，被省掉的存储正好在慢的那一层，这时重算比存下来划算（[单卡 GPU 优化方法第 4 节](/blog/gpu-single-gpu-optimization/#recompute)）。

论文的复杂度结论：标准 attention 要 $\Theta(Nd+N^2)$ 次 HBM 访问，FlashAttention 要 $\Theta(N^2d^2M^{-1})$，$M$ 是 SRAM 的大小。常见的 $d$（64–128）和 $M$（100KB 左右）下，$d^2$ 比 $M$ 小很多倍，所以 FlashAttention 的 HBM 访问也少很多倍；$M=\Theta(d^2)$ 时两者才一样。论文给的下界有意取得很弱：不存在一个精确算法，能对 $[d, Nd]$ 里的**每一个** $M$ 都做到 $o(N^2d^2M^{-1})$ 次访问。实际的效果是算得更多（反向要重算 score tile）、搬得少得多。

显存随 $N$ 线性增长而不是平方增长，序列越长省得越多，算的仍然是**精确**的 softmax，当时的稀疏 attention 和低秩 attention 都做不到这一点。

*RTX 5090 实测待补：标准 attention 和 FlashAttention 的计算量、显存读写量和耗时（前向加反向），以及不同序列长度下的显存。*

FA1 解决了 HBM 的读写，但 kernel 本身离 Tensor core 的峰值还很远，差距出在 kernel 怎么用 GPU 上，这是 FlashAttention-2 要解决的。

## 3 FlashAttention-2：让 GPU 忙起来 {#fa2}

[第 2 节](#fa1)的 FlashAttention 靠 tiling、online softmax 和 recomputation 把 $N \times N$ 的矩阵留在片上，它的 online softmax 每一步都把 $O$ 归一化一次。FlashAttention-2 的数学和它相同，改的是工作怎么分配。

<div class="fa-intuit"><b>思路：</b>v1 搬的数据量已经对了，但 GPU 大部分时候闲着。v2 不改数学，只改调度：分出多得多的互相独立的工作，让每个 SM 都有活干；同时少花指令在 matmul 以外的簿记上。</div>

v1 解决了 IO 的问题，吞吐却还有很大余量，离调得好的 dense GEMM 差得很远。FlashAttention-2（Dao，2023）是一篇讲工作调度的论文：数学不变，改进的是工作怎么切分给 thread block 和 warp。

最要紧的改动有三处。

1. **减少非 matmul 的 FLOPs。** 非 matmul 的计算（exp、行最大值和行和、rescale 的乘法）跑在 FP32 ALU 和特殊函数单元上，不走 Tensor core。A100 的 FP32 算力是 1.95e13 FLOPS，FP16 matmul 是 3.12e14 FLOPS，所以每个非 matmul 的 FLOP 大约贵 16 倍。v1 每一步都让输出保持归一化，除了免不掉的 $\exp(m_{\text{old}}-m_{\text{new}})$ 修正，还要再除以当前的累计和。FA2 的累加器不做归一化，每个 block 只乘 $\exp(m_{\text{old}}-m_{\text{new}})$ 这一项修正，最后才除一次 $\ell$（即 [2.2 节](#online-softmax)的伪代码）；给反向存的也只有每行一个 logsumexp $L=m+\log\ell$，不再把 $m$ 和 $\ell$ 都存下来。
2. **交换内外层循环，在序列长度上并行。** v1 只在 (batch × head) 上并行。现在的序列长、每张卡上的 batch 小，这样分出的 thread block 可能不够占满 GPU 的 SM（[图 3-1](#fig-3-1)）。FA2 把 query block 的循环放到外层，各个行 block 之间互相独立：前向里每个 thread block 负责一个 query 行 block。反向里每个 thread block 负责一个 key/value 列 block，在本地累加自己的 $dK_j$、$dV_j$，$dQ$ 则用 atomic add 累加到一块共享的缓冲区里。

<figure id="fig-3-1">
<svg viewBox="0 0 640 168" width="100%" role="img" aria-label="例子：一条 8K token 的序列、16 个 head，跑在有 108 个 SM 的 A100 上。FlashAttention-1 启动 16 个 thread block，108 个 SM 里只有 16 个在忙；FlashAttention-2 还把每个 head 的 query 切成 64 个 block，启动 1024 个 thread block，108 个 SM 全部在忙">
  <style>
    .sm-h { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .sm-s { font-size: 11px; fill: currentColor; opacity: .75; }
    .sm-f { font-size: 11.5px; font-weight: 600; fill: currentColor; }
  </style>
  <defs>
    <pattern id="sm-busy" x="26" y="46" width="15" height="15" patternUnits="userSpaceOnUse">
      <rect width="12" height="12" rx="2" fill="rgb(var(--color-primary-500))"/>
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
<figcaption><strong>图 3-1</strong> 在序列上并行的作用，以一个示例说明：A100 上一条 8K token 的序列、16 个 head，方块是 A100 的 108 个 SM。每个 (batch, head) 一个 thread block 时，只有 16 个 SM 分到活，其余空闲；把每个 head 的 query 按 128 行切成 block，得到 1024 个互相独立的 thread block，足够把整张卡填满好几遍。</figcaption>
</figure>

3. **更好的 warp 分工。** 在一个 thread block 内，v1 把 $K$/$V$ 分给各个 warp，每个 warp 都能看到全部 $Q$（split-K 方案）；于是每个 warp 都得把自己的部分结果写进 shared memory，同步之后再跨 warp 归约。FA2 反过来，把 $Q$ 分给各个 warp，每个 warp 都能看到全部 $K$/$V$（split-Q），这样每个 warp 算出的就是一段完整的输出，**不需要通过 shared memory 做跨 warp 归约**（[图 3-2](#fig-3-2)）。各个 warp 一起加载每个 $K$/$V$ tile 时，仍然要在整个 block 的 barrier 上会合，但它们从不交换部分输出。

<figure id="fig-3-2">
<svg viewBox="0 0 640 256" width="100%" role="img" aria-label="split-K 的 warp 分工要在 shared memory 里跨 warp 归约部分输出；split-Q 让每个 warp 得到一段完整的输出，不需要跨 warp 归约">
  <style>
    .wp-title { font-size: 13px; font-weight: 600; fill: currentColor; }
    .wp-lab   { font-size: 12px; fill: #fff; text-anchor: middle; }
    .wp-lab-dk { font-size: 12px; fill: rgb(var(--color-neutral-900)); text-anchor: middle; }
    .wp-sub   { font-size: 11px; fill: currentColor; opacity: .7; text-anchor: middle; }
    .wp-box   { font-size: 11px; fill: currentColor; text-anchor: middle; }
  </style>
  <text x="170" y="18" text-anchor="middle" class="wp-title">v1：split-K</text>
  <rect x="30" y="30" width="280" height="30" rx="4" fill="rgba(128,128,128,.12)" stroke="currentColor" stroke-opacity=".35"/>
  <text x="170" y="50" class="wp-box">Q：共享，每个 warp 都能看到</text>
  <g>
    <rect x="30"  y="78" width="60" height="34" rx="4" fill="rgb(var(--color-secondary-500))"/>
    <text x="60"  y="99" class="wp-lab-dk">K/V 0</text>
    <rect x="100" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-secondary-500))"/>
    <text x="130" y="99" class="wp-lab-dk">K/V 1</text>
    <rect x="170" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-secondary-500))"/>
    <text x="200" y="99" class="wp-lab-dk">K/V 2</text>
    <rect x="240" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-secondary-500))"/>
    <text x="270" y="99" class="wp-lab-dk">K/V 3</text>
  </g>
  <text x="170" y="128" class="wp-sub">warp 0–3：各算一段 K/V 的部分和</text>
  <path d="M60,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <path d="M130,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <path d="M210,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <path d="M280,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <rect x="90" y="190" width="160" height="32" rx="4" fill="rgba(var(--color-primary-500), .16)" stroke="rgb(var(--color-primary-500))"/>
  <text x="170" y="211" class="wp-box">在 shared memory 里归约</text>
  <text x="170" y="242" class="wp-sub">写部分和 · barrier 同步 · 相加</text>
  <line x1="330" y1="10" x2="330" y2="250" stroke="currentColor" stroke-opacity=".15"/>
  <text x="490" y="18" text-anchor="middle" class="wp-title">v2：split-Q</text>
  <rect x="350" y="30" width="280" height="30" rx="4" fill="rgba(128,128,128,.12)" stroke="currentColor" stroke-opacity=".35"/>
  <text x="490" y="50" class="wp-box">K、V：共享，每个 warp 都能看到</text>
  <g>
    <rect x="350" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-primary-500))"/>
    <text x="380" y="99" class="wp-lab">Q 0</text>
    <rect x="420" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-primary-500))"/>
    <text x="450" y="99" class="wp-lab">Q 1</text>
    <rect x="490" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-primary-500))"/>
    <text x="520" y="99" class="wp-lab">Q 2</text>
    <rect x="560" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-primary-500))"/>
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
  <rect x="350" y="190" width="60" height="32" rx="4" fill="rgba(var(--color-primary-500), .14)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".5"/>
  <text x="380" y="211" class="wp-box">O 0</text>
  <rect x="420" y="190" width="60" height="32" rx="4" fill="rgba(var(--color-primary-500), .14)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".5"/>
  <text x="450" y="211" class="wp-box">O 1</text>
  <rect x="490" y="190" width="60" height="32" rx="4" fill="rgba(var(--color-primary-500), .14)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".5"/>
  <text x="520" y="211" class="wp-box">O 2</text>
  <rect x="560" y="190" width="60" height="32" rx="4" fill="rgba(var(--color-primary-500), .14)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".5"/>
  <text x="590" y="211" class="wp-box">O 3</text>
  <text x="490" y="242" class="wp-sub">各段输出互相独立，不需要跨 warp 归约</text>
</svg>
<figcaption><strong>图 3-2</strong> 同样四个 warp，切法相反。v1 切 $K$/$V$，四份部分和要经 shared memory 合并；v2 切 $Q$，每个 warp 的 matmul 一算完，它那段输出就已经是完整的。</figcaption>
</figure>

v2 不需要任何新的硬件特性，就是一篇「把手里已有的 GPU 用得更好」的论文，这也是它在 Hopper 以外的 GPU（以及 AMD 的 ROCm 移植版）上部署最广的原因。

*RTX 5090 实测待补：v1 和 v2 的前向、反向吞吐和占 Tensor core 峰值的比例。*

Hopper 新增了几种异步硬件，v1 和 v2 都不是为它们设计的，[第 4 节](#fa3)的 FlashAttention-3 就是围绕这些硬件写的。

## 4 FlashAttention-3：用上 Hopper 新加的硬件 {#fa3}

[第 3 节](#fa2)的 FA2 推迟归一化、交换内外层循环的顺序、沿序列长度并行，再用 split-Q 把工作分给 warp。换到 H100，硬件多了 v1、v2 设计时都没有用到的东西，FA3 就从这里入手。

<div class="fa-intuit"><b>思路：</b>在 Hopper 上，搬 tile 和做矩阵乘由不同的硬件单元负责，两者可以同时进行。v3 把 kernel 改成一条流水线：一个 warp 只负责取数据，其余的只负责计算，两组负责计算的 warpgroup 轮流工作，Tensor core 不用停下来等 softmax。</div>

H100 新加了两个原语，v1 和 v2 设计时都没有用到。一是 **Tensor Memory Accelerator（TMA）**，一个在 HBM 和 shared memory 之间异步整块搬 tile 的硬件单元（TMA 和异步同步见入门篇的 [Appendix C](/blog/gpu-triton-intro/#tma)）。二是 **warpgroup 级的 MMA（WGMMA）**，由 4 个 warp 组成的 warpgroup 一起发出的 Tensor core 指令，比以前的更宽，而且是异步的。FlashAttention-3（Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao，2024）围绕这两个原语来设计，FP8 部分另外用了 block 量化和 incoherent processing。具体有三点。

1. **Warp specialization（producer/consumer）。** FA2 里每个 warp 既要发出自己那份 tile 的拷贝（用 Ampere 的 cp.async，已经是异步的），又要做计算，各阶段之间用整个 block 的 barrier 同步。FA3 改成由一个 producer warp 发 TMA load，把 tile 搬进 shared memory 里一个多级的环形缓冲，缓冲的状态由硬件 barrier 跟踪；register 大部分分给 consumer warpgroup，它们只对已经搬好的 tile 跑 WGMMA（[图 4-1](#fig-4-1)，这种分工在入门篇的 [C.3](/blog/gpu-triton-intro/#tma-ws) 讲过）。这样 kernel 就成了一条显式的软件 pipeline。

<figure id="fig-4-1">
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
  <rect x="136" y="66" width="100" height="58" rx="6" fill="rgba(var(--color-secondary-500), .22)" stroke="rgb(var(--color-secondary-500))"/>
  <text x="186" y="90" class="pc-h">producer warp</text>
  <text x="186" y="107" class="pc-s">只发 TMA load</text>
  <line x1="238" y1="95" x2="261" y2="95" stroke="currentColor" stroke-opacity=".5" stroke-width="1.5"/>
  <polygon points="268,95 260,91 260,99" fill="currentColor" fill-opacity=".5"/>
  <text x="352" y="54" class="pc-s">shared memory 环形缓冲</text>
  <rect x="270" y="66" width="38" height="58" rx="4" fill="rgba(var(--color-secondary-500), .55)" stroke="rgb(var(--color-secondary-500))"/>
  <rect x="312" y="66" width="38" height="58" rx="4" fill="rgba(var(--color-secondary-500), .55)" stroke="rgb(var(--color-secondary-500))"/>
  <rect x="354" y="66" width="38" height="58" rx="4" fill="none" stroke="rgb(var(--color-secondary-500))"/>
  <rect x="354" y="95" width="38" height="29" fill="rgba(var(--color-secondary-500), .55)"/>
  <rect x="396" y="66" width="38" height="58" rx="4" fill="none" stroke="rgb(var(--color-secondary-500))" stroke-dasharray="3 2"/>
  <text x="289" y="140" class="pc-t">已满</text>
  <text x="331" y="140" class="pc-t">已满</text>
  <text x="373" y="140" class="pc-t">装入中</text>
  <text x="415" y="140" class="pc-t">空闲</text>
  <line x1="436" y1="88" x2="467" y2="72" stroke="currentColor" stroke-opacity=".5" stroke-width="1.5"/>
  <polygon points="474,69 465,68 469,76" fill="currentColor" fill-opacity=".5"/>
  <line x1="436" y1="102" x2="467" y2="118" stroke="currentColor" stroke-opacity=".5" stroke-width="1.5"/>
  <polygon points="474,121 469,114 465,122" fill="currentColor" fill-opacity=".5"/>
  <rect x="476" y="48" width="148" height="42" rx="6" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))"/>
  <text x="550" y="66" class="pc-h">consumer warpgroup 1</text>
  <text x="550" y="82" class="pc-s">WGMMA + softmax</text>
  <rect x="476" y="100" width="148" height="42" rx="6" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))"/>
  <text x="550" y="118" class="pc-h">consumer warpgroup 2</text>
  <text x="550" y="134" class="pc-s">WGMMA + softmax</text>
  <path d="M550,144 V168 H186 V133" fill="none" stroke="currentColor" stroke-opacity=".45" stroke-dasharray="4 3"/>
  <polygon points="186,126 182,134 190,134" fill="currentColor" fill-opacity=".45"/>
  <text x="368" y="161" class="pc-s">slot 用完 → 通过硬件 barrier 还给 producer</text>
</svg>
<figcaption><strong>图 4-1</strong> Warp specialization 的分工。producer warp 只发 TMA，把 tile 搬进 shared memory 的环形 slot；consumer warpgroup 只算已经搬满的 slot。只要环形缓冲一直领先，Tensor core 就不用等 load。</figcaption>
</figure>

2. **GEMM 和 softmax 重叠。** 在单个 tile 上，softmax 依赖这个 block 的 $QK^\top$，$PV$ 的 GEMM 又依赖 softmax，所以它们只能串行。FA3 从两个互补的方向拆开这条依赖链。第一个是 warpgroup 之间的 ping-pong（[图 4-2](#fig-4-2)）：两个 consumer warpgroup 各自负责一个不同的 query tile，用 barrier 强制它们交替执行，warpgroup A 做 GEMM 时 warpgroup B 做 softmax，反过来也一样。第二个是 warpgroup 内部的 pipeline：在同一个 warpgroup 里，下一个 block 的 $QK^\top$ 和当前 block 的 $PV$ 这两组异步 WGMMA 按一定的时机发出，让 softmax 在它们执行期间进行。

<figure id="fig-4-2">
<svg viewBox="0 0 640 210" width="100%" role="img" aria-label="Ping-pong 调度：两个 warpgroup 交替做 GEMM 和 softmax，一个做 softmax 的时段和另一个做 matmul 的时段重叠；在 H100 上 softmax 的耗时约为 GEMM 的一半">
  <style>
    .pp-lane { font-size: 12px; fill: currentColor; opacity: .8; }
    .pp-seg  { font-size: 11px; fill: #fff; text-anchor: middle; }
    .pp-seg-dk { font-size: 11px; fill: rgb(var(--color-neutral-900)); text-anchor: middle; }
    .pp-wait { font-size: 11px; fill: currentColor; opacity: .7; text-anchor: middle; }
    .pp-sub  { font-size: 11px; fill: currentColor; opacity: .7; }
    .pp-leg  { font-size: 11px; fill: currentColor; }
  </style>
  <text x="20" y="45" class="pp-lane">Warpgroup A</text>
  <text x="20" y="105" class="pp-lane">Warpgroup B</text>
  <g>
    <rect x="130" y="24" width="120" height="34" fill="rgb(var(--color-primary-500))"/>
    <text x="190" y="45" class="pp-seg">GEMM</text>
    <rect x="250" y="24" width="60" height="34" fill="rgb(var(--color-secondary-500))"/>
    <text x="280" y="45" class="pp-seg-dk">softmax</text>
    <rect x="310" y="24" width="60" height="34" fill="rgba(128,128,128,.10)"/>
    <text x="340" y="45" class="pp-wait">等待</text>
    <rect x="370" y="24" width="120" height="34" fill="rgb(var(--color-primary-500))"/>
    <text x="430" y="45" class="pp-seg">GEMM</text>
    <rect x="490" y="24" width="60" height="34" fill="rgb(var(--color-secondary-500))"/>
    <text x="520" y="45" class="pp-seg-dk">softmax</text>
    <rect x="550" y="24" width="60" height="34" fill="rgba(128,128,128,.10)"/>
    <text x="580" y="45" class="pp-wait">等待</text>
  </g>
  <g>
    <rect x="130" y="84" width="60" height="34" fill="rgb(var(--color-secondary-500))"/>
    <text x="160" y="105" class="pp-seg-dk">softmax</text>
    <rect x="190" y="84" width="60" height="34" fill="rgba(128,128,128,.10)"/>
    <text x="220" y="105" class="pp-wait">等待</text>
    <rect x="250" y="84" width="120" height="34" fill="rgb(var(--color-primary-500))"/>
    <text x="310" y="105" class="pp-seg">GEMM</text>
    <rect x="370" y="84" width="60" height="34" fill="rgb(var(--color-secondary-500))"/>
    <text x="400" y="105" class="pp-seg-dk">softmax</text>
    <rect x="430" y="84" width="60" height="34" fill="rgba(128,128,128,.10)"/>
    <text x="460" y="105" class="pp-wait">等待</text>
    <rect x="490" y="84" width="120" height="34" fill="rgb(var(--color-primary-500))"/>
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
    <rect x="130" y="184" width="14" height="14" fill="rgb(var(--color-primary-500))"/>
    <text x="150" y="195" class="pp-leg">GEMM（Tensor core）</text>
    <rect x="300" y="184" width="14" height="14" fill="rgb(var(--color-secondary-500))"/>
    <text x="320" y="195" class="pp-leg">softmax（exp/ALU 单元）</text>
  </g>
</svg>
<figcaption><strong>图 4-2</strong> 两个 warpgroup 的相位错开一步：一个在做 softmax 的标量运算时，另一个正在 Tensor core 上做 matmul，非 matmul 的开销藏进了 matmul 的时间里，不再和它串行。各段宽度是示意，按 H100、head dim 128 的比例画，exp 的耗时约为与它重叠的 GEMM 的一半。</figcaption>
</figure>

3. **FP8：block 量化和 incoherent processing。** 在 matmul 之前直接把 $Q$、$K$ 量化成 FP8，吞吐上很划算，但精度不够：少数离群值占满了量化范围，其余数值的分辨率被压得很低（FP8 和 block 量化见[单卡优化篇](/blog/gpu-single-gpu-optimization/#precision)）。FA3 用了两个互补的办法。第一是 block 量化：每个 tile 一个 FP8 scale，而不是整个张量一个，离群值的影响只限于它所在的 block；kernel 本来就按 tile 计算，所以这样做没有额外开销。第二是 incoherent processing：量化前把 $Q$ 和 $K$ 乘上同一个随机正交矩阵 $M$。因为 $M$ 正交，$(QM)(KM)^\top = QMM^\top K^\top = QK^\top$，attention 分数在数学上不变；但 $QM$ 的每个元素现在是许多原始元素的随机混合，任何一个坐标都不会被单个离群值主导（[图 4-3](#fig-4-3)）。实现上用的是带随机符号翻转的 Hadamard 变换，长度为 $d$ 的一行代价是 $O(d\log d)$，稠密的旋转则要 $O(d^2)$。

<figure id="fig-4-3">
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
  <rect x="50" y="107" width="20" height="3" fill="rgb(var(--color-primary-500))"/>
  <rect x="80" y="110" width="20" height="2" fill="rgb(var(--color-primary-500))"/>
  <rect x="110" y="109" width="20" height="1" fill="rgb(var(--color-primary-500))"/>
  <rect x="140" y="30" width="20" height="80" fill="rgb(var(--color-secondary-500))"/>
  <rect x="170" y="110" width="20" height="4" fill="rgb(var(--color-primary-500))"/>
  <rect x="200" y="108" width="20" height="2" fill="rgb(var(--color-primary-500))"/>
  <rect x="230" y="109" width="20" height="1" fill="rgb(var(--color-primary-500))"/>
  <rect x="260" y="110" width="20" height="3" fill="rgb(var(--color-primary-500))"/>
  <line x1="364" y1="78.9" x2="616" y2="78.9" stroke="currentColor" stroke-opacity=".4" stroke-dasharray="4 3"/>
  <line x1="364" y1="141.1" x2="616" y2="141.1" stroke="currentColor" stroke-opacity=".4" stroke-dasharray="4 3"/>
  <line x1="364" y1="110" x2="616" y2="110" stroke="currentColor" stroke-opacity=".35"/>
  <text x="366" y="72" class="ic-s">±3.1</text>
  <rect x="370" y="78.9" width="20" height="31.1" fill="rgb(var(--color-primary-500))"/>
  <rect x="400" y="110" width="20" height="26.2" fill="rgb(var(--color-primary-500))"/>
  <rect x="430" y="110" width="20" height="23.3" fill="rgb(var(--color-primary-500))"/>
  <rect x="460" y="81.7" width="20" height="28.3" fill="rgb(var(--color-primary-500))"/>
  <rect x="490" y="80.3" width="20" height="29.7" fill="rgb(var(--color-primary-500))"/>
  <rect x="520" y="110" width="20" height="29" fill="rgb(var(--color-primary-500))"/>
  <rect x="550" y="110" width="20" height="30.4" fill="rgb(var(--color-primary-500))"/>
  <rect x="580" y="81.7" width="20" height="28.3" fill="rgb(var(--color-primary-500))"/>
  <text x="170" y="212" class="ic-c">最大值 8.0 决定了量化的 scale</text>
  <text x="490" y="212" class="ic-c">最大值 3.1 · 长度不变（8.03）</text>
</svg>
<figcaption><strong>图 4-3</strong> 在一个 8 维的玩具向量上做 incoherent processing（数值是精确算出来的）。量化的 scale 由 block 里的最大值决定，一个 8.0 的离群值让其余 7 个小值只能挤在可表示范围里很窄的一段；随机翻转符号再做 Hadamard 变换，把离群值的能量分散到每个坐标上。最大值从 8.0 降到 3.1，向量长度不变；$Q$ 和 $K$ 做的是同一个旋转，所以 $QK^\top$ 里的每个点积也不变。</figcaption>
</figure>

*RTX 5090 实测待补：5090 没有 WGMMA，FA3 的 kernel 跑不了；这里改测同样思路在 5090 上能做到的部分，以及 FP8 路径的误差。*

[图 4-2](#fig-4-2) 里 softmax 只用 GEMM 一半左右的时间，到了 Blackwell 的 B200 上两者一样长，[第 5 节](#fa4)的 FA4 就从这里出发。

## 5 FlashAttention-4：瓶颈随硬件换代而转移 {#fa4}

[第 4 节](#fa3)的 FA3 在 Hopper 上用 warp specialization 搭起异步 pipeline，让 GEMM 和 softmax 重叠，再加上 FP8。到了 Blackwell，硬件各部分的提升幅度不一样，瓶颈又换了位置。

<div class="fa-intuit"><b>思路：</b>Blackwell 把矩阵乘的速度提高了一倍，做 exp 的单元和 shared memory 却没变，原来藏在 matmul 后面的 softmax 现在和 matmul 用的时间一样长。FA4 直接削减这部分工作：一部分 exp 改到别的单元上算，跳过不影响结果的 rescale，把更多数据放进新增的 tensor memory。</div>

FlashAttention-4 的作者是 Zadouri、Hoehnerbach、Shah、Liu、Thakkar 和 Dao（Princeton、Meta、Colfax Research、NVIDIA、Georgia Tech 和 Together AI），2026 年 3 月发在 arXiv 上。论文针对 Blackwell 数据中心 GPU（B200/GB200）；开源实现（flash-attention 仓库里的 `flash_attn/cute`，安装包名是 `flash-attn-4`）也带了 Hopper 的 kernel。它是四代里最新的一代，在生产环境里经过的检验也最少。

论文的出发点是**硬件的非对称扩展**（asymmetric hardware scaling）。从 Hopper 到 Blackwell，dense BF16 的 Tensor core 吞吐翻了一倍（每个 SM 每个时钟 8192 和 4096 FLOPs；整卡约 2.25e15 和 1e15 FLOPS），shared memory 的读带宽（每个 SM 每个时钟 128 B）和做 exp 的 MUFU 单元（每个 SM 每个时钟 16 次）完全没变。以 $d=128$ 时一个 $128\times128$ 的 tile 为例（[图 5-1](#fig-5-1)），前向的两个 MMA 在 Hopper 上要 $4\cdot128^3/4096=2048$ 个周期，$128^2$ 个 exp 要 $128^2/16=1024$ 个周期，exp 只占 matmul 的一半，FA3 的 ping-pong 能把它们藏在 matmul 后面。B200 上 MMA 只要 1024 个周期，exp 仍然要 1024 个，只靠重叠已经藏不住了。这和最初的 FlashAttention 解决的是同一个 IO 与计算的平衡问题，只是发生在更高的一层。

<figure id="fig-5-1">
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
  <rect x="160" y="33" width="420" height="18" fill="rgb(var(--color-primary-500))"/>
  <text x="586" y="46" class="as-val">2048</text>
  <text x="150" y="72" class="as-lab">exp（MUFU）</text>
  <rect x="160" y="59" width="210" height="18" fill="rgba(var(--color-secondary-500), .25)" stroke="rgb(var(--color-secondary-500))" stroke-width="1.5"/>
  <text x="376" y="72" class="as-val">1024</text>
  <text x="420" y="72" class="as-note">MMA 的一半，藏得住</text>
  <text x="20" y="104" class="as-grp">B200</text>
  <text x="150" y="124" class="as-lab">MMA（Tensor core）</text>
  <rect x="160" y="111" width="210" height="18" fill="rgb(var(--color-primary-500))"/>
  <text x="376" y="124" class="as-val">1024</text>
  <text x="150" y="150" class="as-lab">exp（MUFU）</text>
  <rect x="160" y="137" width="210" height="18" fill="rgba(var(--color-secondary-500), .25)" stroke="rgb(var(--color-secondary-500))" stroke-width="1.5"/>
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
<figcaption><strong>图 5-1</strong> 前向一个 tile（$M=N=d=128$）每个 SM 的周期数，是 roofline 估算，不是实测；MMA 按 $4MNd$ 除以每个 SM 每周期的 FLOPs 算（Hopper 是 $4MNd/4096=2048$），exp 按 $MN/16=1024$ 算。B200 的数取自 FA4 论文 Table 1，H100 用同样的公式套 Hopper 的 MMA 速率；Hopper 的 MMA tile 是 64 行，操作数重读的次数不同，所以 H100 没画 shared memory。</figcaption>
</figure>

FA4 的做法有四条：

1. **围绕完全异步的 MMA 和更大的 tile 重建 pipeline。** Blackwell 的 MMA 以 $128\times N$ 的 tile 为单位，累加结果异步写进 tensor memory，不经过 register。FA4 保留了 FA3 的 ping-pong，现在是在每个 thread block 的两个 128 行 $Q$ tile 之间交替；每行由一个 thread 负责，求行最大值和行和都不需要 warp shuffle。$P$ 经由 tensor memory 而不是 register 送进 $PV$ 的 MMA，所以输出的 rescale 可以挪到单独的 correction warpgroup 里做，不再占 softmax 的关键路径。tile 调度器按最长任务优先（longest-processing-time-first）分配，改善了 causal 和变长 batch 的负载均衡。
2. **软件模拟 exp，以及有条件的 softmax rescale。** 软件模拟不替代硬件 exp，而是多出一个算 exp 的地方。FA4 把每行 exp 里调好比例的一部分（10–25%）放到 FMA 单元上算，不走 MUFU：先拆成 $2^x = 2^{\lfloor x\rfloor}\cdot 2^{x-\lfloor x\rfloor}$（Cody–Waite range reduction），对 $f\in[0,1)$ 用 3 次多项式求 $2^f$，再把整数移位放进浮点数的指数位，得到 $2^{\lfloor x\rfloor}$。每次软件模拟的开销比一条硬件 `ex2` 大，但两边并行，总的 exp 吞吐提高了。有条件的 rescale 针对的是另一项非 matmul 开销，即 [2.2 节](#online-softmax)里 $O \leftarrow e^{m_{\text{old}}-m_{\text{new}}}O$ 的修正。这一步从来不是正确性所必需的，只是为了让数值不越界：$O$ 和 $\ell$ 对着同一个参考最大值累加，参考值取多少，$O/\ell$ 都是精确的。所以 FA4 沿用旧的最大值，只有新 block 把最大值抬高超过 $\log_2 256 = 8$ 时才更新，$P$ 里未归一化的元素因此最多放大 256 倍，FP32 累加器完全容得下（[图 5-2](#fig-5-2)）。

<figure id="fig-5-2">
<svg viewBox="0 0 640 240" width="100%" role="img" aria-label="十个 key/value block 上的有条件 rescale：分数的当前最大值在其中九个 block 上升，标准 online softmax 要 rescale 九次；FA4 保留旧的参考最大值，只在第 6 和第 9 个 block 更新，这两处真实最大值比参考值高出超过 8（log₂ 单位）">
  <style>
    .cr-s { font-size: 11px; fill: currentColor; opacity: .75; }
    .cr-t { font-size: 11px; fill: currentColor; }
    .cr-k { font-size: 10.5px; fill: currentColor; opacity: .65; text-anchor: middle; }
    .cr-hl { font-size: 11px; font-weight: 600; fill: currentColor; }
  </style>
  <line x1="60" y1="12" x2="82" y2="12" stroke="rgb(var(--color-primary-500))" stroke-width="2"/>
  <text x="88" y="16" class="cr-t">分数的当前最大值</text>
  <line x1="252" y1="12" x2="274" y2="12" stroke="rgb(var(--color-secondary-500))" stroke-width="3.5"/>
  <text x="280" y="16" class="cr-t">FA4 实际用的最大值</text>
  <rect x="424" y="6" width="12" height="12" fill="rgba(var(--color-secondary-500), .15)"/>
  <text x="442" y="16" class="cr-t">允许的差距：最多 256×</text>
  <rect x="42" y="138.4" width="280" height="44.8" fill="rgba(var(--color-secondary-500), .15)"/>
  <rect x="322" y="85.2" width="168" height="44.8" fill="rgba(var(--color-secondary-500), .15)"/>
  <rect x="490" y="34.8" width="112" height="44.8" fill="rgba(var(--color-secondary-500), .15)"/>
  <line x1="40" y1="30" x2="40" y2="200" stroke="currentColor" stroke-opacity=".4"/>
  <line x1="40" y1="200" x2="604" y2="200" stroke="currentColor" stroke-opacity=".4"/>
  <text x="34" y="203" text-anchor="end" class="cr-k">0</text>
  <text x="34" y="147" text-anchor="end" class="cr-k">10</text>
  <text x="34" y="91" text-anchor="end" class="cr-k">20</text>
  <text x="34" y="35" text-anchor="end" class="cr-k">30</text>
  <text transform="translate(12 118) rotate(-90)" text-anchor="middle" class="cr-k">行最大值（log₂ 单位）</text>
  <polyline points="42,183.2 322,183.2 322,130 490,130 490,79.6 602,79.6" fill="none" stroke="rgb(var(--color-secondary-500))" stroke-width="3.5"/>
  <polyline points="42,183.2 98,183.2 98,169.2 154,169.2 154,165.3 210,165.3 210,149.6 266,149.6 266,147.4 322,147.4 322,130 378,130 378,128.3 434,128.3 434,127.2 490,127.2 490,79.6 546,79.6 546,78.5 602,78.5" fill="none" stroke="rgb(var(--color-primary-500))" stroke-width="2"/>
  <circle cx="70" cy="183.2" r="3" fill="rgb(var(--color-primary-500))"/>
  <circle cx="126" cy="169.2" r="3" fill="rgb(var(--color-primary-500))"/>
  <circle cx="182" cy="165.3" r="3" fill="rgb(var(--color-primary-500))"/>
  <circle cx="238" cy="149.6" r="3" fill="rgb(var(--color-primary-500))"/>
  <circle cx="294" cy="147.4" r="3" fill="rgb(var(--color-primary-500))"/>
  <circle cx="350" cy="130" r="3" fill="rgb(var(--color-primary-500))"/>
  <circle cx="406" cy="128.3" r="3" fill="rgb(var(--color-primary-500))"/>
  <circle cx="462" cy="127.2" r="3" fill="rgb(var(--color-primary-500))"/>
  <circle cx="518" cy="79.6" r="3" fill="rgb(var(--color-primary-500))"/>
  <circle cx="574" cy="78.5" r="3" fill="rgb(var(--color-primary-500))"/>
  <circle cx="322" cy="130" r="6" fill="none" stroke="rgb(var(--color-secondary-500))" stroke-width="2"/>
  <circle cx="490" cy="79.6" r="6" fill="none" stroke="rgb(var(--color-secondary-500))" stroke-width="2"/>
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
<figcaption><strong>图 5-2</strong> 一行上的有条件 rescale（示意数据）。当前最大值在 10 个 block 里有 9 个上升，标准 online softmax 每次都要 rescale $O$ 和 $\ell$；FA4 保留旧的参考最大值（粗线），只在真实最大值比它高出 8 以上（$\log_2$ 单位，阴影带）时才更新，这里只有两次。</figcaption>
</figure>

3. **tensor memory（TMEM）和 2-CTA MMA 模式，主要用在反向。** Blackwell 给每个 SM 加了 256KB 的 tensor memory，Tensor core 直接把累加结果写进去。2-CTA 模式下，同一个 cluster 里一对 SM 上的两个 thread block（CTA）一起执行一次 $M=256$ 的 MMA；每个 CTA 存 A tile 和累加器的一半，B 操作数也只暂存一半，B 的 shared memory 流量减半。B200 上反向受 shared memory 限制：按论文的模型，shared memory 流量约 3328 个周期，MMA 只有 2560 个周期。把更多中间结果放进 TMEM、再用 2-CTA 模式，可以降到约 2688 个周期，对 $dQ$ 的 atomic add 也减少了。
4. **用 CuTe-DSL 写，不用原生的 CUDA C++ 模板。** CuTe-DSL 是 CUTLASS 嵌在 Python 里的 kernel 语言。表达能力和 C++ 模板相当，编译快得多；attention 的变体（ALiBi、sliding window、soft-capping）可以写成普通的 Python 分数修改函数，JIT 编译进 kernel。

作者还和 cuDNN 团队合作，从 9.13/9.14 起把其中几项技术并进了 cuDNN。

*RTX 5090 实测待补：5090（sm_120）没有 TMEM 和 2-CTA MMA，FA4 的 Blackwell kernel 跑不了；这里改测软件模拟 exp 和有条件 rescale 这两项在 5090 上的效果。*

四代各自改了什么到这里就讲完了，下一节看四代之间始终没有变的部分。

## 6 四代没变的东西 {#constant}

前面几节讲的都是每一代改了什么。kernel 技巧一条条列下来，很容易忽略哪些东西始终没变，这里把它们明确写出来：

- **四代算的是同一个数学函数。** 都是 $\mathrm{softmax}(QK^\top/\sqrt d)V$，差别只在浮点舍入；FP8 路径还有量化误差，这部分误差有界，论文也实际测量过，没有含糊带过。
- **没有一代降低了 FLOP 数的渐近量级。** 每个版本的计算量都是 $\Theta(N^2d)$；重计算会多出一点，有 causal mask 时跳过完全被 mask 掉的 tile，实际执行的计算比算完整矩阵再 mask 少一半左右。整条演进线一直在做同一件事：去掉 FLOP 数本身并不需要的数据搬运和非 matmul 开销。
- **每一代针对的都是上一代留下的新瓶颈。** v1 去掉了 $N^2$ 大小的矩阵在 HBM 上的往返。v2 解决了 v1 的调度留下的低 occupancy 和 warp 之间的通信开销。v3 用上了 v1、v2 出现时还没有的异步硬件（TMA/WGMMA），另外用 FP8 处理精度和吞吐之间的取舍。v4 应对的是 Blackwell 的非对称扩展：exp 的 FLOP 数很少，在前向里却和 matmul 占一样多的周期，反向里 shared memory 的流量还超过了 matmul。所以「attention 慢」不是一次能解决的问题：每一代解决的只是当时最紧的约束，下一代 GPU 又会带来新的约束。

最后一节用一张表把四代汇总起来。

## 7 总结 {#summary}

[表 7-1](#tab-7-1) 列出每一代的年份、目标硬件和核心做法。

| 版本 | 年份 | 目标硬件 | 核心做法 |
|---|---|---|---|
| FlashAttention | 2022 | Ampere 时代的 GPU | tiling + online softmax + 重计算，N×N 矩阵从不在 HBM 里实体化 |
| FlashAttention-2 | 2023 | Ampere/Ada/Hopper | 推迟归一化；交换内外层循环，并在序列长度上并行；warp 之间切分 Q（split-Q） |
| FlashAttention-3 | 2024 | Hopper（H100） | warp specialization 的异步 pipeline（TMA + WGMMA）；GEMM 与 softmax 重叠；FP8 加 block 量化和 incoherent processing |
| FlashAttention-4 | 2026 | Blackwell（B200/GB200），代码也能在 Hopper 上跑 | 异步 MMA + 更大的 tile；软件模拟 exp + 有条件的 rescale；TMEM + 2-CTA MMA；CuTe-DSL |
{#tab-7-1 caption="**表 7-1** FlashAttention 四代对比"}

## References {#references}

1. Dao, Fu, Ermon, Rudra, Ré. ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135), NeurIPS 2022. 图 2-2 的 tiling 示意图以这篇论文的 Figure 1 为基础。
2. Dao. ["FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"](https://arxiv.org/abs/2307.08691), ICLR 2024. 另见 [Stanford Hazy Research 的介绍文章](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)。
3. Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. ["FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"](https://arxiv.org/abs/2407.08608), NeurIPS 2024. 另见作者的[博客文章](https://tridao.me/blog/2024/flash3/)。
4. Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao. ["FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"](https://arxiv.org/abs/2603.05451), arXiv:2603.05451, 2026 年 3 月。
5. [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)。四个版本的参考实现。
6. Milakov, Gimelshein. ["Online normalizer calculation for softmax"](https://arxiv.org/abs/1805.02867), arXiv:1805.02867, 2018. online softmax 的出处，2.2 节用到。
7. [Transformer FLOPs 计算器](/projects/flops/)。按同样的思路估算整个 Transformer 的参数量、FLOPs、显存和训练时间。
