---
title: "FlashAttention 1–4：从 IO 出发重写 attention kernel"
date: 2026-10-05
draft: false
math: true
description: "FlashAttention 四代的技术梳理：A100 上 IO-aware 的 tiling，更好的工作划分，Hopper 的异步执行，再到针对 Blackwell 各硬件单元扩展不对称的协同设计。GPU training optimization 系列的第四篇。"
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

四代 FlashAttention 都建立在同一个判断上：标准 attention 慢在搬数据，不在计算。前面的[谁偷走了 5090 的算力和显存](/blog/gpu-training-analysis/)已经在 RTX 5090 上用 roofline 和实测讲过这一点，这里只补 A100 上的两层存储（[图 1-1](#fig-1-1)）和 FlashAttention 论文给的时间拆分。

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

FlashAttention 论文实测了 GPT-2 的 attention 层，时间主要花在了计算量很小的那几个操作上（[图 1-2](#fig-1-2)）。

<figure id="fig-1-2">
<svg viewBox="0 0 640 196" width="100%" role="img" aria-label="GPT-2 的 attention 运行时间，读自 FlashAttention 论文的图 1：PyTorch 在两个矩阵乘上花约 4 ms，在 mask、softmax 和 dropout 上花约 13 ms，共约 17 ms；FlashAttention 的单个融合 kernel 约 2 ms">
  <style>
    .tb-lab { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .tb-sub { font-size: 11px; fill: currentColor; opacity: .7; }
    .tb-in  { font-size: 11px; text-anchor: middle; fill: rgb(var(--color-neutral-900)); }
    .tb-inw { font-size: 11px; text-anchor: middle; fill: #fff; }
    .tb-val { font-size: 12px; font-weight: 600; fill: currentColor; }
    .tb-tick { font-size: 11px; fill: currentColor; opacity: .7; text-anchor: middle; }
    .tb-leg { font-size: 11px; fill: currentColor; }
  </style>
  <path d="M178,30 V24 H518 V30" fill="none" stroke="currentColor" stroke-opacity=".45"/>
  <text x="348" y="16" text-anchor="middle" class="tb-sub">≈ 13 ms：三次逐元素 pass，每次都读写整个 N×N 矩阵</text>
  <text x="110" y="50" text-anchor="end" class="tb-lab">PyTorch</text>
  <text x="110" y="64" text-anchor="end" class="tb-sub">5 个独立的 op</text>
  <rect x="120" y="36" width="54" height="34" fill="rgb(var(--color-primary-500))"/>
  <text x="147" y="57" class="tb-inw">QK<tspan dy="-4" font-size="8">T</tspan></text>
  <rect x="176" y="36" width="120.7" height="34" fill="rgb(var(--color-secondary-500))"/>
  <text x="236.3" y="57" class="tb-in">mask</text>
  <rect x="298.7" y="36" width="96.7" height="34" fill="rgb(var(--color-secondary-500))"/>
  <text x="347" y="57" class="tb-in">softmax</text>
  <rect x="397.3" y="36" width="120.7" height="34" fill="rgb(var(--color-secondary-500))"/>
  <text x="457.7" y="57" class="tb-in">dropout</text>
  <rect x="520" y="36" width="50.7" height="34" fill="rgb(var(--color-primary-500))"/>
  <text x="545.3" y="57" class="tb-inw">×V</text>
  <text x="578" y="58" class="tb-val">≈ 16.9 ms</text>
  <text x="110" y="114" text-anchor="end" class="tb-lab">FlashAttention</text>
  <text x="110" y="128" text-anchor="end" class="tb-sub">1 个融合 kernel</text>
  <rect x="120" y="100" width="58.7" height="34" fill="rgb(var(--color-primary-500))"/>
  <text x="149.3" y="121" class="tb-inw">融合</text>
  <text x="186" y="122" class="tb-val">≈ 2.2 ms（快 7.6×）</text>
  <line x1="120" y1="150" x2="600" y2="150" stroke="currentColor" stroke-opacity=".4"/>
  <line x1="120" y1="150" x2="120" y2="154" stroke="currentColor" stroke-opacity=".4"/>
  <line x1="253.3" y1="150" x2="253.3" y2="154" stroke="currentColor" stroke-opacity=".4"/>
  <line x1="386.7" y1="150" x2="386.7" y2="154" stroke="currentColor" stroke-opacity=".4"/>
  <line x1="520" y1="150" x2="520" y2="154" stroke="currentColor" stroke-opacity=".4"/>
  <text x="120" y="167" class="tb-tick">0</text>
  <text x="253.3" y="167" class="tb-tick">5</text>
  <text x="386.7" y="167" class="tb-tick">10</text>
  <text x="520" y="167" class="tb-tick">15 ms</text>
  <rect x="120" y="179" width="11" height="11" rx="2" fill="rgb(var(--color-primary-500))"/>
  <text x="136" y="188" class="tb-leg">矩阵乘（几乎全部的 FLOPs）</text>
  <rect x="350" y="179" width="11" height="11" rx="2" fill="rgb(var(--color-secondary-500))"/>
  <text x="366" y="188" class="tb-leg">mask · softmax · dropout（memory-bound）</text>
</svg>
<figcaption><strong>图 1-2</strong> 标准 attention 的时间花在哪（GPT-2，A100；数值是从 FlashAttention 论文图 1 读出的近似值）。两个矩阵乘占了几乎全部 FLOPs，却只用了约四分之一的时间；mask、softmax、dropout 几乎不做运算，却各自把整个 $N\times N$ 矩阵从 HBM 读出再写回。论文的 7.6× 加速来自把这些步骤融合成一个从不写出该矩阵的 kernel。</figcaption>
</figure>

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

把 $Q$、$K$、$V$ 沿 $N$ 切成 block，block 小到一个 $Q$ block 加一个 $K$/$V$ block 能放进片上的 SRAM（即 shared memory）。每个局部的 score block $S_{ij}=Q_iK_j^\top/\sqrt d$ 都在片上算，完整的 $S$ 和 $P$ 从不写回 HBM（[图 2-2](#fig-2-2)）。tiling 为什么能少读显存，在[单卡 GPU 优化方法](/blog/gpu-single-gpu-optimization/#tiling)里按 matmul 算过一遍；这里多出来的一步是 softmax。

<figure id="fig-2-2">
<svg viewBox="0 0 640 352" width="100%" role="img" aria-label="Tiling：Q 的第 i 个 block 和 K 的第 j 个 block 相乘，得到 N × N 的 score 矩阵里的一个 tile，此刻只有这一个 tile 存在；它再乘 V 的第 j 个 block，加进 O 的第 i 个 block，内层循环让 j 走过这一行">
  <style>
    .tl-lab { font-size: 13px; font-weight: 600; fill: currentColor; }
    .tl-sub { font-size: 11px; fill: currentColor; opacity: .7; font-weight: 400; }
    .tl-txt { font-size: 11px; fill: currentColor; }
    .tl-in  { font-size: 12.5px; font-weight: 600; text-anchor: middle; fill: rgb(var(--color-neutral-900)); }
  </style>
  <rect x="150" y="30" width="40" height="24" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="190" y="30" width="40" height="24" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="230" y="30" width="40" height="24" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="270" y="30" width="40" height="24" fill="rgb(var(--color-primary-500))" stroke="rgb(var(--color-primary-500))"/>
  <rect x="310" y="30" width="40" height="24" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="350" y="30" width="40" height="24" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="390" y="30" width="40" height="24" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="430" y="30" width="40" height="24" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <text x="310" y="20" text-anchor="middle" class="tl-lab">K<tspan dy="-5" font-size="9">T</tspan><tspan dy="5" class="tl-sub">  d × N</tspan></text>
  <text x="112" y="58" text-anchor="middle" class="tl-lab">Q</text>
  <text x="112" y="72" text-anchor="middle" class="tl-sub">N × d</text>
  <text x="512" y="58" text-anchor="middle" class="tl-lab">V</text>
  <text x="512" y="72" text-anchor="middle" class="tl-sub">N × d</text>
  <text x="584" y="58" text-anchor="middle" class="tl-lab">O</text>
  <text x="584" y="72" text-anchor="middle" class="tl-sub">N × d</text>
  <rect x="150" y="160" width="320" height="40" fill="rgba(var(--color-primary-500), .08)"/>
  <rect x="150" y="160" width="40" height="40" fill="rgba(var(--color-secondary-500), .25)"/>
  <rect x="190" y="160" width="40" height="40" fill="rgba(var(--color-secondary-500), .25)"/>
  <rect x="230" y="160" width="40" height="40" fill="rgba(var(--color-secondary-500), .25)"/>
  <line x1="190" y1="80" x2="190" y2="320" stroke="currentColor" stroke-opacity=".1"/>
  <line x1="230" y1="80" x2="230" y2="320" stroke="currentColor" stroke-opacity=".1"/>
  <line x1="270" y1="80" x2="270" y2="320" stroke="currentColor" stroke-opacity=".1"/>
  <line x1="310" y1="80" x2="310" y2="320" stroke="currentColor" stroke-opacity=".1"/>
  <line x1="350" y1="80" x2="350" y2="320" stroke="currentColor" stroke-opacity=".1"/>
  <line x1="390" y1="80" x2="390" y2="320" stroke="currentColor" stroke-opacity=".1"/>
  <line x1="430" y1="80" x2="430" y2="320" stroke="currentColor" stroke-opacity=".1"/>
  <line x1="150" y1="120" x2="470" y2="120" stroke="currentColor" stroke-opacity=".1"/>
  <line x1="150" y1="160" x2="470" y2="160" stroke="currentColor" stroke-opacity=".1"/>
  <line x1="150" y1="200" x2="470" y2="200" stroke="currentColor" stroke-opacity=".1"/>
  <line x1="150" y1="240" x2="470" y2="240" stroke="currentColor" stroke-opacity=".1"/>
  <line x1="150" y1="280" x2="470" y2="280" stroke="currentColor" stroke-opacity=".1"/>
  <rect x="150" y="80" width="320" height="240" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-dasharray="4 4"/>
  <line x1="312" y1="180" x2="563" y2="180" stroke="currentColor" stroke-opacity=".55" stroke-width="1.5"/>
  <polygon points="570,180 562,176 562,184" fill="currentColor" fill-opacity=".55"/>
  <rect x="100" y="80" width="24" height="40" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="100" y="120" width="24" height="40" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="100" y="160" width="24" height="40" fill="rgb(var(--color-primary-500))" stroke="rgb(var(--color-primary-500))"/>
  <rect x="100" y="200" width="24" height="40" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="100" y="240" width="24" height="40" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="100" y="280" width="24" height="40" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="500" y="80" width="24" height="30" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="500" y="110" width="24" height="30" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="500" y="140" width="24" height="30" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="500" y="170" width="24" height="30" fill="rgb(var(--color-primary-500))" stroke="rgb(var(--color-primary-500))"/>
  <rect x="500" y="200" width="24" height="30" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="500" y="230" width="24" height="30" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="500" y="260" width="24" height="30" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="500" y="290" width="24" height="30" fill="rgba(var(--color-primary-500), .12)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".45"/>
  <rect x="572" y="80" width="24" height="40" fill="none" stroke="currentColor" stroke-opacity=".35"/>
  <rect x="572" y="120" width="24" height="40" fill="none" stroke="currentColor" stroke-opacity=".35"/>
  <rect x="572" y="160" width="24" height="40" fill="rgba(var(--color-primary-500), .35)" stroke="rgb(var(--color-primary-500))"/>
  <rect x="572" y="200" width="24" height="40" fill="none" stroke="currentColor" stroke-opacity=".35"/>
  <rect x="572" y="240" width="24" height="40" fill="none" stroke="currentColor" stroke-opacity=".35"/>
  <rect x="572" y="280" width="24" height="40" fill="none" stroke="currentColor" stroke-opacity=".35"/>
  <rect x="270" y="160" width="40" height="40" fill="rgb(var(--color-secondary-500))"/>
  <text x="290" y="185" class="tl-in">S<tspan dy="3" font-size="9">ij</tspan></text>
  <line x1="290" y1="56" x2="290" y2="151" stroke="currentColor" stroke-opacity=".55" stroke-width="1.5"/>
  <polygon points="290,158 286,150 294,150" fill="currentColor" fill-opacity=".55"/>
  <line x1="126" y1="180" x2="261" y2="180" stroke="currentColor" stroke-opacity=".55" stroke-width="1.5"/>
  <polygon points="268,180 260,176 260,184" fill="currentColor" fill-opacity=".55"/>
  <text x="318" y="154" class="tl-txt">乘 V_j，加进 O_i</text>
  <line x1="156" y1="214" x2="457" y2="214" stroke="currentColor" stroke-opacity=".45"/>
  <polygon points="464,214 456,210 456,218" fill="currentColor" fill-opacity=".45"/>
  <text x="462" y="232" text-anchor="end" class="tl-txt">内层循环：j 走过每个 K/V block</text>
  <line x1="86" y1="84" x2="86" y2="305" stroke="currentColor" stroke-opacity=".45"/>
  <polygon points="86,312 82,304 90,304" fill="currentColor" fill-opacity=".45"/>
  <text transform="translate(78 200) rotate(-90)" text-anchor="middle" class="tl-txt">外层循环：i 走过每个 Q block</text>
  <text x="290" y="264" text-anchor="middle" class="tl-txt">此刻只有这一个 tile</text>
  <text x="290" y="278" text-anchor="middle" class="tl-sub">（在片上，只存在一轮）</text>
  <text x="310" y="342" text-anchor="middle" class="tl-sub">S = QK<tspan dy="-4" font-size="8">T</tspan><tspan dy="4"> 是 N × N，从不完整存下来</tspan></text>
</svg>
<figcaption><strong>图 2-2</strong> 和 FlashAttention 论文的 Figure 1 是同一件事，按 FlashAttention-2 后来采用的循环顺序画。$Q_i$ 和 $K_j$ 得到 score 矩阵的一个 tile $S_{ij}$，它乘上 $V_j$ 加进 $O_i$ 后就丢掉，再算下一个 $j$。写回 HBM 的只有算完的 $O_i$ 和它每行的统计量。FA2 里每个 query block $i$ 由一个 thread block 负责。</figcaption>
</figure>

### 2.2 Online softmax {#online-softmax}

softmax 是按行归一化的：$P_{ij} = \exp(S_{ij})/\sum_k \exp(S_{ik})$，分母要用到一整行的 score，而 tiling 以后一个 tile 只看得到这一行里的 $B_k$ 个。

先看普通的 softmax 怎么算（[图 2-3](#fig-2-3)）。直接算 $e^x$ 容易溢出：fp32 在 $x$ 超过 88.7、fp16 在 $x$ 超过 11.1 时就是 inf。所以实际的实现都先减掉这一行的最大值 $m$，分子分母同乘 $e^{-m}$，结果不变。这样一行要扫三遍：求最大值，求分母，归一化。第 2 遍的每一项 $e^{x-m}$ 都要用到第 1 遍的 $m$；分块以后只看到第一个 block 时，最大值可能还在后面的 block 里，第 2 遍没法开始。

<figure id="fig-2-3">
<svg viewBox="0 0 680 292" width="100%" role="img" aria-label="普通 softmax 一行要扫三遍：第 1 遍求最大值 m = 4，第 2 遍求分母 ℓ，每一项都要用到 m，第 3 遍归一化。分块以后只看到第一个 block 的 2 和 1 时，最大值 4 还在后面的 block 里，第 2 遍没法开始">
  <style>
    .os2 { font-size: 11px; fill: currentColor; }
    .os2-b { font-size: 12px; font-weight: 600; fill: currentColor; }
    .os2-s { font-size: 10.5px; fill: currentColor; opacity: .72; }
    .os2-c { font-size: 12px; font-weight: 600; text-anchor: middle; fill: currentColor; }
    .os2-hi { fill: var(--fig-hi); font-weight: 600; }
  </style>
  <defs><marker id="osA-m" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-1)"/></marker></defs>
<text class="os2-b" x="8" y="20">普通 softmax：一行 score 要扫三遍</text>
<text class="os2" x="8" y="48">一行 score</text>
<rect x="200" y="30" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="226.0" y="48.0">2</text>
<rect x="256" y="30" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="282.0" y="48.0">1</text>
<rect x="312" y="30" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="338.0" y="48.0">4</text>
<rect x="368" y="30" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="394.0" y="48.0">3</text>
<rect x="424" y="30" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="450.0" y="48.0">1</text>
<rect x="480" y="30" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="506.0" y="48.0">0</text>
<text class="os2" x="8" y="88">第 1 遍：求最大值</text>
<line x1="200" y1="84" x2="530" y2="84" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#osA-m)"/>
<text class="os2" x="542" y="88">m = max = 4</text>
<text class="os2" x="8" y="122">第 2 遍：求分母</text>
<line x1="200" y1="118" x2="530" y2="118" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#osA-m)"/>
<text class="os2" x="542" y="122">ℓ = Σ e^(x − m) = 1.62</text>
<text class="os2-s" x="204" y="133">每一项都要用到第 1 遍的 m</text>
<text class="os2" x="8" y="156">第 3 遍：归一化</text>
<line x1="200" y1="152" x2="530" y2="152" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#osA-m)"/>
<text class="os2" x="542" y="156">p = e^(x − m) / ℓ</text>
<line x1="8" y1="190" x2="672" y2="190" stroke="currentColor" stroke-opacity=".15"/>
<text class="os2-b" x="8" y="212">分块以后：一次只看得到一个 block</text>
<rect x="200" y="224" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="226.0" y="242.0">2</text>
<rect x="256" y="224" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="282.0" y="242.0">1</text>
<text class="os2-s" x="254" y="266" text-anchor="middle">block 1</text>
<rect x="322" y="224" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.25" stroke-opacity=".35"/><text class="os2-c" x="348.0" y="242.0">4</text>
<rect x="378" y="224" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.25" stroke-opacity=".35"/><text class="os2-c" x="404.0" y="242.0">3</text>
<text class="os2-s" x="376" y="266" text-anchor="middle">block 2</text>
<rect x="444" y="224" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.25" stroke-opacity=".35"/><text class="os2-c" x="470.0" y="242.0">1</text>
<rect x="500" y="224" width="52" height="28" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.25" stroke-opacity=".35"/><text class="os2-c" x="526.0" y="242.0">0</text>
<text class="os2-s" x="498" y="266" text-anchor="middle">block 3</text>
<text class="os2" x="200" y="284">只看到 block 1 的 2、1 时，最大值 4 还在 block 2 里，第 2 遍没法开始</text>
</svg>
<figcaption><strong>图 2-3</strong> 一行 6 个 score 上的普通 softmax。第 2、3 遍都依赖第 1 遍扫完整行得到的 $m$，所以按 block 分开以后，没法边读 block 边算分母。</figcaption>
</figure>

Online softmax（Milakov 和 Gimelshein 2018）不等整行的最大值：先按到目前为止的最大值算，最大值变了再修正（[图 2-4](#fig-2-4)）。每扫完一个 block 记下两个数，$m$ 是到目前为止的最大值，$\ell$ 是按这个 $m$ 算的分母。下一个 block 里出现更大的值时，$\ell$ 里的每一项都是按旧的 $m$ 算的，比按新最大值算的都大了 $e^{m_{\text{new}} - m_{\text{old}}}$ 倍。所有项差的是同一个倍数，所以整个 $\ell$ 乘一次 $e^{m_{\text{old}} - m_{\text{new}}}$ 就对上了，不用回头重算：

<p align="center">$m_{\text{new}} = \max\big(m_{\text{old}},\ \max_{k \in \text{新 block}} x_k\big), \qquad \ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}}\,\ell_{\text{old}} + \sum_{k \in \text{新 block}} e^{x_k - m_{\text{new}}}$</p>

attention 要的不是 softmax 本身，而是用它对 $V$ 加权求和。$O$ 按同样的办法累加还没归一化的 $e^{x-m} v$，最大值变大时和 $\ell$ 乘同一个修正因子。扫完所有 block 以后 $\ell$ 才是完整的分母，$O$ 除以它一次就是结果，三遍变成了一遍。

<figure id="fig-2-4">
<svg viewBox="0 0 680 300" width="100%" role="img" aria-label="online softmax 在 6 个 score、3 个 block 上的计算。每个 block 只扫一遍，先更新最大值 m，再把之前的分母 ℓ 和累加的 O 乘修正因子 e 的 m_old 减 m_new 次方，再加上这个 block 的项。block 2 把最大值从 2 变成 4，修正因子是 e 的 −2 次方。最后 O 除以 ℓ 等于 3.12，和一次算完的结果相同">
  <style>
    .os2 { font-size: 11px; fill: currentColor; }
    .os2-b { font-size: 12px; font-weight: 600; fill: currentColor; }
    .os2-s { font-size: 10.5px; fill: currentColor; opacity: .72; }
    .os2-c { font-size: 12px; font-weight: 600; text-anchor: middle; fill: currentColor; }
    .os2-hi { fill: var(--fig-hi); font-weight: 600; }
  </style>
  <defs><marker id="osB-m" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-1)"/></marker></defs>
<text class="os2-b" x="8" y="18">online softmax：每个 block 只扫一遍，m、ℓ、O 一路带下去（v 取 1 到 6）</text>
<rect x="4" y="30" width="200" height="208" rx="8" style="fill: var(--fig-3)" fill-opacity=".12" stroke="currentColor" stroke-opacity=".18"/>
<text class="os2-b" x="14" y="50">block 1</text>
<rect x="84" y="36" width="46" height="24" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="107.0" y="52.0">2</text>
<text class="os2-s" x="107" y="74" text-anchor="middle">v = 1</text>
<rect x="136" y="36" width="46" height="24" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="159.0" y="52.0">1</text>
<text class="os2-s" x="159" y="74" text-anchor="middle">v = 2</text>
<text class="os2" x="14" y="96" xml:space="preserve">m = max(−∞, 2, 1) = 2</text>
<text class="os2" x="14" y="114" xml:space="preserve">ℓ = e^(2−2) + e^(1−2)</text>
<text class="os2" x="14" y="132" xml:space="preserve">  = 1 + 0.37 = 1.37</text>
<text class="os2" x="14" y="156" xml:space="preserve">O = 1·1 + 0.37·2</text>
<text class="os2" x="14" y="174" xml:space="preserve">  = 1.74</text>
<text class="os2" x="14" y="192" xml:space="preserve"></text>
<rect x="12" y="208" width="184" height="22" rx="4" style="fill: var(--fig-1)" fill-opacity=".14"/>
<text class="os2" x="104.0" y="223" text-anchor="middle">m = 2，ℓ = 1.37，O = 1.74</text>
<line x1="206" y1="219" x2="236" y2="219" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#osB-m)"/>
<rect x="240" y="30" width="200" height="208" rx="8" style="fill: var(--fig-3)" fill-opacity=".12" stroke="currentColor" stroke-opacity=".18"/>
<text class="os2-b" x="250" y="50">block 2</text>
<rect x="320" y="36" width="46" height="24" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="343.0" y="52.0">4</text>
<text class="os2-s" x="343" y="74" text-anchor="middle">v = 3</text>
<rect x="372" y="36" width="46" height="24" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="395.0" y="52.0">3</text>
<text class="os2-s" x="395" y="74" text-anchor="middle">v = 4</text>
<text class="os2" x="250" y="96" xml:space="preserve">m = max(2, 4, 3) = 4</text>
<text class="os2" x="250" y="114" xml:space="preserve">ℓ = 1.37 <tspan class="os2-hi">× e^(2−4)</tspan> + e^0 + e^(−1)</text>
<text class="os2" x="250" y="132" xml:space="preserve">  = 0.19 + 1 + 0.37 = 1.55</text>
<text class="os2" x="250" y="156" xml:space="preserve">O = 1.74 <tspan class="os2-hi">× e^(2−4)</tspan></text>
<text class="os2" x="250" y="174" xml:space="preserve">  + 1·3 + 0.37·4</text>
<text class="os2" x="250" y="192" xml:space="preserve">  = 0.23 + 3 + 1.47 = 4.71</text>
<rect x="248" y="208" width="184" height="22" rx="4" style="fill: var(--fig-1)" fill-opacity=".14"/>
<text class="os2" x="340.0" y="223" text-anchor="middle">m = 4，ℓ = 1.55，O = 4.71</text>
<line x1="442" y1="219" x2="472" y2="219" style="stroke: var(--fig-1)" stroke-width="1.6" marker-end="url(#osB-m)"/>
<rect x="476" y="30" width="200" height="208" rx="8" style="fill: var(--fig-3)" fill-opacity=".12" stroke="currentColor" stroke-opacity=".18"/>
<text class="os2-b" x="486" y="50">block 3</text>
<rect x="556" y="36" width="46" height="24" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="579.0" y="52.0">1</text>
<text class="os2-s" x="579" y="74" text-anchor="middle">v = 5</text>
<rect x="608" y="36" width="46" height="24" rx="3" style="fill: var(--fig-3); stroke: var(--fig-1)" fill-opacity="0.9" stroke-opacity=".35"/><text class="os2-c" x="631.0" y="52.0">0</text>
<text class="os2-s" x="631" y="74" text-anchor="middle">v = 6</text>
<text class="os2" x="486" y="96" xml:space="preserve">m = max(4, 1, 0) = 4</text>
<text class="os2" x="486" y="114" xml:space="preserve">ℓ = 1.55 <tspan class="os2-hi">× e^(4−4)</tspan> + e^(−3) + e^(−4)</text>
<text class="os2" x="486" y="132" xml:space="preserve">  = 1.55 + 0.05 + 0.02 = 1.62</text>
<text class="os2" x="486" y="156" xml:space="preserve">O = 4.71 <tspan class="os2-hi">× e^(4−4)</tspan></text>
<text class="os2" x="486" y="174" xml:space="preserve">  + 0.05·5 + 0.02·6</text>
<text class="os2" x="486" y="192" xml:space="preserve">  = 4.71 + 0.25 + 0.11 = 5.07</text>
<rect x="484" y="208" width="184" height="22" rx="4" style="fill: var(--fig-1)" fill-opacity=".14"/>
<text class="os2" x="576.0" y="223" text-anchor="middle">m = 4，ℓ = 1.62，O = 5.07</text>
<text class="os2" x="340.0" y="264" text-anchor="middle">最后只除一次：O / ℓ = 5.07 / 1.62 = 3.12，和 6 个 score 一次算出来的 softmax 加权和相同</text>
<text class="os2-s" x="340.0" y="288" text-anchor="middle"><tspan class="os2-hi">红色</tspan>是修正因子 e^(m_old − m_new)：最大值变大时，之前累加的 ℓ 和 O 都要乘它；最大值不变时它等于 1</text>
</svg>
<figcaption><strong>图 2-4</strong> 同一行 6 个 score 分 3 个 block 做 online softmax，每个 score 配一个 $v$（取 1 到 6，相当于 $d = 1$ 的 $V$）。每个 block 先更新 $m$，再把之前的 $\ell$、$O$ 乘修正因子，加上这个 block 的 $e^{x-m}$ 和 $e^{x-m}v$；块底的深色框是带到下一个 block 的状态。数字保留两位小数。</figcaption>
</figure>

两段各自算好的 $(m_1, \ell_1)$、$(m_2, \ell_2)$ 也能直接合并：$m = \max(m_1, m_2)$，$\ell = e^{m_1 - m}\,\ell_1 + e^{m_2 - m}\,\ell_2$。这个合并和加法一样满足结合律，所以既可以一段接一段地扫，也可以各段并行算完再归约；FlashAttention 在一个 thread block 里沿 K/V 一段接一段地扫。写成伪代码，一次处理一个 query block，最后归一化一次：

```
for 每个 query block i:
    O_i ← 0,  ℓ_i ← 0,  m_i ← -∞           # 只放在 shared memory / register 里
    for 每个 key/value block j:
        S_ij  ← Q_i K_jᵀ / sqrt(d)         # B × B，只在片上，不进 HBM
        m_new ← max(m_i, rowmax(S_ij))
        P_ij  ← exp(S_ij − m_new)
        ℓ_i   ← exp(m_i − m_new)·ℓ_i + rowsum(P_ij)
        O_i   ← exp(m_i − m_new)·O_i + P_ij·V_j
        m_i   ← m_new
    O_i ← O_i / ℓ_i                        # O_i（和每行的统计量）只写回 HBM 一次
```

严格说，这是 FlashAttention-2 采用的循环顺序。FA1 论文的 Algorithm 1 是反过来套的：外层循环走 $K$/$V$ block，内层循环走 $Q$ block，所以内层的每一步都要从 HBM 读出 $O_i$、$\ell_i$、$m_i$，更新后再写回去，而且每一步都用 $\mathrm{diag}(\ell_i)^{-1}$ 把 $O_i$ 归一化一次。后面的 IO 复杂度对两种顺序都成立；换循环顺序是 FA2 的改动之一（[第 3 节](#fa2)）。

这样算是精确的，不是近似：修正以后每一项都等于按最终最大值算的值，因为 $e^{S-m_{\text{old}}}\cdot e^{m_{\text{old}}-m_{\text{new}}}=e^{S-m_{\text{new}}}$；和一次算完的实现相比，只差浮点加法的顺序。整个过程没有 $N\times N$ 大小的东西写进 HBM，额外的显存只有每行的 $m$、$\ell$，是 $O(N)$ 而不是 $O(N^2)$。

前向不存 $P$，反向却要用到它，这是下一小节的事。

### 2.3 反向的 recomputation {#fa1-recompute}

attention 的反向一般要再用一次 $P$。为此存下完整的 $N\times N$ 矩阵，就把刚省下的显存又花回去了。FlashAttention 只存 $O$ 和每行的统计量 $(m, \ell)$，反向时在片上把需要的 $S_{ij}$、$P_{ij}$ block 重新算出来。这是用少量额外的 matmul 计算换显存和显存读写的大幅减少，被省掉的存储正好在慢的那一层，这时重算比存下来划算（[单卡 GPU 优化方法第 4 节](/blog/gpu-single-gpu-optimization/#recompute)）。

论文的复杂度结论：标准 attention 要 $\Theta(Nd+N^2)$ 次 HBM 访问，FlashAttention 要 $\Theta(N^2d^2M^{-1})$，$M$ 是 SRAM 的大小。常见的 $d$（64–128）和 $M$（100KB 左右）下，$d^2$ 比 $M$ 小很多倍，所以 FlashAttention 的 HBM 访问也少很多倍；$M=\Theta(d^2)$ 时两者才一样。论文给的下界有意取得很弱：不存在一个精确算法，能对 $[d, Nd]$ 里的**每一个** $M$ 都做到 $o(N^2d^2M^{-1})$ 次访问。论文自己的测量直接给出了这个取舍：算得更多，搬得少得多，耗时短得多（[图 2-5](#fig-2-5)）。

<figure id="fig-2-5">
<svg viewBox="0 0 640 196" width="100%" role="img" aria-label="A100 上 GPT-2 medium 的 attention，前向加反向：标准 attention 计算量 6.66e10 FLOPs，HBM 读写 40.3 GB，耗时 41.7 ms；FlashAttention 计算量 7.52e10 FLOPs，多 13%，但 HBM 读写只有 4.4 GB，是 1/9.2，耗时 7.3 ms，快 5.7 倍">
  <style>
    .ft-lab { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .ft-sub { font-size: 11px; fill: currentColor; opacity: .7; }
    .ft-val { font-size: 11.5px; fill: currentColor; }
    .ft-hl  { font-size: 11.5px; font-weight: 600; fill: currentColor; }
  </style>
  <rect x="170" y="5" width="11" height="11" rx="2" fill="rgba(128,128,128,.45)"/>
  <text x="186" y="14" class="ft-sub">标准 attention</text>
  <rect x="320" y="5" width="11" height="11" rx="2" fill="rgb(var(--color-primary-500))"/>
  <text x="336" y="14" class="ft-sub">FlashAttention</text>
  <text x="160" y="47" text-anchor="end" class="ft-lab">计算量</text>
  <text x="160" y="61" text-anchor="end" class="ft-sub">FLOPs</text>
  <rect x="170" y="30" width="318.8" height="14" fill="rgba(128,128,128,.45)"/>
  <text x="494.8" y="41" class="ft-val">6.66e10</text>
  <rect x="170" y="48" width="360" height="14" fill="rgb(var(--color-primary-500))"/>
  <text x="536" y="59" class="ft-hl">7.52e10（+13%）</text>
  <text x="160" y="107" text-anchor="end" class="ft-lab">HBM 读写量</text>
  <text x="160" y="121" text-anchor="end" class="ft-sub">读 + 写</text>
  <rect x="170" y="90" width="360" height="14" fill="rgba(128,128,128,.45)"/>
  <text x="536" y="101" class="ft-val">40.3 GB</text>
  <rect x="170" y="108" width="39.3" height="14" fill="rgb(var(--color-primary-500))"/>
  <text x="215.3" y="119" class="ft-hl">4.4 GB，是 1/9.2</text>
  <text x="160" y="167" text-anchor="end" class="ft-lab">耗时</text>
  <text x="160" y="181" text-anchor="end" class="ft-sub">前向 + 反向</text>
  <rect x="170" y="150" width="360" height="14" fill="rgba(128,128,128,.45)"/>
  <text x="536" y="161" class="ft-val">41.7 ms</text>
  <rect x="170" y="168" width="63" height="14" fill="rgb(var(--color-primary-500))"/>
  <text x="239" y="179" class="ft-hl">7.3 ms，快 5.7 倍</text>
</svg>
<figcaption><strong>图 2-5</strong> 按 FlashAttention 论文 Figure 2（左）重画：A100 上 GPT-2 medium 的 attention（序列长度 1024，head dimension 64，16 个 head，batch 64），前向加反向，每一行单独缩放。FlashAttention 因为反向要重算 score tile，计算量更多，但 HBM 读写只有 1/9.2，所以快 5.7 倍。耗时跟着显存读写走，不跟着 FLOPs 走。</figcaption>
</figure>

实际效果：序列长度 128–4K 时，attention 这个 op 本身（前向加反向）比 PyTorch 的标准实现快 2–4 倍；端到端训练上，BERT-large（序列长度 512，对比 MLPerf 1.1 的纪录）快 15%，GPT-2（序列长度 1K，对比 HuggingFace）快 3 倍，Long-Range Arena 快 2.4 倍。显存随 $N$ 线性增长而不是平方增长，序列越长省得越多，4K token 时显存少约 20 倍，于是可以用长得多的上下文训练，算的仍然是**精确**的 softmax，当时的稀疏 attention 和低秩 attention 都做不到这一点。

FA1 解决了 HBM 的读写，但在 A100 上前向只用到 Tensor core 峰值的 30–50%（[图 7-1](#fig-7-1)）。剩下的差距出在 kernel 怎么用 GPU 上，这是 FlashAttention-2 要解决的。

## 3 FlashAttention-2：让 GPU 忙起来 {#fa2}

[第 2 节](#fa1)的 FlashAttention 靠 tiling、online softmax 和 recomputation 把 $N \times N$ 的矩阵留在片上，它的 online softmax 每一步都把 $O$ 归一化一次。FlashAttention-2 的数学和它相同，改的是工作怎么分配。

<div class="fa-intuit"><b>思路：</b>v1 搬的数据量已经对了，但 GPU 大部分时候闲着。v2 不改数学，只改调度：分出多得多的互相独立的工作，让每个 SM 都有活干；同时少花指令在 matmul 以外的簿记上。</div>

v1 解决了 IO 的问题，吞吐却还有很大余量：在 A100 上，它的前向只达到 GPU 理论峰值 FLOPS 的 30–50%（反向是 25–35%），远低于调得好的 dense GEMM 能达到的 80–90%。FlashAttention-2（Dao，2023）是一篇讲工作调度的论文：数学不变，改进的是工作怎么切分给 thread block 和 warp。

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

合起来，v2 比 v1 快约 2 倍：在 A100 上，前向最高达到理论峰值 FLOPS 的 73%（约 2.3e14 FLOPS），反向最高 63%，因为反向的数据复用模式本来就差一些。端到端地看，GPT 类模型的训练每张 A100 最高达到 2.25e14 FLOPS（72% 的 model FLOPs utilization）。v2 不需要任何新的硬件特性，就是一篇「把手里已有的 GPU 用得更好」的论文，这也是它在 Hopper 以外的 GPU（以及 AMD 的 ROCm 移植版）上部署最广的原因。

Hopper 新增了几种异步硬件，v1 和 v2 都不是为它们设计的，[第 4 节](#fa3)的 FlashAttention-3 就是围绕这些硬件写的。

## 4 FlashAttention-3：用上 Hopper 新加的硬件 {#fa3}

[第 3 节](#fa2)的 FA2 推迟归一化、交换内外层循环的顺序、沿序列长度并行，再用 split-Q 把工作分给 warp，在 A100 上前向做到了峰值的 73%。换到 H100，硬件多了 v1、v2 设计时都没有用到的东西，FA3 就从这里入手。

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

2. **GEMM 和 softmax 重叠。** 在单个 tile 上，softmax 依赖这个 block 的 $QK^\top$，$PV$ 的 GEMM 又依赖 softmax，所以它们只能串行。FA3 从两个互补的方向拆开这条依赖链。第一个是 warpgroup 之间的 ping-pong（[图 4-2](#fig-4-2)）：两个 consumer warpgroup 各自负责一个不同的 query tile，用 barrier 强制它们交替执行，warpgroup A 做 GEMM 时 warpgroup B 做 softmax，反过来也一样。论文报告，在 H100 上 head dim 128、序列长度 8K 时，这一项把 FP16 前向吞吐从约 5.7e14 FLOPS 提到约 6.2e14 FLOPS。第二个是 warpgroup 内部的 pipeline：在同一个 warpgroup 里，下一个 block 的 $QK^\top$ 和当前 block 的 $PV$ 这两组异步 WGMMA 按一定的时机发出，让 softmax 在它们执行期间进行。

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

3. **FP8：block 量化和 incoherent processing。** 在 matmul 之前直接把 $Q$、$K$ 量化成 FP8，吞吐上很划算，但精度不够：少数离群值占满了量化范围，其余数值的分辨率被压得很低（FP8 和 block 量化见[单卡优化篇](/blog/gpu-single-gpu-optimization/#precision)）。FA3 用了两个互补的办法。第一是 block 量化：每个 tile 一个 FP8 scale，而不是整个张量一个，离群值的影响只限于它所在的 block；kernel 本来就按 tile 计算，所以这样做没有额外开销。第二是 incoherent processing：量化前把 $Q$ 和 $K$ 乘上同一个随机正交矩阵 $M$。因为 $M$ 正交，$(QM)(KM)^\top = QMM^\top K^\top = QK^\top$，attention 分数在数学上不变；但 $QM$ 的每个元素现在是许多原始元素的随机混合，任何一个坐标都不会被单个离群值主导（[图 4-3](#fig-4-3)）。实现上用的是带随机符号翻转的 Hadamard 变换，长度为 $d$ 的一行代价是 $O(d\log d)$，稠密的旋转则要 $O(d^2)$。在模拟了离群特征的测试输入上，这条 FP8 路径的 RMSE 比用 per-tensor scale 的 FP8 attention 基线低 2.6 倍。

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

FA3 在 H100 上比 FA2 快 1.5–2.0 倍，BF16 最高到 8.4e14 FLOPS（约为 H100 SXM5 稠密峰值 9.89e14 FLOPS 的 85%），FP8 最高到 1.3e15 FLOPS。2024 年 7 月的 arXiv v1 报的是 7.4e14 FLOPS 和接近 1.2e15 FLOPS，这里的数字来自 [References](#references) 里引用的 NeurIPS 2024 版本。

[图 4-2](#fig-4-2) 里 softmax 只用 GEMM 一半左右的时间，到了 Blackwell 的 B200 上两者一样长，[第 5 节](#fa4)的 FA4 就从这里出发。

## 5 FlashAttention-4：瓶颈随硬件换代而转移 {#fa4}

[第 4 节](#fa3)的 FA3 在 Hopper 上用 warp specialization 搭起异步 pipeline，让 GEMM 和 softmax 重叠，再加上 FP8，在 H100 上做到了峰值的 85%。到了 Blackwell，硬件各部分的提升幅度不一样，瓶颈又换了位置。

<div class="fa-intuit"><b>思路：</b>Blackwell 把矩阵乘的速度提高了一倍，做 exp 的单元和 shared memory 却没变，原来藏在 matmul 后面的 softmax 现在和 matmul 用的时间一样长。FA4 直接削减这部分工作：一部分 exp 改到别的单元上算，跳过不影响结果的 rescale，把更多数据放进新增的 tensor memory。</div>

FlashAttention-4 的作者是 Zadouri、Hoehnerbach、Shah、Liu、Thakkar 和 Dao（Princeton、Meta、Colfax Research、NVIDIA、Georgia Tech 和 Together AI），2026 年 3 月发在 arXiv 上。论文针对 Blackwell 数据中心 GPU（B200/GB200）；开源实现（flash-attention 仓库里的 `flash_attn/cute`，安装包名是 `flash-attn-4`）也带了 Hopper 的 kernel。它是四代里最新的一代，在生产环境里经过的检验也最少，下面的数字都应当看作作者自己的测量结果。

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
4. **用 CuTe-DSL 写，不用原生的 CUDA C++ 模板。** CuTe-DSL 是 CUTLASS 嵌在 Python 里的 kernel 语言。论文报告它的编译速度比大量使用模板的 C++ 快 20–30×，表达能力相当；attention 的变体（ALiBi、sliding window、soft-capping）可以写成普通的 Python 分数修改函数，JIT 编译进 kernel。

论文报告 B200 上 BF16 前向最高 1.613e15 FLOPS，约为这张卡 dense 峰值 2.25e15 FLOPS 的 71%。这个比例低于 FA3 在 H100 上的 85%，但峰值翻了一倍，exp 单元和 shared memory 却没变，要守住约 70% 需要用上前面所有的改动；按绝对吞吐算，它约是 FA3 的 1.9 倍。和基线相比，FA4 最多比 cuDNN 9.13 快 1.3×，比 Triton 快 2.7×。作者还和 cuDNN 团队合作，从 9.13/9.14 起把其中几项技术并进了 cuDNN，所以 FA4 和最新的 cuDNN（9.19）之间的差距要小得多。

四代各自改了什么到这里就讲完了，下一节看四代之间始终没有变的部分。

## 6 四代没变的东西 {#constant}

前面几节讲的都是每一代改了什么。kernel 技巧一条条列下来，很容易忽略哪些东西始终没变，这里把它们明确写出来：

- **四代算的是同一个数学函数。** 都是 $\mathrm{softmax}(QK^\top/\sqrt d)V$，差别只在浮点舍入；FP8 路径还有量化误差，这部分误差有界，论文也实际测量过，没有含糊带过。
- **没有一代降低了 FLOP 数的渐近量级。** 每个版本的计算量都是 $\Theta(N^2d)$；重计算会多出一点，有 causal mask 时跳过完全被 mask 掉的 tile，实际执行的计算比算完整矩阵再 mask 少一半左右。整条演进线一直在做同一件事：去掉 FLOP 数本身并不需要的数据搬运和非 matmul 开销。
- **每一代针对的都是上一代留下的新瓶颈。** v1 去掉了 $N^2$ 大小的矩阵在 HBM 上的往返。v2 解决了 v1 的调度留下的低 occupancy 和 warp 之间的通信开销。v3 用上了 v1、v2 出现时还没有的异步硬件（TMA/WGMMA），另外用 FP8 处理精度和吞吐之间的取舍。v4 应对的是 Blackwell 的非对称扩展：exp 的 FLOP 数很少，在前向里却和 matmul 占一样多的周期，反向里 shared memory 的流量还超过了 matmul。所以「attention 慢」不是一次能解决的问题：每一代解决的只是当时最紧的约束，下一代 GPU 又会带来新的约束。

最后一节用一张图和一张表把四代汇总起来。

## 7 总结 {#summary}

[图 7-1](#fig-7-1) 是各代报告的前向峰值利用率，[表 7-1](#tab-7-1) 列出每一代的年份、目标硬件、核心做法和主要结果。

<figure id="fig-7-1">
<svg viewBox="0 0 640 260" width="100%" role="img" aria-label="各代 FlashAttention 报告的前向峰值利用率，按各自 GPU 的 dense Tensor core 峰值计算：FlashAttention 在 A100 上 30% 到 50%；FlashAttention-2 在 A100 上 73%；FlashAttention-3 在 H100 上 85%；FlashAttention-4 在 B200 上 71%。">
  <style>
    .bc-val  { font-size: 13px; font-weight: 600; fill: currentColor; text-anchor: middle; }
    .bc-cat  { font-size: 13px; fill: currentColor; text-anchor: middle; }
    .bc-sub  { font-size: 11px; fill: currentColor; opacity: .7; text-anchor: middle; }
    .bc-tick { font-size: 11px; fill: currentColor; opacity: .7; text-anchor: end; }
  </style>
  <line x1="60" y1="40" x2="600" y2="40" stroke="currentColor" stroke-opacity=".12" stroke-dasharray="3 3"/>
  <line x1="60" y1="125" x2="600" y2="125" stroke="currentColor" stroke-opacity=".12" stroke-dasharray="3 3"/>
  <line x1="60" y1="210" x2="600" y2="210" stroke="currentColor" stroke-opacity=".4"/>
  <text x="54" y="44" class="bc-tick">100%</text>
  <text x="54" y="129" class="bc-tick">50%</text>
  <text x="54" y="214" class="bc-tick">0%</text>
  <rect x="95" y="159" width="70" height="51" fill="rgb(var(--color-primary-500))"><title>FlashAttention（v1），A100 前向：峰值的 30–50%（FlashAttention-2 论文的测量）</title></rect>
  <rect x="95" y="125" width="70" height="34" fill="rgba(var(--color-primary-500), .18)" stroke="rgb(var(--color-primary-500))" stroke-dasharray="3 2"/>
  <text x="130" y="115" class="bc-val">30–50%</text>
  <text x="130" y="230" class="bc-cat">FA1</text>
  <text x="130" y="246" class="bc-sub">A100</text>
  <rect x="230" y="85.9" width="70" height="124.1" fill="rgb(var(--color-primary-500))"><title>FlashAttention-2，A100 前向：最高达峰值的 73%（3.12e14 FLOPS 中的 2.3e14）</title></rect>
  <text x="265" y="76" class="bc-val">73%</text>
  <text x="265" y="230" class="bc-cat">FA2</text>
  <text x="265" y="246" class="bc-sub">A100</text>
  <rect x="365" y="65.5" width="70" height="144.5" fill="rgb(var(--color-primary-500))"><title>FlashAttention-3，H100 前向，BF16：最高达峰值的 85%（9.89e14 FLOPS 中的 8.4e14）</title></rect>
  <text x="400" y="56" class="bc-val">85%</text>
  <text x="400" y="230" class="bc-cat">FA3</text>
  <text x="400" y="246" class="bc-sub">H100</text>
  <rect x="500" y="89.3" width="70" height="120.7" fill="rgb(var(--color-primary-500))"><title>FlashAttention-4，B200 前向，BF16：最高达峰值的 71%（2.25e15 FLOPS 中的 1.613e15）</title></rect>
  <text x="535" y="79" class="bc-val">71%</text>
  <text x="535" y="230" class="bc-cat">FA4</text>
  <text x="535" y="246" class="bc-sub">B200</text>
</svg>
<figcaption><strong>图 7-1</strong> 各代主力 kernel 报告的前向峰值利用率，按各自 GPU 的 dense FP16/BF16 Tensor core 峰值算（A100 3.12e14、H100 9.89e14、B200 2.25e15 FLOPS）；FA1 的柱子是 FlashAttention-2 论文给它测的 30–50% 区间。不同 GPU 之间不能直接比：Blackwell 的 Tensor core 翻倍，exp 单元和 shared memory 没有变快，同样的峰值比例更难达到，所以 FA4 在 B200 上的 71% 低于 FA3 在 H100 上的 85%。</figcaption>
</figure>

| 版本 | 年份 | 目标硬件 | 核心做法 | 主要结果 |
|---|---|---|---|---|
| FlashAttention | 2022 | Ampere 时代的 GPU | tiling + online softmax + 重计算，N×N 矩阵从不在 HBM 里实体化 | 额外显存 O(N)，结果精确；attention op 快 2–4×；端到端快 15%（BERT）到 3×（GPT-2） |
| FlashAttention-2 | 2023 | Ampere/Ada/Hopper | 推迟归一化；交换内外层循环，并在序列长度上并行；warp 之间切分 Q（split-Q） | 比 v1 快约 2×，前向最高达 A100 峰值的 73% |
| FlashAttention-3 | 2024 | Hopper（H100） | warp specialization 的异步 pipeline（TMA + WGMMA）；GEMM 与 softmax 重叠；FP8 加 block 量化和 incoherent processing | 比 v2 快 1.5–2.0×；最高达 H100 峰值的 85%（8.4e14 FLOPS）；FP8 达 1.3e15 FLOPS |
| FlashAttention-4 | 2026 | Blackwell（B200/GB200），代码也能在 Hopper 上跑 | 异步 MMA + 更大的 tile；软件模拟 exp + 有条件的 rescale；TMEM + 2-CTA MMA；CuTe-DSL | 达 B200 峰值的 71%（1.613e15 FLOPS）；最多比 cuDNN 9.13 快 1.3×，比 Triton 快 2.7× |
{#tab-7-1 caption="**表 7-1** FlashAttention 四代对比" note="数字都是各代论文报告的结果，FA4 的是作者自己的测量。"}

## References {#references}

1. Dao, Fu, Ermon, Rudra, Ré. ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135), NeurIPS 2022. 前文的时间拆分图、tiling 示意图和 FLOPs 与访存量的对比图都以这篇论文的 Figure 1、Figure 2 为基础。
2. Dao. ["FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"](https://arxiv.org/abs/2307.08691), ICLR 2024. 另见 [Stanford Hazy Research 的介绍文章](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)。
3. Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. ["FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"](https://arxiv.org/abs/2407.08608), NeurIPS 2024. 另见作者的[博客文章](https://tridao.me/blog/2024/flash3/)。
4. Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao. ["FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"](https://arxiv.org/abs/2603.05451), arXiv:2603.05451, 2026 年 3 月。
5. [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)。四个版本的参考实现。
6. Milakov, Gimelshein. ["Online normalizer calculation for softmax"](https://arxiv.org/abs/1805.02867), arXiv:1805.02867, 2018. online softmax 的出处，2.2 节用到。
7. [Transformer FLOPs 计算器](/projects/flops/)。按同样的思路估算整个 Transformer 的参数量、FLOPs、显存和训练时间。
