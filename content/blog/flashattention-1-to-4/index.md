---
title: "FlashAttention 1–4: How IO-Awareness Reshaped the Attention Kernel"
date: 2026-10-05
draft: false
math: true
description: "A technical walkthrough of FlashAttention's four generations — from IO-aware tiling on A100, through better work partitioning and Hopper's asynchrony, to Blackwell's asymmetric-scaling co-design."
tags: ["FlashAttention", "Transformer", "Attention", "GPU", "LLM", "AI"]
categories: ["AI笔记"]
series: ["AI笔记"]
series_order: 3
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

> **Prerequisite:** [GPU 训练分析：FLOPs、Roofline 与显存峰值](/blog/gpu-training-analysis/) (in Chinese) covers FLOPs, arithmetic intensity, the roofline model and peak memory, and measures on an RTX 5090 why the $N\times N$ score matrix dominates both the runtime and the activation memory of standard attention — the two problems this post starts from.

Self-attention is the one layer every Transformer pays for twice: once in FLOPs, and once — far more painfully — in memory traffic. FlashAttention is a family of exact-attention kernels built to fix the second problem, and each new version targets a *different bottleneck* that only becomes visible once the previous one is gone. This post walks through all four generations: what each one actually changed, why that change mattered on the hardware of its time, and what stays constant across all of them.

Here is the whole story at a glance; the rest of the post unpacks each column.

<div class="fa-road">
<div class="fa-step"><b>v1 · 2022</b><i>A100 (Ampere)</i><span>Bottleneck: the N×N score matrix makes round trips through slow HBM.</span><span>Fix: tile it, fuse every step into one kernel, keep it on-chip.</span></div>
<div class="fa-step"><b>v2 · 2023</b><i>A100 (Ampere)</i><span>Bottleneck: most of the GPU sits idle — too few thread blocks, too much non-matmul work.</span><span>Fix: parallelize over the sequence; normalize once at the end.</span></div>
<div class="fa-step"><b>v3 · 2024</b><i>H100 (Hopper)</i><span>Bottleneck: loading tiles and doing math still take turns.</span><span>Fix: async producer/consumer pipeline; overlap softmax with matmul; FP8.</span></div>
<div class="fa-step"><b>v4 · 2026</b><i>B200 (Blackwell)</i><span>Bottleneck: tensor cores doubled, but the exp unit and shared memory did not.</span><span>Fix: compute some exps in software, skip most rescales, use tensor memory.</span></div>
</div>

I'll keep the math in inline notation throughout. The primary sources are linked at the end — I'd encourage reading at least the first paper directly rather than taking any secondhand summary (including this one) at face value.

## Why attention is memory-bound, not compute-bound

<div class="fa-intuit"><b>Intuition.</b> Think of HBM as a big warehouse and on-chip SRAM as a small workbench beside the arithmetic units: the bench is about 10× faster but holds about 2000× less. Standard attention keeps carrying the entire N×N score matrix from the bench to the warehouse and back — once per step. FlashAttention never lets that matrix leave the bench.</div>

Standard self-attention computes $S = QK^\top/\sqrt{d}$, $P = \mathrm{softmax}(S)$, $O = PV$, for $Q, K, V \in \mathbb{R}^{N\times d}$. The FLOP count is $\Theta(N^2 d)$, and it stays $\Theta(N^2 d)$ in every version discussed here (the backward pass even adds a recomputed $QK^\top$, about $2N^2d$ FLOPs on top of the standard backward's $\sim 8N^2d$). **FlashAttention does not reduce the asymptotic amount of arithmetic attention requires.** What it reduces is HBM traffic, and on a modern GPU that is almost always the thing you're actually paying for.

The reason is the memory hierarchy. An A100 has 40–80GB of HBM (the "GPU memory" everyone quotes) with roughly 1.5–2.0TB/s of bandwidth, and 192KB of on-chip SRAM per streaming multiprocessor (SM) — about 20MB across its 108 SMs — with an estimated ~19TB/s of aggregate bandwidth: an order of magnitude faster, three orders of magnitude smaller. A naive implementation materializes $S$ and $P$ — both $N\times N$ — in HBM, writes them, reads them back for the softmax, writes again, reads again for the final matmul. For long sequences this round-tripping dominates wall-clock time. The elementwise passes over the $N\times N$ matrix (softmax, plus masking and dropout in training) do only $O(1)$ FLOPs per element moved, so they are purely bandwidth-bound; even the two matmuls are memory-limited at typical head dimensions, since producing an fp16 $N\times N$ matrix from $2N^2d$ FLOPs gives an arithmetic intensity of only about $d$ FLOPs/byte (64–128), below the A100's roofline ridge point of roughly 150–200 FLOPs/byte. The whole pipeline sits left of the ridge: its speed is set by memory bandwidth, not by peak FLOPs.

<figure>
<svg viewBox="0 0 640 230" width="100%" role="img" aria-label="A100 memory hierarchy: about 20 MB of fast on-chip SRAM above 40 to 80 GB of slower HBM">
  <style>
    .mh-label { font-size: 13px; fill: currentColor; }
    .mh-sub { font-size: 11px; fill: currentColor; opacity: .7; }
    .mh-box { stroke-width: 1.5; }
  </style>
  <rect x="165" y="20" width="310" height="56" rx="6" class="mh-box" fill="rgba(var(--color-primary-500), .16)" stroke="rgb(var(--color-primary-500))"/>
  <text x="320" y="44" text-anchor="middle" class="mh-label">On-chip SRAM</text>
  <text x="320" y="62" text-anchor="middle" class="mh-sub">≈20 MB total (192 KB × 108 SMs) · ~19 TB/s aggregate</text>
  <line x1="320" y1="80" x2="320" y2="106" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-start="url(#mh-arrow)" marker-end="url(#mh-arrow)"/>
  <defs>
    <marker id="mh-arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto-start-reverse">
      <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" opacity=".35"/>
    </marker>
  </defs>
  <rect x="90" y="110" width="460" height="100" rx="6" class="mh-box" fill="rgba(128,128,128,.10)" stroke="currentColor" stroke-opacity=".4"/>
  <text x="320" y="156" text-anchor="middle" class="mh-label">HBM (device memory)</text>
  <text x="320" y="176" text-anchor="middle" class="mh-sub">40–80 GB · ~1.5–2.0 TB/s</text>
  <text x="320" y="224" text-anchor="middle" class="mh-sub">not drawn to scale</text>
</svg>
<figcaption>Two tiers of memory on an A100. HBM holds about 2000× more than all on-chip SRAM combined (40GB vs ~20MB), but SRAM has roughly 10× the bandwidth. Every FlashAttention version is, at its core, a different strategy for keeping the $N\times N$ intermediate out of the bottom tier.</figcaption>
</figure>

The FlashAttention paper makes this concrete with a measurement of GPT-2's attention layer. The surprise is *which* operations take the time:

<figure>
<svg viewBox="0 0 640 196" width="100%" role="img" aria-label="Attention runtime on GPT-2, read off the FlashAttention paper's Figure 1: PyTorch spends about 4 milliseconds in the two matrix multiplies and about 13 milliseconds in mask, softmax and dropout, about 17 milliseconds in total; FlashAttention's single fused kernel takes about 2 milliseconds">
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
  <text x="348" y="16" text-anchor="middle" class="tb-sub">≈ 13 ms: three elementwise passes, each reading and writing the whole N×N matrix</text>
  <text x="110" y="50" text-anchor="end" class="tb-lab">PyTorch</text>
  <text x="110" y="64" text-anchor="end" class="tb-sub">5 separate ops</text>
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
  <text x="110" y="128" text-anchor="end" class="tb-sub">1 fused kernel</text>
  <rect x="120" y="100" width="58.7" height="34" fill="rgb(var(--color-primary-500))"/>
  <text x="149.3" y="121" class="tb-inw">fused</text>
  <text x="186" y="122" class="tb-val">≈ 2.2 ms (7.6× faster)</text>
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
  <text x="136" y="188" class="tb-leg">matrix multiplies (almost all of the FLOPs)</text>
  <rect x="390" y="179" width="11" height="11" rx="2" fill="rgb(var(--color-secondary-500))"/>
  <text x="406" y="188" class="tb-leg">mask · softmax · dropout (memory-bound)</text>
</svg>
<figcaption>Where the time goes in standard attention (GPT-2 on an A100; approximate values read off Figure 1 of the FlashAttention paper). The two matrix multiplies hold nearly all of the FLOPs but take only about a quarter of the runtime. Mask, softmax and dropout do almost no arithmetic, yet each one streams the full $N\times N$ matrix out of HBM and back. Fusing everything into one kernel that never writes that matrix is the paper's 7.6× speedup.</figcaption>
</figure>

## FlashAttention (v1): make the N² matrix never exist in HBM

<div class="fa-intuit"><b>Intuition.</b> Don't build the N×N score matrix at all. Work on one small tile at a time on the workbench, keep two running numbers per row — the largest score seen so far and the sum of exponentials — and fix up your partial answer whenever a larger score shows up. At the end you get exactly the same output.</div>

Dao, Fu, Ermon, Rudra, and Ré's original 2022 paper frames this as an IO-awareness problem and solves it with two classic systems ideas applied to softmax: **tiling** and **recomputation**. Here is the difference in one picture:

<figure>
<svg viewBox="0 0 640 300" width="100%" role="img" aria-label="Standard attention runs three kernels and sends the N by N matrices S and P out to HBM and back; FlashAttention runs one fused kernel that reads Q, K and V and writes O, so only N by d data touches HBM">
  <style>
    .dm-h { font-size: 13px; font-weight: 600; fill: currentColor; }
    .dm-hs { font-size: 11px; fill: currentColor; opacity: .7; }
    .dm-k { font-size: 12.5px; font-weight: 600; fill: currentColor; text-anchor: middle; }
    .dm-ks { font-size: 11px; fill: currentColor; opacity: .75; text-anchor: middle; }
    .dm-m { font-size: 11px; fill: currentColor; text-anchor: middle; }
    .dm-a { font-size: 11px; font-weight: 600; fill: currentColor; }
  </style>
  <text x="166" y="18" text-anchor="middle" class="dm-h">Standard attention</text>
  <text x="166" y="35" text-anchor="middle" class="dm-hs">three kernels, one per step</text>
  <text x="480" y="18" text-anchor="middle" class="dm-h">FlashAttention</text>
  <text x="480" y="35" text-anchor="middle" class="dm-hs">one fused kernel, looping over tiles on-chip</text>
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
  <text x="166" y="276" text-anchor="middle" class="dm-hs">S and P (N×N each) go out to HBM and come back</text>
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
  <text x="480" y="276" text-anchor="middle" class="dm-hs">only N×d tensors ever touch HBM</text>
  <line x1="150" y1="290" x2="176" y2="290" stroke="rgb(var(--color-secondary-500))" stroke-width="7"/>
  <text x="184" y="294" class="dm-hs">an N×N matrix</text>
  <line x1="350" y1="290" x2="376" y2="290" stroke="rgb(var(--color-primary-500))" stroke-width="2"/>
  <text x="384" y="294" class="dm-hs">an N×d matrix</text>
</svg>
<figcaption>Left: the standard implementation (Algorithm 0 in the FlashAttention paper) runs one kernel per step, so the $N\times N$ matrices $S$ and $P$ each make a round trip through HBM — and in training, mask and dropout add more round trips. Right: FlashAttention fuses all three steps into a single kernel that loops over tiles on-chip; the only HBM traffic is reading $Q$, $K$, $V$ and writing $O$.</figcaption>
</figure>

**Tiling.** Split $Q$, $K$, $V$ into blocks along $N$ small enough that a $Q$-block together with a $K$/$V$-block fits in SRAM. Compute each local score block $S_{ij}=Q_iK_j^\top/\sqrt d$ on-chip and never write the full $S$ or $P$ back to HBM:

<figure>
<svg viewBox="0 0 640 352" width="100%" role="img" aria-label="Tiling: query block i is multiplied with key block j to form one tile of the N by N score matrix, which is the only tile that exists at that moment; the tile is multiplied by value block j and added into output block i, and the inner loop moves j across the row">
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
  <text x="318" y="154" class="tl-txt">× V block j → O block i</text>
  <line x1="156" y1="214" x2="457" y2="214" stroke="currentColor" stroke-opacity=".45"/>
  <polygon points="464,214 456,210 456,218" fill="currentColor" fill-opacity=".45"/>
  <text x="320" y="232" class="tl-txt">inner loop: every K/V block j</text>
  <line x1="86" y1="84" x2="86" y2="305" stroke="currentColor" stroke-opacity=".45"/>
  <polygon points="86,312 82,304 90,304" fill="currentColor" fill-opacity=".45"/>
  <text transform="translate(78 200) rotate(-90)" text-anchor="middle" class="tl-txt">outer loop over query blocks i</text>
  <text x="290" y="264" text-anchor="middle" class="tl-txt">only this tile exists</text>
  <text x="290" y="278" text-anchor="middle" class="tl-sub">(on-chip, for one step)</text>
  <text x="310" y="342" text-anchor="middle" class="tl-sub">S = QK<tspan dy="-4" font-size="8">T</tspan><tspan dy="4"> is N × N — it is never stored whole</tspan></text>
</svg>
<figcaption>The same picture as Figure 1 of the FlashAttention paper, drawn in the loop order FlashAttention-2 later adopted. Query block $Q_i$ meets key block $K_j$ to form one tile $S_{ij}$ of the score matrix; that tile is multiplied by $V_j$, added into $O_i$, and thrown away before the next $j$. Only the finished $O_i$ (and its row statistics) is written to HBM. In FA2 each query block $i$ is handled by its own thread block.</figcaption>
</figure>

The one wrinkle is that softmax is a *row-wise global* normalization — $P_{ij} = \exp(S_{ij})/\sum_k \exp(S_{ik})$ needs the sum over the entire row, which you don't have yet when you're only looking at block $j$. The fix is the "online softmax" recurrence: keep a running row-max $m$ and running row-sum $\ell$, and correct the running output every time the max changes. A tiny example with six scores in three blocks shows the whole trick:

<figure>
<svg viewBox="0 0 640 268" width="100%" role="img" aria-label="Online softmax on six scores in three blocks. Block 1, scores 2 and 1: max 2, sum 1.37. Block 2, scores 4 and 3: the max rises to 4, so the earlier weights and the sum shrink by e to the minus 2, about 0.14, and the sum becomes 1.55. Block 3, scores 1 and 0: max unchanged, sum 1.62, identical to computing softmax over all six scores at once">
  <style>
    .os-h { font-size: 12px; font-weight: 600; fill: currentColor; }
    .os-s { font-size: 11px; fill: currentColor; opacity: .75; }
    .os-t { font-size: 11px; fill: currentColor; }
    .os-k { font-size: 10px; fill: currentColor; opacity: .6; text-anchor: middle; }
    .os-arr { font-size: 16px; fill: currentColor; opacity: .45; text-anchor: middle; }
  </style>
  <text x="24" y="20" class="os-h">Block 1 · scores 2, 1</text>
  <text x="24" y="36" class="os-s">first max: m = 2</text>
  <line x1="24" y1="150" x2="194" y2="150" stroke="currentColor" stroke-opacity=".35"/>
  <rect x="34" y="60" width="18" height="90" fill="rgb(var(--color-secondary-500))"/>
  <rect x="60" y="116.9" width="18" height="33.1" fill="rgb(var(--color-secondary-500))"/>
  <text x="43" y="164" class="os-k">k1</text>
  <text x="69" y="164" class="os-k">k2</text>
  <text x="24" y="188" class="os-t">ℓ = 1 + 0.37 = 1.37</text>
  <text x="24" y="204" class="os-s">bar height = e^(score − m)</text>
  <text x="212" y="105" class="os-arr">→</text>
  <text x="232" y="20" class="os-h">Block 2 · scores 4, 3</text>
  <text x="232" y="36" class="os-s">new max: m = 2 → 4</text>
  <line x1="232" y1="150" x2="402" y2="150" stroke="currentColor" stroke-opacity=".35"/>
  <rect x="242" y="60" width="18" height="90" fill="none" stroke="currentColor" stroke-opacity=".45" stroke-dasharray="3 2"/>
  <rect x="268" y="116.9" width="18" height="33.1" fill="none" stroke="currentColor" stroke-opacity=".45" stroke-dasharray="3 2"/>
  <rect x="242" y="137.8" width="18" height="12.2" fill="rgb(var(--color-primary-500))"/>
  <rect x="268" y="145.5" width="18" height="4.5" fill="rgb(var(--color-primary-500))"/>
  <rect x="294" y="60" width="18" height="90" fill="rgb(var(--color-secondary-500))"/>
  <rect x="320" y="116.9" width="18" height="33.1" fill="rgb(var(--color-secondary-500))"/>
  <text x="264" y="52" text-anchor="middle" class="os-s">shrunk × 0.14</text>
  <text x="251" y="164" class="os-k">k1</text>
  <text x="277" y="164" class="os-k">k2</text>
  <text x="303" y="164" class="os-k">k3</text>
  <text x="329" y="164" class="os-k">k4</text>
  <text x="232" y="188" class="os-t">ℓ = 0.14 × 1.37 + 1.37 = 1.55</text>
  <text x="232" y="204" class="os-s">old bars and ℓ shrink by e^(2 − 4)</text>
  <text x="420" y="105" class="os-arr">→</text>
  <text x="440" y="20" class="os-h">Block 3 · scores 1, 0</text>
  <text x="440" y="36" class="os-s">max unchanged: m = 4</text>
  <line x1="440" y1="150" x2="610" y2="150" stroke="currentColor" stroke-opacity=".35"/>
  <rect x="450" y="137.8" width="18" height="12.2" fill="rgb(var(--color-primary-500))"/>
  <rect x="476" y="145.5" width="18" height="4.5" fill="rgb(var(--color-primary-500))"/>
  <rect x="502" y="60" width="18" height="90" fill="rgb(var(--color-primary-500))"/>
  <rect x="528" y="116.9" width="18" height="33.1" fill="rgb(var(--color-primary-500))"/>
  <rect x="554" y="145.5" width="18" height="4.5" fill="rgb(var(--color-secondary-500))"/>
  <rect x="580" y="148.4" width="18" height="1.6" fill="rgb(var(--color-secondary-500))"/>
  <text x="459" y="164" class="os-k">k1</text>
  <text x="485" y="164" class="os-k">k2</text>
  <text x="511" y="164" class="os-k">k3</text>
  <text x="537" y="164" class="os-k">k4</text>
  <text x="563" y="164" class="os-k">k5</text>
  <text x="589" y="164" class="os-k">k6</text>
  <text x="440" y="188" class="os-t">ℓ = 1.55 + 0.07 = 1.62</text>
  <text x="440" y="204" class="os-s">nothing to rescale</text>
  <text x="320" y="234" text-anchor="middle" class="os-t">Check: softmax over all six scores at once has Σ e^(score − 4) = 1.62 — the same ℓ.</text>
  <rect x="96" y="249" width="11" height="11" rx="2" fill="rgb(var(--color-secondary-500))"/>
  <text x="112" y="258" class="os-s">this block's scores</text>
  <rect x="236" y="249" width="11" height="11" rx="2" fill="rgb(var(--color-primary-500))"/>
  <text x="252" y="258" class="os-s">earlier blocks, rescaled</text>
  <rect x="396" y="249" width="11" height="11" rx="2" fill="none" stroke="currentColor" stroke-opacity=".45" stroke-dasharray="3 2"/>
  <text x="412" y="258" class="os-s">before rescaling</text>
</svg>
<figcaption>Online softmax on one row of six scores, two per block. Each bar is a score's unnormalized weight $e^{s-m}$ relative to the current running max $m$. When block 2 raises the max from 2 to 4, everything accumulated so far — the earlier weights, the running sum $\ell$, and (not drawn) the partial output $O$ — is multiplied by $e^{2-4}\approx 0.14$, which is exactly what the earlier terms would have been had we known the max was 4 all along. After the last block, $O/\ell$ is the exact softmax-weighted sum.</figcaption>
</figure>

In code, one query block at a time with a single normalization at the end:

```
for each query block i:
    O_i ← 0,  ℓ_i ← 0,  m_i ← -∞           # kept in SRAM/registers only
    for each key/value block j:
        S_ij  ← Q_i K_jᵗ / sqrt(d)         # B×B, SRAM only, never hits HBM
        m_new ← max(m_i, rowmax(S_ij))
        P_ij  ← exp(S_ij − m_new)
        ℓ_i   ← exp(m_i − m_new)·ℓ_i + rowsum(P_ij)
        O_i   ← exp(m_i − m_new)·O_i + P_ij·V_j
        m_i   ← m_new
    O_i ← O_i / ℓ_i                        # write O_i (and the row statistics) to HBM once
```

Strictly, this is the loop order FlashAttention-2 adopted. The FA1 paper's Algorithm 1 nests the loops the other way: the outer loop walks $K$/$V$ blocks and the inner loop walks $Q$ blocks, so every inner step reads $O_i$, $\ell_i$, $m_i$ from HBM, updates them — keeping $O_i$ normalized by $\mathrm{diag}(\ell_i)^{-1}$ at every step — and writes them back. The IO bound below holds for both orders; the reordering is one of the things FA2 changed.

Why is this exact rather than approximate? After processing blocks $1..j$, the accumulators satisfy $\ell_i=\sum_{k\le j}\mathrm{rowsum}(e^{S_{ik}-m_i})$ and $O_i=\sum_{k\le j}e^{S_{ik}-m_i}V_k$, with $m_i$ the running max. When the max grows, multiplying both by $e^{m_i-m_{\text{new}}}$ re-bases every earlier term, since $e^{S-m_i}\cdot e^{m_i-m_{\text{new}}}=e^{S-m_{\text{new}}}$. After the last block, $O_i/\ell_i=\mathrm{softmax}(S_{i,:})V$, because softmax is invariant to a per-row shift; the only difference from a one-shot implementation is floating-point summation order. Nothing of size $N\times N$ is ever written to HBM: the extra memory is $O(N)$ for the row statistics, instead of $O(N^2)$.

**Recomputation for the backward pass.** The backward pass of attention normally needs $P$ again. Rather than store the full $N\times N$ matrix for that (which is exactly the memory blow-up we just avoided), FlashAttention stores only $O$ and the small per-row statistics $(m, \ell)$, and *recomputes* the needed $S_{ij}$, $P_{ij}$ blocks on the fly inside SRAM during the backward pass. This trades a modest amount of extra matmul FLOPs for a large reduction in memory and memory traffic — a textbook example of recomputation beating storage when the storage is in the slow tier.

The paper's complexity result: standard attention needs $\Theta(Nd+N^2)$ HBM accesses; FlashAttention needs $\Theta(N^2d^2M^{-1})$, where $M$ is the SRAM size. For typical $d$ (64–128) and $M$ (around 100KB), $d^2$ is many times smaller than $M$, so FlashAttention makes many times fewer HBM accesses; at $M=\Theta(d^2)$ the two bounds would coincide. The paper's accompanying lower bound is deliberately weak: no exact algorithm can achieve $o(N^2d^2M^{-1})$ accesses for *every* $M$ in $[d, Nd]$. The paper's own measurement shows the trade directly — more arithmetic, far less data movement, much less time:

<figure>
<svg viewBox="0 0 640 196" width="100%" role="img" aria-label="GPT-2 medium attention, forward plus backward on an A100: standard attention 66.6 GFLOPs, 40.3 GB of HBM traffic, 41.7 ms; FlashAttention 75.2 GFLOPs, 13 percent more, but 4.4 GB of traffic, 9.2 times less, and 7.3 ms, 5.7 times faster">
  <style>
    .ft-lab { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .ft-sub { font-size: 11px; fill: currentColor; opacity: .7; }
    .ft-val { font-size: 11.5px; fill: currentColor; }
    .ft-hl  { font-size: 11.5px; font-weight: 600; fill: currentColor; }
  </style>
  <rect x="170" y="5" width="11" height="11" rx="2" fill="rgba(128,128,128,.45)"/>
  <text x="186" y="14" class="ft-sub">standard attention</text>
  <rect x="320" y="5" width="11" height="11" rx="2" fill="rgb(var(--color-primary-500))"/>
  <text x="336" y="14" class="ft-sub">FlashAttention</text>
  <text x="160" y="47" text-anchor="end" class="ft-lab">GFLOPs</text>
  <text x="160" y="61" text-anchor="end" class="ft-sub">arithmetic</text>
  <rect x="170" y="30" width="318.8" height="14" fill="rgba(128,128,128,.45)"/>
  <text x="494.8" y="41" class="ft-val">66.6</text>
  <rect x="170" y="48" width="360" height="14" fill="rgb(var(--color-primary-500))"/>
  <text x="536" y="59" class="ft-hl">75.2 (+13%)</text>
  <text x="160" y="107" text-anchor="end" class="ft-lab">HBM traffic</text>
  <text x="160" y="121" text-anchor="end" class="ft-sub">reads + writes</text>
  <rect x="170" y="90" width="360" height="14" fill="rgba(128,128,128,.45)"/>
  <text x="536" y="101" class="ft-val">40.3 GB</text>
  <rect x="170" y="108" width="39.3" height="14" fill="rgb(var(--color-primary-500))"/>
  <text x="215.3" y="119" class="ft-hl">4.4 GB · 9.2× less</text>
  <text x="160" y="167" text-anchor="end" class="ft-lab">Runtime</text>
  <text x="160" y="181" text-anchor="end" class="ft-sub">forward + backward</text>
  <rect x="170" y="150" width="360" height="14" fill="rgba(128,128,128,.45)"/>
  <text x="536" y="161" class="ft-val">41.7 ms</text>
  <rect x="170" y="168" width="63" height="14" fill="rgb(var(--color-primary-500))"/>
  <text x="239" y="179" class="ft-hl">7.3 ms · 5.7× faster</text>
</svg>
<figcaption>Figure 2 (left) of the FlashAttention paper, redrawn: GPT-2 medium attention (sequence length 1024, head dimension 64, 16 heads, batch 64) on an A100, forward plus backward. Each row is scaled separately. FlashAttention does <em>more</em> arithmetic — the backward pass recomputes the score tiles — yet runs 5.7× faster, because its HBM traffic is 9× smaller. Runtime follows memory traffic, not FLOPs.</figcaption>
</figure>

In practice the attention op itself (forward + backward) became 2–4× faster than PyTorch's standard implementation at sequence lengths 128–4K, which translated into end-to-end training speedups of 15% on BERT-large (seq. 512, vs. the MLPerf 1.1 record), 3× on GPT-2 (seq. 1K, vs. HuggingFace) and 2.4× on Long-Range Arena. Because memory now grows linearly instead of quadratically in $N$, the saving grows with sequence length — about 20× less memory at 4K tokens — which let people train with substantially longer context, while still computing the *exact* softmax, something the sparse- and low-rank-attention literature at the time could not claim.

## FlashAttention-2: the kernel wasn't even using the GPU well

<div class="fa-intuit"><b>Intuition.</b> v1 moves the right amount of data but leaves most of the GPU idle. v2 keeps the math and fixes the scheduling: hand out many more independent pieces of work so every SM stays busy, and spend fewer instructions on bookkeeping that isn't a matrix multiply.</div>

v1 solved the IO problem but left throughput on the table: on an A100 its forward pass reached only 30–50% of the GPU's theoretical peak FLOPs/s (25–35% for the backward pass), well below the 80–90% that a well-tuned dense GEMM achieves. FlashAttention-2 (Dao, 2023) is a work-scheduling paper — same math, better engineering of how work is split across thread blocks and warps.

Three changes matter most:

1. **Fewer non-matmul FLOPs.** Non-matmul work (exponentials, row max/sum, rescaling multiplies) runs on the FP32 ALUs and special-function units rather than the tensor cores. On A100 that is 19.5 TFLOPs/s of FP32 versus 312 TFLOPs/s of FP16 matmul, so each non-matmul FLOP costs roughly 16× more. v1 keeps the output normalized at every step, dividing by the running sum on top of the unavoidable $\exp(m_{\text{old}}-m_{\text{new}})$ correction. FA2 keeps an *un-normalized* accumulator, applies only the $\exp(m_{\text{old}}-m_{\text{new}})$ correction per block, divides by $\ell$ once at the end (the pseudocode above), and saves a single logsumexp $L=m+\log\ell$ per row for the backward pass instead of both $m$ and $\ell$.
2. **Swapped loops and parallelism over sequence length.** v1 parallelized only over (batch × heads), which can be too few thread blocks to saturate the GPU's SMs with today's long sequences and small per-GPU batches. FA2 makes the query-block loop the outer one, so row blocks are independent: in the forward pass each thread block owns one query row block. In the backward pass each thread block owns one key/value column block, accumulating its $dK_j$, $dV_j$ locally and using atomic adds into a shared $dQ$ buffer.

<figure>
<svg viewBox="0 0 640 168" width="100%" role="img" aria-label="Example with one 8K-token sequence and 16 heads on a 108-SM A100: FlashAttention-1 launches 16 thread blocks, so 16 of 108 SMs are busy; FlashAttention-2 also splits the queries into 64 blocks per head, launching 1,024 thread blocks that keep all 108 SMs busy">
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
  <text x="26" y="18" class="sm-h">v1: one thread block per (batch, head)</text>
  <text x="26" y="34" class="sm-s">batch 1 × 16 heads = 16 thread blocks</text>
  <rect x="26" y="46" width="270" height="90" fill="url(#sm-idle)"/>
  <rect x="26" y="46" width="240" height="15" fill="url(#sm-busy)"/>
  <text x="26" y="156" class="sm-f">16 of 108 SMs busy (15%)</text>
  <text x="341" y="18" class="sm-h">v2: also one per 128-query block</text>
  <text x="341" y="34" class="sm-s">16 heads × 64 query blocks = 1,024 thread blocks</text>
  <rect x="341" y="46" width="270" height="90" fill="url(#sm-busy)"/>
  <text x="341" y="156" class="sm-f">108 of 108 SMs busy (≈ 9.5 waves of blocks)</text>
</svg>
<figcaption>Why parallelizing over the sequence matters, for one illustrative case: a single 8K-token sequence with 16 heads on an A100, whose 108 SMs are drawn as squares. With one thread block per (batch, head), only 16 SMs get work and the rest idle; splitting each head's queries into 128-row blocks gives 1,024 independent thread blocks, enough to fill the GPU many times over.</figcaption>
</figure>

3. **Better warp partitioning.** Within a thread block, v1 split $K$/$V$ across warps while every warp could see all of $Q$ (a "split-K" scheme); each warp then had to write its partial result to shared memory, synchronize, and reduce across warps. FA2 flips this: it splits $Q$ across warps while every warp sees all of $K$/$V$ ("split-Q"), so each warp produces a complete output slice with **no cross-warp reduction through shared memory**. Warps still meet at block-wide barriers when they cooperatively load each $K$/$V$ tile, but they never exchange partial outputs.

<figure>
<svg viewBox="0 0 640 256" width="100%" role="img" aria-label="Split-K warp partitioning needs a shared-memory reduction of partial outputs across warps; split-Q gives each warp a complete output slice with no cross-warp reduction">
  <style>
    .wp-title { font-size: 13px; font-weight: 600; fill: currentColor; }
    .wp-lab   { font-size: 12px; fill: #fff; text-anchor: middle; }
    .wp-lab-dk { font-size: 12px; fill: rgb(var(--color-neutral-900)); text-anchor: middle; }
    .wp-sub   { font-size: 11px; fill: currentColor; opacity: .7; text-anchor: middle; }
    .wp-box   { font-size: 11px; fill: currentColor; text-anchor: middle; }
  </style>
  <text x="170" y="18" text-anchor="middle" class="wp-title">v1 — split-K</text>
  <rect x="30" y="30" width="280" height="30" rx="4" fill="rgba(128,128,128,.12)" stroke="currentColor" stroke-opacity=".35"/>
  <text x="170" y="50" class="wp-box">Q — shared, visible to every warp</text>
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
  <text x="170" y="128" class="wp-sub">warps 0–3: partial sum per K/V slice</text>
  <path d="M60,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <path d="M130,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <path d="M210,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <path d="M280,134 Q170,170 170,186" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <rect x="90" y="190" width="160" height="32" rx="4" fill="rgba(var(--color-primary-500), .16)" stroke="rgb(var(--color-primary-500))"/>
  <text x="170" y="211" class="wp-box">reduce in shared mem</text>
  <text x="170" y="242" class="wp-sub">write partial · barrier sync · add</text>
  <line x1="330" y1="10" x2="330" y2="250" stroke="currentColor" stroke-opacity=".15"/>
  <text x="490" y="18" text-anchor="middle" class="wp-title">v2 — split-Q</text>
  <rect x="350" y="30" width="280" height="30" rx="4" fill="rgba(128,128,128,.12)" stroke="currentColor" stroke-opacity=".35"/>
  <text x="490" y="50" class="wp-box">K, V — shared, visible to every warp</text>
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
  <text x="490" y="128" class="wp-sub">warps 0–3: complete output per Q-slice</text>
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
  <text x="490" y="242" class="wp-sub">independent outputs — no cross-warp reduction</text>
</svg>
<figcaption>Same four warps, opposite split. v1 splits $K$/$V$ and has to reconcile four partial sums through shared memory; v2 splits $Q$ so each warp's output slice is already complete when its matmuls finish.</figcaption>
</figure>

Net effect: roughly 2× over v1, reaching up to 73% of theoretical peak FLOPs/s on A100 for the forward pass (about 230 TFLOPs/s) and up to 63% for the backward pass, which has an inherently less favorable data-reuse pattern. End to end, GPT-style training reached up to 225 TFLOPs/s per A100 (72% model FLOPs utilization). No new hardware feature was required — this is squarely a "use the GPU you already had better" paper, which is also why it's the version most widely deployed across non-Hopper GPUs (and the AMD ROCm port).

## FlashAttention-3: exploiting what Hopper added

<div class="fa-intuit"><b>Intuition.</b> On Hopper, copying tiles and multiplying matrices are done by separate hardware engines that can run at the same time. v3 turns the kernel into an assembly line: one worker only fetches, others only compute, and two compute groups take turns so the tensor cores never sit waiting for softmax.</div>

H100 introduced two primitives that neither v1 nor v2 was designed to use: the **Tensor Memory Accelerator (TMA)**, a hardware unit that copies whole tiles between HBM and shared memory asynchronously, and **warpgroup-level MMA (WGMMA)**, a wider, asynchronous tensor-core instruction issued by a group of four warps. FlashAttention-3 (Shah, Bikshandi, Zhang, Thakkar, Ramani, and Dao, 2024) is built around exploiting both, plus block quantization and incoherent processing for FP8.

1. **Warp specialization (producer/consumer).** In FA2, every warp both issues its share of the tile copies (already asynchronous, via Ampere's cp.async) and computes, with block-wide barriers between stages. FA3 instead assigns a producer warp to issue TMA loads into a multi-stage shared-memory ring tracked by hardware barriers, and gives most of the register file to consumer warpgroups that only run WGMMA on tiles already staged. The kernel becomes an explicit software pipeline.

<figure>
<svg viewBox="0 0 640 186" width="100%" role="img" aria-label="Warp specialization: a producer warp copies K and V tiles from HBM into a four-slot shared-memory ring buffer using TMA, two consumer warpgroups take ready tiles and run WGMMA and softmax, and each consumed slot is released back to the producer through a hardware barrier">
  <style>
    .pc-h { font-size: 12px; font-weight: 600; fill: currentColor; text-anchor: middle; }
    .pc-s { font-size: 11px; fill: currentColor; opacity: .75; text-anchor: middle; }
    .pc-t { font-size: 10.5px; fill: currentColor; opacity: .75; text-anchor: middle; }
  </style>
  <rect x="16" y="58" width="80" height="74" rx="6" fill="rgba(128,128,128,.10)" stroke="currentColor" stroke-opacity=".4"/>
  <text x="56" y="91" class="pc-h">HBM</text>
  <text x="56" y="108" class="pc-s">K, V tiles</text>
  <line x1="98" y1="95" x2="127" y2="95" stroke="currentColor" stroke-opacity=".5" stroke-width="1.5"/>
  <polygon points="134,95 126,91 126,99" fill="currentColor" fill-opacity=".5"/>
  <rect x="136" y="66" width="100" height="58" rx="6" fill="rgba(var(--color-secondary-500), .22)" stroke="rgb(var(--color-secondary-500))"/>
  <text x="186" y="90" class="pc-h">producer warp</text>
  <text x="186" y="107" class="pc-s">TMA loads only</text>
  <line x1="238" y1="95" x2="261" y2="95" stroke="currentColor" stroke-opacity=".5" stroke-width="1.5"/>
  <polygon points="268,95 260,91 260,99" fill="currentColor" fill-opacity=".5"/>
  <text x="352" y="54" class="pc-s">shared-memory ring buffer</text>
  <rect x="270" y="66" width="38" height="58" rx="4" fill="rgba(var(--color-secondary-500), .55)" stroke="rgb(var(--color-secondary-500))"/>
  <rect x="312" y="66" width="38" height="58" rx="4" fill="rgba(var(--color-secondary-500), .55)" stroke="rgb(var(--color-secondary-500))"/>
  <rect x="354" y="66" width="38" height="58" rx="4" fill="none" stroke="rgb(var(--color-secondary-500))"/>
  <rect x="354" y="95" width="38" height="29" fill="rgba(var(--color-secondary-500), .55)"/>
  <rect x="396" y="66" width="38" height="58" rx="4" fill="none" stroke="rgb(var(--color-secondary-500))" stroke-dasharray="3 2"/>
  <text x="289" y="140" class="pc-t">full</text>
  <text x="331" y="140" class="pc-t">full</text>
  <text x="373" y="140" class="pc-t">filling</text>
  <text x="415" y="140" class="pc-t">empty</text>
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
  <text x="368" y="161" class="pc-s">slot consumed → released back to the producer (hardware barrier)</text>
</svg>
<figcaption>Warp specialization as an assembly line. The producer warp does nothing but issue TMA copies into a ring of shared-memory slots; the consumer warpgroups do nothing but compute on slots that are already full. As long as the ring stays ahead, the tensor cores never wait on a load, and the producer gives most of its registers to the consumers.</figcaption>
</figure>

2. **Overlapping GEMM and softmax.** On a single tile, softmax depends on that block's $QK^\top$, and the $PV$ GEMM depends on softmax, so they serialize. FA3 breaks the chain in two complementary ways. *Inter-warpgroup ping-pong:* the two consumer warpgroups each own a different query tile, and barriers force them to alternate, so warpgroup A's GEMMs run while warpgroup B does its softmax, and vice versa; the paper reports this moving FP16 forward throughput from roughly 570 to 620 TFLOPs/s at head-dim 128, seqlen 8K on H100. *Intra-warpgroup pipelining:* within one warpgroup, the asynchronous WGMMAs for the next block's $QK^\top$ and the current block's $PV$ are issued so that softmax runs while they are in flight.

<figure>
<svg viewBox="0 0 640 210" width="100%" role="img" aria-label="Ping-pong scheduling: two warpgroups alternate GEMM and softmax so each one's softmax phase overlaps the other's matmul phase; on H100 the softmax takes about half as long as the GEMMs">
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
    <text x="340" y="45" class="pp-wait">wait</text>
    <rect x="370" y="24" width="120" height="34" fill="rgb(var(--color-primary-500))"/>
    <text x="430" y="45" class="pp-seg">GEMM</text>
    <rect x="490" y="24" width="60" height="34" fill="rgb(var(--color-secondary-500))"/>
    <text x="520" y="45" class="pp-seg-dk">softmax</text>
    <rect x="550" y="24" width="60" height="34" fill="rgba(128,128,128,.10)"/>
    <text x="580" y="45" class="pp-wait">wait</text>
  </g>
  <g>
    <rect x="130" y="84" width="60" height="34" fill="rgb(var(--color-secondary-500))"/>
    <text x="160" y="105" class="pp-seg-dk">softmax</text>
    <rect x="190" y="84" width="60" height="34" fill="rgba(128,128,128,.10)"/>
    <text x="220" y="105" class="pp-wait">wait</text>
    <rect x="250" y="84" width="120" height="34" fill="rgb(var(--color-primary-500))"/>
    <text x="310" y="105" class="pp-seg">GEMM</text>
    <rect x="370" y="84" width="60" height="34" fill="rgb(var(--color-secondary-500))"/>
    <text x="400" y="105" class="pp-seg-dk">softmax</text>
    <rect x="430" y="84" width="60" height="34" fill="rgba(128,128,128,.10)"/>
    <text x="460" y="105" class="pp-wait">wait</text>
    <rect x="490" y="84" width="120" height="34" fill="rgb(var(--color-primary-500))"/>
    <text x="550" y="105" class="pp-seg">GEMM</text>
  </g>
  <line x1="130" y1="18" x2="130" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="250" y1="18" x2="250" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="370" y1="18" x2="370" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="490" y1="18" x2="490" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="610" y1="18" x2="610" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <text x="610" y="145" text-anchor="end" class="pp-sub">time →</text>
  <path d="M250,122 L250,134 L370,134 L370,122" fill="none" stroke="currentColor" stroke-opacity=".4"/>
  <text x="310" y="160" text-anchor="middle" class="pp-sub">A's softmax runs while B's GEMMs run — different functional units</text>
  <g>
    <rect x="130" y="184" width="14" height="14" fill="rgb(var(--color-primary-500))"/>
    <text x="150" y="195" class="pp-leg">GEMM (tensor cores)</text>
    <rect x="300" y="184" width="14" height="14" fill="rgb(var(--color-secondary-500))"/>
    <text x="320" y="195" class="pp-leg">softmax (exp/ALU units)</text>
  </g>
</svg>
<figcaption>Two warpgroups, phases offset by one step. Whenever one warpgroup is doing softmax's scalar work, the other is busy on its tensor-core matmuls, so the "non-matmul tax" disappears into the matmul's shadow instead of serializing with it. Widths are schematic but proportioned for H100 at head dim 128, where the exponentials take about half as long as the GEMMs they hide behind; on B200 the two become equal (see the FA4 section).</figcaption>
</figure>

3. **FP8 with block quantization and incoherent processing.** Quantizing $Q$ and $K$ directly to FP8 before the matmul is attractive for throughput but inaccurate, because a few outlier entries dominate the quantization range and crush the resolution for everything else. FA3 uses two complementary fixes. First, *block quantization*: one FP8 scale per tile instead of one per tensor, which keeps an outlier's damage inside its own block and costs nothing extra in a tiled kernel. Second, *incoherent processing*: multiply $Q$ and $K$ by the same random orthogonal matrix $M$ before quantizing. Since $M$ is orthogonal, $(QM)(KM)^\top = QMM^\top K^\top = QK^\top$ — the attention scores are mathematically unchanged — but each entry of $QM$ is now a random mixture of many original entries, so no single outlier dominates any one coordinate. Implemented as a Hadamard transform with random sign flips, this costs $O(d\log d)$ per length-$d$ row instead of $O(d^2)$ for a dense rotation. On test inputs with simulated outlier features, the FP8 path has 2.6× lower RMSE than a baseline FP8 attention with per-tensor scaling.

<figure>
<svg viewBox="0 0 640 222" width="100%" role="img" aria-label="An 8-entry vector with one outlier of 8.0 and seven entries below 0.4 in magnitude; after random sign flips and a Hadamard rotation every entry is between 2.3 and 3.1 in magnitude, the largest entry drops from 8.0 to 3.1, and the vector's length stays 8.03">
  <style>
    .ic-h { font-size: 12px; font-weight: 600; fill: currentColor; text-anchor: middle; }
    .ic-s { font-size: 11px; fill: currentColor; opacity: .75; }
    .ic-c { font-size: 11px; fill: currentColor; opacity: .75; text-anchor: middle; }
  </style>
  <text x="170" y="16" class="ic-h">Before: one outlier</text>
  <text x="490" y="16" class="ic-h">After random signs + Hadamard rotation</text>
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
  <text x="170" y="212" class="ic-c">largest entry 8.0 sets the quantizer's scale</text>
  <text x="490" y="212" class="ic-c">largest entry 3.1 · same length (8.03)</text>
</svg>
<figcaption>Incoherent processing on a toy 8-entry vector (computed exactly). A quantizer's scale is set by the largest entry in its block, so one outlier of 8.0 forces the seven small entries to share a tiny slice of the representable range. Random sign flips followed by a Hadamard transform spread the outlier's energy across every coordinate: the largest entry drops from 8.0 to 3.1, while the vector's length — and every dot product in $QK^\top$, since $Q$ and $K$ get the same rotation — is unchanged.</figcaption>
</figure>

Results: 1.5–2.0× over FA2 on H100, up to 840 TFLOPs/s in BF16 (about 85% of H100 SXM5's 989 TFLOPs/s dense peak), and up to 1.3 PFLOPs/s in FP8. (The July 2024 arXiv v1 reported 740 TFLOPs/s and close to 1.2 PFLOPs/s; the numbers here are from the NeurIPS 2024 version cited below.)

## FlashAttention-4: when the bottleneck itself shifts hardware generation

<div class="fa-intuit"><b>Intuition.</b> Blackwell made matrix multiplies twice as fast but left the exponential unit and shared memory alone, so the softmax that used to hide behind the matmul now takes just as long as it. FA4 goes after that work directly: compute some exponentials on other units, skip rescales that don't change the answer, and keep more data in the new tensor memory.</div>

FlashAttention-4 (Zadouri, Hoehnerbach, Shah, Liu, Thakkar, and Dao; Princeton, Meta, Colfax Research, NVIDIA, Georgia Tech, and Together AI) appeared on arXiv in March 2026. The paper targets Blackwell datacenter GPUs (B200/GB200); the open-source implementation (`flash_attn/cute` in the flash-attention repo, installed as `flash-attn-4`) also ships Hopper kernels. It is the newest generation here and the least battle-tested in production, so treat its numbers as the authors' own measurements.

The paper's motivating observation is **asymmetric hardware scaling**. Going from Hopper to Blackwell, dense BF16 tensor-core throughput doubled (8192 vs 4096 FLOPs per clock per SM; about 2.25 vs 1 PFLOPs/s per GPU), while shared-memory read bandwidth (128 B per clock per SM) and the MUFU exponential unit (16 ops per clock per SM) did not change at all. For a $128\times128$ tile at $d=128$, the two forward MMAs take $4\cdot128^3/4096=2048$ cycles on Hopper and the $128^2$ exponentials take $128^2/16=1024$, so FA3's ping-pong could hide the exponentials behind a matmul twice as long. On B200 the MMAs take 1024 cycles and the exponentials still take 1024: overlap alone can no longer hide them. It is the same IO/compute-balance problem the original paper solved, one level further up the stack.

<figure>
<svg viewBox="0 0 640 250" width="100%" role="img" aria-label="Roofline cycle budget for one forward-pass tile, 128 by 128 with head dimension 128. H100: tensor-core MMA 2048 cycles, exponential unit 1024 cycles. B200: MMA 1024 cycles, exponential unit 1024 cycles, shared-memory reads 768 cycles. The MMA time halves from H100 to B200 while the exponential time does not change.">
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
  <text x="150" y="46" class="as-lab">MMA (tensor cores)</text>
  <rect x="160" y="33" width="420" height="18" fill="rgb(var(--color-primary-500))"/>
  <text x="586" y="46" class="as-val">2048</text>
  <text x="150" y="72" class="as-lab">exp (MUFU)</text>
  <rect x="160" y="59" width="210" height="18" fill="rgba(var(--color-secondary-500), .25)" stroke="rgb(var(--color-secondary-500))" stroke-width="1.5"/>
  <text x="376" y="72" class="as-val">1024</text>
  <text x="420" y="72" class="as-note">½ of MMA: hides in its shadow</text>
  <text x="20" y="104" class="as-grp">B200</text>
  <text x="150" y="124" class="as-lab">MMA (tensor cores)</text>
  <rect x="160" y="111" width="210" height="18" fill="rgb(var(--color-primary-500))"/>
  <text x="376" y="124" class="as-val">1024</text>
  <text x="150" y="150" class="as-lab">exp (MUFU)</text>
  <rect x="160" y="137" width="210" height="18" fill="rgba(var(--color-secondary-500), .25)" stroke="rgb(var(--color-secondary-500))" stroke-width="1.5"/>
  <text x="376" y="150" class="as-val">1024</text>
  <text x="420" y="150" class="as-note">= MMA: nothing left to hide</text>
  <text x="150" y="176" class="as-lab">shared-mem reads</text>
  <rect x="160" y="163" width="157.5" height="18" fill="rgba(128,128,128,.12)" stroke="currentColor" stroke-opacity=".5" stroke-dasharray="3 2"/>
  <text x="323.5" y="176" class="as-val">768</text>
  <line x1="160" y1="196" x2="580" y2="196" stroke="currentColor" stroke-opacity=".4"/>
  <text x="160" y="211" class="as-tick">0</text>
  <text x="265" y="211" class="as-tick">512</text>
  <text x="370" y="211" class="as-tick">1024</text>
  <text x="475" y="211" class="as-tick">1536</text>
  <text x="580" y="211" class="as-tick">2048</text>
  <text x="370" y="232" class="as-tick">cycles per SM for one 128×128 tile, head dim 128 (fewer is faster)</text>
</svg>
<figcaption>A roofline estimate for one forward-pass tile ($M=N=d=128$), not a measurement. MMA time per tile halves from Hopper to Blackwell, from $4MNd/4096 = 2048$ to $1024$ cycles, but the $MN/16 = 1024$ cycles of exponentials do not: on H100 the exp work fits in half the MMA's shadow, on B200 it takes as long as the MMA itself. B200 bars follow Table 1 of the FA4 paper; the H100 bars apply the same formulas with Hopper's MMA rate (H100 shared-memory time is omitted because Hopper's 64-row MMA tiles change how often operands are re-read). In the backward pass the paper's estimate makes shared memory the binding resource instead: 3328 cycles against 2560 for MMA.</figcaption>
</figure>

FA4's answers:

1. **A pipeline rebuilt around fully asynchronous MMA and larger tiles.** Blackwell's MMA works on $128\times N$ tiles and writes its accumulator to tensor memory asynchronously, without going through registers. FA4 keeps FA3's ping-pong, now between two 128-row $Q$ tiles per thread block, with one thread per row so the row max and row sum need no warp shuffles. Because $P$ reaches the $PV$ MMA through tensor memory rather than registers, the output rescale moves to a separate correction warpgroup, off the softmax critical path. A longest-processing-time-first tile scheduler improves load balance for causal and variable-length batches.
2. **Software-emulated exponential and conditional softmax rescaling.** The emulation does not replace the hardware exp; it adds a second source of it. FA4 computes a tuned fraction (10–25%) of each row's exponentials on the FMA pipes instead of MUFU: split $2^x = 2^{\lfloor x\rfloor}\cdot 2^{x-\lfloor x\rfloor}$ (Cody–Waite range reduction), evaluate $2^f$ for $f\in[0,1)$ with a degree-3 polynomial, and build $2^{\lfloor x\rfloor}$ by shifting the integer into the float's exponent bits. Each emulated evaluation costs more than one hardware `ex2`, but running both in parallel raises total exp throughput. Conditional rescaling attacks the other non-matmul cost, the $O \leftarrow e^{m_{\text{old}}-m_{\text{new}}}O$ correction. That rescale was never needed for correctness, only to keep numbers in range: $O$ and $\ell$ are accumulated against the same reference max, so $O/\ell$ is exact for any reference. FA4 therefore keeps the stale max unless a new block raises it by more than $\log_2 256 = 8$, which bounds the unnormalized entries of $P$ by a factor of 256 — harmless for FP32 accumulators.

<figure>
<svg viewBox="0 0 640 240" width="100%" role="img" aria-label="Conditional rescaling over ten key/value blocks: the running max of the scores rises at nine of the blocks, so standard online softmax rescales nine times; FA4 keeps a stale reference max and only updates it at blocks 6 and 9, when the true max exceeds the reference by more than 8 in log-2 units">
  <style>
    .cr-s { font-size: 11px; fill: currentColor; opacity: .75; }
    .cr-t { font-size: 11px; fill: currentColor; }
    .cr-k { font-size: 10.5px; fill: currentColor; opacity: .65; text-anchor: middle; }
    .cr-hl { font-size: 11px; font-weight: 600; fill: currentColor; }
  </style>
  <line x1="60" y1="12" x2="82" y2="12" stroke="rgb(var(--color-primary-500))" stroke-width="2"/>
  <text x="88" y="16" class="cr-t">running max of the scores</text>
  <line x1="252" y1="12" x2="274" y2="12" stroke="rgb(var(--color-secondary-500))" stroke-width="3.5"/>
  <text x="280" y="16" class="cr-t">max FA4 actually uses</text>
  <rect x="424" y="6" width="12" height="12" fill="rgba(var(--color-secondary-500), .15)"/>
  <text x="442" y="16" class="cr-t">allowed gap: up to 256×</text>
  <rect x="42" y="138.4" width="280" height="44.8" fill="rgba(var(--color-secondary-500), .15)"/>
  <rect x="322" y="85.2" width="168" height="44.8" fill="rgba(var(--color-secondary-500), .15)"/>
  <rect x="490" y="34.8" width="112" height="44.8" fill="rgba(var(--color-secondary-500), .15)"/>
  <line x1="40" y1="30" x2="40" y2="200" stroke="currentColor" stroke-opacity=".4"/>
  <line x1="40" y1="200" x2="604" y2="200" stroke="currentColor" stroke-opacity=".4"/>
  <text x="34" y="203" text-anchor="end" class="cr-k">0</text>
  <text x="34" y="147" text-anchor="end" class="cr-k">10</text>
  <text x="34" y="91" text-anchor="end" class="cr-k">20</text>
  <text x="34" y="35" text-anchor="end" class="cr-k">30</text>
  <text transform="translate(12 118) rotate(-90)" text-anchor="middle" class="cr-k">row max (log₂ units)</text>
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
  <text x="56" y="50" class="cr-s">Standard online softmax rescales at all 9 increases; FA4 only twice.</text>
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
  <text x="322" y="232" text-anchor="middle" class="cr-k">key/value block</text>
</svg>
<figcaption>Conditional rescaling on an illustrative row. The running max rises at nine of the ten blocks, and standard online softmax rescales $O$ and $\ell$ every time. FA4 keeps a stale reference max (thick line) and only moves it when the true max gets more than 8 above it in $\log_2$ units (the shaded band) — here twice. Between updates the unnormalized weights can reach $2^8 = 256$, which FP32 accumulators handle easily, and because $O$ and $\ell$ always share the same reference, $O/\ell$ is still exact.</figcaption>
</figure>

3. **Tensor memory (TMEM) and 2-CTA MMA mode, mainly for the backward pass.** Blackwell adds 256KB per SM of tensor memory that the tensor cores write accumulators into directly. In 2-CTA mode, two thread blocks (CTAs) on a pair of SMs in the same cluster execute one MMA with $M=256$; each keeps half of the A tile and the accumulator and stages only half of operand B, halving shared-memory traffic for B. The backward pass is shared-memory-bound on B200 (about 3328 cycles of shared-memory traffic against 2560 of MMA in the paper's model); keeping more intermediates in TMEM and using 2-CTA mode brings it down to about 2688 cycles and reduces the atomic adds into $dQ$.
4. **Written in CuTe-DSL rather than raw CUDA C++ templates.** CuTe-DSL is CUTLASS's Python-embedded kernel language; the paper reports 20–30× faster compile times than template-heavy C++ with comparable expressiveness, and attention variants (ALiBi, sliding window, soft-capping) can be written as plain Python score-modification functions that get JIT-compiled into the kernel.

The paper reports up to 1613 TFLOPs/s in the BF16 forward pass on B200, about 71% of the GPU's 2.25 PFLOPs/s dense peak. That is a lower fraction than FA3's 85% on H100 — but against a peak that doubled while the exp unit and shared memory did not, holding ~70% took every change above, and it is about 1.9× FA3's absolute throughput. Against baselines, FA4 is up to 1.3× faster than cuDNN 9.13 and up to 2.7× faster than Triton. The authors also worked with the cuDNN team to fold several of these techniques into cuDNN from 9.13/9.14 on, so the gap to the newest cuDNN (9.19) is much smaller.

## What's actually constant across all four

It's worth being explicit about what *didn't* change, because that's the part that's easy to lose in a list of kernel tricks:

- **All four compute the same mathematical function.** $\mathrm{softmax}(QK^\top/\sqrt d)V$, up to floating-point rounding (and, for the FP8 path, quantization — which is itself bounded and measured, not hand-waved away).
- **None of them reduce the asymptotic FLOP count.** Compute stays $\Theta(N^2d)$ in every version; recomputation adds a little, and with a causal mask, skipping fully masked tiles roughly halves the work actually executed compared with computing and then masking the full matrix. The entire lineage is a sequence of answers to "how do we stop paying for data movement and non-matmul overhead that the FLOP count doesn't actually require."
- **Each version targets whatever the *previous* version turned into the new bottleneck.** v1 removes the $N^2$ HBM round-trip. v2 fixes the low occupancy and warp-communication overhead that v1's scheduling left on the table. v3 exploits new async hardware (TMA/WGMMA) that v1/v2 predate, and separately attacks the precision/throughput trade-off with FP8. v4 responds to Blackwell's asymmetric scaling: the exponentials are tiny in FLOP count but now take as many cycles as the matmuls in the forward pass, and shared-memory traffic exceeds the matmuls in the backward pass. This is the general pattern of hardware/software co-design: you don't get to solve "attention is slow" once — you resolve whichever constraint is currently binding, and the next GPU generation hands you a new one.

## Summary

<figure>
<svg viewBox="0 0 640 260" width="100%" role="img" aria-label="Peak forward-pass utilization reported for each FlashAttention generation, as a percentage of its GPU's dense tensor-core peak: FlashAttention on A100, 30 to 50 percent; FlashAttention-2 on A100, 73 percent; FlashAttention-3 on H100, 85 percent; FlashAttention-4 on B200, 71 percent.">
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
  <rect x="95" y="159" width="70" height="51" fill="rgb(var(--color-primary-500))"><title>FlashAttention (v1), A100 forward pass: 30–50% of peak (FlashAttention-2 paper)</title></rect>
  <rect x="95" y="125" width="70" height="34" fill="rgba(var(--color-primary-500), .18)" stroke="rgb(var(--color-primary-500))" stroke-dasharray="3 2"/>
  <text x="130" y="115" class="bc-val">30–50%</text>
  <text x="130" y="230" class="bc-cat">FA1</text>
  <text x="130" y="246" class="bc-sub">A100</text>
  <rect x="230" y="85.9" width="70" height="124.1" fill="rgb(var(--color-primary-500))"><title>FlashAttention-2, A100 forward pass: up to 73% of peak (230 of 312 TFLOPs/s)</title></rect>
  <text x="265" y="76" class="bc-val">73%</text>
  <text x="265" y="230" class="bc-cat">FA2</text>
  <text x="265" y="246" class="bc-sub">A100</text>
  <rect x="365" y="65.5" width="70" height="144.5" fill="rgb(var(--color-primary-500))"><title>FlashAttention-3, H100 forward pass, BF16: up to 85% of peak (840 of 989 TFLOPs/s)</title></rect>
  <text x="400" y="56" class="bc-val">85%</text>
  <text x="400" y="230" class="bc-cat">FA3</text>
  <text x="400" y="246" class="bc-sub">H100</text>
  <rect x="500" y="89.3" width="70" height="120.7" fill="rgb(var(--color-primary-500))"><title>FlashAttention-4, B200 forward pass, BF16: up to 71% of peak (1613 of 2250 TFLOPs/s)</title></rect>
  <text x="535" y="79" class="bc-val">71%</text>
  <text x="535" y="230" class="bc-cat">FA4</text>
  <text x="535" y="246" class="bc-sub">B200</text>
</svg>
<figcaption>Peak forward-pass utilization reported for each generation's flagship kernel, as a share of its own GPU's dense FP16/BF16 tensor-core peak (A100 312, H100 989, B200 2250 TFLOPs/s). FA1's bar spans the 30–50% range the FlashAttention-2 paper measured for it. Not apples-to-apples across GPUs: FA4's 71% on B200 sits below FA3's 85% on H100 because Blackwell doubled the tensor cores without speeding up the exponential unit or shared memory, so the same fraction of peak is harder to reach.</figcaption>
</figure>

| | Year | Target HW | Core idea | Headline result |
|---|---|---|---|---|
| FlashAttention | 2022 | Ampere-era GPUs | Tiling + online softmax + recomputation → never materialize the N×N matrix in HBM | O(N) extra memory, exact; 2–4× faster attention op; 15% (BERT) to 3× (GPT-2) end-to-end |
| FlashAttention-2 | 2023 | Ampere/Ada/Hopper | Deferred normalization; swapped loops + parallelism over sequence length; split-Q warps | ~2× over v1, up to 73% of A100 peak (forward) |
| FlashAttention-3 | 2024 | Hopper (H100) | Warp-specialized async pipeline (TMA + WGMMA); GEMM/softmax overlap; FP8 with block quantization + incoherent processing | 1.5–2.0× over v2; up to 85% of H100 peak (840 TFLOPs/s); 1.3 PFLOPs/s FP8 |
| FlashAttention-4 | 2026 | Blackwell (B200/GB200); code also runs on Hopper | Async MMA + larger tiles; software-emulated exp + conditional rescaling; TMEM + 2-CTA MMA; CuTe-DSL | 71% of B200 peak (1613 TFLOPs/s); up to 1.3× over cuDNN 9.13, 2.7× over Triton |

## Sources and further reading

- Dao, Fu, Ermon, Rudra, Ré. ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135), NeurIPS 2022. Figures 1 and 2 of this paper are the basis for the time-breakdown, tiling and FLOPs-vs-traffic diagrams above.
- Dao. ["FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"](https://arxiv.org/abs/2307.08691), ICLR 2024; see also the [Stanford Hazy Research write-up](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2).
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. ["FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"](https://arxiv.org/abs/2407.08608), NeurIPS 2024; author's [blog post](https://tridao.me/blog/2024/flash3/).
- Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao. ["FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"](https://arxiv.org/abs/2603.05451), arXiv:2603.05451, March 2026 (the PDF also ships in the repo's `assets/` folder).
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — reference implementation for all four versions.
- Want the same kind of accounting for a whole Transformer — parameters, FLOPs, memory, training time? See my [Transformer FLOPs calculator](/projects/flops/).
