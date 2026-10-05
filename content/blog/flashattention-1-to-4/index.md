---
title: "FlashAttention 1–4: How IO-Awareness Reshaped the Attention Kernel"
date: 2026-10-05
draft: false
math: true
description: "A technical walkthrough of FlashAttention's four generations — from IO-aware tiling on A100, through better work partitioning and Hopper's asynchrony, to Blackwell's asymmetric-scaling co-design."
tags: ["FlashAttention", "Transformer", "Attention", "GPU", "LLM", "AI"]
categories: ["AI笔记"]
series: ["AI笔记"]
series_order: 2
---

Self-attention is the one layer every Transformer pays for twice: once in FLOPs, and once — far more painfully — in memory traffic. FlashAttention is a family of exact-attention kernels built to fix the second problem, and each new version targets a *different bottleneck* that only becomes visible once the previous one is gone. This post walks through all four generations: what each one actually changed, why that change mattered on the hardware of its time, and what stays constant across all of them.

I'll keep the math in inline notation throughout. The primary sources are linked at the end — I'd encourage reading at least the first paper directly rather than taking any secondhand summary (including this one) at face value.

## Why attention is memory-bound, not compute-bound

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

## FlashAttention (v1): make the N² matrix never exist in HBM

Dao, Fu, Ermon, Rudra, and Ré's original 2022 paper frames this as an IO-awareness problem and solves it with two classic systems ideas applied to softmax: **tiling** and **recomputation**.

**Tiling.** Split $Q$, $K$, $V$ into blocks along $N$ small enough that a $Q$-block together with a $K$/$V$-block fits in SRAM. Compute each local score block $S_{ij}=Q_iK_j^\top/\sqrt d$ on-chip and never write the full $S$ or $P$ back to HBM. The one wrinkle is that softmax is a *row-wise global* normalization — $P_{ij} = \exp(S_{ij})/\sum_k \exp(S_{ik})$ needs the sum over the entire row, which you don't have yet when you're only looking at block $j$. The fix is the "online softmax" recurrence: keep a running row-max $m$ and running row-sum $\ell$, and correct the running output every time the max changes. Here it is in its cleanest form, one query block at a time with a single normalization at the end:

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

<figure>
<svg viewBox="0 0 640 270" width="100%" role="img" aria-label="Tiling in the loop order FlashAttention-2 adopted: one query block swept against a sequence of key/value blocks on-chip, with the output written to HBM once">
  <style>
    .tg-lab  { font-size: 13px; fill: currentColor; }
    .tg-sub  { font-size: 11px; fill: currentColor; opacity: .7; }
    .tg-idx  { font-size: 12px; font-weight: 600; text-anchor: middle; fill: rgb(var(--color-neutral-900)); }
  </style>
  <rect x="40" y="20" width="560" height="140" rx="6" fill="none" stroke="currentColor" stroke-opacity=".25" stroke-dasharray="4 4"/>
  <text x="46" y="14" class="tg-sub">full N×N score matrix — never materialized</text>
  <line x1="40" y1="55"  x2="600" y2="55"  stroke="currentColor" stroke-opacity=".12"/>
  <line x1="40" y1="90"  x2="600" y2="90"  stroke="currentColor" stroke-opacity=".12"/>
  <line x1="40" y1="125" x2="600" y2="125" stroke="currentColor" stroke-opacity=".12"/>
  <rect x="40" y="90" width="560" height="35" fill="rgba(var(--color-primary-500), .10)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".6"/>
  <text x="30" y="112" text-anchor="end" class="tg-lab">Q<tspan dy="3" font-size="9">i</tspan></text>
  <g>
    <rect x="64"  y="90" width="92" height="35" rx="4" fill="rgb(var(--color-secondary-500))"/>
    <text x="110" y="112" class="tg-idx">S<tspan dy="3" font-size="9">i1</tspan></text>
    <rect x="186" y="90" width="92" height="35" rx="4" fill="rgb(var(--color-secondary-500))"/>
    <text x="232" y="112" class="tg-idx">S<tspan dy="3" font-size="9">i2</tspan></text>
    <rect x="308" y="90" width="92" height="35" rx="4" fill="rgb(var(--color-secondary-500))"/>
    <text x="354" y="112" class="tg-idx">S<tspan dy="3" font-size="9">i3</tspan></text>
    <rect x="430" y="90" width="92" height="35" rx="4" fill="rgb(var(--color-secondary-500))"/>
    <text x="476" y="112" class="tg-idx">S<tspan dy="3" font-size="9">i4</tspan></text>
    <text x="560" y="112" class="tg-sub" text-anchor="middle">…</text>
  </g>
  <text x="320" y="147" text-anchor="middle" class="tg-sub">each K/V block streamed in from HBM, used in SRAM, then discarded</text>
  <line x1="320" y1="160" x2="320" y2="192" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#tg-arrow)"/>
  <defs>
    <marker id="tg-arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" opacity=".35"/>
    </marker>
  </defs>
  <rect x="280" y="194" width="80" height="26" rx="4" fill="rgba(var(--color-primary-500), .16)" stroke="rgb(var(--color-primary-500))"/>
  <text x="320" y="212" text-anchor="middle" class="tg-lab">O<tspan dy="3" font-size="9">i</tspan></text>
  <line x1="320" y1="220" x2="320" y2="234" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#tg-arrow)"/>
  <rect x="125" y="236" width="390" height="28" rx="4" fill="rgba(128,128,128,.10)" stroke="currentColor" stroke-opacity=".4"/>
  <text x="320" y="255" text-anchor="middle" class="tg-sub">HBM: final Oᵢ + row log-sum-exp written once</text>
</svg>
<figcaption>One query block $Q_i$ swept against every key/value block, in the loop order FlashAttention-2 adopted. Each local score block $S_{ij}$ exists only in SRAM for one iteration; the accumulators $(O_i, m_i, \ell_i)$ stay on-chip, and only the finished $O_i$ and its row statistics are written to HBM. (FA1's own Algorithm 1 instead round-trips $O_i, \ell_i, m_i$ through HBM on every outer $K$/$V$ step.)</figcaption>
</figure>

This is exact, not approximate. After processing blocks $1..j$, the accumulators satisfy $\ell_i=\sum_{k\le j}\mathrm{rowsum}(e^{S_{ik}-m_i})$ and $O_i=\sum_{k\le j}e^{S_{ik}-m_i}V_k$, with $m_i$ the running max. When the max grows, multiplying both by $e^{m_i-m_{\text{new}}}$ re-bases every earlier term, since $e^{S-m_i}\cdot e^{m_i-m_{\text{new}}}=e^{S-m_{\text{new}}}$. After the last block, $O_i/\ell_i=\mathrm{softmax}(S_{i,:})V$, because softmax is invariant to a per-row shift; the only difference from a one-shot implementation is floating-point summation order. Nothing of size $N\times N$ is ever written to HBM: the extra memory is $O(N)$ for the row statistics, instead of $O(N^2)$.

**Recomputation for the backward pass.** The backward pass of attention normally needs $P$ again. Rather than store the full $N\times N$ matrix for that (which is exactly the memory blow-up we just avoided), FlashAttention stores only $O$ and the small per-row statistics $(m, \ell)$, and *recomputes* the needed $S_{ij}$, $P_{ij}$ blocks on the fly inside SRAM during the backward pass. This trades a modest amount of extra matmul FLOPs for a large reduction in memory and memory traffic — a textbook example of recomputation beating storage when the storage is in the slow tier.

The paper's complexity result: standard attention needs $\Theta(Nd+N^2)$ HBM accesses; FlashAttention needs $\Theta(N^2d^2M^{-1})$, where $M$ is the SRAM size. For typical $d$ (64–128) and $M$ (around 100KB), $d^2$ is many times smaller than $M$, so FlashAttention makes many times fewer HBM accesses; at $M=\Theta(d^2)$ the two bounds would coincide. The paper's accompanying lower bound is deliberately weak: no exact algorithm can achieve $o(N^2d^2M^{-1})$ accesses for *every* $M$ in $[d, Nd]$. In practice the attention op itself (forward + backward) became 2–4× faster than PyTorch's standard implementation at sequence lengths 128–4K, which translated into end-to-end training speedups of 15% on BERT-large (seq. 512, vs. the MLPerf 1.1 record), 3× on GPT-2 (seq. 1K, vs. HuggingFace) and 2.4× on Long-Range Arena. Removing the memory wall also let people train with substantially longer context — while still computing the *exact* softmax, something the sparse- and low-rank-attention literature at the time could not claim.

## FlashAttention-2: the kernel wasn't even using the GPU well

v1 solved the IO problem but left throughput on the table: on an A100 its forward pass reached only 30–50% of the GPU's theoretical peak FLOPs/s (25–35% for the backward pass), well below the 80–90% that a well-tuned dense GEMM achieves. FlashAttention-2 (Dao, 2023) is a work-scheduling paper — same math, better engineering of how work is split across thread blocks and warps.

Three changes matter most:

1. **Fewer non-matmul FLOPs.** Non-matmul work (exponentials, row max/sum, rescaling multiplies) runs on the FP32 ALUs and special-function units rather than the tensor cores. On A100 that is 19.5 TFLOPs/s of FP32 versus 312 TFLOPs/s of FP16 matmul, so each non-matmul FLOP costs roughly 16× more. v1 keeps the output normalized at every step, dividing by the running sum on top of the unavoidable $\exp(m_{\text{old}}-m_{\text{new}})$ correction. FA2 keeps an *un-normalized* accumulator, applies only the $\exp(m_{\text{old}}-m_{\text{new}})$ correction per block, divides by $\ell$ once at the end (the pseudocode above), and saves a single logsumexp $L=m+\log\ell$ per row for the backward pass instead of both $m$ and $\ell$.
2. **Swapped loops and parallelism over sequence length.** v1 parallelized only over (batch × heads), which can be too few thread blocks to saturate the GPU's SMs with today's long sequences and small per-GPU batches. FA2 makes the query-block loop the outer one, so row blocks are independent: in the forward pass each thread block owns one query row block. In the backward pass each thread block owns one key/value column block, accumulating its $dK_j$, $dV_j$ locally and using atomic adds into a shared $dQ$ buffer.
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

H100 introduced two primitives that neither v1 nor v2 was designed to use: the **Tensor Memory Accelerator (TMA)**, a hardware unit that copies whole tiles between HBM and shared memory asynchronously, and **warpgroup-level MMA (WGMMA)**, a wider, asynchronous tensor-core instruction issued by a group of four warps. FlashAttention-3 (Shah, Bikshandi, Zhang, Thakkar, Ramani, and Dao, 2024) is built around exploiting both, plus block quantization and incoherent processing for FP8.

1. **Warp specialization (producer/consumer).** In FA2, every warp both issues its share of the tile copies (already asynchronous, via Ampere's cp.async) and computes, with block-wide barriers between stages. FA3 instead assigns a producer warp to issue TMA loads into a multi-stage shared-memory ring tracked by hardware barriers, and gives most of the register file to consumer warpgroups that only run WGMMA on tiles already staged. The kernel becomes an explicit software pipeline.
2. **Overlapping GEMM and softmax.** On a single tile, softmax depends on that block's $QK^\top$, and the $PV$ GEMM depends on softmax, so they serialize. FA3 breaks the chain in two complementary ways. *Inter-warpgroup ping-pong:* the two consumer warpgroups each own a different query tile, and barriers force them to alternate, so warpgroup A's GEMMs run while warpgroup B does its softmax, and vice versa; the paper reports this moving FP16 forward throughput from roughly 570 to 620 TFLOPs/s at head-dim 128, seqlen 8K on H100. *Intra-warpgroup pipelining:* within one warpgroup, the asynchronous WGMMAs for the next block's $QK^\top$ and the current block's $PV$ are issued so that softmax runs while they are in flight.
3. **FP8 with block quantization and incoherent processing.** Quantizing $Q$ and $K$ directly to FP8 before the matmul is attractive for throughput but inaccurate, because a few outlier entries dominate the quantization range and crush the resolution for everything else. FA3 uses two complementary fixes. First, *block quantization*: one FP8 scale per tile instead of one per tensor, which keeps an outlier's damage inside its own block and costs nothing extra in a tiled kernel. Second, *incoherent processing*: multiply $Q$ and $K$ by the same random orthogonal matrix $M$ before quantizing. Since $M$ is orthogonal, $(QM)(KM)^\top = QMM^\top K^\top = QK^\top$ — the attention scores are mathematically unchanged — but each entry of $QM$ is now a random mixture of many original entries, so no single outlier dominates any one coordinate. Implemented as a Hadamard transform with random sign flips, this costs $O(d\log d)$ per length-$d$ row instead of $O(d^2)$ for a dense rotation. On test inputs with simulated outlier features, the FP8 path has 2.6× lower RMSE than a baseline FP8 attention with per-tensor scaling.

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

Results: 1.5–2.0× over FA2 on H100, up to 840 TFLOPs/s in BF16 (about 85% of H100 SXM5's 989 TFLOPs/s dense peak), and up to 1.3 PFLOPs/s in FP8. (The July 2024 arXiv v1 reported 740 TFLOPs/s and close to 1.2 PFLOPs/s; the numbers here are from the NeurIPS 2024 version cited below.)

## FlashAttention-4: when the bottleneck itself shifts hardware generation

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

- Dao, Fu, Ermon, Rudra, Ré. ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
- Dao. ["FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"](https://arxiv.org/abs/2307.08691), ICLR 2024; see also the [Stanford Hazy Research write-up](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2).
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. ["FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"](https://arxiv.org/abs/2407.08608), NeurIPS 2024; author's [blog post](https://tridao.me/blog/2024/flash3/).
- Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao. ["FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"](https://arxiv.org/abs/2603.05451), arXiv:2603.05451, March 2026 (the PDF also ships in the repo's `assets/` folder).
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — reference implementation for all four versions.
