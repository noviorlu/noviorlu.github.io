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

I'll keep the math in inline notation throughout. If you want the primary sources, they're linked at the end — I'd encourage reading at least the first paper directly rather than taking any secondhand summary (including this one) at face value.

## Why attention is memory-bound, not compute-bound

Standard self-attention computes $S = QK^\top/\sqrt{d}$, $P = \mathrm{softmax}(S)$, $O = PV$, for $Q, K, V \in \mathbb{R}^{N\times d}$. The FLOP count is $O(N^2 d)$ — identical for every version of FlashAttention discussed here. **FlashAttention does not reduce the number of floating-point operations attention requires.** What it reduces is HBM traffic, and on a modern GPU that is almost always the thing you're actually paying for.

The reason is the memory hierarchy. An A100, for instance, has on the order of 40–80GB of HBM (the big, "GPU memory" everyone quotes) with roughly 1.5–2.0TB/s of bandwidth, and roughly 192KB of on-chip SRAM *per streaming multiprocessor*, an order of magnitude faster at upward of ~19TB/s but minuscule in capacity. A naive implementation of the formula above materializes $S$ and $P$ — both $N\times N$ — in HBM, writes them, reads them back for the softmax, writes again, reads again for the final matmul. For long sequences this round-tripping dominates wall-clock time even though the matmuls themselves are cheap in FLOPs. This is a textbook roofline-model situation: the kernel is bottlenecked by arithmetic intensity (FLOPs per byte moved), not by peak FLOPs.

<figure>
<svg viewBox="0 0 640 230" width="100%" role="img" aria-label="GPU memory hierarchy: small fast SRAM on top of large slow HBM">
  <style>
    .mh-label { font: 13px/1.4 inherit; fill: currentColor; }
    .mh-sub { font: 11px/1.4 inherit; fill: currentColor; opacity: .6; }
    .mh-box { stroke-width: 1.5; }
  </style>
  <rect x="220" y="20" width="200" height="56" rx="6" class="mh-box" fill="rgba(var(--color-primary-500), .16)" stroke="rgb(var(--color-primary-500))"/>
  <text x="320" y="44" text-anchor="middle" class="mh-label">On-chip SRAM</text>
  <text x="320" y="62" text-anchor="middle" class="mh-sub">~192KB / SM · ~19 TB/s</text>
  <line x1="320" y1="76" x2="320" y2="104" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#mh-arrow)"/>
  <defs>
    <marker id="mh-arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" opacity=".35"/>
    </marker>
  </defs>
  <rect x="90" y="110" width="460" height="100" rx="6" class="mh-box" fill="rgba(128,128,128,.10)" stroke="currentColor" stroke-opacity=".4"/>
  <text x="320" y="156" text-anchor="middle" class="mh-label">HBM (device memory)</text>
  <text x="320" y="176" text-anchor="middle" class="mh-sub">40–80GB · ~1.5–2.0 TB/s</text>
  <text x="320" y="224" text-anchor="middle" class="mh-sub">capacity grows downward; bandwidth grows upward — not drawn to scale</text>
</svg>
<figcaption>Two tiers of memory, roughly three orders of magnitude apart in both capacity and bandwidth. Every FlashAttention version is, at its core, a different strategy for keeping the $N\times N$ intermediate out of the bottom tier.</figcaption>
</figure>

## FlashAttention (v1): make the $N^2$ matrix never exist in HBM

Dao, Fu, Ermon, Rudra, and Ré's original 2022 paper frames this as an IO-awareness problem and solves it with two classic systems ideas applied to softmax: **tiling** and **recomputation**.

**Tiling.** Split $Q$, $K$, $V$ into blocks along $N$ small enough that a $Q$-block together with a $K$/$V$-block fits in SRAM. Stream the $K$/$V$ blocks past a fixed $Q$-block, compute each local score block $S_{ij}$ on-chip, and never write the full $S$ or $P$ back to HBM. The one wrinkle is that softmax is a *row-wise global* normalization — $P_{ij} = \exp(S_{ij})/\sum_k \exp(S_{ik})$ needs the sum over the entire row, which you don't have yet when you're only looking at block $j$. The fix is the "online softmax" recurrence: keep a running row-max $m$ and running row-sum $\ell$, and correct the running output every time the max changes:

```
for each query block i:
    O_i ← 0,  ℓ_i ← 0,  m_i ← -∞          # kept in SRAM/registers only
    for each key/value block j:
        S_ij  ← Q_i K_jᵗ / sqrt(d)         # B×B, SRAM only, never hits HBM
        m_new ← max(m_i, rowmax(S_ij))
        P_ij  ← exp(S_ij − m_new)
        ℓ_i   ← exp(m_i − m_new)·ℓ_i + rowsum(P_ij)
        O_i   ← exp(m_i − m_new)·O_i + P_ij·V_j
        m_i   ← m_new
    O_i ← O_i / ℓ_i                        # one write of O_i to HBM, that's it
```

<figure>
<svg viewBox="0 0 640 270" width="100%" role="img" aria-label="Tiling: one query block swept against a sequence of key/value blocks entirely on-chip, with a single output write to HBM">
  <style>
    .tg-lab  { font: 13px/1.4 inherit; fill: currentColor; }
    .tg-sub  { font: 11px/1.4 inherit; fill: currentColor; opacity: .6; }
    .tg-idx  { font: 11px/1.4 inherit; fill: #fff; font-weight: 600; text-anchor: middle; }
  </style>
  <rect x="40" y="20" width="560" height="140" rx="6" fill="none" stroke="currentColor" stroke-opacity=".25" stroke-dasharray="4 4"/>
  <text x="46" y="14" class="tg-sub">full N×N score matrix — never materialized</text>
  <line x1="40" y1="55"  x2="600" y2="55"  stroke="currentColor" stroke-opacity=".12"/>
  <line x1="40" y1="90"  x2="600" y2="90"  stroke="currentColor" stroke-opacity=".12"/>
  <line x1="40" y1="125" x2="600" y2="125" stroke="currentColor" stroke-opacity=".12"/>
  <rect x="40" y="90" width="560" height="35" fill="rgba(var(--color-primary-500), .10)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".6"/>
  <text x="30" y="112" text-anchor="end" class="tg-lab">Qᵢ</text>
  <g>
    <rect x="64"  y="90" width="92" height="35" rx="4" fill="rgb(var(--color-secondary-500))"/>
    <text x="110" y="112" class="tg-idx">S_i1</text>
    <rect x="186" y="90" width="92" height="35" rx="4" fill="rgb(var(--color-secondary-500))" opacity=".85"/>
    <text x="232" y="112" class="tg-idx">S_i2</text>
    <rect x="308" y="90" width="92" height="35" rx="4" fill="rgb(var(--color-secondary-500))" opacity=".7"/>
    <text x="354" y="112" class="tg-idx">S_i3</text>
    <rect x="430" y="90" width="92" height="35" rx="4" fill="rgb(var(--color-secondary-500))" opacity=".55"/>
    <text x="476" y="112" class="tg-idx">S_i4</text>
    <text x="560" y="112" class="tg-sub" text-anchor="middle">…</text>
  </g>
  <text x="320" y="178" text-anchor="middle" class="tg-sub">each K_j/V_j block streamed in, computed in SRAM, then discarded</text>
  <line x1="320" y1="160" x2="320" y2="192" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#tg-arrow)"/>
  <defs>
    <marker id="tg-arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" opacity=".35"/>
    </marker>
  </defs>
  <rect x="280" y="194" width="80" height="26" rx="4" fill="rgba(var(--color-primary-500), .16)" stroke="rgb(var(--color-primary-500))"/>
  <text x="320" y="212" text-anchor="middle" class="tg-lab">Oᵢ</text>
  <line x1="320" y1="220" x2="320" y2="234" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#tg-arrow)"/>
  <rect x="140" y="236" width="360" height="28" rx="4" fill="rgba(128,128,128,.10)" stroke="currentColor" stroke-opacity=".4"/>
  <text x="320" y="255" text-anchor="middle" class="tg-sub">HBM — one write of Oᵢ, that's the only traffic for this row</text>
</svg>
<figcaption>One query block $Q_i$ swept against every key/value block in sequence. Each local score block $S_{ij}$ exists only in SRAM for the duration of one iteration; the running accumulators $(O_i, m_i, \ell_i)$ live in registers/SRAM the whole time, and only the finished $O_i$ ever touches HBM.</figcaption>
</figure>

This is exact, not approximate: because $\exp(S_{ij}-m) = \exp(S_{ij}-m_{\text{true}})\cdot\exp(m_{\text{true}}-m)$, every rescaling is an algebraically valid correction, and by the time you've seen every block the accumulators hold exactly what a one-shot softmax would have produced. The only thing ever written to HBM per query block is the final $O_i$ — $O(N)$ total, instead of $O(N^2)$.

**Recomputation for the backward pass.** The backward pass of attention normally needs $P$ again. Rather than store the full $N\times N$ matrix for that (which is exactly the memory blow-up we just avoided), FlashAttention stores only $O$ and the small per-row statistics $(m, \ell)$, and *recomputes* the needed $S_{ij}$, $P_{ij}$ blocks on the fly inside SRAM during the backward pass. This trades a modest amount of extra matmul FLOPs for a large reduction in memory and memory traffic — a textbook example of recomputation beating storage when the storage is in the slow tier.

The paper's complexity result: standard attention needs $\Theta(Nd+N^2)$ HBM accesses; FlashAttention needs $\Theta(N^2d^2M^{-1})$, where $M$ is the SRAM size — and this is provably optimal (within a constant factor) for any exact-attention algorithm, achieved when $M=\Theta(d^2)$. Practically, this took attention's memory footprint from $O(N^2)$ to $O(N)$, gave roughly a 2–4× end-to-end training speedup and, by removing the memory wall, let people train with substantially longer context than was previously feasible — while still computing the *exact* softmax, something the sparse- and low-rank-attention literature at the time could not claim.

## FlashAttention-2: the kernel wasn't even using the GPU well

v1 solved the IO problem but left throughput on the table: on an A100 it reached only on the order of 25–40% of the GPU's theoretical peak FLOPs/s, well below the ~80–90% that a well-tuned dense GEMM achieves. FlashAttention-2 (Dao, 2023) is a pure work-scheduling paper — same algorithm, better engineering of how work is split across thread blocks and warps.

Two changes matter most:

1. **Fewer non-matmul FLOPs.** v1's inner loop rescales the output accumulator by $\exp(m_{\text{old}}-m_{\text{new}})$ at *every* block iteration. Non-matmul instructions (exponentials, the rescale multiply) run on the GPU's much lower-throughput special-function/ALU paths, not the tensor cores, so every one of them is disproportionately expensive relative to a matmul of the same "size." FA2 restructures the recurrence so that this rescaling work is deferred and only the minimum necessary bookkeeping happens per block, cutting the non-matmul FLOP count significantly without changing the result.
2. **Better parallelism and warp partitioning.** v1 parallelized only over (batch × heads) — fine when batch×heads is large, but with today's larger models, smaller per-GPU batch sizes, and long-context training/inference, that grid can be too small to saturate a GPU's streaming multiprocessors. FA2 additionally parallelizes over the query-sequence dimension. Within a thread block, v1 split $K$/$V$ across warps while every warp could see all of $Q$ (a "split-K" scheme); each warp then had to write its partial result to shared memory, synchronize, and reduce across warps — extra shared-memory traffic. FA2 flips this: it splits $Q$ across warps while every warp sees all of $K$/$V$ ("split-Q"), so each warp produces a complete, independent output slice with **no inter-warp communication or synchronization** at all.

<figure>
<svg viewBox="0 0 640 300" width="100%" role="img" aria-label="Split-K warp partitioning requires a shared-memory reduction across warps; split-Q lets each warp finish independently">
  <style>
    .wp-title { font: 600 13px/1.4 inherit; fill: currentColor; }
    .wp-lab   { font: 12px/1.4 inherit; fill: #fff; text-anchor: middle; }
    .wp-sub   { font: 11px/1.4 inherit; fill: currentColor; opacity: .65; text-anchor: middle; }
  </style>
  <text x="170" y="18" text-anchor="middle" class="wp-title">v1 — split-K</text>
  <rect x="30" y="30" width="280" height="30" rx="4" fill="rgba(128,128,128,.12)" stroke="currentColor" stroke-opacity=".35"/>
  <text x="170" y="50" class="wp-sub" fill-opacity="1">Q — shared, visible to every warp</text>
  <g>
    <rect x="30"  y="78" width="60" height="34" rx="4" fill="rgb(var(--color-secondary-500))"/>
    <text x="60"  y="99" class="wp-lab">K/V₀</text>
    <rect x="100" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-secondary-500))" opacity=".85"/>
    <text x="130" y="99" class="wp-lab">K/V₁</text>
    <rect x="170" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-secondary-500))" opacity=".7"/>
    <text x="200" y="99" class="wp-lab">K/V₂</text>
    <rect x="240" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-secondary-500))" opacity=".55"/>
    <text x="270" y="99" class="wp-lab">K/V₃</text>
  </g>
  <text x="170" y="128" class="wp-sub">warp₀…₃: partial sum per K/V slice</text>
  <path d="M60,130 Q170,175 170,188 M130,130 Q170,175 170,188 M200,130 Q170,175 170,188 M270,130 Q170,175 170,188"
    fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5"/>
  <rect x="90" y="190" width="160" height="32" rx="4" fill="rgba(var(--color-primary-500), .16)" stroke="rgb(var(--color-primary-500))"/>
  <text x="170" y="211" text-anchor="middle" class="wp-sub" fill-opacity="1">reduce in shared mem</text>
  <text x="170" y="240" text-anchor="middle" class="wp-sub">write partial · barrier sync · add</text>
  <line x1="330" y1="10" x2="330" y2="260" stroke="currentColor" stroke-opacity=".15"/>
  <text x="490" y="18" text-anchor="middle" class="wp-title">v2 — split-Q</text>
  <rect x="350" y="30" width="280" height="30" rx="4" fill="rgba(128,128,128,.12)" stroke="currentColor" stroke-opacity=".35"/>
  <text x="490" y="50" class="wp-sub" fill-opacity="1">K, V — shared, visible to every warp</text>
  <g>
    <rect x="350" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-primary-500))"/>
    <text x="380" y="99" class="wp-lab">Q₀</text>
    <rect x="420" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-primary-500))" opacity=".85"/>
    <text x="450" y="99" class="wp-lab">Q₁</text>
    <rect x="490" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-primary-500))" opacity=".7"/>
    <text x="520" y="99" class="wp-lab">Q₂</text>
    <rect x="560" y="78" width="60" height="34" rx="4" fill="rgb(var(--color-primary-500))" opacity=".55"/>
    <text x="590" y="99" class="wp-lab">Q₃</text>
  </g>
  <text x="490" y="128" class="wp-sub">warp₀…₃: complete output per Q-slice</text>
  <line x1="380" y1="130" x2="380" y2="188" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <line x1="450" y1="130" x2="450" y2="188" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <line x1="520" y1="130" x2="520" y2="188" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <line x1="590" y1="130" x2="590" y2="188" stroke="currentColor" stroke-opacity=".35" stroke-width="1.5" marker-end="url(#wp-arrow)"/>
  <defs>
    <marker id="wp-arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" opacity=".35"/>
    </marker>
  </defs>
  <rect x="350" y="190" width="60" height="32" rx="4" fill="rgba(var(--color-primary-500), .14)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".5"/>
  <text x="380" y="211" class="wp-sub" fill-opacity="1">O₀</text>
  <rect x="420" y="190" width="60" height="32" rx="4" fill="rgba(var(--color-primary-500), .14)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".5"/>
  <text x="450" y="211" class="wp-sub" fill-opacity="1">O₁</text>
  <rect x="490" y="190" width="60" height="32" rx="4" fill="rgba(var(--color-primary-500), .14)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".5"/>
  <text x="520" y="211" class="wp-sub" fill-opacity="1">O₂</text>
  <rect x="560" y="190" width="60" height="32" rx="4" fill="rgba(var(--color-primary-500), .14)" stroke="rgb(var(--color-primary-500))" stroke-opacity=".5"/>
  <text x="590" y="211" class="wp-sub" fill-opacity="1">O₃</text>
  <text x="490" y="240" text-anchor="middle" class="wp-sub">independent — no cross-warp sync at all</text>
</svg>
<figcaption>Same four warps, opposite split. v1 splits $K$/$V$ and has to reconcile four partial sums afterward; v2 splits $Q$ so each warp's output slice is already complete when its matmul finishes.</figcaption>
</figure>

Net effect: roughly 2× over v1, reaching up to ~70% of theoretical peak FLOPs/s on A100 for the forward pass (and somewhat less, ~63%, for the backward pass, which has an inherently less favorable data-reuse pattern). No new hardware feature was required — this is squarely a "use the GPU you already had better" paper, which is also why it's the version most widely deployed today across non-Hopper GPUs (and the AMD ROCm port).

## FlashAttention-3: exploiting what Hopper added

H100 introduced two primitives that neither v1 nor v2 was designed to use: the **Tensor Memory Accelerator (TMA)**, which moves tiles between HBM and SRAM *asynchronously* without occupying a warp's registers/compute, and **warpgroup-level MMA (WGMMA)**, a wider, asynchronous tensor-core instruction. FlashAttention-3 (Shah, Bikshandi, Zhang, Thakkar, Ramani, and Dao, 2024) is built around exploiting both, plus a precision trick for FP8.

1. **Warp specialization (producer/consumer).** Rather than have every warp alternate between "load a tile" and "compute on it" — which under synchronous loads means compute stalls waiting on memory — FA3 assigns some warps to be *producers* that issue TMA loads continuously, and other warps to be *consumers* that only run WGMMA on tiles the producers have already staged. This turns the kernel into an explicit software pipeline instead of a single warp doing everything serially.
2. **Overlapping GEMM and softmax ("ping-pong" scheduling).** Within one iteration, softmax depends on that iteration's GEMM output, so they're sequentially dependent — but *across* warpgroups and iterations they don't have to be. With two warpgroups, FA3 schedules it so warpgroup A does its softmax (non-matmul, scalar-ish units) at the same time warpgroup B is running its WGMMA (tensor cores) for the next tile, and vice versa on the next step. Since the two phases use largely disjoint functional units, this overlap is close to free, and the paper reports it moving FP16 forward throughput from roughly 570 to 620 TFLOPs/s at head-dim 128, seqlen 8K on H100.
3. **FP8 with incoherent processing.** Quantizing $Q$ and $K$ directly to FP8 before the matmul is attractive for throughput but inaccurate, because a few outlier entries dominate the quantization range and crush the resolution for everything else. FA3's fix is elegant: multiply $Q$ and $K$ by the same random orthogonal matrix $M$ before quantizing. Since $M$ is orthogonal, $(QM)(KM)^\top = QMM^\top K^\top = QK^\top$ — the attention scores are mathematically unchanged — but each entry of $QM$ is now a random mixture of many original entries, so no single outlier dominates any one coordinate anymore. Implemented efficiently as a (signed) Hadamard transform, this runs in $O(d\log d)$ per head instead of $O(d^2)$ and is reported to cut FP8 quantization error by about 2.6× versus naive FP8 quantization.

<figure>
<svg viewBox="0 0 640 210" width="100%" role="img" aria-label="Ping-pong scheduling: two warpgroups alternate GEMM and softmax so each one's softmax phase overlaps the other's matmul phase">
  <style>
    .pp-lane { font: 12px/1.4 inherit; fill: currentColor; opacity: .75; }
    .pp-seg  { font: 11px/1.4 inherit; fill: #fff; text-anchor: middle; }
    .pp-sub  { font: 11px/1.4 inherit; fill: currentColor; opacity: .6; }
  </style>
  <text x="20" y="42" class="pp-lane">Warpgroup A</text>
  <text x="20" y="102" class="pp-lane">Warpgroup B</text>
  <g>
    <rect x="130" y="24" width="120" height="34" fill="rgb(var(--color-primary-500))"/>
    <text x="190" y="45" class="pp-seg">GEMM</text>
    <rect x="250" y="24" width="120" height="34" fill="rgb(var(--color-secondary-500))"/>
    <text x="310" y="45" class="pp-seg">softmax</text>
    <rect x="370" y="24" width="120" height="34" fill="rgb(var(--color-primary-500))"/>
    <text x="430" y="45" class="pp-seg">GEMM</text>
    <rect x="490" y="24" width="120" height="34" fill="rgb(var(--color-secondary-500))"/>
    <text x="550" y="45" class="pp-seg">softmax</text>
  </g>
  <g>
    <rect x="130" y="84" width="120" height="34" fill="rgb(var(--color-secondary-500))"/>
    <text x="190" y="105" class="pp-seg">softmax</text>
    <rect x="250" y="84" width="120" height="34" fill="rgb(var(--color-primary-500))"/>
    <text x="310" y="105" class="pp-seg">GEMM</text>
    <rect x="370" y="84" width="120" height="34" fill="rgb(var(--color-secondary-500))"/>
    <text x="430" y="105" class="pp-seg">softmax</text>
    <rect x="490" y="84" width="120" height="34" fill="rgb(var(--color-primary-500))"/>
    <text x="550" y="105" class="pp-seg">GEMM</text>
  </g>
  <line x1="130" y1="18" x2="130" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="250" y1="18" x2="250" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="370" y1="18" x2="370" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="490" y1="18" x2="490" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <line x1="610" y1="18" x2="610" y2="124" stroke="currentColor" stroke-opacity=".2" stroke-dasharray="3 3"/>
  <text x="610" y="145" text-anchor="end" class="pp-sub">time →</text>
  <path d="M250,122 L250,138 L370,138 L370,122" fill="none" stroke="currentColor" stroke-opacity=".4"/>
  <text x="310" y="155" text-anchor="middle" class="pp-sub">A's softmax runs while B's GEMM runs — different functional units, ~free overlap</text>
  <g>
    <rect x="130" y="184" width="14" height="14" fill="rgb(var(--color-primary-500))"/>
    <text x="150" y="195" class="pp-sub" fill-opacity="1">GEMM (tensor cores)</text>
    <rect x="330" y="184" width="14" height="14" fill="rgb(var(--color-secondary-500))"/>
    <text x="350" y="195" class="pp-sub" fill-opacity="1">softmax (non-matmul units)</text>
  </g>
</svg>
<figcaption>Two warpgroups, phases offset by one step. Whenever one warpgroup is stuck doing softmax's scalar work, the other is busy on its tensor-core matmul — so the "non-matmul tax" mostly disappears into the matmul's shadow instead of serializing with it.</figcaption>
</figure>

Results: 1.5–2.0× over FA2 in BF16 on H100 (up to ~740 TFLOPs/s, ~75% of H100's theoretical peak), and close to 1.2 PFLOPs/s in FP8 with the reduced quantization error from incoherent processing.

## FlashAttention-4: when the bottleneck itself shifts hardware generation

FlashAttention-4 (Zadouri, Hoehnerbach, Shah, Liu, Thakkar, and Dao; Princeton, Together AI, Meta, NVIDIA, Colfax Research, and Georgia Tech) was published in March 2026, targeting Blackwell GPUs (B200/GB200). I'll flag upfront that this is the newest and least independently battle-tested generation here — arXiv access was unavailable while researching this post, so the description below is reconstructed from the paper's abstract and third-party technical summaries rather than a direct read of the full text; treat the specifics with slightly more caution than v1–v3.

The paper's motivating observation is **asymmetric hardware scaling**: going from Hopper to Blackwell, tensor-core throughput roughly doubled, but the other functional units attention depends on — shared-memory bandwidth, and critically the special-function units that compute $\exp$ for softmax — scaled much less. FA3's ping-pong trick worked because softmax's non-matmul cost was *small relative to* the matmul it was hidden behind. Double the matmul speed without touching the exp-unit speed, and that same non-matmul tax that used to hide for free starts to show up as a real bottleneck again — a new instance of the same IO/compute-balance problem the original paper solved, just one level up the stack.

FA4's answers, as reported:

1. **Redesigned, more asynchronous pipelining with larger tiles**, sized for Blackwell's bigger and faster tensor cores so they stay fed rather than idling between async MMA issues.
2. **Software-emulated exponential and conditional softmax rescaling**: approximating the hardware transcendental-unit $\exp$ with a cheaper software routine tuned for the softmax use case, and skipping the online-softmax rescale step when it's not numerically necessary — directly attacking the exp-unit bottleneck the asymmetric scaling exposed.
3. **Tensor memory (TMEM) and 2-CTA MMA mode**: Blackwell adds a dedicated on-chip memory for MMA accumulators, and a mode where two cooperative thread arrays (effectively, two SM partitions) jointly issue one larger MMA while sharing a loaded operand. FA4 reportedly uses both to cut shared-memory traffic and reduce the atomic-add traffic the backward pass needs for accumulating gradients.
4. **Written in CuTe-DSL rather than raw CUDA C++ templates.** CuTe-DSL is CUTLASS's Python-embedded kernel language; the claimed benefit is 20–30× faster compile times than template-heavy C++ with comparable expressiveness, and it's reported to let users write masking/bias variants (ALiBi, sliding window, soft-capping) as plain Python functions that get JIT-compiled into the kernel, rather than hand-writing new CUDA for each variant.

Reported results: up to ~1613 TFLOPs/s at 71% utilization on B200 in BF16 — notable because attention kernels have historically struggled to clear 50–60% utilization even on earlier hardware — a 1.3× speedup over cuDNN 9.13 and 2.7× over a Triton-based implementation.

## What's actually constant across all four

It's worth being explicit about what *didn't* change, because that's the part that's easy to lose in a list of kernel tricks:

- **All four compute the exact same mathematical function.** $\mathrm{softmax}(QK^\top/\sqrt d)V$, bit-for-bit up to floating-point rounding (and, for the FP8 path, quantization — which is itself bounded and measured, not hand-waved away).
- **None of them reduce FLOPs.** The $O(N^2d)$ compute cost is identical to naive attention in every version. The entire lineage is a sequence of answers to "how do we stop paying for data movement/non-matmul overhead that the FLOP count doesn't actually require."
- **Each version targets whatever the *previous* version turned into the new bottleneck.** v1 removes the $N^2$ HBM round-trip. v2 fixes the resulting low occupancy and warp-communication overhead that v1's scheduling left on the table. v3 exploits new async hardware (TMA/WGMMA) that v1/v2 predate, and separately attacks the precision/throughput trade-off with FP8. v4 responds to Blackwell's asymmetric scaling making the (now comparatively tiny) non-matmul softmax cost visible again. This is the general pattern of hardware/software co-design: you don't get to solve "attention is slow" once — you resolve whichever constraint is currently binding, and the next GPU generation hands you a new one.

## Summary

<figure>
<svg viewBox="0 0 640 260" width="100%" role="img" aria-label="Forward-pass utilization of each FlashAttention generation's flagship kernel, as a percentage of its target GPU's theoretical peak FLOPs per second">
  <style>
    .bc-val  { font: 600 13px/1.4 inherit; fill: currentColor; text-anchor: middle; }
    .bc-cat  { font: 13px/1.4 inherit; fill: currentColor; text-anchor: middle; }
    .bc-sub  { font: 11px/1.4 inherit; fill: currentColor; opacity: .6; text-anchor: middle; }
  </style>
  <line x1="60" y1="40" x2="60" y2="210" stroke="currentColor" stroke-opacity=".15"/>
  <line x1="60" y1="210" x2="600" y2="210" stroke="currentColor" stroke-opacity=".4"/>
  <g>
    <rect x="95"  y="156" width="70" height="54" rx="4" fill="rgb(var(--color-primary-500))" opacity=".55">
      <title>FlashAttention (v1), A100 forward pass: roughly 25–40% of theoretical peak FLOPs/s</title>
    </rect>
    <text x="130" y="146" class="bc-val">~25–40%</text>
    <text x="130" y="230" class="bc-cat">FA1</text>
    <text x="130" y="246" class="bc-sub">A100</text>
  </g>
  <g>
    <rect x="230" y="91" width="70" height="119" rx="4" fill="rgb(var(--color-primary-500))" opacity=".75">
      <title>FlashAttention-2, A100 forward pass: up to ~70% of theoretical peak FLOPs/s</title>
    </rect>
    <text x="265" y="81" class="bc-val">~70%</text>
    <text x="265" y="230" class="bc-cat">FA2</text>
    <text x="265" y="246" class="bc-sub">A100</text>
  </g>
  <g>
    <rect x="365" y="82" width="70" height="128" rx="4" fill="rgb(var(--color-primary-500))" opacity=".9">
      <title>FlashAttention-3, H100 forward pass, BF16: ~75% of theoretical peak FLOPs/s</title>
    </rect>
    <text x="400" y="72" class="bc-val">~75%</text>
    <text x="400" y="230" class="bc-cat">FA3</text>
    <text x="400" y="246" class="bc-sub">H100</text>
  </g>
  <g>
    <rect x="500" y="89" width="70" height="121" rx="4" fill="rgb(var(--color-primary-500))">
      <title>FlashAttention-4, B200 forward pass, BF16: 71% of theoretical peak FLOPs/s</title>
    </rect>
    <text x="535" y="79" class="bc-val">71%</text>
    <text x="535" y="230" class="bc-cat">FA4</text>
    <text x="535" y="246" class="bc-sub">B200</text>
  </g>
</svg>
<figcaption>Forward-pass utilization of each generation's flagship kernel, as a percentage of its own target GPU's theoretical peak — not strictly apples-to-apples across hardware generations (A100 → A100 → H100 → B200), but the climb toward consistently saturating the tensor cores is the real, shared story across all four papers.</figcaption>
</figure>

| | Year | Target HW | Core idea | Headline result |
|---|---|---|---|---|
| FlashAttention | 2022 | Ampere-era GPUs | Tiling + online softmax + recomputation → avoid materializing the $N\times N$ matrix in HBM | $O(N)$ memory, exact attention, ~2–4× training speedup |
| FlashAttention-2 | 2023 | Ampere/Ada/Hopper | Fewer non-matmul FLOPs; split-Q warp partitioning; parallelize over seqlen too | ~2× over v1, up to ~70% of A100 peak FLOPs/s |
| FlashAttention-3 | 2024 | Hopper (H100) | Warp-specialized async pipeline (TMA + WGMMA); ping-pong GEMM/softmax overlap; FP8 incoherent processing | 1.5–2.0× over v2, ~75% of H100 peak; ~1.2 PFLOPs/s in FP8 |
| FlashAttention-4 | 2026 | Blackwell (B200/GB200) | Software-emulated exp + conditional rescale; tensor memory + 2-CTA MMA; CuTe-DSL implementation | 71% of B200 peak; 1.3× over cuDNN, 2.7× over Triton |

## Sources and further reading

- Dao, Fu, Ermon, Rudra, Ré. ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
- Dao. ["FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"](https://arxiv.org/abs/2307.08691), ICLR 2024; see also the [Stanford Hazy Research write-up](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2).
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. ["FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"](https://arxiv.org/abs/2407.08608), NeurIPS 2024; author's [blog post](https://tridao.me/blog/2024/flash3/).
- Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao. "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling," arXiv:2603.05451, March 2026.
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — reference implementation and README for all four versions.

*A note on sourcing: the FlashAttention-4 section above was written from secondary summaries and the paper's abstract, since direct arXiv access wasn't available while writing this. The v1–v3 numbers were cross-checked against the original papers and author blog posts. If you're citing any of this in something that matters, go read the primary sources.*
