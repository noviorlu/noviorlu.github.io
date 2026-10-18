---
title: "单卡 GPU 优化方法：从 divergence 到 tiling"
date: 2026-10-18
draft: false
math: true
description: "单卡上常用的七种优化方法：control divergence、低精度、fusion、recomputation、memory coalescing、bank conflict 和 tiling。每种方法省的是什么、对应哪一层硬件，配上前后几篇里的实测。GPU training optimization 系列的第二篇。"
tags: ["GPU", "Triton", "CUDA", "优化", "LLM", "AI"]
categories: ["AI笔记"]
series: ["GPU training optimization"]
series_order: 2
---

[上一篇](/blog/gpu-triton-intro/)介绍了 GPU 的硬件和 Triton 的写法。这一篇把单卡上常用的优化方法放在一起：每种方法省的是什么，对应哪一层硬件，前后几篇里哪里用到了它。七种方法按从小到大的粒度排：先看一个 warp 内部（[第 1 节](#divergence)），再看数据的位宽（[第 2 节](#precision)）、kernel 之间（[第 3、4 节](#fusion)），最后是访存的方式（[第 5、6、7 节](#coalescing)）。[下一篇](/blog/gpu-training-analysis/)在 RTX 5090 上拆解一步训练，用的就是这里的几种方法。

> 环境：RTX 5090 32 GB，torch 2.11.0+cu130，Triton 3.6.0。文中的 KB、MB、GB 都按 1024 进位。

---

## 1 Control divergence {#divergence}

一个 warp 的 32 个 thread 同一时刻执行同一条指令（[入门篇 1.3 节](/blog/gpu-triton-intro/#hw-model)）。如果遇到分支，warp 里一部分 thread 走 if、另一部分走 else，硬件只能把两边都执行一遍：执行 if 那段时，走 else 的 thread 被屏蔽掉，什么也不做，反过来也一样。这叫 control divergence，代价是两段的时间加起来。分支在不同 warp 之间不一样没关系，每个 warp 只走自己那条路（[图 1-1](#fig-o-3)）。

两边怎么排，从 Volta 开始变了。Pascal 及以前，一个 warp 只有一个程序计数器，分支两边串行：if 那边全部执行完，才轮到 else，结束后 warp 在 Z 合起来，32 个 thread 又一起执行（图 1-1 上）。Volta 起每个 thread 有自己的程序计数器，两边的语句可以交替执行，A、X、B、Y 轮流推进；调度器还要保守地假设 Z 可能用到别的分支产生的数据，所以不保证在 Z 合起来，Z 也可能分两次执行（图 1-1 中）。A、B、X、Y 里没有同步操作时，调度器能判断合起来是安全的，就会和以前一样在 Z 合起来；也可以在 Z 前面调用 `__syncwarp()` 强制合起来（[Volta 白皮书](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)）。不管哪种排法，每一拍仍然只执行一条指令，同一时刻只有一边的 thread 在算，空着的 lane 一样多，divergence 的代价没有变。

<figure id="fig-o-3" class="fg-fig">
<svg class="fg" viewBox="0 0 640 446" width="100%" role="img" aria-label="control divergence：if (lane < 4) { A; B } else { X; Y }; Z。Pascal 及以前分支两边串行，先 A、B 再 X、Y，执行一边时另一半 lane 被屏蔽，最后一起执行 Z，共 5 段；Volta 起两边可以交替执行，A、X、B、Y 轮流，Z 也可能分两次执行，共 6 段，同一时刻仍只有一半 lane 在执行；分支按 warp 一致时，每个 warp 只走自己那条路">
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
  <defs><marker id="fig-o-3-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <text class="tb" x="8" y="22">Pascal 及以前：分支两边串行，结束后在 Z 合起来</text>
  <text class="s" x="118.0" y="40" text-anchor="middle">A</text>
  <text class="s" x="162.0" y="40" text-anchor="middle">B</text>
  <text class="s" x="206.0" y="40" text-anchor="middle">X</text>
  <text class="s" x="250.0" y="40" text-anchor="middle">Y</text>
  <text class="s" x="294.0" y="40" text-anchor="middle">Z</text>
  <text class="s" x="88" y="56" text-anchor="end">lane 0</text>
  <rect x="97" y="46" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="141" y="46" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="185" y="46" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="229" y="46" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="273" y="46" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="88" y="71" text-anchor="end">lane 1</text>
  <rect x="97" y="61" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="141" y="61" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="185" y="61" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="229" y="61" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="273" y="61" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="88" y="86" text-anchor="end">lane 2</text>
  <rect x="97" y="76" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="141" y="76" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="185" y="76" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="229" y="76" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="273" y="76" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="88" y="101" text-anchor="end">lane 3</text>
  <rect x="97" y="91" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="141" y="91" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="185" y="91" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="229" y="91" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="273" y="91" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="88" y="116" text-anchor="end">lane 4</text>
  <rect x="97" y="106" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="141" y="106" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="185" y="106" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="229" y="106" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="273" y="106" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="88" y="131" text-anchor="end">lane 5</text>
  <rect x="97" y="121" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="141" y="121" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="185" y="121" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="229" y="121" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="273" y="121" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="88" y="146" text-anchor="end">lane 6</text>
  <rect x="97" y="136" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="141" y="136" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="185" y="136" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="229" y="136" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="273" y="136" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="88" y="161" text-anchor="end">lane 7</text>
  <rect x="97" y="151" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="141" y="151" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="185" y="151" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="229" y="151" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="273" y="151" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="326" y="109">5 段，前 4 段都有一半 lane 空着</text>
  <text class="tb" x="8" y="196">Volta 起：两边可以交替执行，Z 不一定合起来</text>
  <text class="s" x="118.0" y="214" text-anchor="middle">A</text>
  <text class="s" x="162.0" y="214" text-anchor="middle">X</text>
  <text class="s" x="206.0" y="214" text-anchor="middle">B</text>
  <text class="s" x="250.0" y="214" text-anchor="middle">Y</text>
  <text class="s" x="294.0" y="214" text-anchor="middle">Z</text>
  <text class="s" x="338.0" y="214" text-anchor="middle">Z</text>
  <text class="s" x="88" y="230" text-anchor="end">lane 0</text>
  <rect x="97" y="220" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="141" y="220" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="185" y="220" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="229" y="220" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="273" y="220" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="317" y="220" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <text class="s" x="88" y="245" text-anchor="end">lane 1</text>
  <rect x="97" y="235" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="141" y="235" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="185" y="235" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="229" y="235" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="273" y="235" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="317" y="235" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <text class="s" x="88" y="260" text-anchor="end">lane 2</text>
  <rect x="97" y="250" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="141" y="250" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="185" y="250" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="229" y="250" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="273" y="250" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="317" y="250" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <text class="s" x="88" y="275" text-anchor="end">lane 3</text>
  <rect x="97" y="265" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="141" y="265" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="185" y="265" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="229" y="265" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="273" y="265" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="317" y="265" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <text class="s" x="88" y="290" text-anchor="end">lane 4</text>
  <rect x="97" y="280" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="141" y="280" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="185" y="280" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="229" y="280" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="273" y="280" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="317" y="280" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="88" y="305" text-anchor="end">lane 5</text>
  <rect x="97" y="295" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="141" y="295" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="185" y="295" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="229" y="295" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="273" y="295" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="317" y="295" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="88" y="320" text-anchor="end">lane 6</text>
  <rect x="97" y="310" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="141" y="310" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="185" y="310" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="229" y="310" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="273" y="310" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="317" y="310" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="88" y="335" text-anchor="end">lane 7</text>
  <rect x="97" y="325" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="141" y="325" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="185" y="325" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="229" y="325" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <rect x="273" y="325" width="42" height="12" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"/>
  <rect x="317" y="325" width="42" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/>
  <text class="s" x="370" y="283">6 段，同一时刻仍只有一半 lane 在执行</text>
  <text class="tb" x="8" y="374">分支按 warp 一致：warp 0 都走 if，warp 1 都走 else</text>
  <text class="s" x="88" y="396" text-anchor="end">warp 0</text>
  <rect x="97" y="386" width="42" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/><text class="s" x="118.0" y="397" text-anchor="middle">A</text>
  <rect x="141" y="386" width="42" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/><text class="s" x="162.0" y="397" text-anchor="middle">B</text>
  <rect x="185" y="386" width="42" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/><text class="s" x="206.0" y="397" text-anchor="middle">Z</text>
  <text class="s" x="88" y="416" text-anchor="end">warp 1</text>
  <rect x="97" y="406" width="42" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/><text class="s" x="118.0" y="417" text-anchor="middle">X</text>
  <rect x="141" y="406" width="42" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/><text class="s" x="162.0" y="417" text-anchor="middle">Y</text>
  <rect x="185" y="406" width="42" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 40%, transparent); stroke: var(--fig-1)" stroke-width="1"/><text class="s" x="206.0" y="417" text-anchor="middle">Z</text>
  <text class="s" x="238" y="417">每个 warp 3 段，没有空着的 lane</text>
  <line x1="96.0" y1="440.0" x2="360.0" y2="440.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-o-3-m0)"/>
  <text class="s" x="366" y="444">时间</text>
</svg>
<figcaption><strong>图 1-1</strong> control divergence（按 8 条 lane 画，实际一个 warp 32 条），代码是 <code>if (lane &lt; 4) { A; B } else { X; Y }; Z</code>。虚线是这一段被屏蔽、没在执行的 lane。上：Pascal 及以前一个 warp 只有一个程序计数器，分支两边串行执行。中：Volta 起每个 thread 有自己的程序计数器，两边的语句可以交替执行，调度器也不保证在 Z 合起来；但每一拍仍然只执行一条指令，空着的 lane 一样多。下：分支按 warp 一致时没有空着的 lane。上、中两层的画法参考 <a href="https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf">Volta 白皮书</a> Figure 20、22 和 <a href="https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf">CS336 lecture 5</a>。</figcaption>
</figure>

所以分支条件最好在一个 warp 里一致，比如按 program、按行来分，而不是按单个元素。逐元素的条件可以改成计算：两边都算出来，再用 select 挑一个，没有跳转，也就没有 divergence。Triton 里 `tl.where(cond, a, b)` 就是这样，`tl.load` 的 `mask` 编出来是带谓词的指令：入门篇 GELU 的 PTX 里，`@%p1 ld.global.v4.b32` 前面的 `%p1` 就是谓词，越界的 thread 不读，但大家执行的是同一条指令。Triton 里的 `if`、`for` 只能用在整个 program 共有的标量上，所以写 Triton 时很少遇到这个问题；写 CUDA 时才要自己留意。

warp 内部的问题说完，下一个是每个数占多少位。

## 2 低精度 {#precision}

把 fp32 换成 bf16、fp8、fp4，一是每个数的字节少了，同样的带宽能搬更多的数（[图 2-1](#fig-o-4)）；二是能用上更快的 Tensor core。5090 上 fp32 走 CUDA core，峰值 1.05e14 FLOPS；Tensor core 在 fp32 累加时，bf16 是 2.1e14，fp8 是 4.19e14，fp4 是 1.68e15（dense，[RTX Blackwell 白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)）。Tensor core 的输入是低精度，乘积按全精度算，再加到一个 fp32 的累加器上，所以矩阵乘的累加本来就是 fp32。各种格式差在指数位和尾数位怎么分（[图 2-1](#fig-o-4)）：指数位决定范围，尾数位决定精度。

<figure id="fig-o-4" class="fg-fig">
<svg class="fg" viewBox="0 0 640 292" width="100%" role="img" aria-label="几种浮点格式的位：符号位、指数位、尾数位各几位，能表示的最大值，以及一条 128 B 的 cache line 装几个数。fp32 是 1、8、23；bf16 是 1、8、7；fp16 是 1、5、10；fp8 E4M3 是 1、4、3；fp8 E5M2 是 1、5、2；fp4 E2M1 是 1、2、1">
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
  <text class="s" x="485.5" y="22">最大值</text><text class="s" x="543.5" y="22">128 B 装几个</text>
  <text class="lab" x="8" y="47">fp32</text>
  <rect x="92.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="103.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="115.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="126.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="138.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="149.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="161.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="172.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="184.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="195.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="207.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="218.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="230.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="241.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="253.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="264.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="276.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="287.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="299.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="310.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="322.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="333.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="345.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="356.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="368.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="379.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="391.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="402.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="414.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="425.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="437.0" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="448.5" y="32" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <text class="s" x="485.5" y="47">3.4e38</text><text class="s" x="543.5" y="47">32 个</text>
  <text class="lab" x="8" y="81">bf16</text>
  <rect x="92.0" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="103.5" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="115.0" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="126.5" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="138.0" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="149.5" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="161.0" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="172.5" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="184.0" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="195.5" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="207.0" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="218.5" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="230.0" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="241.5" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="253.0" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="264.5" y="66" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <text class="s" x="485.5" y="81">3.4e38</text><text class="s" x="543.5" y="81">64 个</text>
  <text class="lab" x="8" y="115">fp16</text>
  <rect x="92.0" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="103.5" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="115.0" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="126.5" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="138.0" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="149.5" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="161.0" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="172.5" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="184.0" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="195.5" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="207.0" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="218.5" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="230.0" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="241.5" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="253.0" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="264.5" y="100" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <text class="s" x="485.5" y="115">65504</text><text class="s" x="543.5" y="115">64 个</text>
  <text class="lab" x="8" y="149">fp8 E4M3</text>
  <rect x="92.0" y="134" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="103.5" y="134" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="115.0" y="134" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="126.5" y="134" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="138.0" y="134" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="149.5" y="134" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="161.0" y="134" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="172.5" y="134" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <text class="s" x="485.5" y="149">448</text><text class="s" x="543.5" y="149">128 个</text>
  <text class="lab" x="8" y="183">fp8 E5M2</text>
  <rect x="92.0" y="168" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="103.5" y="168" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="115.0" y="168" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="126.5" y="168" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="138.0" y="168" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="149.5" y="168" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="161.0" y="168" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <rect x="172.5" y="168" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <text class="s" x="485.5" y="183">57344</text><text class="s" x="543.5" y="183">128 个</text>
  <text class="lab" x="8" y="217">fp4 E2M1</text>
  <rect x="92.0" y="202" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="103.5" y="202" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="115.0" y="202" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="126.5" y="202" width="10.0" height="20" rx="1.5" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/>
  <text class="s" x="485.5" y="217">6</text><text class="s" x="543.5" y="217">256 个</text>
  <rect x="8" y="246" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/><text class="s" x="28" y="256">符号位</text>
  <rect x="79.5" y="246" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/><text class="s" x="99.5" y="256">指数位：决定范围</text>
  <rect x="203.5" y="246" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-3)" stroke-width="0.8"/><text class="s" x="223.5" y="256">尾数位：决定精度</text>
</svg>
<figcaption><strong>图 2-1</strong> 几种浮点格式的位分配。bf16 和 fp32 的指数位一样多，范围相同，只是尾数短；fp16 范围小得多，最大只到 65504；fp8 两种格式在范围和精度之间取舍，E4M3 精度高、E5M2 范围大。显存、L2 都按 128 B 的 cache line 搬数据，位数减半，同样的搬运量就多一倍的数。画法参考 CS336 lecture 5。</figcaption>
</figure>

所以训练时不是所有 op 都能换成低精度（[表 2-1](#tab-2-1)）。

| | 哪些 op | 原因 |
|:--|:--|:--|
| 可以用 16 位存 | 矩阵乘；relu、tanh、add、mul 这类逐元素 op | 每个输出只和几个输入有关，误差不会累积 |
| 要更高精度（fp32） | sum、softmax、normalization 这类归约；把小数加到很大的和上 | 尾数短，小数加到大数上会被舍掉 |
| 要更大的范围（fp32 或 bf16） | exp、log、pow 这类输出远大于输入的逐元素 op；loss | fp16 最大只到 65504，容易溢出 |
{#tab-2-1 caption="**表 2-1** 训练时哪些 op 能用低精度" note="分类来自 NVIDIA 的 <a href=\"https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf\">mixed precision 教程</a>，CS336 lecture 5 也用了它。"}

fp8 和 fp4 能表示的范围很窄，E4M3 最大只到 448，不能直接把 bf16 的数截过去，要先除以一个 scale 再取整，算完再乘回来。scale 怎么选是量化的核心问题。整个张量只用一个 scale 时，一个很大的离群值就会把 scale 撑大，其他数都挤到很少几个格子里。所以现在的格式都按 block 给 scale：[OCP 的 MX 格式](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)每 32 个连续的数共用一个 E8M0 的 scale，E8M0 只有指数位，也就是一个 2 的幂，元素是 MXFP8（E4M3 或 E5M2）或 MXFP4（E2M1）；NVIDIA 的 [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) 每 16 个 E2M1 的数共用一个 E4M3 的 scale，外面再乘一个 fp32 的 per-tensor scale。E2M1 一共只有 16 个编码：0、0.5、1、1.5、2、3、4、6 和它们的负数，fp4 能用全靠 scale。DeepSeek-V3 的 fp8 训练也是这个思路：activation 每 1 × 128 个数一个 scale，权重每 128 × 128 一个，Tensor core 每累加 128 个数就把部分和搬到 CUDA core 上用 fp32 加，避免累加误差变大（[DeepSeek-V3](https://arxiv.org/abs/2412.19437)）。

block scaling 带来一个训练时才有的麻烦：转置（[图 2-2](#fig-o-9)）。一个 block 里的数共用一个 scale，block 必须沿矩阵乘的归约方向连续。权重按 PyTorch 的 Linear 存成 [N, K]，前向 Y = X Wᵀ 沿 K 归约，block 横着分；反向算 dX = dY W 沿 N 归约，block 要竖着分，原来那组 block 和 scale 用不上。量化过的 W 再转置重新量化会叠加误差，所以 [MXFP8 的训练方案](https://arxiv.org/abs/2506.08027)给每个张量存两份，一份沿行量化、一份沿列量化，都从高精度的数量化出来。不只是 W：一个 Linear 有三个矩阵乘，每个输入张量都要在其中两个里出现，归约方向各不相同（[表 2-2](#tab-2-2)）。

| | 前向 Y = X Wᵀ | 反向 dX = dY W | 反向 dW = dYᵀ X |
|:--|:--|:--|:--|
| X [M, K] | 沿行（K） | | 沿列（M） |
| W [N, K] | 沿行（K） | 沿列（N） | |
| dY [M, N] | | 沿行（N） | 沿列（M） |
{#tab-2-2 caption="**表 2-2** 一个 Linear 的三个矩阵乘，各要沿哪个方向量化" note="block 要沿归约方向连续，所以 X、W、dY 各要量化两份。MXFP8 的方案里三个矩阵乘的输入都是 MXFP8，前向和 dX 输出 bf16，dW 输出 fp32 给优化器。"}
DeepSeek-V3 的 activation 在反向时也要从 1 × 128 的 tile 转成 128 × 1，它把 scale 都取成 2 的幂，让重新量化不再引入额外误差，但这一来一回要先反量化、转置、再量化，还得经过一次显存。

<figure id="fig-o-9" class="fg-fig">
<svg class="fg" viewBox="0 0 640 290" width="100%" role="img" aria-label="block scaling 和转置：W 是 [N, K]，前向 Y = X Wᵀ 沿 K 归约，每行 4 个数一个 block、一个 scale；反向 dX = dY W 沿 N 归约，block 要竖着分，原来的 scale 对不上，只能从高精度的 W 再量化一份">
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
  <defs><marker id="fig-o-9-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <text class="tb" x="40" y="22">前向 Y = X Wᵀ：沿 K 分块</text>
  <rect x="40.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="60.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="80.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="100.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="120.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="140.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="160.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="180.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="40.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="60.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="80.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="100.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="120.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="140.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="160.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="180.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="40.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="60.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="80.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="100.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="120.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="140.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="160.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="180.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="40.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="60.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="80.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="100.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="120.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="140.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="160.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="180.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="40.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="60.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="80.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="100.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="120.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="140.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="160.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="180.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="40.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="60.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="80.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="100.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="120.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="140.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="160.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="180.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="40.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="60.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="80.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="100.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="120.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="140.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="160.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="180.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="40.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="60.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="80.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="100.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="120.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="140.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="160.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="180.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="208" y="43" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="222" y="43" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="208" y="63" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="222" y="63" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="208" y="83" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="222" y="83" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="208" y="103" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="222" y="103" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="208" y="123" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="222" y="123" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="208" y="143" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="222" y="143" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="208" y="163" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="222" y="163" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="208" y="183" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="222" y="183" width="12" height="14" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="208" y="34">scale</text>
  <text class="s" x="40" y="34">K →</text>
  <text class="s" x="40" y="222">W 是 [N, K]，每行 2 个 block，各一个 scale</text>
  <text class="tb" x="380" y="22">反向 dX = dY W：沿 N 分块</text>
  <rect x="380.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="400.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="420.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="440.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="460.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="480.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="500.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="520.5" y="40.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="380.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="400.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="420.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="440.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="460.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="480.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="500.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="520.5" y="60.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="380.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="400.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="420.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="440.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="460.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="480.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="500.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="520.5" y="80.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="380.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="400.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="420.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="440.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="460.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="480.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="500.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="520.5" y="100.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="380.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="400.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="420.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="440.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="460.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="480.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="500.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="520.5" y="120.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="380.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="400.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="420.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="440.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="460.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="480.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="500.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="520.5" y="140.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="380.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="400.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="420.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="440.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="460.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="480.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="500.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="520.5" y="160.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="380.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="400.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="420.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="440.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="460.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="480.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="500.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 38%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="520.5" y="180.5" width="19" height="19" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 16%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <rect x="383" y="208" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="383" y="222" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="403" y="208" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="403" y="222" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="423" y="208" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="423" y="222" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="443" y="208" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="443" y="222" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="463" y="208" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="463" y="222" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="483" y="208" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="483" y="222" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="503" y="208" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="503" y="222" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="523" y="208" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="523" y="222" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="546" y="222">scale</text>
  <text class="s" x="358" y="52">N</text><text class="s" x="358" y="66">↓</text>
  <text class="s" x="380" y="256">同一个 W，每列 2 个 block，scale 和左边对不上</text>
  <text class="s" x="380" y="271">只能从高精度的 W 再量化一份</text>
  <line x1="300.0" y1="120.0" x2="360.0" y2="120.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.4" marker-end="url(#fig-o-9-m0)"/>
  <text class="s" x="330" y="112" text-anchor="middle">换个方向分块</text>
</svg>
<figcaption><strong>图 2-2</strong> block scaling 为什么转置以后要再量化一份（示意，一个 block 按 4 个数画，MX 格式是 32 个）。颜色深浅区分不同的 block，红色是每个 block 的 scale。W 按 PyTorch 的 Linear 存成 [N, K]。同一个 block 里的数共用一个 scale，所以 block 必须沿矩阵乘的归约方向连续：前向沿 K，反向算 dX 时沿 N，原来的 block 和 scale 用不上。</figcaption>
</figure>

在 Transformer 里，fp8 已经能用在训练的矩阵乘上：MXFP8 的方案在 8B 模型、15T token 上和 bf16 训练的结果对得上，权重、activation、梯度都用 E4M3。用 MXFP8 的只是 QKV、attention 输出的 projection、FFN 的两个 Linear 这几处矩阵乘；LayerNorm、softmax、attention 里的两个 BMM、残差加法、loss 和优化器状态都还是 bf16 或 fp32，和表 2-1 的分类一致。fp4 也开始用在预训练上（[NVFP4 预训练](https://arxiv.org/abs/2509.25149)），推理时更常见。再往下是只有三个值的权重：[BitNet b1.58](https://arxiv.org/abs/2402.17764) 的权重取 −1、0、+1（log₂3 ≈ 1.58 bit），先除以整个矩阵绝对值的平均再取整截断，activation 按 token 量化成 8 bit；乘以 ±1、0 只是加、减或跳过，矩阵乘几乎没有乘法。它是从头按这种精度训练的，不是训完再量化。推理时还常把训好的权重压到 2 到 4 bit，比如 [QuIP#](https://arxiv.org/abs/2402.04396)，算之前再反量化回 bf16。

还有两点容易踩。一是 Triton 的 `tl.dot` 遇到 fp32 输入默认走 tf32 的 Tensor core，结果和 fp32 矩阵乘差 0.1 左右，要精确的 fp32 得写 `input_precision="ieee"`（[入门篇 2.4 节](/blog/gpu-triton-intro/#triton-matmul)）。二是低精度省的主要是时间，显存不一定省得多：下一篇在 5090 上用 bf16 autocast 训练，前向快了 1.9–2.3 倍，显存只省了 18–21%，因为权重、梯度、Adam 状态还是 fp32，norm、softmax 这些也还在 fp32 下算（[下一篇 4.1 节](/blog/gpu-training-analysis/#bf16)）。

位宽省的是每个数的字节，下面两节省的是读写的次数。

## 3 Fusion {#fusion}

[Horace He](https://horace.io/brrr_intro.html) 把 GPU 比作一座工厂：计算单元是车间，显存是仓库，每个 kernel 都要把原料从仓库运进车间，做完再把产品运回仓库。逐元素和归约类的 op 算得很少，时间几乎都花在这两趟运输上。CS336 lecture 5 的例子是 $\sin^2 x + \cos^2 x$：eager 下 sin、cos、两个平方和一个加法是 5 个 kernel，中间结果在显存和计算单元之间来回搬了 4 趟。几个这样的 op 接在一起时，每个 op 都把中间结果写回显存、下一个再读回来，fusion 把它们合成一个 kernel，中间结果留在 register 里，显存只读写一次（[图 3-1](#fig-o-5)）。入门篇[表 2-1](/blog/gpu-triton-intro/#tab-2-1) 里，GELU 按公式用张量运算写，eager 要跑 8 个 kernel，194.7 µs；合成一个 Triton kernel 是 46.6 µs，快 4.2 倍。softmax 从 5 个 kernel 合成 1 个，快 2.6 倍。`torch.compile` 自动做的就是这件事，它在 GPU 上生成的也是 Triton kernel。

<figure id="fig-o-5" class="fg-fig">
<svg class="fg" viewBox="0 0 640 262" width="100%" role="img" aria-label="fusion：eager 每个 op 是一个 kernel，中间结果都写回显存、下一个再读回来；fused 一个 kernel 读一次 x、写一次 y，中间结果留在 register 里">
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
  <defs><marker id="fig-o-5-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker><marker id="fig-o-5-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <text class="tb" x="8" y="22">eager：每个 op 一个 kernel</text>
  <rect x="20" y="34" width="70" height="214" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 14%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/><text class="s" x="55" y="50" text-anchor="middle">显存</text>
  <rect x="34" y="60" width="42" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1"/><text class="s" x="55" y="73" text-anchor="middle">x</text>
  <rect x="34" y="112" width="42" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1"/><text class="s" x="55" y="125" text-anchor="middle">t1</text>
  <rect x="34" y="164" width="42" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1"/><text class="s" x="55" y="177" text-anchor="middle">t2</text>
  <rect x="34" y="216" width="42" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1"/><text class="s" x="55" y="229" text-anchor="middle">y</text>
  <rect x="190" y="85" width="74" height="26" rx="6" class="op"/><text class="t" x="227" y="103" text-anchor="middle">op 1</text>
  <line x1="78.0" y1="69.0" x2="186.0" y2="93.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-o-5-m0)"/>
  <line x1="186.0" y1="105.0" x2="78.0" y2="121.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-o-5-m0)"/>
  <rect x="190" y="137" width="74" height="26" rx="6" class="op"/><text class="t" x="227" y="155" text-anchor="middle">op 2</text>
  <line x1="78.0" y1="121.0" x2="186.0" y2="145.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-o-5-m0)"/>
  <line x1="186.0" y1="157.0" x2="78.0" y2="173.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-o-5-m0)"/>
  <rect x="190" y="189" width="74" height="26" rx="6" class="op"/><text class="t" x="227" y="207" text-anchor="middle">op 3</text>
  <line x1="78.0" y1="173.0" x2="186.0" y2="197.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-o-5-m0)"/>
  <line x1="186.0" y1="209.0" x2="78.0" y2="225.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-o-5-m0)"/>
  <text class="s" x="227" y="246" text-anchor="middle">读 3 次、写 3 次</text>
  <text class="tb" x="340" y="22">fused：一个 kernel</text>
  <rect x="352" y="34" width="70" height="214" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 14%, transparent); stroke: var(--fig-mute)" stroke-width="1.4"/><text class="s" x="387" y="50" text-anchor="middle">显存</text>
  <rect x="366" y="60" width="42" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1"/><text class="s" x="387" y="73" text-anchor="middle">x</text>
  <rect x="366" y="216" width="42" height="18" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 22%, transparent); stroke: var(--fig-2)" stroke-width="1"/><text class="s" x="387" y="229" text-anchor="middle">y</text>
  <rect x="510" y="60" width="94" height="152" rx="8" style="fill: color-mix(in srgb, var(--fig-hi) 10%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/><text class="s" x="557" y="74" text-anchor="middle">register</text>
  <rect x="520" y="84" width="74" height="26" rx="6" class="op"/><text class="t" x="557" y="102" text-anchor="middle">op 1</text>
  <line x1="557.0" y1="111.0" x2="557.0" y2="123.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-o-5-m1)"/>
  <rect x="520" y="124" width="74" height="26" rx="6" class="op"/><text class="t" x="557" y="142" text-anchor="middle">op 2</text>
  <line x1="557.0" y1="151.0" x2="557.0" y2="163.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-o-5-m1)"/>
  <rect x="520" y="164" width="74" height="26" rx="6" class="op"/><text class="t" x="557" y="182" text-anchor="middle">op 3</text>
  <line x1="410.0" y1="69.0" x2="508.0" y2="92.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-o-5-m0)"/>
  <line x1="508.0" y1="200.0" x2="410.0" y2="225.0" style="stroke: var(--fig-hi)" stroke-width="1.4" marker-end="url(#fig-o-5-m0)"/>
  <text class="s" x="557" y="246" text-anchor="middle">读 1 次、写 1 次</text>
</svg>
<figcaption><strong>图 3-1</strong> fusion 省掉中间结果的往返（示意，按 3 个 op 画）。eager 下每个 op 是一个 kernel，t1、t2 都要写回显存、下一个 kernel 再读回来；合成一个 kernel 以后，中间结果留在 register 里。画法参考 CS336 lecture 5。</figcaption>
</figure>

fusion 对一个 kernel 本来就受算力限制的 op 帮助不大：matmul 后面接的 ReLU 单独跑也只要 4 µs，矩阵乘本身要 150 µs 左右。下一篇把 RMSNorm 的 5 个 op 合成一个 kernel，前向、反向都少读写一遍中间结果（[下一篇 3.1 节](/blog/gpu-training-analysis/#rmsnorm)）。

合成一个 kernel 以后，中间结果不用写回显存了，但反向要用它们的时候怎么办，就是下一节的事。

## 4 Recomputation {#recompute}

训练时反向要用前向的中间结果。存下来要占显存，读回来要花带宽；如果重新算一遍比读回来便宜，就不存，反向时重算。这是拿算力换显存和带宽。三层 sigmoid 就是一个例子（[图 4-1](#fig-o-6)）：存下中间结果时，前向 1 读 3 写，反向 3 读 1 写，一共 8 次显存读写；sigmoid 的导数只要它自己的值，反向时读 x 重算一遍，前向 1 读 1 写，反向 2 读 1 写，一共 5 次。

<figure id="fig-o-6" class="fg-fig">
<svg class="fg" viewBox="0 0 640 250" width="100%" role="img" aria-label="三层 sigmoid 的前向和反向。左：前向把 s1、s2 写回显存给反向用，前向 1 读 3 写，反向 3 读 1 写，一共 8 次；右：前向只写 out，反向读 x 重算 s1、s2，前向 1 读 1 写，反向 2 读 1 写，一共 5 次">
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
  <defs><marker id="fig-o-6-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-o-6-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker></defs>
  <text class="tb" x="8" y="22">存下中间结果</text>
  <text class="s" x="58" y="44" text-anchor="middle">x</text>
  <rect x="22" y="54" width="72" height="24" rx="5" class="op"/><text class="t" x="58" y="70" text-anchor="middle">sigmoid</text>
  <line x1="58.0" y1="48.0" x2="58.0" y2="53.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-o-6-m0)"/>
  <line x1="96.0" y1="72.0" x2="136.0" y2="80.0" style="stroke: var(--fig-hi)" stroke-width="1.3" marker-end="url(#fig-o-6-m1)"/>
  <text class="s" x="140" y="84">s1 写回显存</text>
  <rect x="22" y="96" width="72" height="24" rx="5" class="op"/><text class="t" x="58" y="112" text-anchor="middle">sigmoid</text>
  <line x1="58.0" y1="79.0" x2="58.0" y2="95.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-o-6-m0)"/>
  <line x1="96.0" y1="114.0" x2="136.0" y2="122.0" style="stroke: var(--fig-hi)" stroke-width="1.3" marker-end="url(#fig-o-6-m1)"/>
  <text class="s" x="140" y="126">s2 写回显存</text>
  <rect x="22" y="138" width="72" height="24" rx="5" class="op"/><text class="t" x="58" y="154" text-anchor="middle">sigmoid</text>
  <line x1="58.0" y1="121.0" x2="58.0" y2="137.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-o-6-m0)"/>
  <line x1="58.0" y1="163.0" x2="58.0" y2="178.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-o-6-m0)"/>
  <text class="s" x="58" y="190" text-anchor="middle">out</text>
  <text class="tb" x="330" y="22">反向时重算</text>
  <text class="s" x="380" y="44" text-anchor="middle">x</text>
  <rect x="344" y="54" width="72" height="24" rx="5" class="op"/><text class="t" x="380" y="70" text-anchor="middle">sigmoid</text>
  <line x1="380.0" y1="48.0" x2="380.0" y2="53.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-o-6-m0)"/>
  <rect x="344" y="96" width="72" height="24" rx="5" class="op"/><text class="t" x="380" y="112" text-anchor="middle">sigmoid</text>
  <line x1="380.0" y1="79.0" x2="380.0" y2="95.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-o-6-m0)"/>
  <rect x="344" y="138" width="72" height="24" rx="5" class="op"/><text class="t" x="380" y="154" text-anchor="middle">sigmoid</text>
  <line x1="380.0" y1="121.0" x2="380.0" y2="137.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-o-6-m0)"/>
  <line x1="380.0" y1="163.0" x2="380.0" y2="178.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1.2" marker-end="url(#fig-o-6-m0)"/>
  <text class="s" x="380" y="190" text-anchor="middle">out</text>
  <text class="s" x="8" y="212">前向</text><text class="s" x="44" y="212">读 x，写 s1、s2、out：1 读 3 写</text><text class="s" x="330" y="212">读 x，写 out：1 读 1 写</text>
  <text class="s" x="8" y="228">反向</text><text class="s" x="44" y="228">读 s1、s2、dout，写 dx：3 读 1 写</text><text class="s" x="330" y="228">读 x、dout 重算 s1、s2，写 dx：2 读 1 写</text>
  <text class="s" x="8" y="244">合计</text><text class="s" x="44" y="244">8 次显存读写</text><text class="s" x="330" y="244">5 次，少了 3/8</text>
</svg>
<figcaption><strong>图 4-1</strong> 三层 sigmoid，存中间结果和反向重算各要几次显存读写。例子来自 <a href="https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467">PyTorch dev-discuss 的 min-cut recomputation 一文</a>，CS336 lecture 5 也用了它。sigmoid 的导数只要 sigmoid 自己的值，所以反向读 x 就能重算出 s1、s2。</figcaption>
</figure>

它有两个尺度。小的在一个 op 里：融合后的 RMSNorm 不存 $\hat x$，反向时用存下来的 x 和每行一个数的 r 重算。大的跨好几层，叫 activation checkpoint：前向只存每一段的入口，反向走到这一段时把前向重跑一遍。下一篇在 5090 上测过，每层一个 checkpoint 时显存从 15.0 GiB 降到 7.8 GiB，step 慢 28–33%，正好是一步从 3 次前向的量变成 4 次（[下一篇 4.2 节](/blog/gpu-training-analysis/#checkpoint)）。FlashAttention 的反向也是重算：不存 seq × seq 的 P，反向时用存下来的每行 logsumexp 重算出来（[FlashAttention 1–4](/blog/flashattention-1-to-4/)）。

前面几节都在少搬数据，后面三节看同样的数据怎么搬更快。

## 5 Memory coalescing（HBM） {#coalescing}

先看显存本身。DRAM 读一个地址时，要先把这个地址所在的一整行搬进 sense amplifier，这一步很慢；搬进来以后，同一行附近的一段数据（burst）可以接着连续送出来，几乎不花额外时间。所以显存总是一段一段地读，GPU 这边对应的单位是 32 B 的 sector 和 128 B 的 cache line。

一个 warp 发一条 load 时，硬件把 32 个 thread 的地址合起来，按 32 B 的 sector 去显存取，128 B 是一条 cache line。相邻的 thread 读相邻的地址，32 个 thread 的请求就落在几条连续的 cache line 里，取上来的字节都用得上，这叫 coalescing。相邻的 thread 隔得很远，比如按列读一个行优先的矩阵，每个 thread 都要单独取一个 sector，大部分字节是白取的（[图 5-1](#fig-o-7)）。

<figure id="fig-o-7" class="fg-fig">
<svg class="fg" viewBox="0 0 640 250" width="100%" role="img" aria-label="memory coalescing：8 个 thread 各读一个 4 B 的数。上：相邻 thread 读相邻地址，32 B 正好是一个 sector，取 1 个 sector；下：相邻 thread 隔 32 B，每个 thread 落在不同的 sector，要取 8 个 sector，每个只用到 4 B">
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
  <defs><marker id="fig-o-7-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <text class="tb" x="8" y="34">相邻 thread 读相邻的 4 B：合并</text>
  <rect x="60.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="69.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="78.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="87.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="96.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="105.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="114.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="123.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="59.5" y="72" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.6" stroke-width="1"/>
  <rect x="132.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="141.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="150.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="159.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="168.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="177.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="186.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="195.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="131.5" y="72" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.15" stroke-width="1"/>
  <rect x="204.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="213.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="222.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="231.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="240.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="249.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="258.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="267.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="203.5" y="72" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.15" stroke-width="1"/>
  <rect x="276.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="285.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="294.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="303.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="312.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="321.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="330.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="339.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="275.5" y="72" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.15" stroke-width="1"/>
  <rect x="348.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="357.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="366.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="375.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="384.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="393.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="402.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="411.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="347.5" y="72" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.15" stroke-width="1"/>
  <rect x="420.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="429.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="438.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="447.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="456.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="465.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="474.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="483.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="419.5" y="72" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.15" stroke-width="1"/>
  <rect x="492.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="501.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="510.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="519.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="528.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="537.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="546.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="555.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="491.5" y="72" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.15" stroke-width="1"/>
  <rect x="564.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="573.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="582.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="591.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="600.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="609.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="618.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="627.0" y="74" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="0.8"/>
  <rect x="563.5" y="72" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.15" stroke-width="1"/>
  <line x1="64.0" y1="54.0" x2="64.0" y2="71.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <text class="s" x="64.0" y="52" text-anchor="middle" style="font-size: 9px">T0</text>
  <line x1="73.0" y1="54.0" x2="73.0" y2="71.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <line x1="82.0" y1="54.0" x2="82.0" y2="71.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <line x1="91.0" y1="54.0" x2="91.0" y2="71.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <line x1="100.0" y1="54.0" x2="100.0" y2="71.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <line x1="109.0" y1="54.0" x2="109.0" y2="71.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <line x1="118.0" y1="54.0" x2="118.0" y2="71.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <line x1="127.0" y1="54.0" x2="127.0" y2="71.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <text class="s" x="127.0" y="52" text-anchor="middle" style="font-size: 9px">T7</text>
  <text class="s" x="60" y="106">取 1 个 sector（32 B），每个字节都用上</text>
  <text class="tb" x="8" y="150">相邻 thread 隔 32 B（比如按列读行优先的矩阵）：不合并</text>
  <rect x="60.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="69.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="78.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="87.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="96.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="105.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="114.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="123.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="59.5" y="188" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.6" stroke-width="1"/>
  <rect x="132.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="141.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="150.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="159.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="168.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="177.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="186.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="195.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="131.5" y="188" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.6" stroke-width="1"/>
  <rect x="204.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="213.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="222.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="231.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="240.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="249.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="258.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="267.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="203.5" y="188" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.6" stroke-width="1"/>
  <rect x="276.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="285.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="294.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="303.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="312.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="321.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="330.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="339.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="275.5" y="188" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.6" stroke-width="1"/>
  <rect x="348.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="357.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="366.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="375.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="384.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="393.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="402.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="411.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="347.5" y="188" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.6" stroke-width="1"/>
  <rect x="420.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="429.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="438.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="447.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="456.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="465.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="474.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="483.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="419.5" y="188" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.6" stroke-width="1"/>
  <rect x="492.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="501.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="510.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="519.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="528.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="537.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="546.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="555.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="491.5" y="188" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.6" stroke-width="1"/>
  <rect x="564.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <rect x="573.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="582.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="591.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="600.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="609.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="618.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="627.0" y="190" width="8.0" height="16" style="fill: color-mix(in srgb, var(--fig-2) 18%, transparent); stroke: var(--fig-2)" stroke-width="0.8"/>
  <rect x="563.5" y="188" width="72.0" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="0.6" stroke-width="1"/>
  <line x1="64.0" y1="170.0" x2="64.0" y2="187.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <text class="s" x="64.0" y="168" text-anchor="middle" style="font-size: 9px">T0</text>
  <line x1="136.0" y1="170.0" x2="136.0" y2="187.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <text class="s" x="136.0" y="168" text-anchor="middle" style="font-size: 9px">T1</text>
  <line x1="208.0" y1="170.0" x2="208.0" y2="187.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <text class="s" x="208.0" y="168" text-anchor="middle" style="font-size: 9px">T2</text>
  <line x1="280.0" y1="170.0" x2="280.0" y2="187.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <text class="s" x="280.0" y="168" text-anchor="middle" style="font-size: 9px">T3</text>
  <line x1="352.0" y1="170.0" x2="352.0" y2="187.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <text class="s" x="352.0" y="168" text-anchor="middle" style="font-size: 9px">T4</text>
  <line x1="424.0" y1="170.0" x2="424.0" y2="187.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <text class="s" x="424.0" y="168" text-anchor="middle" style="font-size: 9px">T5</text>
  <line x1="496.0" y1="170.0" x2="496.0" y2="187.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <text class="s" x="496.0" y="168" text-anchor="middle" style="font-size: 9px">T6</text>
  <line x1="568.0" y1="170.0" x2="568.0" y2="187.0" stroke="currentColor" stroke-opacity=".6" stroke-width="1" marker-end="url(#fig-o-7-m0)"/>
  <text class="s" x="568.0" y="168" text-anchor="middle" style="font-size: 9px">T7</text>
  <text class="s" x="60" y="222">取 8 个 sector（256 B），只用到 32 B</text>
  <text class="s" x="648" y="88">一格 4 B</text><text class="s" x="648" y="102">一个框 32 B</text>
</svg>
<figcaption><strong>图 5-1</strong> memory coalescing（按 8 个 thread 画，实际一个 warp 32 个）。红色是被读的数，浅蓝是同一个 sector 里顺带取上来、没用到的字节。</figcaption>
</figure>

每个 thread 一次读几个相邻的数也有关系。入门篇的 GELU 里，n 是 16 的倍数时，Triton 知道地址按 16 B 对齐，一个 thread 用一条 `ld.global.v4` 读 4 个 fp32；n 不是 16 的倍数时，就变成 4 条 4 B 的 `ld.global`（[入门篇 2.1 节 notes](/blog/gpu-triton-intro/#triton-gelu)），条数多了 4 倍。矩阵乘里最容易踩：A 是行优先存的，如果让相邻的 thread 各负责一行、每个 thread 沿着自己那一行往右读，同一时刻 32 个 thread 读的是同一列上的 32 个数，相隔一整行，完全不合并；让相邻的 thread 负责相邻的列、一起往下走，同一时刻读的是同一行上连续的数，才是合并的。写 Triton 时，让 `tl.arange` 沿着内存里连续的那一维，block 的这一维取得够长，编译器就会排成合并的读写。

显存这边讲完，shared memory 也有自己的规矩。

## 6 Bank conflict（shmem） {#bank}

shared memory 分成 32 个 bank，每个 bank 宽 4 B，地址按 4 B 轮流分到各个 bank 上（[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)）。一个 warp 读 shared memory 时，32 个 thread 落在 32 个不同的 bank 上，一拍就能读完；如果有几个 thread 落在同一个 bank 的不同地址上，这几个请求只能排队，一拍拆成几拍，这叫 bank conflict。几个 thread 读同一个地址不算冲突，会广播。

最常见的例子是按列读一个 32 × 32 的 fp32 矩阵：一列的 32 个数隔 128 B，全落在同一个 bank 上，要排 32 拍。matmul 里这种访问躲不掉：A 的 block 要按行读，B 的 block 要按列读，总有一个方向会撞上同一个 bank。常见的解法有两种：每行多留一个数（padding），让一列的数错开到不同的 bank；或者按规律打乱存放的位置（swizzle），比如把第 r 行第 c 列放进第 (c xor r) 个 bank，这样按行读、按列读都落在不同的 bank 上，也不多占空间（[图 6-1](#fig-o-8)）。

<figure id="fig-o-8" class="fg-fig">
<svg class="fg" viewBox="0 0 640 264" width="100%" role="img" aria-label="bank conflict（按 8 个 bank 画）：8 × 8 的 fp32 矩阵按行存进 shared memory，一列的 8 个数都落在 bank 0，8 个 thread 同时读这一列要排 8 拍；每行多留一个数（padding），或者把第 r 行第 c 列放进第 (c xor r) 个 bank（swizzle），一列的数都错开到 8 个不同的 bank，一拍读完">
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
  <text class="s" x="8" y="40">bank</text>
  <text class="tb" x="36" y="22">按行存</text>
  <text class="s" x="47.5" y="44" text-anchor="middle">0</text>
  <text class="s" x="70.5" y="44" text-anchor="middle">1</text>
  <text class="s" x="93.5" y="44" text-anchor="middle">2</text>
  <text class="s" x="116.5" y="44" text-anchor="middle">3</text>
  <text class="s" x="139.5" y="44" text-anchor="middle">4</text>
  <text class="s" x="162.5" y="44" text-anchor="middle">5</text>
  <text class="s" x="185.5" y="44" text-anchor="middle">6</text>
  <text class="s" x="208.5" y="44" text-anchor="middle">7</text>
  <rect x="37" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="47.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,0</text>
  <rect x="60" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="70.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,1</text>
  <rect x="83" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="93.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,2</text>
  <rect x="106" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="116.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,3</text>
  <rect x="129" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="139.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,4</text>
  <rect x="152" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="162.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,5</text>
  <rect x="175" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="185.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,6</text>
  <rect x="198" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="208.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,7</text>
  <rect x="37" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="47.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,0</text>
  <rect x="60" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="70.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,1</text>
  <rect x="83" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="93.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,2</text>
  <rect x="106" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="116.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,3</text>
  <rect x="129" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="139.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,4</text>
  <rect x="152" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="162.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,5</text>
  <rect x="175" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="185.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,6</text>
  <rect x="198" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="208.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,7</text>
  <rect x="37" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="47.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,0</text>
  <rect x="60" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="70.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,1</text>
  <rect x="83" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="93.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,2</text>
  <rect x="106" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="116.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,3</text>
  <rect x="129" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="139.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,4</text>
  <rect x="152" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="162.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,5</text>
  <rect x="175" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="185.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,6</text>
  <rect x="198" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="208.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,7</text>
  <rect x="37" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="47.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,0</text>
  <rect x="60" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="70.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,1</text>
  <rect x="83" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="93.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,2</text>
  <rect x="106" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="116.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,3</text>
  <rect x="129" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="139.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,4</text>
  <rect x="152" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="162.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,5</text>
  <rect x="175" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="185.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,6</text>
  <rect x="198" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="208.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,7</text>
  <rect x="37" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="47.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,0</text>
  <rect x="60" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="70.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,1</text>
  <rect x="83" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="93.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,2</text>
  <rect x="106" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="116.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,3</text>
  <rect x="129" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="139.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,4</text>
  <rect x="152" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="162.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,5</text>
  <rect x="175" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="185.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,6</text>
  <rect x="198" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="208.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,7</text>
  <rect x="37" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="47.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,0</text>
  <rect x="60" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="70.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,1</text>
  <rect x="83" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="93.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,2</text>
  <rect x="106" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="116.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,3</text>
  <rect x="129" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="139.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,4</text>
  <rect x="152" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="162.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,5</text>
  <rect x="175" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="185.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,6</text>
  <rect x="198" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="208.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,7</text>
  <rect x="37" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="47.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,0</text>
  <rect x="60" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="70.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,1</text>
  <rect x="83" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="93.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,2</text>
  <rect x="106" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="116.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,3</text>
  <rect x="129" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="139.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,4</text>
  <rect x="152" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="162.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,5</text>
  <rect x="175" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="185.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,6</text>
  <rect x="198" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="208.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,7</text>
  <rect x="37" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="47.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,0</text>
  <rect x="60" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="70.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,1</text>
  <rect x="83" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="93.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,2</text>
  <rect x="106" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="116.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,3</text>
  <rect x="129" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="139.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,4</text>
  <rect x="152" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="162.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,5</text>
  <rect x="175" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="185.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,6</text>
  <rect x="198" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="208.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,7</text>
  <text class="s" x="36" y="245">第 0 列全在 bank 0：排 8 拍</text>
  <text class="tb" x="240" y="22">每行补 1 个空位（padding）</text>
  <text class="s" x="251.5" y="44" text-anchor="middle">0</text>
  <text class="s" x="274.5" y="44" text-anchor="middle">1</text>
  <text class="s" x="297.5" y="44" text-anchor="middle">2</text>
  <text class="s" x="320.5" y="44" text-anchor="middle">3</text>
  <text class="s" x="343.5" y="44" text-anchor="middle">4</text>
  <text class="s" x="366.5" y="44" text-anchor="middle">5</text>
  <text class="s" x="389.5" y="44" text-anchor="middle">6</text>
  <text class="s" x="412.5" y="44" text-anchor="middle">7</text>
  <rect x="241" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="251.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,0</text>
  <rect x="264" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="274.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,1</text>
  <rect x="287" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="297.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,2</text>
  <rect x="310" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="320.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,3</text>
  <rect x="333" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="343.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,4</text>
  <rect x="356" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="366.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,5</text>
  <rect x="379" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="389.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,6</text>
  <rect x="402" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="412.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,7</text>
  <rect x="241" y="71" width="21" height="17" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".35" stroke-dasharray="3 2"/>
  <rect x="264" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="274.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,0</text>
  <rect x="287" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="297.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,1</text>
  <rect x="310" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="320.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,2</text>
  <rect x="333" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="343.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,3</text>
  <rect x="356" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="366.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,4</text>
  <rect x="379" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="389.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,5</text>
  <rect x="402" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="412.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,6</text>
  <rect x="241" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="251.5" y="102" text-anchor="middle" style="font-size: 8.5px">1,7</text>
  <rect x="264" y="90" width="21" height="17" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".35" stroke-dasharray="3 2"/>
  <rect x="287" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="297.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,0</text>
  <rect x="310" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="320.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,1</text>
  <rect x="333" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="343.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,2</text>
  <rect x="356" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="366.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,3</text>
  <rect x="379" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="389.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,4</text>
  <rect x="402" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="412.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,5</text>
  <rect x="241" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="251.5" y="121" text-anchor="middle" style="font-size: 8.5px">2,6</text>
  <rect x="264" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="274.5" y="121" text-anchor="middle" style="font-size: 8.5px">2,7</text>
  <rect x="287" y="109" width="21" height="17" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".35" stroke-dasharray="3 2"/>
  <rect x="310" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="320.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,0</text>
  <rect x="333" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="343.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,1</text>
  <rect x="356" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="366.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,2</text>
  <rect x="379" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="389.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,3</text>
  <rect x="402" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="412.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,4</text>
  <rect x="241" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="251.5" y="140" text-anchor="middle" style="font-size: 8.5px">3,5</text>
  <rect x="264" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="274.5" y="140" text-anchor="middle" style="font-size: 8.5px">3,6</text>
  <rect x="287" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="297.5" y="140" text-anchor="middle" style="font-size: 8.5px">3,7</text>
  <rect x="310" y="128" width="21" height="17" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".35" stroke-dasharray="3 2"/>
  <rect x="333" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="343.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,0</text>
  <rect x="356" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="366.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,1</text>
  <rect x="379" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="389.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,2</text>
  <rect x="402" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="412.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,3</text>
  <rect x="241" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="251.5" y="159" text-anchor="middle" style="font-size: 8.5px">4,4</text>
  <rect x="264" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="274.5" y="159" text-anchor="middle" style="font-size: 8.5px">4,5</text>
  <rect x="287" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="297.5" y="159" text-anchor="middle" style="font-size: 8.5px">4,6</text>
  <rect x="310" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="320.5" y="159" text-anchor="middle" style="font-size: 8.5px">4,7</text>
  <rect x="333" y="147" width="21" height="17" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".35" stroke-dasharray="3 2"/>
  <rect x="356" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="366.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,0</text>
  <rect x="379" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="389.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,1</text>
  <rect x="402" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="412.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,2</text>
  <rect x="241" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="251.5" y="178" text-anchor="middle" style="font-size: 8.5px">5,3</text>
  <rect x="264" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="274.5" y="178" text-anchor="middle" style="font-size: 8.5px">5,4</text>
  <rect x="287" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="297.5" y="178" text-anchor="middle" style="font-size: 8.5px">5,5</text>
  <rect x="310" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="320.5" y="178" text-anchor="middle" style="font-size: 8.5px">5,6</text>
  <rect x="333" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="343.5" y="178" text-anchor="middle" style="font-size: 8.5px">5,7</text>
  <rect x="356" y="166" width="21" height="17" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".35" stroke-dasharray="3 2"/>
  <rect x="379" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="389.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,0</text>
  <rect x="402" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="412.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,1</text>
  <rect x="241" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="251.5" y="197" text-anchor="middle" style="font-size: 8.5px">6,2</text>
  <rect x="264" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="274.5" y="197" text-anchor="middle" style="font-size: 8.5px">6,3</text>
  <rect x="287" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="297.5" y="197" text-anchor="middle" style="font-size: 8.5px">6,4</text>
  <rect x="310" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="320.5" y="197" text-anchor="middle" style="font-size: 8.5px">6,5</text>
  <rect x="333" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="343.5" y="197" text-anchor="middle" style="font-size: 8.5px">6,6</text>
  <rect x="356" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="366.5" y="197" text-anchor="middle" style="font-size: 8.5px">6,7</text>
  <rect x="379" y="185" width="21" height="17" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".35" stroke-dasharray="3 2"/>
  <rect x="402" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="412.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,0</text>
  <rect x="241" y="204" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="251.5" y="216" text-anchor="middle" style="font-size: 8.5px">7,1</text>
  <rect x="264" y="204" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="274.5" y="216" text-anchor="middle" style="font-size: 8.5px">7,2</text>
  <rect x="287" y="204" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="297.5" y="216" text-anchor="middle" style="font-size: 8.5px">7,3</text>
  <rect x="310" y="204" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="320.5" y="216" text-anchor="middle" style="font-size: 8.5px">7,4</text>
  <rect x="333" y="204" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="343.5" y="216" text-anchor="middle" style="font-size: 8.5px">7,5</text>
  <rect x="356" y="204" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="366.5" y="216" text-anchor="middle" style="font-size: 8.5px">7,6</text>
  <rect x="379" y="204" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="389.5" y="216" text-anchor="middle" style="font-size: 8.5px">7,7</text>
  <rect x="402" y="204" width="21" height="17" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".35" stroke-dasharray="3 2"/>
  <text class="s" x="240" y="245">错开到 8 个 bank：一拍</text>
  <text class="tb" x="444" y="22">swizzle：bank = c xor r</text>
  <text class="s" x="455.5" y="44" text-anchor="middle">0</text>
  <text class="s" x="478.5" y="44" text-anchor="middle">1</text>
  <text class="s" x="501.5" y="44" text-anchor="middle">2</text>
  <text class="s" x="524.5" y="44" text-anchor="middle">3</text>
  <text class="s" x="547.5" y="44" text-anchor="middle">4</text>
  <text class="s" x="570.5" y="44" text-anchor="middle">5</text>
  <text class="s" x="593.5" y="44" text-anchor="middle">6</text>
  <text class="s" x="616.5" y="44" text-anchor="middle">7</text>
  <rect x="445" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="455.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,0</text>
  <rect x="468" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="478.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,1</text>
  <rect x="491" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="501.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,2</text>
  <rect x="514" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="524.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,3</text>
  <rect x="537" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="547.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,4</text>
  <rect x="560" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="570.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,5</text>
  <rect x="583" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="593.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,6</text>
  <rect x="606" y="52" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="616.5" y="64" text-anchor="middle" style="font-size: 8.5px">0,7</text>
  <rect x="468" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="478.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,0</text>
  <rect x="445" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="455.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,1</text>
  <rect x="514" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="524.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,2</text>
  <rect x="491" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="501.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,3</text>
  <rect x="560" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="570.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,4</text>
  <rect x="537" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="547.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,5</text>
  <rect x="606" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="616.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,6</text>
  <rect x="583" y="71" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="593.5" y="83" text-anchor="middle" style="font-size: 8.5px">1,7</text>
  <rect x="491" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="501.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,0</text>
  <rect x="514" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="524.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,1</text>
  <rect x="445" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="455.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,2</text>
  <rect x="468" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="478.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,3</text>
  <rect x="583" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="593.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,4</text>
  <rect x="606" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="616.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,5</text>
  <rect x="537" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="547.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,6</text>
  <rect x="560" y="90" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="570.5" y="102" text-anchor="middle" style="font-size: 8.5px">2,7</text>
  <rect x="514" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="524.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,0</text>
  <rect x="491" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="501.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,1</text>
  <rect x="468" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="478.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,2</text>
  <rect x="445" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="455.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,3</text>
  <rect x="606" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="616.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,4</text>
  <rect x="583" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="593.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,5</text>
  <rect x="560" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="570.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,6</text>
  <rect x="537" y="109" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="547.5" y="121" text-anchor="middle" style="font-size: 8.5px">3,7</text>
  <rect x="537" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="547.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,0</text>
  <rect x="560" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="570.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,1</text>
  <rect x="583" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="593.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,2</text>
  <rect x="606" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="616.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,3</text>
  <rect x="445" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="455.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,4</text>
  <rect x="468" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="478.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,5</text>
  <rect x="491" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="501.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,6</text>
  <rect x="514" y="128" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="524.5" y="140" text-anchor="middle" style="font-size: 8.5px">4,7</text>
  <rect x="560" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="570.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,0</text>
  <rect x="537" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="547.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,1</text>
  <rect x="606" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="616.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,2</text>
  <rect x="583" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="593.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,3</text>
  <rect x="468" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="478.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,4</text>
  <rect x="445" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="455.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,5</text>
  <rect x="514" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="524.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,6</text>
  <rect x="491" y="147" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="501.5" y="159" text-anchor="middle" style="font-size: 8.5px">5,7</text>
  <rect x="583" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="593.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,0</text>
  <rect x="606" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="616.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,1</text>
  <rect x="537" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="547.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,2</text>
  <rect x="560" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="570.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,3</text>
  <rect x="491" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="501.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,4</text>
  <rect x="514" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="524.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,5</text>
  <rect x="445" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="455.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,6</text>
  <rect x="468" y="166" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="478.5" y="178" text-anchor="middle" style="font-size: 8.5px">6,7</text>
  <rect x="606" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 45%, transparent); stroke: var(--fig-hi)" stroke-width="0.8"/>
  <text class="s" x="616.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,0</text>
  <rect x="583" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="593.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,1</text>
  <rect x="560" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="570.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,2</text>
  <rect x="537" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="547.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,3</text>
  <rect x="514" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="524.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,4</text>
  <rect x="491" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="501.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,5</text>
  <rect x="468" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="478.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,6</text>
  <rect x="445" y="185" width="21" height="17" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 10%, transparent); stroke: var(--fig-1)" stroke-width="0.8"/>
  <text class="s" x="455.5" y="197" text-anchor="middle" style="font-size: 8.5px">7,7</text>
  <text class="s" x="444" y="245">错开到 8 个 bank：一拍</text>
</svg>
<figcaption><strong>图 6-1</strong> bank conflict 的两种解法（按 8 个 bank、8 × 8 的矩阵画，实际是 32 个 bank）。一格是一个 4 B 的数，标的是它在矩阵里的 (行, 列)；同一列的格子在同一个 bank，同一个 bank 一拍只能读一个地址。红色是 8 个 thread 同时要读的第 0 列。padding 多占一点空间；swizzle 不多占空间，按行读、按列读都不冲突。</figcaption>
</figure>

写 Triton 时这件事由编译器决定，shared memory 的布局写在 TTGIR 里：入门篇 2.4 节那个 fp32 的 matmul，A、B 的 slot 用的是 `swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1}>`，也就是没有打乱；走 Tensor core 的 `tl.dot` 编译器会换成别的布局。

最后一种方法把前面几件事都用上：让一个数从显存读进来以后，在片上被用很多次。

## 7 Tiling {#tiling}

matmul $C = A B$ 里，A 是 M × K，B 是 K × N。A 的每个数要和 B 的 N 列各乘一次，B 的每个数要和 A 的 M 行各乘一次。naive 的写法每个输出自己去显存读它要的那一行和那一列，A 的每个数就被读 N 次。tiling 把 C 切成 T × T 的 block，一个 program 把 A、B 对应的 block 读进 shared memory，给这个 block 里的 T × T 个输出共用（[图 7-1](#fig-o-2)）：A 的每个数只在 C 的同一行 block 里被读进来一次，一共 N / T 次，每次读进来以后在 shared memory 里被用 T 次。显存访问因此少了 T 倍（[表 7-1](#tab-7-1)）。

| | A 的每个数从显存读 | B 的每个数从显存读 | 整个 kernel 从显存读 |
|:--|--:|--:|--:|
| naive | N 次（512 次） | M 次（4096 次） | 32 GB |
| tiling（T × T 的 block） | N / T 次（8 次），每次在 shared memory 里用 T 次 | M / T 次（64 次），每次用 T 次 | 512 MB |
{#tab-7-1 caption="**表 7-1** naive 和 tiling：A、B 的每个数从显存读几次" note="括号里是 <a href=\"/blog/gpu-triton-intro/#triton-matmul\">入门篇 2.4 节</a>的形状：A 4096 × 2048、B 2048 × 512，fp32，T = 64。整个 kernel 的读量是 A 的 M × K 个数乘读的次数，加上 B 的 K × N 个数乘读的次数；512 MB 和入门篇 2.4 节 notes 里从 PTX 数出来的一致。"}

<figure id="fig-o-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 236" width="100%" role="img" aria-label="matmul 切 block：C 的 block (1, 1) 由 A 的第 1 个 row block 和 B 的第 1 个 column block 算出；沿 K 一轮读一对 block，实心是第 0 轮，斜线是第 1 到 3 轮">
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
  <defs><pattern id="fig-o-2-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" style="stroke: var(--fig-1)" stroke-opacity=".55" stroke-width="1.6"/></pattern></defs>
  <text class="tb" x="104.0" y="26" text-anchor="middle">C</text><text class="s" x="104.0" y="40" text-anchor="middle">M × N</text>
  <text class="tb" x="286.0" y="26" text-anchor="middle">A</text><text class="s" x="286.0" y="40" text-anchor="middle">M × K</text>
  <text class="tb" x="468.0" y="26" text-anchor="middle">B</text><text class="s" x="468.0" y="40" text-anchor="middle">K × N</text>
  <text class="tb" x="195" y="117.0" text-anchor="middle" style="font-size: 18px">=</text>
  <text class="tb" x="377" y="117.0" text-anchor="middle" style="font-size: 18px">@</text>
  <rect x="41.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="223.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="405.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="73.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="255.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="437.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <rect x="105.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="287.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="469.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="137.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="319.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="501.0" y="49.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="41.0" y="81.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="223.0" y="81.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/>
  <rect x="405.0" y="81.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="73.0" y="81.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-hi) 40%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <rect x="255.0" y="81.0" width="30" height="30" rx="3" style="fill: url(#fig-o-2-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1.4"/>
  <rect x="437.0" y="81.0" width="30" height="30" rx="3" style="fill: url(#fig-o-2-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1.4"/>
  <rect x="105.0" y="81.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="287.0" y="81.0" width="30" height="30" rx="3" style="fill: url(#fig-o-2-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1.4"/>
  <rect x="469.0" y="81.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="137.0" y="81.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="319.0" y="81.0" width="30" height="30" rx="3" style="fill: url(#fig-o-2-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1.4"/>
  <rect x="501.0" y="81.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="41.0" y="113.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="223.0" y="113.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="405.0" y="113.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="73.0" y="113.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="255.0" y="113.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="437.0" y="113.0" width="30" height="30" rx="3" style="fill: url(#fig-o-2-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1.4"/>
  <rect x="105.0" y="113.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="287.0" y="113.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="469.0" y="113.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="137.0" y="113.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="319.0" y="113.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="501.0" y="113.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="41.0" y="145.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="223.0" y="145.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="405.0" y="145.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="73.0" y="145.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="255.0" y="145.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="437.0" y="145.0" width="30" height="30" rx="3" style="fill: url(#fig-o-2-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1.4"/>
  <rect x="105.0" y="145.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="287.0" y="145.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="469.0" y="145.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="137.0" y="145.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="319.0" y="145.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <rect x="501.0" y="145.0" width="30" height="30" rx="3" style="fill: color-mix(in srgb, var(--fig-mute) 8%, transparent); stroke: var(--fig-mute)" stroke-width="1"/>
  <text class="s" x="88.0" y="100.0" text-anchor="middle">(1, 1)</text>
  <text class="s" x="238.0" y="192" text-anchor="middle">T=0</text>
  <text class="s" x="270.0" y="192" text-anchor="middle">T=1</text>
  <text class="s" x="302.0" y="192" text-anchor="middle">T=2</text>
  <text class="s" x="334.0" y="192" text-anchor="middle">T=3</text>
  <text class="s" x="538" y="68.0">T=0</text>
  <text class="s" x="538" y="100.0">T=1</text>
  <text class="s" x="538" y="132.0">T=2</text>
  <text class="s" x="538" y="164.0">T=3</text>
  <text class="s" x="104.0" y="192" text-anchor="middle">一个 program 负责的 C block</text>
  <rect x="40" y="210" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)" stroke-width="1.4"/><text class="s" x="60" y="220">第 0 轮读的一对 block</text>
  <rect x="200" y="210" width="14" height="12" rx="2" style="fill: url(#fig-o-2-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1.4"/><text class="s" x="220" y="220">之后几轮（沿 K）</text>
  <rect x="340" y="210" width="14" height="12" rx="2" style="fill: color-mix(in srgb, var(--fig-hi) 40%, transparent); stroke: var(--fig-hi)" stroke-width="1.4"/><text class="s" x="360" y="220">累加器：C block 一直留在 register 里</text>
</svg>
<figcaption><strong>图 7-1</strong> matmul 怎么切 block，一格是一个 block。C 的 block (1, 1) 只用到 A 的第 1 个 row block 和 B 的第 1 个 column block；沿 K 一轮读一对 block，乘完加到累加器上，实心是第 0 轮，斜线是之后几轮。A block 里的每个数被 C block 里同一行的 BN 个输出用到，B block 里的每个数被同一列的 BM 个输出用到。</figcaption>
</figure>

tile 越大，显存读得越少，但一个 program 要占更多 register 和 shared memory，怎么取见入门篇 [3.5 节](/blog/gpu-triton-intro/#tile)。在 Triton 里 BM、BN、BK 都得是 2 的幂，这是 `tl.arange` 的要求；`tl.dot` 还要求 BK 至少 16。tile 的大小还带来两种 quantization。一是矩阵的维度不是 tile 的整数倍时，边上的 tile 只有一部分是有用的数，叫 tile quantization；行长不是 burst 的整数倍时，一个 tile 的一行还会跨过两个 burst，读起来也慢。二是 tile 的个数不是全卡一批能放下的整数倍时，最后一批只有几个 tile 在跑，叫 wave quantization。CS336 lecture 5 引的例子是 A100 上的方阵乘：tile 取 256 × 128，N = 1792 时有 (1792 / 256) × (1792 / 128) = 7 × 14 = 98 个 tile，A100 的 108 个 SM 一批放得下；N = 1793 时变成 8 × 15 = 120 个，要分两批，FLOPS 突然掉下去（[thonking.ai](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications)）。入门篇 3.4 节的 matmul 也是这样：A 从 4032 行变成 4096 行，计算量只多 1.6%，耗时多 28%，因为 512 个 program 比全卡一批能放的 510 个多了 2 个（[入门篇 3.4 节](/blog/gpu-triton-intro/#sched-wave)）。

tiling 不只在 shared memory 这一层。每个 thread 从 shared memory 读进 A、B 的一小段，再和自己 register 里的一小块累加器一起算，一个数在 register 里也被用好几次；入门篇 [3.5 节](/blog/gpu-triton-intro/#tile)扫 matmul 的 block 大小时，最快的配置每个 thread 存 64 个累加器，就是这一层的 tiling。

---

这七种方法里，fusion、recomputation 和 tiling 省的是显存读写的次数，低精度省的是每次的字节，coalescing 和 bank conflict 让同样的读写更快，divergence 让 warp 少做无用功。[下一篇](/blog/gpu-training-analysis/)在 5090 上拆一步 Transformer 训练，会看到这些方法各自能省多少，以及哪里光靠它们不够。

---

## References {#references}

1. Stanford CS336. [Lecture 5: GPUs](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)。这一篇七种方法的顺序和第 1、3、4、5、7 节示意图的画法。
2. NVIDIA. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)。SIMT 和分支、global memory 的合并访问、shared memory 的 bank。
3. NVIDIA. [NVIDIA Tesla V100 GPU Architecture](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)（白皮书）。第 1 节的 independent thread scheduling、分支交替执行和 `__syncwarp()`，图 1-1 上、中两层的画法（Figure 20、22）。
4. NVIDIA. [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。谓词执行和 `ld.global.v4` 等访存指令。
5. NVIDIA. [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)（白皮书）。第 2 节 5090 上 bf16、fp8、fp4 Tensor core 的峰值。
6. Open Compute Project. [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。MX 格式：32 个数一个 E8M0 的 scale。
7. NVIDIA. [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)。NVFP4：16 个数一个 E4M3 的 scale，外加 per-tensor 的 fp32 scale。
8. Mishra, Stosic, Layton, Micikevicius. [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027). arXiv:2506.08027, 2025. 所有张量用 E4M3，每个张量沿行、沿列各量化一份。
9. DeepSeek-AI. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). arXiv:2412.19437, 2024. fp8 训练的 1 × 128、128 × 128 scale，累加提升到 fp32，反向时 tile 的转置。
10. NVIDIA. [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149). arXiv:2509.25149, 2025.
11. Ma et al. [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764). arXiv:2402.17764, 2024. BitNet b1.58。
12. Tseng, Chee, Sun, Kuleshov, De Sa. [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396). ICML 2024. 推理时 2 到 4 bit 的权重量化。
13. Chillee (Horace He). [Min-cut optimal(*) recomputation (i.e. activation checkpointing) with AOTAutograd](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467)。第 4 节三层 sigmoid 的例子。
14. Chen, Xu, Zhang, Guestrin. [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174). arXiv:1604.06174, 2016. activation checkpoint。
15. Dao, Fu, Ermon, Rudra, Ré. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). NeurIPS 2022. 反向时重算 P。
16. [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)。fused softmax、matmul 的写法。
17. NVIDIA (Dusan Stosic). [Training Neural Networks with Tensor Cores](https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf)（ECCV 2020 mixed precision tutorial）。表 2-1 的分类，Tensor core 的全精度乘积和 fp32 累加。
18. Horace He. [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)。第 3 节把 GPU 比作工厂、显存比作仓库的说法。
19. Horace He. [What Shapes Do Matrix Multiplications Like?](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications)。第 7 节对齐和 wave quantization 的例子。
20. NVIDIA. [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。tile quantization 和 wave quantization。
