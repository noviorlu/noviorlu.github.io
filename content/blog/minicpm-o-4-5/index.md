---
title: "MiniCPM-o 4.5 解读：全双工 Omni 模型的架构、时序与训练"
date: 2026-10-18
draft: false
math: true
description: "读 MiniCPM-o 4.5 的论文和 HuggingFace 推理代码：三块结构各自的维度和速率，Omni-Flow 怎么把音视频输入和语音输出排在一条 1 秒一格的时间轴上，TAIL 怎么让语音跟上时间，四个训练阶段各训哪些参数，以及代码里有、论文里没写的细节。"
tags: ["Omni", "全双工", "语音", "多模态", "LLM", "AI"]
categories: ["AI笔记"]
series: ["全双工 Omni 模型"]
series_order: 1
---

注：本文是 [MiniCPM-o 4.5](https://arxiv.org/abs/2604.27393)（OpenBMB，2026 年 4 月）的论文解读，外加对 [HuggingFace 上推理代码](https://huggingface.co/openbmb/MiniCPM-o-4_5)的阅读。论文里的数字直接引用，代码里的行为标明出自哪个函数；两者对不上的地方单独列在[第 5 节](#code)。我写这篇是为了给自己的全双工项目定方案，所以会多写一些「这个设计在哪里定死了、哪里还能改」。

**(Why)** 现在的多模态模型大多是轮流的：先听完，再说，说的时候不再看输入。MiniCPM-o 4.5 想做的是边看、边听、边说，并且能在没人提问时主动开口，比如看到场景变化时提醒一句。论文把这两点叫做 full-duplex 和 proactive。

**(What)** 它的做法是把对话摊在一条时间轴上：每 1 秒切一格（论文叫 chunk，代码里叫 unit），这一秒的视频帧和音频先写进 LLM，LLM 再决定这一秒是继续听还是说，说的话交给一个小的语音解码器转成语音。整套模型 9B，Qwen3-8B 做 backbone，INT4 下 11 GB 显存就能在 RTX 4090 上实时跑。

**(How)** 先在[第 1 节](#arch)看三块结构和它们的速率，再在[第 2 节](#omniflow)看 Omni-Flow 怎么排序列、推理时各模块的先后，[第 3 节](#tail)看语音怎么和时间对齐（TAIL），[第 4 节](#training)看四个训练阶段，[第 5 节](#code)是读代码看到的细节，最后在[第 6 节](#implications)整理哪些设计已经被它定下、哪些还开着。

---

## 1 架构：三块，两种速率 {#arch}

先来看整体结构。如[图 1-1](#fig-1-1)，MiniCPM-o 4.5 由三块组成：编码器把视频和音频变成 LLM 的输入嵌入；LLM 出文本，同时把每个文本 token 的隐状态交给语音解码器；语音解码器出离散的语音 token，最后由 flow matching 合成波形。

<figure id="fig-1-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 756" width="100%" role="img" aria-label="MiniCPM-o 4.5 架构（从上到下）：视频和音频分别经 SigLIP 加 Resampler、Whisper 加 MLP 和均值池化变成 4096 维嵌入，每秒按 unit 标记、视觉、音频的顺序沿序列拼成一段，接在 KV cache 后面交给 Qwen3-8B；LLM 出文本，同时把文本 token 和末层隐状态交给 0.3B 的语音 token 解码器，出 25 Hz 的 S3 token，再由 flow matching 合成 24 kHz 音频">
  <style>
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .tick { stroke: currentColor; stroke-opacity: .35; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-1-1-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker><marker id="fig-1-1-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-2)"/></marker><marker id="fig-1-1-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-3)"/></marker><marker id="fig-1-1-m3" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker></defs>
  <text class="lab2" x="20" y="18" text-anchor="start">箭头旁的数字是进入下一模块的 token 数</text>
  <rect x="90" y="32" width="140" height="40" rx="6" class="op"/>
  <text class="t" x="160.0" y="48.5" text-anchor="middle">视频帧</text>
  <text class="s" x="160.0" y="63.5" text-anchor="middle">1–5 FPS</text>
  <rect x="370" y="32" width="140" height="40" rx="6" class="op"/>
  <text class="t" x="440.0" y="48.5" text-anchor="middle">音频</text>
  <text class="s" x="440.0" y="63.5" text-anchor="middle">16 kHz</text>
  <rect x="55.0" y="96" width="210" height="56" rx="6" style="fill: color-mix(in srgb, var(--fig-2) 16%, var(--fig-bg)); stroke: var(--fig-2)" stroke-width="1.6"/>
  <text class="tb" x="160.0" y="113.0" text-anchor="middle">SigLIP ViT</text>
  <text class="s" x="160.0" y="128.0" text-anchor="middle">0.4B</text>
  <text class="s" x="160.0" y="143.0" text-anchor="middle">448², 1024 patch</text>
  <rect x="335.0" y="96" width="210" height="56" rx="6" style="fill: color-mix(in srgb, var(--fig-3) 16%, var(--fig-bg)); stroke: var(--fig-3)" stroke-width="1.6"/>
  <text class="tb" x="440.0" y="113.0" text-anchor="middle">Whisper-medium</text>
  <text class="s" x="440.0" y="128.0" text-anchor="middle">0.3B</text>
  <text class="s" x="440.0" y="143.0" text-anchor="middle">mel 100 Hz → 50 Hz</text>
  <rect x="55.0" y="176" width="210" height="50" rx="6" style="fill: color-mix(in srgb, var(--fig-2) 16%, var(--fig-bg)); stroke: var(--fig-2)" stroke-width="1.6"/>
  <text class="tb" x="160.0" y="197.5" text-anchor="middle">Resampler</text>
  <text class="s" x="160.0" y="212.5" text-anchor="middle">64 个 query</text>
  <rect x="335.0" y="176" width="210" height="56" rx="6" style="fill: color-mix(in srgb, var(--fig-3) 16%, var(--fig-bg)); stroke: var(--fig-3)" stroke-width="1.6"/>
  <text class="tb" x="440.0" y="193.0" text-anchor="middle">MLP + 均值池化</text>
  <text class="s" x="440.0" y="208.0" text-anchor="middle">1024→4096→4096</text>
  <text class="s" x="440.0" y="223.0" text-anchor="middle">5 帧取平均</text>
  <path d="M160,72 L160,94" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-1-m0)"/>
  <path d="M440,72 L440,94" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-1-m0)"/>
  <path d="M160,152 L160,174" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-1-m0)"/>
  <path d="M440,152 L440,174" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-1-m0)"/>
  <rect class="band" x="20" y="262" width="600" height="92" rx="8"/>
  <text class="ttl" x="32" y="322" text-anchor="start">按时间拼成一条序列：每秒一个 unit，沿序列方向拼接（不相加）</text>
  <rect x="32" y="270" width="48" height="30" rx="4" class="op"/>
  <text x="56.0" y="289" text-anchor="middle" font-size="10.5" fill="currentColor">&lt;unit&gt;</text>
  <rect x="84" y="270" width="54" height="30" rx="4" class="op"/>
  <text x="111.0" y="289" text-anchor="middle" font-size="10.5" fill="currentColor">&lt;image&gt;</text>
  <rect x="142" y="270" width="104" height="30" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 18%, var(--fig-bg)); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text x="194.0" y="289" text-anchor="middle" font-size="10.5" fill="currentColor">64 个视觉嵌入</text>
  <rect x="250" y="270" width="58" height="30" rx="4" class="op"/>
  <text x="279.0" y="289" text-anchor="middle" font-size="10.5" fill="currentColor">&lt;/image&gt;</text>
  <rect x="312" y="270" width="98" height="30" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 18%, var(--fig-bg)); stroke: var(--fig-3)" stroke-width="1.4"/>
  <text x="361.0" y="289" text-anchor="middle" font-size="10.5" fill="currentColor">10 个音频嵌入</text>
  <rect x="414" y="270" width="70" height="30" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 18%, var(--fig-bg)); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text x="449.0" y="289" text-anchor="middle" font-size="10.5" fill="currentColor">LLM 生成</text>
  <rect x="488" y="270" width="48" height="30" rx="4" class="op"/>
  <text x="512.0" y="289" text-anchor="middle" font-size="10.5" fill="currentColor">&lt;/unit&gt;</text>
  <rect x="540" y="270" width="62" height="30" rx="4" class="op" stroke-dasharray="4 3"/>
  <text x="571.0" y="289" text-anchor="middle" font-size="10.5" fill="currentColor">下一秒 …</text>
  <text class="s" x="32" y="342" text-anchor="start">尖括号是词表里的特殊 token（查词表嵌入）；视觉、音频嵌入直接写进对应位置，都是 4096 维</text>
  <path d="M160,226 L160,244 L194,244 L194,268" fill="none" style="stroke: var(--fig-2)" stroke-width="1.5" marker-end="url(#fig-1-1-m1)"/>
  <text class="s" x="168" y="240" text-anchor="start">64 / 帧</text>
  <path d="M440,232 L440,250 L361,250 L361,268" fill="none" style="stroke: var(--fig-3)" stroke-width="1.5" marker-end="url(#fig-1-1-m2)"/>
  <text class="s" x="448" y="246" text-anchor="start">10 / s</text>
  <path d="M320,354 L320,374" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-1-m0)"/>
  <rect x="50" y="376" width="500" height="64" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 16%, var(--fig-bg)); stroke: var(--fig-1)" stroke-width="1.6"/>
  <text class="tb" x="300.0" y="397.0" text-anchor="middle">Qwen3-8B</text>
  <text class="s" x="300.0" y="412.0" text-anchor="middle">36 层，d = 4096</text>
  <text class="s" x="300.0" y="427.0" text-anchor="middle">每秒 3–4 个文本 token</text>
  <rect x="60" y="482" width="140" height="40" rx="6" class="op"/>
  <text class="t" x="130.0" y="506.0" text-anchor="middle">文本输出</text>
  <path d="M130,440 L130,480" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-1-m0)"/>
  <rect x="380" y="482" width="170" height="44" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 16%, var(--fig-bg)); stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="tb" x="465.0" y="500.5" text-anchor="middle">Projector</text>
  <text class="s" x="465.0" y="515.5" text-anchor="middle">4096→768→768，L2</text>
  <path d="M465,440 L465,480" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.5" marker-end="url(#fig-1-1-m3)"/>
  <text class="s" x="473" y="466" text-anchor="start">末层 h</text>
  <rect x="260" y="562" width="290" height="60" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 16%, var(--fig-bg)); stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="tb" x="405.0" y="588.5" text-anchor="middle">语音 token 解码器</text>
  <text class="s" x="405.0" y="603.5" text-anchor="middle">0.3B，Llama 20 层，d 768</text>
  <path d="M300,440 L300,560" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-1-m0)"/>
  <text class="s" x="308" y="502" text-anchor="start">文本 token</text>
  <path d="M465,526 L465,560" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.5" marker-end="url(#fig-1-1-m3)"/>
  <text class="tb" x="473" y="548" text-anchor="start">+</text>
  <rect x="315" y="650" width="180" height="56" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 16%, var(--fig-bg)); stroke: var(--fig-mute)" stroke-width="1.6"/>
  <text class="tb" x="405.0" y="667.0" text-anchor="middle">flow matching</text>
  <text class="s" x="405.0" y="682.0" text-anchor="middle">+ HiFT 声码器</text>
  <text class="s" x="405.0" y="697.0" text-anchor="middle">参考音频定音色</text>
  <path d="M405,622 L405,648" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-1-m0)"/>
  <text class="s" x="413" y="640" text-anchor="start">S3，25 / s</text>
  <rect x="345" y="722" width="120" height="24" rx="6" class="op"/>
  <text class="t" x="405.0" y="738.0" text-anchor="middle">音频输出 24 kHz</text>
  <path d="M405,706 L405,720" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-1-1-m0)"/>
  <text class="s" x="60" y="580" text-anchor="start">语音 token 解码器的输入</text>
  <text class="s" x="60" y="596" text-anchor="start">= 文本 token 的嵌入</text>
  <text class="s" x="60" y="612" text-anchor="start">+ 投影后的 h</text>
  <text class="s" x="60" y="628" text-anchor="start">（逐元素相加）</text>
</svg>
<figcaption><strong>图 1-1</strong> MiniCPM-o 4.5 的三块：编码器、LLM、语音解码器；中间一排是音视频嵌入怎么拼进 LLM 的输入序列。数字取自论文表 13 和 HuggingFace 上的推理代码；音频 50 Hz 到 10 Hz 那一步，代码里是先过 MLP 再做 5 帧均值池化。</figcaption>
</figure>

| 模块 | 结构 | 参数量 | 速率 |
|:--|:--|--:|:--|
| 视觉编码器 | SigLIP ViT，27 层，d = 1152，patch 14 | 417.8M | 每帧 1024 个 patch |
| Resampler | 一层 cross-attention，64 个 query | 88.9M | 每帧压到 64 个 token |
| 音频编码器 | Whisper-medium encoder，24 层，d = 1024 | 307.2M | 50 Hz |
| 音频投影 | MLP 1024→4096→4096 + 5 帧均值池化 | 21.0M | 10 Hz |
| LLM | Qwen3-8B，36 层，d = 4096，GQA 32/8 | 8189.2M | 每秒 3–4 个文本 token |
| 隐状态投影 | MLP 4096→768→768，再 L2 归一化 | 10.5M | 每个文本 token 一次 |
| 语音 token 解码器 | Llama，20 层，d = 768，文本嵌入表 152064 | 116.8M + 188.8M | S3 token，25 Hz |
| 波形合成 | flow matching + 声码器（代码从 `stepaudio2` 导入） | 不计入 9B | 24 kHz |
{#tab-1-1 caption="**表 1-1** MiniCPM-o 4.5 各模块" note="参数量来自论文表 13，合计 9.34B 可训练参数（bf16）。结构细节和速率来自 HuggingFace 上的 `config.json` 和 `modeling_minicpmo.py`。"}

几个值得注意的地方：

- **视觉**：全双工模式下图像最大 448×448，只取原图不切片，所以每帧就是 64 个 token（加上首尾两个标记）。非全双工模式可以切片到 2240×2240。
- **音频**：1 秒音频先变成 100 帧 log-mel，Whisper 的第二个卷积步长为 2，出 50 帧；每帧过 MLP 映射到 4096 维，再每 5 帧取平均，变成 10 个 token。论文说的「两层 MLP 做 5 倍时间压缩」，代码里压缩这一步是均值池化，不是学出来的。
- **音视频怎么合在一起**：进 LLM 之前，视觉和音频不在嵌入层面相加或做任何融合，只是按时间沿序列拼接（图 1-1 中间那一排）。每秒一个 unit：先是 `<unit>`、`<image>` 这类特殊 token，再是 64 个视觉嵌入、10 个音频嵌入，然后才是 LLM 这一秒生成的内容。两种模态之间的交互全部交给 LLM 的 attention。序列的细节见[第 2 节](#sequence)。
- **LLM 只出文本**：论文的理由是，如果让 backbone 直接出 25 Hz 的语音 token，每秒的解码步数要多六到八倍，语言能力也会掉。所以每秒只出 3–4 个文本 token，语音交给小解码器。
- **语音解码器的输入**：每个文本 token 对应一个输入向量，等于解码器自己的文本嵌入加上 LLM 末层隐状态经投影、L2 归一化之后的向量，两者逐元素相加。论文的说法是 LLM 已经把韵律决策编码进隐状态，小解码器只管发音。
- **音色**：语音 token 解码器不接收说话人向量。音色由 flow matching 那一步的参考音频决定；参考音频同时也作为系统提示的一部分送进 LLM。

---

## 2 Omni-Flow：把对话摊在时间轴上 {#omniflow}

### 2.1 序列怎么排 {#sequence}

Omni-Flow 把交互看成三条按时间对齐的流：环境视频、环境音频（用户说话也在里面），以及模型自己的输出。用户的话不再是一个特殊的 user 角色，只是环境音频的一部分。第 $k$ 个 chunk 的视觉 token $v_k$、音频 token $a_k$ 和输出 token $o_k$ 拼成一组 $g_k = [v_k; a_k; o_k]$，所有组按时间顺序接成一条序列，交给普通的 causal LM。不该说话时，$o_k$ 只有一个 listen token。

落到代码里，一个 unit 的样子如[图 2-1](#fig-2-1)：

<figure id="fig-2-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 150" width="100%" role="img" aria-label="全双工模式下一个 1 秒 unit 的 token 顺序：unit 开始标记、图像 64 个嵌入、10 个音频嵌入，然后 LLM 采样 listen 或 speak，speak 后跟文本 token，最后 chunk_eos 和 unit 结束标记">
  <style>
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .tick { stroke: currentColor; stroke-opacity: .35; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <rect x="8" y="44" width="50" height="34" rx="4" class="op"/>
  <text x="33.0" y="65" text-anchor="middle" font-size="10.5" fill="currentColor">&lt;unit&gt;</text>
  <rect x="60" y="44" width="54" height="34" rx="4" class="op"/>
  <text x="87.0" y="65" text-anchor="middle" font-size="10.5" fill="currentColor">&lt;image&gt;</text>
  <rect x="116" y="44" width="96" height="34" rx="4" style="fill: color-mix(in srgb, var(--fig-2) 18%, var(--fig-bg)); stroke: var(--fig-2)" stroke-width="1.4"/>
  <text x="164.0" y="65" text-anchor="middle" font-size="10.5" fill="currentColor">64 个视觉嵌入</text>
  <rect x="214" y="44" width="58" height="34" rx="4" class="op"/>
  <text x="243.0" y="65" text-anchor="middle" font-size="10.5" fill="currentColor">&lt;/image&gt;</text>
  <rect x="274" y="44" width="92" height="34" rx="4" style="fill: color-mix(in srgb, var(--fig-3) 18%, var(--fig-bg)); stroke: var(--fig-3)" stroke-width="1.4"/>
  <text x="320.0" y="65" text-anchor="middle" font-size="10.5" fill="currentColor">10 个音频嵌入</text>
  <rect x="368" y="44" width="64" height="34" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 18%, var(--fig-bg)); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text x="400.0" y="65" text-anchor="middle" font-size="10.5" fill="currentColor">&lt;|speak|&gt;</text>
  <rect x="434" y="44" width="70" height="34" rx="4" style="fill: color-mix(in srgb, var(--fig-1) 18%, var(--fig-bg)); stroke: var(--fig-1)" stroke-width="1.4"/>
  <text x="469.0" y="65" text-anchor="middle" font-size="10.5" fill="currentColor">文本 ≤ 19</text>
  <rect x="506" y="44" width="80" height="34" rx="4" style="fill: color-mix(in srgb, var(--fig-hi) 18%, var(--fig-bg)); stroke: var(--fig-hi)" stroke-width="1.4"/>
  <text x="546.0" y="65" text-anchor="middle" font-size="10.5" fill="currentColor">&lt;|chunk_eos|&gt;</text>
  <rect x="588" y="44" width="50" height="34" rx="4" class="op"/>
  <text x="613.0" y="65" text-anchor="middle" font-size="10.5" fill="currentColor">&lt;/unit&gt;</text>
  <path d="M8,92 L8,100 L364,100 L364,92" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5"/>
  <text class="s" x="186" y="116" text-anchor="middle">prefill：音视频按到达顺序写进 KV cache</text>
  <path d="M366,92 L366,100 L632,100 L632,92" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.5"/>
  <text class="s" x="499" y="116" text-anchor="middle">LLM 自回归生成，遇到终止 token 停</text>
  <text class="lab2" x="8" y="26" text-anchor="start">不说话时 &lt;|speak|&gt; 处是 &lt;|listen|&gt;，后面没有文本；一轮话说完先出 &lt;|turn_eos|&gt; 才能 listen</text>
  <text class="lab2" x="8" y="140" text-anchor="start">每个 unit 对应真实时间 1 秒；视觉帧数由调用方决定，图中按 1 帧画</text>
</svg>
<figcaption><strong>图 2-1</strong> 全双工模式下一个 unit（1 秒）的 token 顺序，取自推理代码 <code>MiniCPMODuplex.streaming_prefill</code> 和 <code>streaming_generate</code>。</figcaption>
</figure>

- 系统提示之后不再有 `<|im_start|>user` 这类角色标记，后面全是 unit。
- 每个 unit 先写入这一秒的视觉和音频嵌入（prefill），然后 LLM 采样第一个 token：`<|listen|>` 或 `<|speak|>`。说话时接着出文本，遇到 `<|chunk_eos|>`、`<|listen|>` 或 `<|chunk_tts_eos|>` 就结束这个 unit。
- 一个 unit 最多 20 步（`max_new_speak_tokens_per_chunk`），最后一步由代码直接写入 `<|chunk_eos|>`，所以最多采样 19 个 token。
- 一轮话说完要先出 `<|turn_eos|>`。代码里有一条硬规则：一轮话没结束时如果采样到 `<|listen|>`，会被替换成 `<|tts_bos|>`，也就是说话说到一半不允许直接停下来听。

### 2.2 推理时的先后 {#timing}

再来看时间上谁先谁后。如[图 2-2](#fig-2-2)，每一秒的音视频要攒满才能写进 LLM；写完后 LLM 生成这一 unit 的文本，语音解码器接着生成 25 个 S3 token，合成的约 1 秒语音在下一秒播放。

<figure id="fig-2-2" class="fg-fig">
<svg class="fg" viewBox="0 0 640 250" width="100%" role="img" aria-label="全双工推理的时序：第 k 秒的音视频攒满后才 prefill，LLM 生成这一 unit 的文本，语音解码器紧接着出 25 个 S3 token，合成出的约 1 秒语音在下一秒播放">
  <style>
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .tick { stroke: currentColor; stroke-opacity: .35; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <defs><marker id="fig-2-2-m0" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-3)"/></marker><marker id="fig-2-2-m1" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill: var(--fig-hi)"/></marker><marker id="fig-2-2-m2" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker></defs>
  <rect class="band" x="4" y="12" width="632" height="44" rx="6"/>
  <text class="ttl" x="14" y="38" text-anchor="start">输入</text>
  <rect class="band" x="4" y="70" width="632" height="44" rx="6"/>
  <text class="ttl" x="14" y="96" text-anchor="start">LLM</text>
  <rect class="band" x="4" y="128" width="632" height="44" rx="6"/>
  <text class="ttl" x="14" y="154" text-anchor="start">语音解码器</text>
  <rect class="band" x="4" y="186" width="632" height="44" rx="6"/>
  <text class="ttl" x="14" y="212" text-anchor="start">播放</text>
  <line class="tick" x1="96" y1="10" x2="96" y2="236" stroke-dasharray="3 3"/>
  <text class="s" x="96" y="246" text-anchor="middle">0 s</text>
  <line class="tick" x1="246" y1="10" x2="246" y2="236" stroke-dasharray="3 3"/>
  <text class="s" x="246" y="246" text-anchor="middle">1 s</text>
  <line class="tick" x1="396" y1="10" x2="396" y2="236" stroke-dasharray="3 3"/>
  <text class="s" x="396" y="246" text-anchor="middle">2 s</text>
  <line class="tick" x1="546" y1="10" x2="546" y2="236" stroke-dasharray="3 3"/>
  <text class="s" x="546" y="246" text-anchor="middle">3 s</text>
  <rect x="100" y="20" width="142" height="28" rx="6" style="fill: color-mix(in srgb, var(--fig-3) 16%, var(--fig-bg)); stroke: var(--fig-3)" stroke-width="1.6"/>
  <text class="t" x="171.0" y="38.0" text-anchor="middle">第 1 秒音视频</text>
  <rect x="250" y="20" width="142" height="28" rx="6" style="fill: color-mix(in srgb, var(--fig-3) 16%, var(--fig-bg)); stroke: var(--fig-3)" stroke-width="1.6"/>
  <text class="t" x="321.0" y="38.0" text-anchor="middle">第 2 秒音视频</text>
  <rect x="400" y="20" width="142" height="28" rx="6" style="fill: color-mix(in srgb, var(--fig-3) 16%, var(--fig-bg)); stroke: var(--fig-3)" stroke-width="1.6"/>
  <text class="t" x="471.0" y="38.0" text-anchor="middle">第 3 秒音视频</text>
  <rect x="248" y="78" width="92" height="28" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 16%, var(--fig-bg)); stroke: var(--fig-1)" stroke-width="1.6"/>
  <text class="t" x="294.0" y="96.0" text-anchor="middle">listen</text>
  <path d="M240,50 L266,76" fill="none" style="stroke: var(--fig-3)" stroke-width="1.5" marker-end="url(#fig-2-2-m0)"/>
  <rect x="398" y="78" width="92" height="28" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 16%, var(--fig-bg)); stroke: var(--fig-1)" stroke-width="1.6"/>
  <text class="t" x="444.0" y="96.0" text-anchor="middle">speak + 文本</text>
  <path d="M390,50 L416,76" fill="none" style="stroke: var(--fig-3)" stroke-width="1.5" marker-end="url(#fig-2-2-m0)"/>
  <rect x="476" y="136" width="66" height="28" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 16%, var(--fig-bg)); stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="t" x="509.0" y="154.0" text-anchor="middle">25 个 S3</text>
  <path d="M490,106 L500,134" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.5" marker-end="url(#fig-2-2-m1)"/>
  <rect x="540" y="194" width="92" height="28" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 16%, var(--fig-bg)); stroke: var(--fig-mute)" stroke-width="1.6"/>
  <text class="t" x="586.0" y="212.0" text-anchor="middle">约 1 秒语音</text>
  <path d="M536,164 L552,192" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-end="url(#fig-2-2-m2)"/>
  <text class="s" x="100" y="146" text-anchor="start">和 LLM 在同一次调用里先后执行，</text>
  <text class="s" x="100" y="160" text-anchor="start">等这一 unit 的文本出完才开始</text>
</svg>
<figcaption><strong>图 2-2</strong> 推理时序示意：第 1 秒 listen，第 2 秒开始说话。每秒的输入攒满后才进 LLM，语音在下一秒播出，所以从收到输入到播出语音至少隔一个 chunk。</figcaption>
</figure>

这里有两点和直觉不太一样：

1. **语音解码器是同步的**。它在 `streaming_generate` 里排在 LLM 循环之后执行，拿到这一 unit 的全部文本和隐状态才开始，不和 LLM 并行。每个 unit 固定出 25 个语音 token（一轮开头和结尾可以少出），正好 1 秒。
2. **反应粒度是 1 秒**。是否开口、是否被打断，都只在 unit 边界上由 LLM 决定。代码里的打断机制是外部的 `break_event`：被置位后这一 unit 直接返回静音并结束这一轮，模型自身没有 `<|interrupt|>` 的用法（这个 token 在词表里，但全双工路径没用到）。

### 2.3 为什么是 1 秒 {#ablation}

chunk 取多长，论文做了消融（[表 2-1](#tab-2-1)），同时比较了两个设计选择：组和组之间是否放显式的边界 token，以及「说不说」和「说什么」是分两步预测（LS：先出 listen/speak 控制 token），还是放在同一个输出空间里预测（LT：直接出 listen 或文本 token）。

| chunk | 边界 | 控制 | AdvBench | AlpacaEval | IFEval | SDQA | MMLU |
|:--|:--|:--|--:|--:|--:|--:|--:|
| 1.0 s | 显式 | LS | **0.98** | 3.56 | **0.29** | **0.36** | **0.65** |
| 1.0 s | 显式 | LT | 0.92 | **3.60** | 0.24 | 0.35 | 0.56 |
| 1.0 s | 隐式 | LT | 0.96 | 3.31 | 0.22 | 0.28 | 0.45 |
| 0.2 s | 显式 | LS | 0.81 | 1.22 | 0.10 | 0.09 | 0.45 |
| 0.1 s | 显式 | LS | 0.67 | 2.40 | 0.10 | 0.13 | 0.32 |
{#tab-2-1 caption="**表 2-1** 全双工设计选择的消融（论文表 1）" note="数字照抄论文。论文没有说明这组消融的训练规模，MMLU 0.65 远低于最终模型的 77.0，应该是缩小规模的实验。"}

论文的结论有三条：1 秒最好，chunk 太短时每格里的信息不够做稳定决策；显式边界更好；LS 好于 LT，「要不要说」和「说什么」应该分开预测。

读这张表时我会注意它的前提：在这个设计里，每个 chunk 都要由 8B 的 LLM 完成「读入、决策、出文本」整个过程，chunk 越短，LLM 每格能出的文本越少。所以「1 秒最好」说的是 LLM 自己按 chunk 决策和生成时的最优值，并不能直接推出一个更快的独立模块也只能 1 秒一格。另外 0.2 s 和 0.1 s 两行并不单调（AlpacaEval 1.22 和 2.40），论文没报告种子数，这种差异里可能有不少噪声。

---

## 3 TAIL：让语音跟上时间 {#tail}

**(Difficulty)** 文本和语音的时长对不上。LLM 在 1 秒里出的文本，读出来可能要 2 秒；如果一直这样，播放会越来越落后于模型当前的状态，用户听到的是模型很早以前想说的话。已有的流式 TTS 要么先出一大段文本再合成，要么按固定的文本/语音 token 比例交织，都没有把语音对齐到真实时间。

**(Method)** TAIL（Time-Aligned Interleaving）在构造训练数据时就按时间分组，如[图 3-1](#fig-3-1)：记下每个文本 token 在语音里的开始和结束时间，开始时间落在 $[(k-1)t,\ kt)$ 的文本 token 和它对应的语音 token 一起归到第 $k$ 个 chunk。这样每个 chunk 的文本数不固定，随累计的播放进度变化：前面已经慢了，下一个 chunk 就少出几个文本 token。

<figure id="fig-3-1" class="fg-fig">
<svg class="fg" viewBox="0 0 640 232" width="100%" role="img" aria-label="TAIL 的分组方式：每个文本 token 按它在语音里的开始时间归到对应的 1 秒 chunk；跨过边界的词，读音落在下一 chunk；每个 chunk 末尾几个 token 的语音推迟到下一 chunk 生成，给发音留一点下文">
  <style>
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .tick { stroke: currentColor; stroke-opacity: .35; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>
  <line class="tick" x1="70" y1="22" x2="70" y2="206" stroke-dasharray="3 3"/>
  <text class="s" x="70" y="220" text-anchor="middle">0 s</text>
  <line class="tick" x1="250" y1="22" x2="250" y2="206" stroke-dasharray="3 3"/>
  <text class="s" x="250" y="220" text-anchor="middle">1 s</text>
  <line class="tick" x1="430" y1="22" x2="430" y2="206" stroke-dasharray="3 3"/>
  <text class="s" x="430" y="220" text-anchor="middle">2 s</text>
  <line class="tick" x1="610" y1="22" x2="610" y2="206" stroke-dasharray="3 3"/>
  <text class="s" x="610" y="220" text-anchor="middle">3 s</text>
  <text class="lab2" x="8" y="16" text-anchor="start">示意：一句话的读音沿时间铺开，竖线是 chunk 边界</text>
  <text class="ttl" x="8" y="56" text-anchor="start">语音</text>
  <rect x="79.0" y="38" width="34.0" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 16%, var(--fig-bg)); stroke: var(--fig-mute)" stroke-width="1.6"/>
  <text class="t" x="96.0" y="55.0" text-anchor="middle">The</text>
  <rect x="120.4" y="38" width="77.19999999999999" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 16%, var(--fig-bg)); stroke: var(--fig-mute)" stroke-width="1.6"/>
  <text class="t" x="159.0" y="55.0" text-anchor="middle">apple</text>
  <rect x="206.8" y="38" width="26.800000000000004" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 16%, var(--fig-bg)); stroke: var(--fig-mute)" stroke-width="1.6"/>
  <text class="t" x="220.20000000000002" y="55.0" text-anchor="middle">on</text>
  <rect x="241.0" y="38" width="28.600000000000026" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 16%, var(--fig-bg)); stroke: var(--fig-mute)" stroke-width="1.6"/>
  <text class="t" x="255.3" y="55.0" text-anchor="middle">the</text>
  <rect x="277.0" y="38" width="79.00000000000003" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 16%, var(--fig-bg)); stroke: var(--fig-mute)" stroke-width="1.6"/>
  <text class="t" x="316.5" y="55.0" text-anchor="middle">table</text>
  <rect x="368.8" y="38" width="62.80000000000001" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 16%, var(--fig-bg)); stroke: var(--fig-mute)" stroke-width="1.6"/>
  <text class="t" x="400.20000000000005" y="55.0" text-anchor="middle">looks</text>
  <rect x="440.8" y="38" width="55.59999999999997" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 16%, var(--fig-bg)); stroke: var(--fig-mute)" stroke-width="1.6"/>
  <text class="t" x="468.6" y="55.0" text-anchor="middle">very</text>
  <rect x="505.59999999999997" y="38" width="93.40000000000005" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-mute) 16%, var(--fig-bg)); stroke: var(--fig-mute)" stroke-width="1.6"/>
  <text class="t" x="552.3" y="55.0" text-anchor="middle">fresh</text>
  <text class="ttl" x="8" y="112" text-anchor="start">文本</text>
  <rect x="76" y="94" width="38" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 16%, var(--fig-bg)); stroke: var(--fig-1)" stroke-width="1.6"/>
  <text class="t" x="95.0" y="111.0" text-anchor="middle">The</text>
  <path d="M95,93 L97.0,66" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.1" stroke-dasharray="5 3"/>
  <rect x="118" y="94" width="38" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 16%, var(--fig-bg)); stroke: var(--fig-1)" stroke-width="1.6"/>
  <text class="t" x="137.0" y="111.0" text-anchor="middle">apple</text>
  <path d="M137,93 L160.0,66" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.1" stroke-dasharray="5 3"/>
  <rect x="160" y="94" width="38" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 16%, var(--fig-bg)); stroke: var(--fig-1)" stroke-width="1.6"/>
  <text class="t" x="179.0" y="111.0" text-anchor="middle">on</text>
  <path d="M179,93 L221.2,66" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.1" stroke-dasharray="5 3"/>
  <rect x="202" y="94" width="38" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 16%, var(--fig-bg)); stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="t" x="221.0" y="111.0" text-anchor="middle">the</text>
  <path d="M221,93 L256.3,66" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.1" stroke-dasharray="5 3"/>
  <rect x="256" y="94" width="38" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 16%, var(--fig-bg)); stroke: var(--fig-1)" stroke-width="1.6"/>
  <text class="t" x="275.0" y="111.0" text-anchor="middle">table</text>
  <path d="M275,93 L317.5,66" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.1" stroke-dasharray="5 3"/>
  <rect x="298" y="94" width="38" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 16%, var(--fig-bg)); stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="t" x="317.0" y="111.0" text-anchor="middle">looks</text>
  <path d="M317,93 L401.2,66" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.1" stroke-dasharray="5 3"/>
  <rect x="436" y="94" width="38" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-1) 16%, var(--fig-bg)); stroke: var(--fig-1)" stroke-width="1.6"/>
  <text class="t" x="455.0" y="111.0" text-anchor="middle">very</text>
  <path d="M455,93 L469.6,66" fill="none" stroke="currentColor" stroke-opacity=".6" stroke-width="1.1" stroke-dasharray="5 3"/>
  <rect x="478" y="94" width="38" height="26" rx="6" style="fill: color-mix(in srgb, var(--fig-hi) 16%, var(--fig-bg)); stroke: var(--fig-hi)" stroke-width="1.6"/>
  <text class="t" x="497.0" y="111.0" text-anchor="middle">fresh</text>
  <path d="M497,93 L553.3,66" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.1" stroke-dasharray="5 3"/>
  <text class="ttl" x="8" y="160" text-anchor="start">规则</text>
  <text class="s" x="70" y="150" text-anchor="start">开始时间落在 [(k−1)t, kt) 的文本 token 归第 k 个 chunk（t = 1 s），对应的语音 token 一起放进这个 chunk</text>
  <text class="s" x="70" y="166" text-anchor="start">红色是 chunk 末尾的 token：它的语音推迟到下一 chunk 生成（look-ahead），读 the 时已经知道后面是 table</text>
  <text class="s" x="70" y="182" text-anchor="start">前面若已慢了，下一 chunk 少出几个文本 token，让语音追上；每 chunk 的文本数随累计进度变化</text>
</svg>
<figcaption><strong>图 3-1</strong> TAIL 的训练数据怎么分组（示意，词和时间是编的）。规则来自论文 3.4 节；推理代码里找不到显式的推迟逻辑，我的理解是解码器从这种数据排布里学会了推迟。</figcaption>
</figure>

此外还有一个有限的 look-ahead：每个 chunk 末尾几个文本 token 的语音推迟到下一个 chunk 再生成。论文举的例子是英文的 the，读 "the apple" 和 "the car" 时发音不同，要看到下一个词才知道怎么读。

代价体现在发音准确率上（[表 3-1](#tab-3-1)）：

| 模式 | ZH CER ↓ | ZH SIM ↑ | EN WER ↓ | EN SIM ↑ |
|:--|--:|--:|--:|--:|
| 不交织 | 1.44 | 74.1 | 2.70 | 64.9 |
| 固定文本数交织 | **0.86** | **74.5** | **2.38** | 64.9 |
| TAIL（按时间） | 1.04 | 74.1 | 3.93 | **65.1** |
{#tab-3-1 caption="**表 3-1** 三种语音生成模式（Seed TTS 测试集，论文表 10）" note="论文表 5 里 MiniCPM-o 4.5 的 TTS 成绩（0.86 / 74.5 / 2.38 / 64.9）和这里「固定文本数交织」一行完全相同，也就是说标题级的 TTS 数字不是全双工模式下的。"}

TAIL 的英文 WER 从 2.38 涨到 3.93。论文把这个当成为了时间对齐付出的代价。

读代码时我没找到推迟的逻辑：全双工推理里，第 $k$ 个 unit 的文本只作为第 $k$ 个 unit 语音的条件，解码器每次固定出 25 个 token，KV cache 在一轮话里一直保留。我的理解是推迟完全由训练数据的排布教给了解码器：训练时 chunk 末尾那个词的语音出现在下一个 chunk 里，解码器就学会了「这个词先不读完」。这一点代码证明不了，只是和代码相容。

---

## 4 训练：四个阶段 {#training}

训练从 MiniCPM-V 4.5 的预训练 checkpoint 开始（视觉部分和 Qwen3-8B 都已经对齐过），再加上预训练好的 Whisper 和随机初始化的语音相关模块。

| 阶段 | 训练哪些参数 | 数据 | 目的 |
|:--|:--|:--|:--|
| 1 语音预训练 | 只训新加的：音频投影、LLM 到语音的投影、语音 token 解码器；其余冻结 | 论文没写，从目的看是语音数据（4.1 节：百万小时级语音处理成的 ASR、零样本 TTS、多说话人对话） | 把 Whisper 特征对齐到 LLM 隐空间；让解码器学会从隐状态出语音 token |
| 2 联合预训练 | 全部解冻 | 视觉语言、语音、全模态数据按固定比例混合，含按时间对齐的全双工和主动交互数据 | 统一的跨模态表示，获得实时交互能力 |
| 3 联合 SFT | 论文没单独说明 | 先大规模指令数据，再人工标注的高质量数据；帧分辨率随机 0.2–0.4 MP，帧率随机 1–5 FPS | 指令跟随和交互质量 |
| 4 强化学习 | 论文没单独说明 | GRPO（答案正确性 + 格式 + 长度奖励）；RLAIF-V 降幻觉 | 推理能力、少幻觉 |
{#tab-4-1 caption="**表 4-1** MiniCPM-o 4.5 的训练阶段（论文第 5 节）" note="论文没有给各阶段的步数、学习率和数据量。"}

几个细节：

- **梯度怎么流**：论文强调从编码器、LLM 到语音 token 解码器约 9B 参数是「token 级可微连接」的，可以端到端反传。第 2 阶段全部解冻、用统一的 next-token 目标训练，所以语音 token 的损失会经过隐状态投影流进 LLM，再流进 Whisper 编码器；编码器同时收到文本和语音两路梯度。flow matching 不在这 9B 里，论文没说它是否另外训过。
- **数据并行的配比**：第 2 阶段把不同的模态组合分配给不同的 data-parallel rank，保证每一步的数据比例固定。
- **全双工数据**：一部分是网络音视频，过滤掉单人独白和音画无关的片段，再用 OCR 去字幕、检测 talking head、按 ASR 转写过滤；另一部分是人工构造的场景和标注（连续描述、主动提醒）。每条样本里视觉、音频、输出文本和输出语音都带时间戳。
- **语音对话数据**：先用文本 LLM 生成口语化的对话，再请配音演员在录音棚里录一部分，要求用对话的方式说而不是照稿念。
- **长度奖励**：在 Kimi-K1.5 的长度奖励上做了平滑，前 480 步不加。消融里 thinking 模式回答长度减少 35.3%，平均分从 73.5 升到 74.3。
- **文本能力**：和 Qwen3-8B-Instruct 比，8 项文本基准平均 82.1 对 81.6，但 MMLU 从 81.7 降到 77.0，Math500 从 84.0 降到 77.0。

训练代码没有公开。HuggingFace 仓库里只有推理代码，`config.json` 里有 `long_weight`、`short_weight`、`aug_loss_weight` 等看起来和语音损失加权有关的字段，但代码里没有任何地方用到它们，具体怎么加权无从得知。

---

## 5 代码里看到的 {#code}

下面这些是读 `modeling_minicpmo.py`、`utils.py`、`processing_minicpmo.py` 得到的，论文没有写或写得不同。函数名对应 HuggingFace 上 2026 年 10 月的版本。

1. **音频降采样是均值池化**：`get_audio_embedding_streaming` 里顺序是 Whisper → `audio_projection_layer`（MLP）→ `AvgPool1d(5)`。论文写的是 MLP 做 5 倍压缩。
2. **流式 Whisper 用 KV cache**：每秒的 50 帧在秒内双向 attention，并通过 cache 看到之前所有帧；cache 超过 1500 个位置（30 秒）就清空重来。卷积边界两侧各多取 2 帧再裁掉。
3. **隐状态取末层**：语音解码器用的是 LLM 最后一层（过了 final norm、直接接 lm_head 的那个）隐状态。每个 unit 第一个 token（listen/speak 决策）的隐状态不交给解码器。
4. **解码器条件是相加**：`_convert_results_to_tts_input` 里，`emb_text(token) + normalize(projector(h))`，末尾接 `audio_bos`。L2 归一化在全双工路径里是写死的。
5. **语音解码器每 unit 出 25 个 token，KV cache 保留到一轮结束**，所以它能看到这一轮之前所有 unit 的文本条件和自己出过的语音 token。
6. **全双工下语音 token 的采样**：温度 0.8，最近 16 个 token 上的重复惩罚 1.05。top-p、top-k 在 `generate_chunk` 里实际上没有生效。
7. **话说到一半不能 listen**：采样到 `<|listen|>` 而本轮没结束时，替换成 `<|tts_bos|>`。
8. **长会话**：LLM 侧有两种滑窗模式（按 unit 丢最旧的，或把旧 unit 的文本摘成一段 "previous:" 放回系统提示后面），默认关闭。
9. **waveform 合成**：每次取 25 + 3 个 token（3 个是 flow matching 的 look-ahead）合成，再前进 25 个；每轮开头先垫 3 个静音 token。

---

## 6 哪些定下了，哪些还开着 {#implications}

最后整理一下，对想在 MiniCPM-o 4.5 上做改动的人（包括我自己），这篇论文和代码给出了什么。

**已经被它验证、可以直接沿用的**

- LLM 只出文本、每秒几步，语音交给 0.3B 的小解码器；解码器以「文本嵌入 + 投影后的隐状态」为条件。8B backbone 在全双工训练之后，文本能力大体保住了（平均分持平，MMLU 掉了约 5 分）。
- 输出链路用单码本 S3 token（6562，25 Hz）加 flow matching，音色由参考音频决定。这是 CosyVoice 系的做法，MiniCPM-o 把它接进了全双工。
- 在「LLM 每格自己决策并生成」的设计里，1 秒、显式边界、先出 listen/speak 再出内容，是消融里最好的组合。

**它没有做、留给后续设计的**

- **反应速度被 chunk 绑定**。语音解码器同步执行，第 $k$ 秒说的话第 $k+1$ 秒才播；打断和开口都只能在 1 秒边界上由 LLM 决定。想让反应比 1 秒快，要么把 chunk 缩小（消融显示 LLM 会变差），要么让解码器脱离 LLM 的节拍、自己看更新的输入，后者 MiniCPM-o 里没有对应的通路：解码器只看文本和隐状态，看不到音频。
- **编码器被所有损失塑形**。第 2 阶段起编码器同时收到文本和语音 token 的梯度。如果希望编码器保留文本损失不在乎的信息（韵律、说话人、对方的停顿），需要另加约束，比如在编码器出口截断梯度，MiniCPM-o 没有这样做。
- **没有预测未来的目标**。所有损失都是 next-token：文本 token 和语音 token。LLM 的隐状态里有多少「接下来会发生什么」的信息，论文没有测。
- **TAIL 有代价**。按时间对齐让英文 WER 从 2.38 涨到 3.93，标题级的 TTS 成绩用的是固定比例交织，不是全双工模式。
- **训练代码不公开**。要改训练，只能按论文的四阶段描述自己重建，损失加权等细节需要自己定。

**改动的挂点**

如果要在推理代码上做实验，几个位置是现成的接口：

| 想改的 | 位置 |
|:--|:--|
| 音频进 LLM 前的处理（池化方式、加旁路） | `MiniCPMO.get_audio_embedding_streaming` |
| 每个 unit 的调度（决策、生成上限、终止） | `MiniCPMODuplex.streaming_generate` 的 LLM 循环 |
| 语音解码器的条件（换隐状态来源、加输入） | `MiniCPMODuplex._convert_results_to_tts_input` |
| 语音解码器每 unit 的步数和 KV 管理 | `MiniCPMTTS.generate_chunk` 及其调用处 |
| 波形合成的分块 | `MiniCPMODuplex._generate_waveform_from_tokens` |
{#tab-6-1 caption="**表 6-1** 推理代码里的改动挂点" note="函数名对应 HuggingFace `openbmb/MiniCPM-o-4_5` 仓库的 `modeling_minicpmo.py`。"}

---

## References {#references}

1. MiniCPM-o Team. [MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction](https://arxiv.org/abs/2604.27393). arXiv:2604.27393, 2026.
2. OpenBMB. [openbmb/MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5)（HuggingFace 模型仓库，含 `modeling_minicpmo.py` 推理代码）。
3. Radford et al. [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356). ICML 2023. Whisper。
4. Zhai et al. [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343). ICCV 2023. SigLIP。
5. Du et al. [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models](https://arxiv.org/abs/2412.10117). arXiv:2412.10117, 2024. S3 token 与 flow matching。
6. Qwen Team. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). arXiv:2505.09388, 2025.
7. Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). arXiv:2402.03300, 2024. GRPO。
8. Yu et al. [RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness](https://arxiv.org/abs/2405.17220). CVPR 2025.
