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

{{fig-1-1}}

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
- **LLM 只出文本**：论文的理由是，如果让 backbone 直接出 25 Hz 的语音 token，每秒的解码步数要多六到八倍，语言能力也会掉。所以每秒只出 3–4 个文本 token，语音交给小解码器。
- **语音解码器的输入**：每个文本 token 对应一个输入向量，等于解码器自己的文本嵌入加上 LLM 末层隐状态经投影、L2 归一化之后的向量，两者逐元素相加。论文的说法是 LLM 已经把韵律决策编码进隐状态，小解码器只管发音。
- **音色**：语音 token 解码器不接收说话人向量。音色由 flow matching 那一步的参考音频决定；参考音频同时也作为系统提示的一部分送进 LLM。

---

## 2 Omni-Flow：把对话摊在时间轴上 {#omniflow}

### 2.1 序列怎么排 {#sequence}

Omni-Flow 把交互看成三条按时间对齐的流：环境视频、环境音频（用户说话也在里面），以及模型自己的输出。用户的话不再是一个特殊的 user 角色，只是环境音频的一部分。第 $k$ 个 chunk 的视觉 token $v_k$、音频 token $a_k$ 和输出 token $o_k$ 拼成一组 $g_k = [v_k; a_k; o_k]$，所有组按时间顺序接成一条序列，交给普通的 causal LM。不该说话时，$o_k$ 只有一个 listen token。

落到代码里，一个 unit 的样子如[图 2-1](#fig-2-1)：

{{fig-2-1}}

- 系统提示之后不再有 `<|im_start|>user` 这类角色标记，后面全是 unit。
- 每个 unit 先写入这一秒的视觉和音频嵌入（prefill），然后 LLM 采样第一个 token：`<|listen|>` 或 `<|speak|>`。说话时接着出文本，遇到 `<|chunk_eos|>`、`<|listen|>` 或 `<|chunk_tts_eos|>` 就结束这个 unit。
- 一个 unit 最多 20 步（`max_new_speak_tokens_per_chunk`），最后一步由代码直接写入 `<|chunk_eos|>`，所以最多采样 19 个 token。
- 一轮话说完要先出 `<|turn_eos|>`。代码里有一条硬规则：一轮话没结束时如果采样到 `<|listen|>`，会被替换成 `<|tts_bos|>`，也就是说话说到一半不允许直接停下来听。

### 2.2 推理时的先后 {#timing}

再来看时间上谁先谁后。如[图 2-2](#fig-2-2)，每一秒的音视频要攒满才能写进 LLM；写完后 LLM 生成这一 unit 的文本，语音解码器接着生成 25 个 S3 token，合成的约 1 秒语音在下一秒播放。

{{fig-2-2}}

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

{{fig-3-1}}

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
