---
title: "AI 简报 - 2026/10/25"
date: 2026-10-25
draft: false
description: "连续潜空间 TTS 里线性可解码的说话人信息不一定能用来控制音色；JEPA 世界模型的潜状态要靠逆动力学监督才能学到实体之间的关系；全模态长期记忆基准把失败拆到写入、检索、传递和读取四个环节。"
tags: ["AI", "Daily Report"]
categories: ["AI Notes"]
series: ["AI Daily Reports"]
---

## 本期值得看

- [Decodability Is Not Controllability](#decodability-is-not-controllability-local-and-anisotropic-speaker-control-in-continuous-latent-tts) 在 VoxCPM2 上发现，最后一层里探针准确率 0.970 的说话人信息改了也换不了音色，而在最早的参考音频位置上做对齐编辑，60 组目标说话人全部生效。
- [Inside-JEPA](#inside-jepa-from-prediction-to-relational-grounding-in-latent-world-models) 发现只做潜空间预测学不到实体之间的关系，加上从关系通道变化反推动作的逆动力学损失后才能在没有关系标注的情况下学到。
- [MOSAIC](#mosaic-an-omni-modal-benchmark-for-long-term-memory-and-cross-modal-reasoning-in-conversational-agents) 用跨文本、图像、视频和音频的多轮历史考察对话 agent 的长期记忆，并把失败定位到表示、检索、传递和读取各个环节。

## 表示学习与世界模型

### [Inside-JEPA: From Prediction to Relational Grounding in Latent World Models](https://arxiv.org/abs/2610.31194)
Hossein Resani, Javen Qinfeng Shi · arXiv 2026-10-22

做了什么：JEPA 在潜空间里学世界模型，便于规划，但潜状态难以解释。Inside-JEPA 在潜状态里为有序的实体对放入关系通道，并用逆动力学损失（从通道的变化反推动作）在没有关系标注的情况下让这些通道对应上真实关系。在 Something-Something V2 上，只做潜空间预测的效果不比随机的关系头好，加上逆动力学监督后关系 AUROC 在十种设置下提高 0.065 到 0.103，在留出任务的 RH20T 机器人数据上提高 0.081；打乱标签、两种无标签目标、只给 5% 的实体对动作标签都无法学到关系。

为什么值得看：它给出了一个具体的反例，说明潜空间预测目标本身不保证潜状态里出现可读的关系结构，需要动作信号来约束。

### [LocaLeJEPA: Maximum Entropy Representations under Geometric Priors](https://arxiv.org/abs/2610.28458)
Kartik Tandon, Alessio Borgi, Pavel Snopov et al. · arXiv 2026-10-20 · [代码](https://github.com/alessioborgi/LocaLeJEPA)

做了什么：LeJEPA 用分布匹配目标把嵌入推向各向同性高斯来防止坍缩，但这个目标依赖欧氏空间假设，也不反映双曲、等变等嵌入几何。作者从流形的角度重新表述 LeJEPA 的最大熵性质，提出局部欧氏的分布匹配正则 LocaLeReg。在 ImageNet-1K 上，搭配双曲 ViT-S/8 和等变 ResNet-50 时明显优于 LeJEPA，换成普通欧氏 ViT-S/8 和 ViT-L/14 也更好，线性和 k-NN 分类以及多个下游任务都有提升。

为什么值得看：它把 LeJEPA 的防坍缩正则从全局约束改成局部约束，在普通欧氏主干上也有提升，说明全局各向同性的目标不一定是最合适的。

## 语音合成

### [Decodability Is Not Controllability: Local and Anisotropic Speaker Control in Continuous-Latent TTS](https://arxiv.org/abs/2610.29345)
Yuanzhong Zheng, Wei Huang, Ming Ma et al. · arXiv 2026-10-21

做了什么：作者在连续潜空间 TTS 模型 VoxCPM2 上，用位置匹配的探针和等范数干预考察 30 个说话人、60 组有向说话人对。最后一层残差声学语言模型（RALM L7）里参考音频状态的说话人探针准确率是 0.970，但对齐编辑后 60 组里没有一组变成目标说话人；在最早的参考音频位置（L0）编辑则相反，目标相似度提高 0.326，60 组全部由目标说话人主导，而任何等范数的方向或位置对照最多只有 1 组。独立的说话人编码器、留出说话人原型检索和双盲听测都确认了这一结果；在 dots.tts 上，主干编辑只能让声音偏离原说话人，去掉 x-vector 则让原说话人相似度从 0.849 降到 0.006。

为什么值得看：它说明在连续潜空间语音模型里，探针能读出某种属性和能通过编辑控制这种属性是两回事，免训练的音色控制要找对位置和信息流经的路径。

### [StyleStream 2.0: Fast and Controllable Streaming Voice Style Conversion](https://arxiv.org/abs/2610.28940)
Yisi Liu, Nicholas Lee, Gopala K. Anumanchipalli · arXiv 2026-10-20 · [代码](https://github.com/Berkeley-Speech-Group/StyleStream2.0)

![Stylizer 和 Voice Controller 的结构](https://arxiv.org/html/2610.28940v1/stylizer-voice-controller_cropped.png)

*原文 Figure 1：左边是 Stylizer 的结构，$[r,t]$ 是 MeanFlow 的区间，$\omega$ 是 CFG 强度；右边是 Voice Controller 的结构，做音色设计时辅助输入为空，做音色编辑时源语音的风格嵌入也作为输入。*

做了什么：语音风格转换要把源语音的音色、口音和情绪换成目标声音，同时保留内容。StyleStream 2.0 训练少步数的 pixel MeanFlow 模型，并借鉴 Self Forcing 用自回归反馈微调，提高流式推理的稳健性，在消费级 GPU 上端到端延迟降到 520 ms 左右，比 1.0 快一倍。它去掉梅尔谱上下文，改用紧凑的风格嵌入传递目标风格，并在这个嵌入空间里训练一个文本条件的 flow matching 模型，同时支持用自然语言设计声音和按指令修改某一个属性。在风格转换、声音设计和声音编辑上目标风格的还原度都最好，可懂度也有竞争力。

为什么值得看：它把流式的低延迟转换和用文字描述来设计、修改声音放进同一个连续风格嵌入空间里。

## 记忆与社交

### [MOSAIC: An Omni-Modal Benchmark for Long-Term Memory and Cross-Modal Reasoning in Conversational Agents](https://arxiv.org/abs/2610.29249)
Hong Ting Tsang, Boyi Zhang, Haoyu Huang et al. · arXiv 2026-10-21

![MOSAIC 的整体设计](https://arxiv.org/html/2610.29249v1/MOSAIC_diagnostic_figure_B_refined.png)

*原文 Figure 2：MOSAIC 概览。上半部分是基准考察的内容：证据分散在几个月的文本、图像、视频和音频会话里，需要跨模态串起来；构造时针对的是图像里的空间布局和背景细节、视频里的动作顺序等原生信号，而不是装饰性的媒体。*

做了什么：全模态助手不只要记住用户说了什么，还要记住用户展示和表达了什么。MOSAIC 包含 923 个问题，建立在以人设为基础、跨 11 到 24 个会话的历史上，涵盖文本、图像、视频和音频，并专门构造空间、时间和韵律信号。配套的诊断协议固定问题、读取模型、提示和预算，只对表示、检索和传递分别做干预。结果显示表示损失随模态而不同，即使给出正确证据，读取环节的错误仍占主导，模型在该拒答时也很少拒答。

为什么值得看：它把多模态长期记忆的评测从一个总分拆成写入、检索、传递和读取几个环节，并把语气韵律也算作需要记住的信息。

## 推荐读的论文

### [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
Mido Assran, Adrien Bardes, David Fan et al. · 2025 · [代码](https://github.com/facebookresearch/vjepa2)

![V-JEPA 2 的整体流程](https://arxiv.org/html/2506.09985v1/flowchart.png)

*原文 Figure 1：V-JEPA 2 概览。用 100 万小时的互联网视频和 100 万张图像，以视觉掩码去噪目标预训练 V-JEPA 2 视频模型，再把它用于动作分类、物体识别等下游任务。*

讲了什么：先在超过 100 万小时的互联网视频和图像上预训练不带动作的 JEPA 模型 V-JEPA 2，在 Something-Something v2 上取得 77.3 的 top-1，在 Epic-Kitchens-100 的动作预判上达到 39.7 的 recall-at-5。和大语言模型对齐后，在 8B 规模的多个视频问答任务上取得最好结果。再用不到 62 小时的无标注机器人视频后训练一个以动作为条件的潜空间世界模型 V-JEPA 2-AC，零样本部署到两个实验室的 Franka 机械臂上，以图像为目标规划完成抓取和放置。

为什么现在读：它展示了同一个自监督潜空间模型可以同时用于视频理解、预测和规划，是 JEPA 路线目前规模最大的公开工作之一；本期的《[Inside-JEPA](#inside-jepa-from-prediction-to-relational-grounding-in-latent-world-models)》引用了它。

### [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai et al. · 2023

讲了什么：提出生成式 agent，在大语言模型外加一个架构：用自然语言保存 agent 的全部经历，随时间把记忆归纳成更高层的反思，并在规划行为时动态检索。作者在一个类似《模拟人生》的沙盒小镇里放了 25 个 agent，用户可以用自然语言和它们交互；只给一个 agent 设定「想办情人节派对」，其他 agent 就在两天里自行传播邀请、结识新人、相约赴会并准时出现。消融实验显示观察、规划和反思三个组件对行为的可信度都很关键。

为什么现在读：它最早把记忆流、反思和检索组成一套让 agent 行为保持连贯可信的架构，后续大量 agent 记忆工作沿用了这种拆分。

## 行业动态

- **[Block’s new voice model sounds eerily human](https://techcrunch.com/2026/10/24/blocks-new-voice-model-sounds-eerily-human/)**（TechCrunch）：Block 发布新的语音模型，称它可以在 MacBook M4 Max 上本地运行。
- **[OpenAI says it will no longer train its AI models on the live internet, citing “significant risks”](https://techcrunch.com/2026/10/23/openai-says-it-will-no-longer-train-its-ai-models-on-the-live-internet-citing-significant-risks/)**（TechCrunch）：OpenAI 表示将让前沿模型的训练与互联网隔离，此前几周它的一位工程师还说过这种做法行不通。
- **[LLMs are making formal software verification feasible at a much larger scale](https://www.anthropic.com/research/formal-verification)**（Anthropic，[HN 讨论](https://news.ycombinator.com/item?id=50202217)）：Anthropic 发布研究文章，讨论大语言模型让形式化软件验证可以在更大规模上进行。

---

> 本期由定时任务根据 arXiv、Hugging Face Daily Papers、Hacker News 和几家公司的博客自动整理，条目均附原文链接；点评是自动生成的判断，以原文为准。
