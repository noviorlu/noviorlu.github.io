---
title: "AI 日报 - 2026/10/18"
date: 2026-10-18
draft: false
description: "全双工对话模型同时生成手势和表情；用一个残差方向控制全双工模型抢话的多少；JEPA 世界模型的一步预测误差和规划成功率几乎无关。"
tags: ["AI", "Daily Report"]
categories: ["AI Notes"]
series: ["AI Daily Reports"]
---

## 今天值得看

- [ELLSA 2](#full-duplex-interaction-with-speech-expressions-and-gestures) 把全双工语音对话和流式手势、表情生成放进一个端到端模型，不说话时也继续生成非语言动作。
- [Initiative vector](#reference-guided-turn-taking-style-transfer-for-full-duplex-spoken-language-models) 发现全双工模型「说多少话」由一个后层残差方向决定，不用训练就能读出和调节。
- [JEPA 世界模型](#one-step-prediction-fidelity-is-a-poor-proxy-for-planning-performance-in-jepa-world-models)的一步预测误差在训练一个 epoch 之后就和规划成功率几乎不相关（r = −0.010）。

## 全双工语音对话

### [Full-Duplex Interaction with Speech, Expressions and Gestures](https://arxiv.org/abs/2610.21452)
Siyin Wang, Wenyi Yu, Terumi Chiba et al. · arXiv 2026-10-15

![ELLSA 2 的整体结构](https://arxiv.org/html/2610.21452v1/seg_1.png)

*原文 Figure 1：ELLSA 2 同时建模全双工语音、手势和表情生成；手势表情合成器以用户和模型两侧语音的隐状态为条件，每 800 ms 自回归生成一段离散的手势和表情编码。*

做了什么：提出 ELLSA 2，一个端到端的全双工对话模型，同时优化语音交互和流式的手势、表情生成。一个按块自回归的合成器结合预训练声学特征和对话主干在「听」「说」两种状态下的隐状态，所以模型不说话时也能持续生成非语言动作。语义手势分成「要不要调用」和「怎么做」两步。实验里语音交互和开源全双工模型相当，手势和表情指标好于级联基线。

为什么值得看：语音之外的动作输出接在对话主干的隐状态上，是全双工模型带动作输出的一个具体做法。

### [Reference-Guided Turn-Taking Style Transfer for Full-Duplex Spoken Language Models](https://arxiv.org/abs/2610.20763)
Yuxuan Chen, Yang Xiao, Jack Sun et al. · arXiv 2026-10-14

![initiative vector 的提取方式](https://arxiv.org/html/2610.20763v1/fig1_overview.png)

*原文 Figure 1：在模型保持沉默的帧里，分别对「800 ms 内会开口」和「继续沉默」两类帧的残差取平均，两者之差归一化后就是 initiative vector，在第 30 层读取和写入。*

做了什么：在全双工语音模型的后层残差流里找到一个方向（initiative vector），不需要训练或标注，从模型自己的对话里提取。这个方向能提前最多 800 ms 预判开口时机；生成时按比例放大或缩小它，说话时长占比可以从 6–12% 调到 42–61%（不干预时 21–30%）。用一段参考录音就能把某个人的轮流说话风格迁移给模型。

为什么值得看：说明「什么时候开口」这类交互性格可能是模型内部一个可读、可调的量，而不是写死在权重里。

### [MultiPartyTurnBench: Benchmarking Floor Control in Multi-Party Spoken Dialogue](https://arxiv.org/abs/2610.20172)
Puneet Mathur, Dinesh Manocha · arXiv 2026-10-14

![MultiPartyTurnBench 的数据生成流程](https://arxiv.org/html/2610.20172v1/conversation_generation.png)

*原文 Figure 2：五步流程把带事件条件的对话脚本变成同步的多说话人音频，并给出精确的话轮转换标注。*

做了什么：240 段合成的、分声道的多人对话（2–5 人，6 种互动类型），评测两件事：判断一轮话是否真的结束，以及判断别人插话是不是真的要抢话。结果不对称：结束检测召回 82.5–92.6%，抢话检测要么漏掉、要么误触发；五个全双工 agent 在「何时进入」上平衡准确率 86.1–95.8%，在另一项上接近随机（46.9–54.7%）。

为什么值得看：现有轮流说话评测多是两人对话，这个基准专门测多人场景，暴露的短板很具体。

### [VoxTurn: Answering Several Turn-Taking Questions in One Pass with One Model](https://arxiv.org/abs/2610.21150)
Yang Xiao · arXiv 2026-10-15

做了什么：一个 0.8B 的语音 LLM，把「用户说完了吗」「是插话还是附和」「这是不是语音」等判断写成选择题，用块隔离的 attention mask 一次前向同时回答。在真实对话停顿上训练，英语在 300 ms 延迟预算下误打断 11.5%，14 种语言平均 19.9%，每次判断 8.3 ms。

为什么值得看：给级联式语音 agent 的轮流说话判断提供了一个小而快的开源方案和数字参照。

## 表示学习与世界模型

### [One-Step Prediction Fidelity Is a Poor Proxy for Planning Performance in JEPA World Models](https://arxiv.org/abs/2610.19375)
Joshua Cheung, Zhang He · arXiv 2026-10-14

做了什么：潜空间世界模型通常按一步预测损失来挑 checkpoint。作者用冻结解码器把一步误差换算成物理单位，在一个端到端 JEPA 世界模型、三个连续控制环境上测：第一个 epoch 一步误差改善 32%、规划成功率提高 4.4 个百分点；之后的 15 个 checkpoint 一步误差又降了 15%，但它和规划成功率的相关系数只有 r = −0.010。

为什么值得看：直接质疑「下一个 latent 预测得更准就是更好的模型」这个常用的选模型标准。

### [Model What Matters: Decision-Guided Selective World Modeling for Language Agents](https://arxiv.org/abs/2610.19642)
Maolin Ran, Minda Chen, Di Huang et al. · arXiv 2026-10-14

![DGS-WM 与普通联合训练的对比](https://arxiv.org/html/2610.19642v1/intro.png)

*原文 Figure 1：普通的策略与世界模型联合训练对所有观测一视同仁（左），DGS-WM 只监督对决策有用的部分（右）。*

做了什么：语言 agent 做策略和世界模型联合训练时，不再把整个观测都当预测目标，而是只挑对后续决策有用的观测单元作为世界模型目标，并按模型对这些目标的置信度调整监督强度，和 RL 损失一起优化。

为什么值得看：「只预测对决策重要的部分」是压缩和预测之间的一种具体取舍。

### [Does Compression Imply Generalization? A Minimum Description Length Perspective](https://arxiv.org/abs/2610.21376)
Lincen Yang, Jiayang Shi, Nan Pu · arXiv 2026-10-15

做了什么：从最小描述长度（MDL）出发，提出只用训练轨迹就能算的 prequential regret，证明在接近插值的区间它上界参数的描述长度，并由此得到 PAC-MDL 泛化界；还证明信息瓶颈（IB）里新加的编码器复杂度修正项被 MDL 参数描述长度上界，把两种压缩观点连起来。

为什么值得看：给「压缩和泛化的关系」一个不依赖高维互信息估计的可计算量。

### [Dataset-Level Regularization for Joint-Embedding Self-Supervised Learning](https://arxiv.org/abs/2610.20158)
Konstantinos Almpanakis, Anna Kreshuk · arXiv 2026-10-14

![minibatch 正则与 OOB 正则](https://arxiv.org/html/2610.20158v1/oob_overview.png)

*原文 Figure 1：(a) 批内方法每一步只用当前样本估计防坍缩项；(b) OOB 定期编码整个数据集，构造逐样本的锚点，训练时按 minibatch 取出。*

做了什么：联合嵌入自监督防坍缩的正则项通常只在当前 minibatch 上算。作者在 CPU 内存里维护数据集级的状态，给每个样本构造全局目标，小 batch 下更接近全数据集的正则梯度；用在 SimCLR 和 SIGReg 上都有提升，batch 越小收益越大。

为什么值得看：SIGReg 一类防坍缩正则在小 batch 下的一个改进，batch 大小不再影响正则的几何。

### [Can World and Action Modeling Benefit Each Other under Unified Causal Training?](https://arxiv.org/abs/2610.21095)
Yue Su, Bokai Lin, Xinkai Wang et al. · arXiv 2026-10-15 · [代码](https://github.com/Interactive-Active-AI/OneWM)

![OneWM 的结构](https://arxiv.org/html/2610.21095v1/onewm-method-pipeline.png)

*原文 Figure 1：VAE 把视频帧编码成潜变量，视频专家对未来的潜变量去噪，动作专家处理动作块，两个专家之间用因果的联合注意力耦合。*

做了什么：OneWM 在同一个视频 + 动作架构里，把动作既当世界预测的条件、又当策略学习的目标。在 LIBERO、RoboCasa365 等环境上，学「给定动作的视觉后果」提高了策略性能和泛化，反过来学生成动作也提高了多视角世界预测的保真度。

为什么值得看：世界建模和动作生成互相帮助的一个对照实验，两个方向都有正结果。

## GUI agent 与动作

### [OpenGCUA: Toward a Fully Open Stack for Generalist Computer Use](https://arxiv.org/abs/2610.22200)
Hanzhang Zhou, Long Li, Quyu Kong et al. · arXiv 2026-10-15 · [代码](https://github.com/Tongyi-MAI/OpenGCUA)

![OpenGCUA 的整体结构](https://arxiv.org/html/2610.22200v1/fig2_v3_candidate_B.png)

*原文 Figure 2：Open Generalist CUA 的整体结构。*

做了什么：开源通用电脑操作 agent 的完整技术栈，第一版。设计重点是让模型执行代码，用中间结果决定后续操作；并把部分观测和任务状态决策从固定的 harness 规则交给模型。在 OSWorld-v2 上，八个模型的平均部分得分提高 19.6 分，每个任务的步数平均减少 53.0%。

为什么值得看：键鼠和代码混合操作电脑的开源基线。

### [MAI-Gym: An Open All-in-One Platform for GUI Agent Evaluation and RL Training across Mobile and Computer Environments](https://arxiv.org/abs/2610.20939)
Panrong Tong, Quyu Kong, Pengxiang Li et al. · arXiv 2026-10-14 · [代码](https://github.com/Tongyi-MAI/MAI-Gym)

![MAI-Gym 的整体结构](https://arxiv.org/html/2610.20939v1/fig1_mai_gym_overview.png)

*原文 Figure 1：左边是在线 GUI 环境，中间是核心能力，下方是「评测、诊断、合成、再训练」的循环，右边是 MAI-Gym-9B 与不同规模模型的对比。*

做了什么：把 GUI agent 的评测、数据收集和在线 RL 训练统一到一个平台，支持 10 个基准、2 万个并行环境，还能合成可重置的模拟应用。训练出的 9B 模型在 MobileWorld 78.4%、OSWorld-Verified 76.2%；在模拟购票应用里做 RL 后，真机上的任务成功率从 28.8% 提到 48.1%。

为什么值得看：模拟环境里的 RL 能迁移到真实应用，是训练操作类 agent 的一个可用路线。

## 语音合成

### [Long-Prompt Zero-Shot TTS: Benchmarking Prosody Reproduction Beyond Voice Cloning](https://arxiv.org/abs/2610.19774)
Dekun Chen, Renjie Xu, Yingda Shen et al. · arXiv 2026-10-14

![LPTTS 系统结构](https://arxiv.org/html/2610.19774v1/icassp2027-main.png)

*原文 Figure 2：长提示编码器从长语音提示里提取全局和目标相关的韵律表示，通过门控 cross-attention 注入 Qwen3TTS；音色仍由原有的说话人编码器提供。*

做了什么：短语音提示只能传达音色，很难传达一个人的韵律习惯。作者提出用长语音提示（最长约 10 分钟）复现说话人的韵律，发布双语基准 LP-TTS-Bench（37 位说话人、63.95 小时），并给出一个把长提示压成固定数量连续韵律向量的参考系统；收益随提示时长增长，约 10 分钟后饱和。

为什么值得看：角色化语音不只是音色，这个基准把语速、重音、语调分开评测。

## 记忆与社交

### [Hy-Memory-30B-A3B: Attributed Reinforcement Learning for Long-Term Memory Writing](https://arxiv.org/abs/2610.22196)
Tianxiang Fei, Jiazheng Kang, Mao Zheng et al. · arXiv 2026-10-15

做了什么：训练一个专门「写记忆」的 30B-A3B 模型，跨 Mem0、MemOS、Hy-Memory 三种记忆框架通用：先做按框架协议的 on-policy 蒸馏，再用逐块探针和跨块归因奖励做 RL。在四个评测 harness 上宏平均 55.49%，Claude Opus 5 是 54.62%。

为什么值得看：长期记忆的质量取决于「写什么、怎么更新」，这篇把写入单独训成一个模型。

### [Beyond Downstream QA: Rethinking How We Evaluate Agent Memory](https://arxiv.org/abs/2610.22209)
Sebastian-Antonio Toma, Jean de Kernier, Mike Ply · arXiv 2026-10-15

做了什么：指出用下游问答准确率评估记忆会把记忆错误和检索、生成错误混在一起。提出 FEB-30，30 条事实随时间演变的轨迹，可以在每次写入后直接检查记忆库。在一个代表性记忆系统上，内在和外在评测平均差 16 个百分点，记忆操作（新建、更新、强化、取代）的分类只有 20% 正确。

为什么值得看：做长期记忆时，评测方法本身就可能给出错误结论。

## 推理与部署

### [Compression Does Not End at Selection: Post-Selection Positional Geometry for Sparse KV Caches](https://arxiv.org/abs/2610.21126)
Wanyang Hong, Zhaoning Zhang · arXiv 2026-10-15

做了什么：稀疏 KV 压缩保留下来的 token 仍带着原来的位置，在 RoPE 下被丢掉的 token 留下的空隙会影响注意力匹配。RoPERelay 给保留的 token 重新分配紧凑、保序的 RoPE 坐标（对 key 做逆旋转再旋转），不需要重新 prefill。36 组对比里 32 组提升，Llama-3.1-8B 在 LongMemEval-S 上从 23.6% 到 35.4%，额外耗时 1–3%。

为什么值得看：在 dense 训练的模型上推理时丢弃 KV，位置怎么处理是一个单独的变量。

### [JointServe: Serving Multi-Model Agents on Memory-Constrained GPUs](https://arxiv.org/abs/2610.19822)
Daijin Hu · arXiv 2026-10-14

做了什么：多模型 agent 在显存有限的 GPU 上部署时，模型权重和 KV 块放进同一个显存预算里统一调度：从调用方的输出流里解析意图提前预取模型，只换入部分层，并给正在进行的会话的 KV 保留带宽。会话完成时间比最强的整模型换入换出基线少 21.7%。

为什么值得看：单卡上同时跑多个模型（比如编码器、主干和语音解码器）时的显存调度思路。

## 行业动态

- **[Hybrid Reasoning Agents: LLM agents that think in latent space, act through text](https://arxiv.org/abs/2610.18861)**（Hacker News 130 分，[讨论](https://news.ycombinator.com/item?id=50145337)）：在潜空间里思考、用文本行动的 LLM agent。
- **[Hyper Optimizing Kimi K3 in vLLM](https://vllm.ai/blog/2026-10-16-kimi-k3-megakernel)**（vLLM 博客）：在 vLLM 里为 Kimi K3 做的 megakernel 优化。
- **[Show HN: Llmspeed](https://llmspeed.dev/)**（Hacker News 321 分）：买硬件之前先体验本地 LLM 在不同配置上跑多快。
- **[Disney agrees to hire the majority of Character.AI's workforce](https://techcrunch.com/2026/10/18/disney-agrees-to-hire-the-majority-of-character-ais-workforce/)**（TechCrunch）：文件显示收到 Disney offer 的 Character.AI 员工绝大多数已接受；这是一个 stalking horse 出价，仍可能有其他公司出更高价。
- **[Replit makes OpenAI its default over Anthropic, CEO says](https://techcrunch.com/2026/10/17/replit-is-no-longer-working-with-anthropic-ceo-says/)**（TechCrunch）：Replit 的默认模型提供方改为 OpenAI，平台上仍可使用 Claude。

---

> 本期由定时任务根据 arXiv、Hugging Face Daily Papers、Hacker News 和几家公司的博客自动整理，条目均附原文链接；点评是自动生成的判断，以原文为准。
