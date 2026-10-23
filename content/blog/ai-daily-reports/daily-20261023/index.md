---
title: "AI 简报 - 2026/10/23"
date: 2026-10-23
draft: false
description: "全双工模型在连续音视频流里该不该开口、何时开口的基准；JEPA 会保留哪些特征取决于特征之间的预测关系；用原生键鼠操作玩 Minecraft 的分层 VLA agent。"
tags: ["AI", "Daily Report"]
categories: ["AI Notes"]
series: ["AI Daily Reports"]
---

## 本期值得看

- [DIVE](#dive-benchmarking-full-duplex-interaction-with-visual-engagement-in-continuous-streams) 在连续的音视频流上分别评测是否回应、何时回应和回应质量，领先的实时模型平均延迟不到一秒，但在应答窗口内的回应率不超过 53.77%。
- [Predictive Closure](#predictive-closure-a-set-level-characterization-of-jepa) 指出 JEPA 选择哪些特征取决于特征集合之间的预测关系，本身难以预测的特征只要能帮助预测别的特征，也会被保留。
- [HiSTEVE](#histeve-10-harnessing-human-interface-vla-agents-for-open-world-games) 用单个 VLA 只看画面、只输出原生键鼠动作完成 Minecraft 里的所有子任务，并能从零做出铁镐。

## 全双工语音对话

### [DIVE: Benchmarking Full-Duplex Interaction with Visual Engagement in Continuous Streams](https://arxiv.org/abs/2610.28405)
Yuhang Chen, Guangyue Liu, Shilong Liu · arXiv 2026-10-20

![DIVE 的评测框架](https://arxiv.org/html/2610.28405v1/evaluate.png)

*原文 Figure 4：DIVE 评测框架概览。交互决策衡量模型是否在需要时回应、在合适时保持沉默；时间对齐把需要的回应按相对标注窗口的位置分为提前、准时、延迟和遗漏。*

做了什么：现有流式基准主要用带时间戳的问答评测因果感知，DIVE 改为评测模型在连续音视频流中能否持续识别相关线索，并决定是否、何时以及如何回应。六个任务分成三类场景：主动介入、持续参与，以及重叠和多人交互下的听说协调，共 525 个带细粒度人工标注和精确时间信息的音视频样本。评测把回应质量、是否参与和回应时机分开打分；领先的实时模型平均延迟不到一秒，但窗口回应率最高只有 53.77%，根据新到的信息调整正在进行的回应尤其困难。

为什么值得看：它把全双工交互里「什么时候该说话」从回答质量中单独拆出来测量，并覆盖了多人交互。

### [DuplexLedger: Benchmarking Transactional Memory in Full-Duplex Speech Agents](https://arxiv.org/abs/2610.27581)
Yuxuan Li, Xiangchen Guan, Kunrong Li et al. · arXiv 2026-10-20 · [代码](https://github.com/lyx-researcher/DuplexLedger)

![DuplexLedger 的整体结构](https://arxiv.org/html/2610.27581v1/figures/figure2_3.png)

*原文 Figure 2：DuplexLedger 概览。一个持续的会话把自适应触发的全双工挑战、事件级的记忆状态变化和语音形式的记忆提问对齐在一起。*

做了什么：全双工语音 agent 要在边听边说的同时更新持久记忆，但哪些事件该写入、修改或忽略并不清楚。DuplexLedger 是一个中英双语基准，包含 1,094 个持续会话，覆盖四种全双工场景，给出事件级的 Write/Update/Ignore/Retain 标注、记忆前后状态和 6,564 个同会话问题。评测五个全双工模型后发现，没有系统能同时做好交互边界控制和最终记忆一致性，最终状态事实覆盖率最高只有 2.01%。作者提出 8.92M 参数的旁路模块 ACT-Mem，接在冻结的 MiniCPM-o 4.5 上，把最新值准确率从 4.11% 提高到 8.68%，但让被打断时停下来的速度变慢。

为什么值得看：它说明全双工模型的记忆和打断时机是互相牵制的两个能力，需要一起评测。

## 表示学习与世界模型

### [Predictive Closure: A Set-Level Characterization of JEPA's Feature Selection](https://arxiv.org/abs/2610.27364)
Mosh Levy, Aly Lidayan, Yoav Goldberg · arXiv 2026-10-19

做了什么：JEPA 的编码器和预测器一起训练，当前和下一时刻的表示都是学出来的，所以表示里会包含哪些特征并不明确。以往的刻画只看单个特征是否可预测，由此推断像红绿灯这样难预测的特征会被丢掉。作者通过改写 JEPA 的训练目标，提出 predictive closure 刻画，说明特征的取舍取决于特征集合的性质：特征独立演化时与以往结论一致，不独立时，红绿灯这类特征只要能帮助预测其他特征（比如车流是否停下）就仍会被保留。受控实验验证了这一点，并显示对一个特征加辅助目标或把它作为预测器的附加输入，可以间接加强另一个特征的表示。

为什么值得看：它给出了一个判断 JEPA 会编码什么的具体规则，也给出了间接引导特征选择的办法。

### [Learning from Next-Frame Prediction: Towards Unified and Scalable Visual Representation Learning](https://arxiv.org/abs/2610.30724)
Yifeng Yao, Jinghan Li, Zhihao Zhang et al. · arXiv 2026-10-22

![VANE 的整体结构](https://arxiv.org/html/2610.30724v1/method.png)

*原文 Figure 2：VANE 概览。VANE 根据已观测的帧预测下一帧的嵌入：因果编码器把观测帧映射成上下文状态，与预测 token 拼接后交给状态预测器。*

做了什么：掩码视频预训练是从上下文恢复缺失内容，而不是从过去预测未来；逐 token 的自回归方法表示质量又不如联合嵌入方法。VANE 在嵌入空间里做整帧的下一帧预测：因果编码器汇总过去的观测，状态预测器推断下一整帧的表示，预测目标是冻结语义 tokenizer 给出的固定嵌入空间，另有一个轻量的生成解码器根据预测表示重建未来帧的 patch 以保留细节。对静态图像构造从局部到全局的裁剪序列，让同一个编码器统一处理图像和视频。模型随规模和数据增长持续变好，在 EPIC-KITCHENS-100 动作预判上 mean-class recall@5 比 V-JEPA 2.1 高 7.1 个点。

为什么值得看：它把「从过去预测未来」作为视频表示学习的主要信号，并在预判类任务上超过了 V-JEPA 2.1。

## GUI agent 与动作

### [HiSTEVE 1.0: Harnessing Human-Interface VLA Agents for Open-World Games](https://arxiv.org/abs/2610.31162)
Zecong Tang, Zian Guan, Yuncheng Guo et al. · arXiv 2026-10-22

![HiSTEVE 的系统结构](https://arxiv.org/html/2610.31162v1/Harness_cropped.png)

*原文 Figure 1：HiSTEVE 的系统概览，展示了 agent harness 的工作机制。*

做了什么：现有游戏 agent 要么有原生视觉运动控制但缺少长程自主性，要么通过环境 API 或预定义技能执行高层规划。HiSTEVE 是分层的 VLA agent：编排层的 harness 带有记录进度的结构化记忆、在 VLA 能力范围内选子任务的 VLM 规划器，以及依据 VLA 策略上下文判断子任务完成的检测器；控制层由单个 VLA 只用画面输入、以原生键鼠动作执行所有子任务。为减少 GUI 和非 GUI 交互之间的干扰，它使用分场景的 RMSNorm 分支和单独的鼠标移动 token 嵌入，训练结合了世界知识注入、模仿学习和带环境反馈的强化学习。在 MineStudio Bench 上得分 85.0%，OpenHA 平均成功率 37.4%，并能像人类玩家一样从零做出铁镐。

为什么值得看：它展示了只靠画面和原生键鼠动作完成长程任务的一套具体做法，包括鼠标动作的单独编码。

## 语音合成

### [Character-Level Prosody Control for Mandarin Speech Synthesis:An Architecture-Preserving Approach](https://arxiv.org/abs/2610.29123)
Diankun An, Lu Ma, Yongliang Wang et al. · arXiv 2026-10-21

做了什么：针对基于 LLM 的中文 TTS，作者从原始音频中大规模自动提取音高、时长、能量和停顿等细粒度韵律标记，以词表外的 Unicode 符号嵌入文本流，再微调现有 TTS 系统的 LLM 去理解这些标记，不改动架构，从而在整句风格控制之外得到逐字的时间控制。可选的文本 LLM 前端能把自然语言指令翻译成标记序列。以 CosyVoice2 为主干、在 2.15M 条（5,858 小时）中文语音上的实验显示，字的时长、音高、停顿和整体语速都能可靠控制，感知质量基本不变。跨 tokenizer 分析还发现，主干中为 ASR 优化的 tokenizer 会削弱韵律变化。

为什么值得看：它给出了一种不改模型结构就能逐字控制中文韵律的方法，并指出语音 tokenizer 的训练目标会影响韵律表达。

## 推理与部署

### [Drift Says When, Budget Says How Much: Rate-Controlled KV Re-selection for Long-Form Multimodal Generation](https://arxiv.org/abs/2610.31433)
Bin Hu, Yingbo Xiao · arXiv 2026-10-22

做了什么：多模态大模型的 KV 压缩通常在 prefill 时选一次要保留的视觉 token，之后整个生成过程都沿用。作者发现在长描述生成中这样做受限明显：即使用未压缩答案的注意力来选，与未压缩输出的 ROUGE-L 一致性也只有 0.44 到 0.52，因为模型读取的证据会随描述推进而移动；而固定的漂移阈值又因为各样本漂移量不可比而系统性失效。RCR 让漂移决定何时、在哪里重新选择，由每个样本的控制器调节频率，并配合低秩漂移检测器、限定候选的重选和全在 GPU 上的决策。在六个模型与任务组合中的五个上比逐步检索的 InfiniGen 高 1.6 到 3.8 个 ROUGE-L 点，与逐步精确重选的差距在 1.7 点以内，检测器小 4 倍。

为什么值得看：它说明长时间生成时一次性的 KV 选择不够，给出了按需重新选择并控制频率的做法。

## 推荐读的论文

### [Learning Individual Styles of Conversational Gesture](https://arxiv.org/abs/1906.04160)
Shiry Ginosar, Amir Bar, Gefen Kohavi et al. · 2019

讲了什么：给定语音音频，生成与之匹配的手部和手臂动作，做法是从单个说话人「真实场景」下的独白语音到其手势的跨模态转换。训练只用无标注视频，监督信号是自动姿态检测给出的带噪伪标签。模型在定量比较中明显优于基线方法，作者还发布了一个大规模的个人手势视频数据集。

为什么现在读：它是从语音直接生成伴随手势的早期代表工作，把个人风格和野外视频的弱监督引入了这个问题。

### [Joint Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for Self Supervised Learning](https://arxiv.org/abs/2505.12477)
Hugues Van Assel, Mark Ibrahim, Tommaso Biancalani et al. · 2025

讲了什么：重建和联合嵌入是自监督学习的两大范式，前者在输入空间恢复样本，后者在潜空间对齐不同视图的表示。作者利用两种方法的闭式解，精确刻画了视图生成过程（如数据增强）如何影响学到的表示，并证明两种范式都需要增强与无关特征之间有最低程度的对齐才能随样本量增加达到渐近最优。当无关特征幅度较大时，联合嵌入方法的对齐条件严格更弱，因此更可取。

为什么现在读：它从理论上回答了在潜空间预测和在输入空间重建之间如何选择；本期的《[Predictive Closure](#predictive-closure-a-set-level-characterization-of-jepa)》引用了它。

## 行业动态

- **[Claude Duet app is Anthropic's bet on agents](https://techcrunch.com/2026/10/22/claude-duet-app-is-anthropics-bet-on-agents/)**（TechCrunch）：Anthropic 推出面向 agent 的 Claude Duet 应用，Pro 和 Max 用户可通过候补名单使用。
- **[Left alone with nothing to do, AI agents started playing](https://techcrunch.com/2026/10/22/left-alone-with-nothing-to-do-ai-agents-started-playing/)**（TechCrunch）：MIT 研究者把 13 个 AI 放在一个虚拟岛上，它们发明了游戏、为丢失的戒指哀悼，最后在各自的工作上做得更好。
- **[Nirva raises $33M for a pendant that only listens to you](https://techcrunch.com/2026/10/22/nirva-raises-33m-for-a-pendant-that-only-listens-to-you/)**（TechCrunch）：Nirva 融资 3300 万美元，计划在年底前交付售价 249 美元的环境感知 AI 吊坠。

---

> 本期由定时任务根据 arXiv、Hugging Face Daily Papers、Hacker News 和几家公司的博客自动整理，条目均附原文链接；点评是自动生成的判断，以原文为准。
