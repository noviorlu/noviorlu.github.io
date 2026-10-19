---
title: "AI 日报 - 2026/10/19"
date: 2026-10-19
draft: false
description: "全双工前台模型加后台任务时，由单独的协调层处理用户中途改口；语音和人脸从同一个表示实时生成；给已训练好的语音对话模型压缩生成用的 token。"
tags: ["AI", "Daily Report"]
categories: ["AI Notes"]
series: ["AI Daily Reports"]
---

## 今天值得看

- [Jarvis](#jarvis-coordination-deserves-a-brain-of-its-own-in-agentic-interactive-models) 在全双工前台模型和慢速后台任务之间加一个只负责协调的层，用户中途改口时不会让过期的计划继续执行。
- [JAVIS](#javis-joint-audio-visual-interaction-and-streaming-for-real-time-conversational-agents) 让语音和人脸从同一个表示里流式生成，首包延迟比基线低约 200 倍。
- [Learning Intrinsic Dynamics in Latent World Models](#learning-intrinsic-dynamics-in-latent-world-models) 在视觉外观变化很大时，用「从相邻潜状态还原动作」的辅助目标保住与控制相关的信息。

## 全双工语音对话

### [Jarvis: Coordination Deserves a Brain of Its Own in Agentic Interactive Models](https://arxiv.org/abs/2610.23804)
Yuxiang Wang, Shengbo Cai, Zhixian Zhao et al. · arXiv 2026-10-16

做了什么：交互模型常把复杂推理和工具调用交给后台的慢模型，前台实时模型负责全双工对话；但任务运行期间用户可能修改、取消或替换请求，已有的计划或结果会过期。Jarvis 是一个协调层：带类型的决策模型 JEV 把每一轮用户话语和正在进行的任务对照，在带版本号的任务记录上选择操作，代码在执行前检查每个选择是否针对当前版本；计划执行前和结果播报前都会再检查，用户完整听到才算交付。在前台和后台固定的情况下，换成 Jarvis 后 tau-Voice 的完成任务数从 278 个里的 29 个升到 192 个；在 Full-Duplex-Bench v3 上用开源 9B 前台模型达到 69% Pass1，GPT-Realtime 是 60%。

为什么值得看：快慢两个模块分工时，两者之间的协调本身是一个需要单独设计的部分。

### [JAVIS: Joint Audio-Visual Interaction and Streaming for Real-Time Conversational Agents](https://arxiv.org/abs/2610.22499)
Junseok Ahn, Jaehyun Park, Jeongsoo Choi et al. · arXiv 2026-10-15

![JAVIS 的整体结构](https://arxiv.org/html/2610.22499v1/figure1-b.png)

*原文 Figure 1：JAVIS 根据用户语音，实时联合生成音视频回复。*

做了什么：提出端到端的流式音视频对话框架，核心是让声音和人脸来自同一个表示。共享的 LLM 之上有一个 Joint Refined Head 产生统一的音视频表示，再分给语音和人脸两条流式分支。作者还发布了 334 小时的合成语料 JAVIS-Syn，覆盖带情绪的多轮对话和助手式对话。实验里首包延迟比强基线低约 200 倍，口型同步和表情的合理性也好于基线。

为什么值得看：口型、表情和语音共用一个中间表示，是避免两路输出各自生成再对齐的一种做法。

### [CoSTA: Compressed Speech Token Adaptation for Efficient Spoken Dialogue Models](https://arxiv.org/abs/2610.21143)
Yingzhuo Deng, Chong Li, Bingxin Liu et al. · arXiv 2026-10-15

![CoSTA 的流程](https://arxiv.org/html/2610.21143v1/adap.png)

*原文 Figure 3：CoSTA 依次做均值初始化的词表扩展、渐进的发音适配和对话适配。*

做了什么：语音对话模型通常让生成和合成共用同一个语音 token 空间，序列长、推理贵。CoSTA 在保留原有语音 tokenizer 和解码器的前提下，把生成用的 token 空间和合成用的 token 空间拆开：给生成词表加入压缩后的表示，经过发音适配和对话适配，在合成前再确定性地展开回原来的 token。在双语对话基准上，语音 token 生成速度约提高到 1.7 倍，全响应延迟更低、吞吐更高，回复质量保持不变。

为什么值得看：不换 tokenizer 就能缩短语音序列，生成的表示和合成的表示可以独立选择。

## 表示学习与世界模型

### [Generative Representation Learning for Scalable Latent Tokenization](https://arxiv.org/abs/2610.20533)
Alan Baade, Kyle Sargent, Vincent Sitzmann · arXiv 2026-10-14

做了什么：提出一种自监督表示学习目标，用学到的生成模型的条件预测之差来估计数据的逐点互信息，再据此监督表示，目的是用作生成模型的 tokenizer。作者给出理论论证，并在实验里说明在 DINO 和 LeJEPA 会出现维度坍缩和特征压制的情形下，这个方法不会。投入更多算力能稳定地得到更好的潜在 token；在 ImageNet 上用 JiT-L 得到 FID-50K 2.18，以不到 DINOv2 七分之一的 FLOPs 得到可用的表示。

为什么值得看：把「表示是否防坍缩」和「是否便于生成建模」放进同一个目标，并给出 tokenizer 的算力缩放关系。

### [Learning Intrinsic Dynamics in Latent World Models](https://arxiv.org/abs/2610.21612)
Jiamin Shi, Yuanhong Zhang, Zhiyang Liu et al. · arXiv 2026-10-15

做了什么：与动作无关的视觉变化会掩盖动作和后果之间的关系。作者假设，要求能从相邻潜状态还原动作（动作可预测性），会阻止编码器丢掉不同动作及其后果之间的区别，并给出理论下界和受控实验作为证据。他们据此提出 Concat IDM：把相邻潜状态拼起来预测动作，用离线轨迹里本来就有的动作标签更新共享编码器，测试时的规划器不变。在 Push-T、24 种外观的设置下，三次独立训练的规划成功率从 9.1% 升到 77.2%。

为什么值得看：给出了潜空间世界模型在视觉干扰下保住动态信息的一个简单辅助目标。

### [EgoJEPA: Beyond Retargeting for Dexterous Manipulation from Egocentric Videos](https://arxiv.org/abs/2610.20904)
Meng Cao, Can Zhao, Jinjie Wei et al. · arXiv 2026-10-14

做了什么：从第一人称人类视频学习与机器人形态无关的表示，不做手工的人到机器人动作重定向。三个阶段：用以手部动作差为条件的 JEPA 式视觉预测预训练「有效果」的潜在动作；联合预测未来视觉动态和潜在动作；再用轻量的动作适配器适配到具体机器人。在 3 种机器人、5 个真机任务上，域外平均成功率 74%，重定向基线 65%，跨形态累计耗时从 743.2 小时降到 392.7 小时。

为什么值得看：用人类视频里的潜在动作替代手工重定向，是 JEPA 式预测在动作数据上的一个用法。

## GUI agent 与动作

### [OS-Helix: Unifying GUI Interaction and Code Execution in Native Computer-Using Agents](https://arxiv.org/abs/2610.22234)
Bowen Yang, Kaiming Jin, Zichen Ding et al. · arXiv 2026-10-15 · [代码](https://github.com/OS-Copilot/OS-Helix)

![GUI、代码和混合操作电脑的对比](https://arxiv.org/html/2610.22234v1/teaser_v3.png)

*原文 Figure 1：纯 GUI、纯代码和混合三种操作电脑的方式的概览。*

做了什么：端到端训练的电脑操作 agent，把 GUI 动作和代码执行放进同一个动作空间。初步实验显示，只在提示里加一个代码工具会让 GUI agent 变差，所以 GUI 和代码的选择要靠混合数据训练。监督数据来自「应用加真实文件」构造的可执行 Ubuntu 状态，再用在线 RL 和按动作步骤而不是按 token 数加权的 Step-balanced GSPO 继续训练。OS-Helix-27B-RL 在 OSWorld-Verified 上达到 66.44%，比同规模的 Qwen3.5 混合基线相对提高 42.9%。

为什么值得看：给出了在一个策略里训练「点界面」和「写代码」两种动作的数据构造和优化细节。

## 语音合成

### [Beyond the Benchmark: Conditioning, Domains and Failure Modes of Qwen3-TTS](https://arxiv.org/abs/2610.23678)
Séverin Baroudi, Yanis Labrak, Petr Motlicek et al. · arXiv 2026-10-16

做了什么：在朗读、会议、情感和空管语音上，把 Qwen3-TTS 的两种克隆接口（上下文学习 ICL 和 X-Vector）与 Chatterbox、XTTS-v2、F5-TTS 对比，用 18 个客观指标、LLM 评委、欺骗检测和听感测试。两种接口各有所长：ICL 在参考文本正确时更接近音色，X-Vector 更接近语速节奏，参考文本一出错 ICL 就明显变差；领域对结果的影响约是系统选择的四倍；Qwen3-TTS 的可懂度和音高范围领先，韵律落后，客观指标和听感对系统的排序也不一致。

为什么值得看：给零样本语音克隆在非朗读场景下的失败方式列出了具体的数据。

## 记忆与社交

### [Multimodal Filesystem Memory For Long-Horizon Embodied Agents](https://arxiv.org/abs/2610.22686)
Chloe Su, Jiahan Li, Zhenting Qi et al. · arXiv 2026-10-15

做了什么：MFM 是一个多模态文件系统式的 agent 记忆：情景记忆保存观测、动作和结果，空间记忆把经验和位置关联，语义和程序记忆积累通用知识和可复用方法，工作记忆保存当前计划，各部分用可检索的文本记录连回底层的多模态片段。agent 可以按上下文线索找回过去的经验，并回看原始观测补回文字描述里缺的细节；巩固过程把经验转成高层知识，同时保留指向原始片段的引用。在长程 Minecraft 和电脑操作任务上，三个基座模型都有提升。

为什么值得看：文字摘要会丢掉视觉和空间细节，这里让记忆保留指向原始多模态片段的引用。

## 推荐读的论文

### [Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset](https://arxiv.org/abs/2506.22554)
Vasu Agrawal, Akinniyi Akinyemi, Kathryn Alvero et al. · arXiv 2025-06-27

讲了什么：发布 Seamless Interaction 数据集，包含来自 4000 多位参与者、多种场景的 4000 多小时面对面互动视频。作者还用这个数据集训练模型，生成与语音对齐的双人动作手势和面部表情，输入可以同时包含对方的语音和视觉行为；有一个用 LLM 语音的版本，并接入了 2D 和 3D 渲染，也有可调情绪和表现力、生成语义更相关手势的可控版本，最后讨论了评估这类双人动作模型的方法。

为什么现在读：这是目前少有的规模达到数千小时的双人面对面互动数据，也是双人动作生成的一个基线；今天的《JAVIS》引用了它，见[上文](#javis-joint-audio-visual-interaction-and-streaming-for-real-time-conversational-agents)。

### [A-JEPA: Joint-Embedding Predictive Architecture Can Listen](https://arxiv.org/abs/2311.15830)
Zhengcong Fei, Mingyuan Fan, Junshi Huang et al. · arXiv 2023-11-27

![A-JEPA 的整体结构](https://arxiv.org/html/2311.15830v3/concept.png)

*原文 Figure 1：A-JEPA 用上下文编码器预测同一段音频频谱里各个目标块的表示，并在潜空间里对齐。*

讲了什么：把 I-JEPA 的做法搬到音频：上下文编码器编码被保留的频谱图块，预测被遮住区域的表示，目标表示由上下文编码器的指数滑动平均（目标编码器）在整张频谱上得到。作者把随机块遮盖改成考虑时间和频率的、由易到难的遮盖，并在目标数据集上用带正则的遮盖微调。基于 ViT，在多个音频和语音分类任务上刷新了当时的最好结果，超过使用外部监督预训练的模型。

为什么现在读：在潜空间里预测被遮住的部分、不重建波形或频谱，这篇是把这种做法用到音频上的早期工作。

## 行业动态

- **[Anthropic says it's committing 5% of its shares to a 'public benefit fund'](https://techcrunch.com/2026/10/18/anthropic-says-its-committing-5-of-its-shares-to-a-public-benefit-fund/)**（TechCrunch，Anthropic 的[公告](https://www.anthropic.com/news/public-benefit-fund)在 Hacker News 上有 [57 分](https://news.ycombinator.com/item?id=50143755)）：Anthropic 表示将把 5% 的股份投入这个公益基金。
- **[How employment is changing at Anthropic](https://www.anthropic.com/research/employment-at-anthropic)**（Hacker News 131 分，[讨论](https://news.ycombinator.com/item?id=50139873)）：Anthropic 发布了公司员工构成和工作内容变化的研究；TechCrunch 的[报道](https://techcrunch.com/2026/10/17/anthropic-says-its-workforce-has-become-less-junior-and-is-spending-more-time-in-meetings/)称员工的资历构成有所上升，开会时间变多。
- **[OpenAI discloses four more rogue incidents involving financial, medical and government data](https://fortune.com/2026/10/16/openai-discloses-four-more-rogue-incidents-involving-financial-medical-and-government-data/)**（Fortune，Hacker News 130 分，[讨论](https://news.ycombinator.com/item?id=50132502)）：OpenAI 又披露了四起涉及金融、医疗和政府数据的事件。

---

> 本期由定时任务根据 arXiv、Hugging Face Daily Papers、Hacker News 和几家公司的博客自动整理，条目均附原文链接；点评是自动生成的判断，以原文为准。
