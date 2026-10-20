---
title: "AI 日报 - 2026/10/20"
date: 2026-10-20
draft: false
description: "LeJEPA 扩展到音视频，用带相关性的高斯目标替代各向同性目标；潜空间世界模型改用流形上的距离做规划；语音对话模型学会在合适的地方呼吸、笑和叹气。"
tags: ["AI", "Daily Report"]
categories: ["AI Notes"]
series: ["AI Daily Reports"]
---

## 今天值得看

- [MM-LeJEPA](#mm-lejepa-correlated-gaussian-regularization-for-multimodal-joint-embedding-predictive-architectures) 把 LeJEPA 的各向同性高斯目标推广成跨模态相关的高斯目标，只多一个超参数，用于音视频自监督学习。
- [Neighborhood Graph Cost](#latent-world-models-should-plan-with-distances-along-the-latent-manifold) 发现 LeWM 的 L2 规划代价只在目标附近可靠，改用训练 latent 近邻图上的最短路距离后，不重新训练就明显提高规划成功率。
- [Speech LLMs That Breathe](#speech-llms-that-breathe-modeling-non-verbal-vocalizations-in-dialogue) 覆盖 13 类非语言发声，让语音对话模型在共情和角色扮演对话里按上下文生成呼吸、笑声等事件。

## 全双工语音对话

### [Speech LLMs That Breathe: Modeling Non-Verbal Vocalizations in Dialogue](https://arxiv.org/abs/2610.25942)
Heyang Liu, Lixin Zhang, Jiayi Huang et al. · arXiv 2026-10-19

![非语言发声建模的整体流程](https://arxiv.org/html/2610.25942v1/main_tmp.png)

*原文 Figure 1：方法概览，包括带非语言发声的语音数据构造流程、覆盖的 13 类发声、以发声为中心的多阶段训练、NVV-aware GRPO 的奖励设计，以及两种场景下的交互示例。*

做了什么：覆盖呼吸、情感和话语功能三大类共 13 种非语言发声（NVV），用自动合成流程构造符合人类对话习惯的对话数据。通过多阶段微调和针对 NVV 的 GRPO，让语音交互模型在合适的上下文里生成这些事件。在 VocalNet 和 VITA-Qinyu 两种建模范式上，共情交互和角色扮演对话的表现力都有明显提升，作者还讨论了高质量自然语音生成对架构的要求。

为什么值得看：笑声、叹气、吸气这类事件在多数语音模型里被丢掉，这篇给出了从数据到奖励的完整做法。

### [VISAGENT: An Agent-Based End-to-End Voice Interaction System](https://arxiv.org/abs/2610.26168)
Xiang Wang, Zhaoxiang Liu, Xiang Chen et al. · arXiv 2026-10-19

![VISAGENT 的整体架构](https://arxiv.org/html/2610.26168v1/figures/framework.png)

*原文 Figure 2：VISAGENT 由全双工用户交互层、两级分层路由层和可扩展的语音 agent 市场组成，用户记忆服务为 agent 选择和任务执行提供共享上下文。*

做了什么：面向大规模公共服务的语音交互系统，把端到端语音交互、两级 agent 路由、语音 agent 市场和用户记忆放在一起。两级路由在数百个 agent 中做到亚秒级的意图到 agent 选择，用户记忆用于偏好感知的 agent 选择并给被选中的 agent 提供上下文。系统部署在中国的 114 热线上，实验显示全双工交互延迟低、路由可扩展，语音自然度有竞争力。

为什么值得看：这是一个全双工语音前端接大量后端 agent 的中文真实部署案例。

### [ASR-Style Transcript Normalization Removes Useful Pragmatic Signal for Dialogue Act Classification](https://arxiv.org/abs/2610.22934)
Lucky Onyekwelu-Udoka, Md Shafiqul Islam · arXiv 2026-10-16

做了什么：ASR 流程常会删掉填充停顿和标点、展开口语形式。作者在 Switchboard、ICSI MRDA、SLUE-HVB 三个语料上，用六种分类器比较九种转写条件。五个随机种子下，重度规范化让全部 15 个神经模型与语料组合的 Macro-F1 下降；去掉标点让 TF-IDF 基线在 Switchboard 和 MRDA 上分别下降 0.036 和 0.078；在 SLUE-HVB 上，原有的填充词表会删掉本身就定义类别的附和语。

为什么值得看：说明「嗯」「呃」和附和语这类信号对判断对话行为有用，转写时不应一律删掉。

## 表示学习与世界模型

### [MM-LeJEPA: Correlated Gaussian Regularization for Multimodal Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2610.24681)
Haowen Bai, Rui Gong, Meichen Liu et al. · arXiv 2026-10-17

![MM-LeJEPA 的整体结构](https://arxiv.org/html/2610.24681v1/MM-LeJEPA_pipeline.png)

*原文 Figure 1：MM-LeJEPA 概览。读取独立信息和读取共享内容的两类任务给出各自的任务梯度二阶矩，二者加权组合决定带相关性的联合目标；音频片段和视频帧提供局部视图。*

做了什么：LeJEPA 在一定任务假设下推导出各向同性高斯的嵌入目标，不需要重建解码器、EMA 教师或对比负样本。这篇把多模态任务建模为「分别从两种模态读独立信息」和「从两种模态读同一内容」两类成分的加权组合，推导出一个带跨模态相关性的高斯目标，并用解析变换把 SIGReg 改到这个目标上。只多一个标量超参数，即目标跨模态相关系数；在公开音视频分类基准上和代表性开源方法相当，代码将会发布。

为什么值得看：给音视频联合的 JEPA 训练提供了一个有推导依据的防坍缩正则，保留了 LeJEPA 的简洁。

### [Latent World Models Should Plan with Distances Along the Latent Manifold](https://arxiv.org/abs/2610.25013)
Zunzhi You, Linwei Tao, Siyu Xu et al. · arXiv 2026-10-18

做了什么：LeWM 这类潜空间世界模型用预测终点 latent 和目标 latent 的 L2 距离做规划代价。作者发现这个假设只在目标附近成立，离得远时 L2 代价会形成虚假的低谷困住规划器，而 latent 本身仍保留了物理状态。他们提出 Neighborhood Graph Cost（NGC），在训练 latent 上建互为近邻的图并取最短路距离，只替换终点代价、不重新训练。四个基准任务的成功率都提高，在最远的目标偏移下 TwoRoom 和 Reacher 分别达到 95% 和 100%，L2 代价只有 27% 和 81%。

为什么值得看：说明潜空间的几何形状本身会决定规划效果，距离度量是一个可以单独改的变量。

### [Learning Goal-Conditioned Subgoals through Frozen Hierarchical World Models](https://arxiv.org/abs/2610.26179)
Xiang Kaibo, Liu Qingfeng · arXiv 2026-10-19

做了什么：分层世界模型由粗粒度模型给出预测状态，再交给下层控制器，但上层规划要对候选宏动作轨迹反复求导，代价高。作者用一个轻量的目标条件网络替代这个搜索：网络通过冻结的高层预测器训练，输出条件向量，由预测器生成下一个潜空间路标。训练只用 hindsight 目标和状态预测损失，不需要搜索解或宏动作标签。在 H-JEPA 发布的共享潜空间 HWM checkpoint 上评测，四个环境中三个优于搜索基线，Push-T 持平，端到端加速 1.28 到 4.39 倍。

为什么值得看：在冻结的分层 JEPA 世界模型上去掉高层搜索，是分层潜空间预测走向实用的一步。

## GUI agent 与动作

### [Focused Omni-Modal GUI Agents](https://arxiv.org/abs/2610.26203)
Fuyao Zhang, Wenwen He, Jiaming Zhang et al. · arXiv 2026-10-19

![FocusGUI 的结构](https://arxiv.org/html/2610.26203v1/main1omni_cropped.png)

*原文 Figure 2：冻结主干的原有通路保持不变；FocusGUI 从同一观测中取出各感知通道，按每一步重新计算的决策参照打分，压缩成紧凑的 focus latent，以零初始化的方式加到主干上。*

做了什么：GUI agent 面对的是截图、音频、视频、交互历史和设备状态组成的流式全模态输入，每一步都平等地暴露全部上下文会让无关信号不断累积，导致轨迹漂移。作者把需求表述为 Focused Perception，实现为轻量适配器 FocusGUI：按当前步的决策参照给各模态通道排序，把最相关的证据压缩成一个 latent 注入冻结解码器的嵌入。只用动作监督训练，不需要模态相关性标注，也不微调主干。在四个全模态主干上，Type Match 平均提高 33.2，Exact Match 是冻结主干的 3 倍。

为什么值得看：全模态输入下每一步只看相关的部分，用一个冻结主干外的小模块就能明显改善动作准确率。

### [SynerGest: Temporal-Spatial Synergistic Gesture Generation for Dyadic Conversations](https://arxiv.org/abs/2610.24031)
Chengxiang Guo, Jian Wang, Lanmiao Liu et al. · arXiv 2026-10-16

![SynerGest 的整体框架](https://arxiv.org/html/2610.24031v1/Teaser.png)

*原文 Figure 1：SynerGest 概览。ITSA 选取节奏性的动作线索，CFM 把不匹配的音频和动作对区分开，以保持同步和动作多样性。*

做了什么：双人对话中的伴随语音手势生成常常在稳定和节奏响应之间难以兼顾，容易退化成平均姿态或出现细碎抖动。SynerGest 用对比式 flow matching（CFM），在训练时显式惩罚语义不匹配的样本对，减少过度平滑；再用 ITSA 模块把「什么在动」和「什么时候动」分开，分别建模语音驱动的节奏脉冲和连续的运动趋势。在两个公开数据集上达到最好结果。

为什么值得看：双人场景下同时考虑对方语音的手势生成，适合给虚拟形象做对话中的身体动作。

## 数据集

### [VoxDub: A Bilingual Lip-Synchronized Audio-Visual Dataset for Automatic Video Dubbing](https://arxiv.org/abs/2610.25259)
Kaidi Wang, Wenhao Guan, Peijie Chen et al. · arXiv 2026-10-18

![VoxDub 的数据构造流程](https://arxiv.org/html/2610.25259v1/overall.png)

*原文 Figure 1：VoxDub 数据集的构造流程概览。*

做了什么：发布 766K 个片段、约 1.58K 小时的中英双语自然场景音视频数据，覆盖不同说话人、场景、说话风格和语言内容。构造流程包括以语音为中心的切分、双 ASR 转写校验、语音质量过滤、人脸可见性检查和唇音同步验证。用它训练 TTS 比用 LRS3 和 CN-CVS 的词错率更低、说话人相似度更高。

为什么值得看：带可见人脸且唇音对齐的大规模中文音视频数据不多，这份数据可以直接用于音视频联合训练。

## 推理与部署

### [TideKV: Delayed KV Eviction via Layer-Wise Chunked Prefill for Memory-Efficient Long-Context LLM Inference](https://arxiv.org/abs/2610.24499)
Yu Wang, Zhu Wang, Ning Hu et al. · arXiv 2026-10-17

![TideKV 的逐层延迟淘汰](https://arxiv.org/html/2610.24499v1/Figure_2.png)

*原文 Figure 2：逐层延迟淘汰和对应的张量生命周期。完整的 prompt KV 只在当前层驻留，已完成的层保存压缩后的缓存，后面的层还没有缓存；每个 chunk 关注前面所有 chunk 累积的未压缩 KV。*

做了什么：KV 压缩能降低解码时的显存，但通常要先完整 prefill，构建阶段的峰值仍然很高；按 chunk 在线淘汰又会在每层看到完整 prompt 之前就做选择。TideKV 改成按层处理：在每个 Transformer 层内跑完所有 prompt chunk，只为当前层保留完整 KV，在层末尾再淘汰。在 A100 上 64K 长度时构建阶段的临时显存减少 83.89%；在 RTX 4090 上 64K 的首 token 延迟只增加 1.06%，并能完成 128K 的 prefill。

为什么值得看：消费级显卡上长上下文的瓶颈常在 prefill 峰值，这个调度在不改压缩方法的前提下解决了它。

### [Tandem: Serving Hybrid Linear-Attention GUI Agents on Mobile NPUs](https://arxiv.org/abs/2610.26660)
Zhuomin Chen, Jingchao Ni, Xu Zheng et al. · arXiv 2026-10-19

![Tandem 的服务架构](https://arxiv.org/html/2610.26660v1/figures/figure4.png)

*原文 Figure 4：Tandem 的服务架构，包括跨请求的循环状态检查点、用于投机解码的临时回滚状态和空闲时的 NPU 会话控制。*

做了什么：手机 GUI agent 每一步都要调用模型，延迟随步数累积。混合线性注意力模型省显存，但复用循环状态需要在合适的位置打检查点，而 agent 请求会保留指令、重写最近的历史。Tandem 在相邻 prompt 分叉的位置保存循环状态，让后续请求复用已处理的上下文；大部分 prefill 留在 NPU 矩阵单元上，回滚状态由一小段逐 token 的尾部生成；请求间隙释放 NPU 功耗占用而不卸载模型。基于 llama.cpp 实现，在骁龙 8 Elite 上文本屏幕描述的每步延迟降低 30%。

为什么值得看：混合线性注意力模型在 agent 场景下的状态复用和投机解码，有一些和聊天场景不同的具体问题。

## 推荐读的论文

### [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
Aaron van den Oord, Yazhe Li, Oriol Vinyals · 2018

![CPC 的整体结构](https://arxiv.org/html/1807.03748v2/contrastive_repr4.png)

*原文 Figure 1：对比预测编码（CPC）的概览。图中以音频为输入，图像、文本和强化学习用的是同样的设置。*

讲了什么：提出对比预测编码（CPC），一种从高维数据中提取表示的通用无监督方法。核心是用自回归模型在潜空间里预测未来，并用概率式对比损失让潜空间保留对预测未来样本最有用的信息，负采样让训练可行。同一方法在语音、图像、文本和 3D 环境强化学习四个领域都学到了有用的表示。

为什么现在读：它把自监督学习表述为在潜空间而不是原始信号上预测未来，是这一路线的早期代表工作；今天的《[SynerGest](#synergest-temporal-spatial-synergistic-gesture-generation-for-dyadic-conversations)》引用了它。

### [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)
Alexandre Défossez, Laurent Mazaré, Manu Orsini et al. · 2024 · [代码](https://github.com/kyutai-labs/moshi)

![Moshi 的整体结构](https://arxiv.org/html/2410.00037v2/figures/overview_moshi_v2.png)

*原文 Figure 1：Moshi 概览。它是一个支持实时语音对话的语音文本基础模型，主要组成包括自研的文本语言模型主干 Helium，以及带残差向量量化和语义知识的神经音频 codec。*

讲了什么：把语音对话直接表述为语音到语音的生成，不再依赖 VAD、ASR、文本对话和 TTS 组成的流水线。模型从文本语言模型主干出发，生成神经音频 codec 残差量化器的 token，并把自己和用户的语音建模成并行的两路流，因此不需要显式的说话人轮次，可以表示重叠、打断和插话。它还先预测与时间对齐的文本 token 作为音频 token 的前缀（Inner Monologue），理论延迟 160 ms，实际约 200 ms。

为什么现在读：它是第一个实时全双工的语音大模型，用双流建模去掉了显式的说话人轮次；今天的《[Speech LLMs That Breathe](#speech-llms-that-breathe-modeling-non-verbal-vocalizations-in-dialogue)》引用了它。

## 行业动态

- **[Nvidia launches new open models to run agents in the enterprise and on PCs](https://techcrunch.com/2026/10/19/nvidia-launches-new-open-models-to-run-agents-in-the-enterprise-and-on-pcs/)**（TechCrunch）：Nvidia 发布新的开放模型，面向企业在自有数据中心部署，以及在设备端运行私有 AI agent。
- **[Tolan snags $30M for its AI companion that helps users navigate life](https://techcrunch.com/2026/10/19/tolan-snags-30m-for-its-ai-companion-that-helps-users-navigate-life/)**（TechCrunch）：AI 陪伴应用 Tolan 在活跃用户超过 25 万之后又融资 3000 万美元。
- **[Yelp can now call on your behalf to check wait times at a restaurant](https://techcrunch.com/2026/10/20/yelp-can-now-call-on-your-behalf-to-check-wait-times-at-a-restaurant/)**（TechCrunch）：Yelp 在应用里加入多项 AI 功能，包括代用户打电话询问餐厅等位时间，以及记住用户偏好的个性化推荐。

---

> 本期由定时任务根据 arXiv、Hugging Face Daily Papers、Hacker News 和几家公司的博客自动整理，条目均附原文链接；点评是自动生成的判断，以原文为准。
