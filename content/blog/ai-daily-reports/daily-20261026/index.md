---
title: "AI 简报 - 2026/10/26"
date: 2026-10-26
draft: false
description: "全双工语音模型在几十分钟的长对话里都会退化，滑窗模型丢掉第一个 KV 项后出现乱码和沉默；I-JEPA 的预测器加宽到编码器宽度后，共享因子的可读性反超 SimCLR 和 BYOL；一个流式视频语言模型同时为电竞直播导播和解说。"
tags: ["AI", "Daily Report"]
categories: ["AI Notes"]
series: ["AI Daily Reports"]
---

## 本期值得看

- [Why Full-Duplex Speech Models Collapse in Long Conversations](#why-full-duplex-speech-models-collapse-in-long-conversations) 发现五个全双工语音模型在长对话里都会退化或耗尽上下文，滑窗模型的乱码和沉默与第一个 KV 项被淘汰有关，固定保留它并加一个重复惩罚，不重新训练就能撑过一小时的回放。
- [Shared Is Not Enough](#shared-is-not-enough-predictor-width-and-factor-readability-under-masked-latent-prediction) 在因子已知的模拟器里发现，I-JEPA 用半宽的预测器时共享因子的可读性只有 SimCLR 的四成左右，把预测器加宽到编码器宽度后排序在每个随机种子上都反过来。
- [DirCast](#dircast-live-esports-directing-and-commentary-with-a-single-streaming-video-language-model) 用一个流式视频语言模型从每位选手的原始画面出发，每秒决定切不切镜头并同时解说。

## 全双工语音对话

### [Why Full-Duplex Speech Models Collapse in Long Conversations](https://arxiv.org/abs/2610.32349)
Yeonju Kim, Bella Godiva, Se Jin Park et al. · arXiv 2026-10-23

做了什么：全双工语音模型在听的时候也每帧生成一个 token，几分钟就会用完上下文窗口，而现有基准只测单轮或几轮。作者提出 FD-LHB 基准，用长对话回放和与 LLM 对话方的实时对话测试五个模型，发现所有模型最终都会退化或耗尽上下文：保留完整历史的模型往往在上下文用完之前就退化，滑窗模型也会陷入重复、沉默和乱码。乱码和沉默与滑窗丢掉第一个 KV 项有关（Initialization-Anchor Eviction），重复还会被模型自己最近的输出强化；据此提出固定保留第一个 KV 项的 IAP 和抑制复读、同时防止长时间沉默的 DRP，不重新训练就在最长一小时的回放中明显减少了对话崩溃，并保持实时运行。

为什么值得看：它把全双工模型的长时稳定性问题定位到滑窗淘汰的具体位置，给出的两个修复都只在推理时生效。

### [CounselPlex: A Stressor-Aware Full-Duplex Counseling Model](https://arxiv.org/abs/2610.32702)
Yeonju Kim, Bella Godiva, Yong Man Ro · arXiv 2026-10-23

![CounselPlex 的整体框架](https://arxiv.org/html/2610.32702v1/figure2.png)

*原文 Figure 2：CounselPlex 框架概览。对方沉默时，模型以一段简短的思考生成更新后的压力源状态；对方重新开口后，这个状态被强制注入到 \<PAD\> 位置，让 agent 用它来理解接下来的话。*

做了什么：全双工语音模型在心理咨询中容易只关注最近一句话，忘掉求助者背后的问题，作者把这种失败称为 Stressor Blindness。CounselPlex 在求助者沉默时更新一个简短的压力源状态，在听的时候把这个状态注入 \<PAD\> 位置。作者还提出 RoPE 位置偏移增强（RPSA），在训练时扩大位置覆盖范围，在更深的轮次上表现更好。LLM 评审显示咨询质量和对压力源的把握都优于强基线。

为什么值得看：全双工模型在听的时候输出的 \<PAD\> 位置通常是空的，这篇把这些位置用来放持续更新的对话状态。

## 表示学习与世界模型

### [Shared Is Not Enough: Predictor Width and Factor Readability Under Masked Latent Prediction](https://arxiv.org/abs/2610.32562)
Nilotpal Kakati · arXiv 2026-10-23

做了什么：I-JEPA 这类掩码潜空间预测方法学的是可预测的特征，但哪些信息会被保留并不清楚。作者搭了一个图像背后每个因子都已知的模拟器，用探针测量训练后各因子的可读程度，在相同的视图对上用 I-JEPA、SimCLR 和 BYOL 训练 0.7M 参数的小编码器，并探测两个视图共享的因子。I-JEPA 的预测器宽度和深度是编码器的一半（与它在 ViT-B 上的比例相同）时，中位分数约为 SimCLR 的 40%；把预测器加宽到编码器宽度后，在每个训练种子上都反超另外两种方法。模拟器和训练好的模型都已发布。

为什么值得看：它说明 JEPA 学到什么信息和预测器的容量直接相关，预测器宽度是一个需要单独调的设计变量。

### [Cola2: Co-Adaptive Hierarchical Language Modeling with Continuous Semantic Compression and Balanced Discrete Generation](https://arxiv.org/abs/2610.31305)
Hongcan Guo, Yian Zhao, Feng Wang et al. · arXiv 2026-10-22

做了什么：分层语言生成把连续的语义建模和离散的文本生成分开。Cola2 把文本动态切成语义片段，用不需要辅助损失的负反馈控制压缩率，并结合重建和生成两种监督学习紧凑、可预测的潜表示；解码器是块因果的掩码扩散模型，用干净、加噪和丢弃三种潜条件训练，用无分类器引导调节语义重建和自主生成之间的平衡。token 预测的梯度经过潜表示更新编码器，而潜空间上的 flow matching 先验通过 stop-gradient 与表示学习解耦；推理时生成的文本按固定的片段划分重新编码，让潜历史和实际文本一致。在约 2100 亿训练 token 上的五个下游任务验证了这些设计，扩展趋势还没有饱和；一个未收敛的 checkpoint 在八个任务上平均 60.10，在 HellaSwag 和 StoryCloze 上超过 Llama-3-8B，而后者报告的预训练量大 43 倍以上。

为什么值得看：它给出了连续潜空间预测加离散解码这种分层结构在语言建模上可以扩展的证据，并逐项说明了压缩率、梯度路径和潜历史一致性的做法。

## 流式视频理解

### [DirCast: Live Esports Directing and Commentary with a Single Streaming Video-Language Model](https://arxiv.org/abs/2610.31734)
Zixian Gao, Lixin Xiu, Penghao Luo et al. · arXiv 2026-10-22

![DirCast 的整体框架](https://arxiv.org/html/2610.31734v1/framework_v3.png)

*原文 Figure 2：框架。每秒用一个低分辨率 ViT 编码十路画面及其状态，导播决定保持还是切换，被选中的画面以高分辨率读入，然后解说员开口。低分辨率画面会过期，播出内容保留。*

做了什么：电竞直播需要导播每秒决定播哪个机位，解说员跟着画面说话，而且都不知道接下来会发生什么。已有的流式视频语言模型只会对人工导播后的画面做解说，这篇研究的是直接接收每位选手的原始画面、只依据过去信息同时导播和解说。DirCast 每秒以低分辨率扫描所有画面，用保持或切换的导播头决定是否切、切到哪里，再以高分辨率读入选中的画面后说话；上下文只保留播出过的画面和说过的话，其他画面过期，因此任意长度的直播开销都有上界。作者构建了 27 场职业 Dota 2 比赛的数据集，每个英雄一路画面，并带有人工导播的选择和解说。DirCast 的导播效果好于基于规则和零样本的导播（包括 GPT-5），在 Proact-VL 协议下解说对 GPT-4o 的胜率为 29.2%，十一个观看人工导播画面的流式解说模型中最强的是 19.4%。

为什么值得看：它把「看哪里」和「说什么」放进同一个流式模型，并用过期机制给长时间直播的上下文开销设了上界。

## 推荐读的论文

### [SALMONN-omni: A Codec-free LLM for Full-duplex Speech Understanding and Generation](https://arxiv.org/abs/2411.18138)
Wenyi Yu, Siyin Wang, Xiaoyu Yang et al. · 2024

![SALMONN-omni 的整体结构](https://arxiv.org/html/2411.18138v1/SALMONN-omni-3.png)

*原文 Figure 2：SALMONN-omni 是一个端到端模型，由流式语音编码器、LLM 和流式语音合成器组成，三者之间都通过嵌入连接；它采用新的无 codec 全双工语音对话框架，并加入了「思考」机制。*

讲了什么：提出 SALMONN-omni，一个不用 codec 的全双工语音理解与生成模型，说话的同时能听到自己生成的语音和背景声音。它的全双工对话框架带有「思考」机制，文本和语音可以异步生成，各部分之间传递的是嵌入而不是量化的语音和音频 token。在语音识别、语音增强和语音问答等流式任务上表现全面，在轮次转换、打断和回声消除场景下处理得好。

为什么现在读：多数全双工语音模型建立在离散音频 token 上，它是第一个用连续嵌入连接各部分的同类模型；本期的《[CounselPlex](#counselplex-a-stressor-aware-full-duplex-counseling-model)》引用了它。

### [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
Zirui Liu, Jiayi Yuan, Hongye Jin et al. · 2024 · [代码](https://github.com/jy-yuan/KIVI)

讲了什么：批量变大、上下文变长后，KV cache 的显存占用成为推理速度和内存的瓶颈，加载 KV cache 时计算单元还会空等。作者系统研究了常见 LLM 中 KV cache 元素的分布，发现 key cache 应按通道量化，value cache 应按 token 量化，据此提出无需调参的 2 bit KV cache 量化算法 KIVI。在 Llama、Falcon 和 Mistral 上质量几乎不变，峰值显存（含模型权重）降低 2.6 倍，批量可以增大到 4 倍，实际推理吞吐提高 2.35 到 3.47 倍。

为什么现在读：它从 key 和 value 的分布差异出发给出了非对称的量化方式，是后续 KV cache 量化工作常用的基线。

## 行业动态

- **[Show HN: Hurston-1 open source TTS that speak Southern American and Afro-English](https://huggingface.co/destined/hurston-1)**（Hacker News）：开源 TTS 模型 Hurston-1 发布在 Hugging Face 上，能说美国南方口音英语和非裔英语。

---

> 本期由定时任务根据 arXiv、Hugging Face Daily Papers、Hacker News 和几家公司的博客自动整理，条目均附原文链接；点评是自动生成的判断，以原文为准。
