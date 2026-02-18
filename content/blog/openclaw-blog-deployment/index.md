---
title: "用 OpenClaw + Claude 从零搭建个人博客的全过程"
date: 2026-02-17
draft: false
description: "记录如何使用 OpenClaw AI Agent 搭配 Claude Opus，通过自然语言对话完成 Hugo 博客的搭建、主题定制、部署上线的完整过程。一次人机协作的实验。"
tags: ["OpenClaw", "AI", "Hugo", "博客搭建", "人机协作"]
categories: ["AI笔记"]
series: ["AI笔记"]
series_order: 1
---

## 起因

我一直想搭一个个人技术博客，但每次都卡在"选主题 → 改样式 → 写配置 → 部署"的循环里。作为一个图形学方向的程序员，我对前端没什么执念，只想要一个**好看、能用、能写东西**的地方。

然后我遇到了 [OpenClaw](https://openclaw.ai)。

OpenClaw 是一个 AI Agent 平台，可以让 Claude 这样的大模型直接操控你的电脑——读写文件、执行命令、操作浏览器、管理定时任务，甚至通过 WhatsApp/Telegram 跟你对话。它不是一个简单的聊天机器人，而是一个**有手有脚的 AI 助手**。

我决定做一个实验：**完全通过自然语言对话，让 AI 帮我从零搭建一个博客。**

---

## Day 1：从零到上线（2026-02-13）

### "帮我搭一个博客"

我在 OpenClaw 的 Web Chat 里打了这句话。接下来的事情几乎全是 AI 自动完成的：

1. **选型**：AI 推荐了 Hugo（静态站点生成器）+ Blowfish 主题（功能丰富、支持暗色模式）
2. **安装**：自动检测到我的 Windows 环境，找到 Hugo 的安装路径
3. **初始化**：`hugo new site`，配置 Blowfish 作为 Git submodule
4. **GitHub 部署**：创建 `noviorlu.github.io` 仓库，写好 GitHub Actions workflow，推送上线

整个过程大概 **20 分钟**，期间我只做了一件事：去 GitHub 把 Pages 的 source 从 branch 切换到 GitHub Actions。

### 第一个问题：Hugo 不在 PATH 里

AI 在第一次运行 `hugo` 命令时失败了，因为我是通过 WinGet 安装的 Hugo，路径是一个超长的：

```
C:\Users\ychen\AppData\Local\Microsoft\WinGet\Packages\Hugo.Hugo.Extended_Microsoft.Winget.Source_8wekyb3d8bbwe\hugo.exe
```

AI 自己发现了这个问题，搜索了文件系统找到了正确路径，之后所有命令都用了完整路径。**没有问我，自己解决了。**

---

## Day 1-2：Endfield 主题定制

我是 [明日方舟：终末地](https://endfield.hypergryph.com/) 的玩家，很喜欢它的工业废土美学——暗色调、`#FFD100` 黄色高亮、几何线条、数据终端风格。

我跟 AI 说："我想要 Arknights Endfield 风格的暗黑工业设计。"

AI 随后做了这些：

### 视觉系统
- **配色方案**：深灰/黑色背景 + `#FFD100` 黄色主色调
- **字体**：`JetBrains Mono` 等宽字体用于标题和 UI 元素
- **标题 Glitch 效果**：微妙的故障动画（透明度 20%/15%，偏移 ≤1px）
- **粒子背景**：浮动的几何粒子，Endfield 数据终端感
- **阅读进度条**：页面顶部的黄色进度条

### 标签多色系统
这个比较有意思。AI 设计了一套 10 色标签系统——黄、冰蓝、琥珀、青绿、紫、绿、青、粉、红、橙——每个标签根据文字内容 hash 到固定颜色。

但第一版没生效！原因是 Blowfish 的标签 HTML 结构是 `<a> → <span> → <span>`，Tailwind 的 `dark:text-primary-400` 样式加在内层 span 上，CSS 优先级打不过。

AI 花了两轮迭代才修好：
1. 第一轮：只给外层 `<a>` 加 class → 无效（内层 span 的 Tailwind 覆盖了）
2. 第二轮：JS 改为同时给 `<a>` 和所有内层 `<span>` 都加 class，CSS 用 `span.ef-tag-X` 提高优先级 → 成功

**这就是真实的调试过程——AI 也会犯错，但它能自己发现问题、分析原因、迭代修复。**

---

## 过程中的 Bug 修复

搭建过程中遇到了不少问题，列几个印象深刻的：

### 内容居中问题
Blowfish 的默认布局在宽屏上内容太散。AI 用 `main.grow > *` 选择器统一限制了所有直接子元素的宽度，配合 CSS 变量 `--ef-wide-width: 1000px` 和 `--ef-content-width: 800px`。

### Series 里的 § 符号
文章底部的系列列表显示 `§ : 本文` 而不是正常的编号。原因是 Blowfish 的 i18n 文件把 `article.part` 翻译成了 `§`。

解决方案很简单：在站点根目录创建 `i18n/zh-CN.yaml`，覆盖翻译：
```yaml
article:
  part: "第"
  this_article: "本文"
```

### 分页组件乱掉
文章底部的上一篇/下一篇排版混乱。AI 直接 override 了 `layouts/partials/article-pagination.html`，用 inline grid 布局重写。

---

## WhatsApp 远程操控

OpenClaw 最酷的功能之一：**你可以通过 WhatsApp 跟 AI 对话，它会在你的电脑上执行操作。**

我后来的大部分修改都是在手机上通过 WhatsApp 发消息完成的。比如：

> 我："标签颜色太单调了，参考 Endfield 的配色多选几个"

AI 就会去改 CSS 和 JS，commit，push，然后告诉我刷新看效果。

这种体验很奇妙——**你在任何地方，用手机发一条消息，AI 就帮你改代码、部署上线。**

---

## 技术栈总结

| 组件 | 选择 |
|------|------|
| 静态站点生成器 | Hugo |
| 主题 | Blowfish |
| 部署 | GitHub Pages + GitHub Actions |
| AI Agent | OpenClaw + Claude Opus |
| 沟通渠道 | Web Chat + WhatsApp |
| 版本控制 | Git → GitHub |
| 自定义样式 | CSS + JS（约 900 行） |

---

## 感受

### AI 做得好的地方
- **自动发现和解决环境问题**（Hugo 路径、PowerShell 语法）
- **迭代式修复**：遇到 bug 会分析原因，提出解决方案，验证结果
- **代码质量还行**：CSS/JS 结构清晰，有注释，变量命名合理
- **记忆力**：OpenClaw 有持久化记忆系统，跨 session 能记住之前的上下文

### AI 做得不好的地方
- **CSS 优先级问题**需要多轮才能修好——AI 对 Tailwind 和自定义 CSS 的优先级交互理解不够深
- **有时过度工程化**：比如一些简单的样式修改，AI 会创建一套完整的系统
- **看不到页面**：AI 无法直接看到渲染效果（浏览器连接不稳定），需要依赖我截图反馈

### 这种协作模式的本质

这不是"AI 替你写代码"——更像是**你有一个不知疲倦的初级工程师**，你说需求，他去实现，遇到问题会自己想办法，做完了来找你 review。

你仍然需要：
- 知道自己想要什么
- 能判断结果好不好
- 在 AI 卡住时给出方向

但你不需要：
- 记住每个配置文件的语法
- 手动调 CSS
- 写 GitHub Actions workflow

---

## 下一步

- 把知乎上的图形学笔记迁移过来
- 继续完善 Endfield 主题设计
- 看看 OpenClaw 的定时任务功能能玩出什么花样

---

*这篇文章的撰写过程本身也是通过 OpenClaw 完成的——我在 WhatsApp 上说"把我们的对话写成博客"，AI 就生成了这篇文章，然后自动 commit 和 push 到 GitHub。*

*Meta enough?* 🤖
