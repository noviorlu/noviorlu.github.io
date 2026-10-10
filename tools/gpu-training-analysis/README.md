# gpu-training-analysis 的图和正文源

- `post_v2.md`：正文模板，图用 `{{fig-x-y}}` 占位。改正文改这里。
- `figs.py`：生成全部 inline SVG 图（Nilou 配色，走 `--fig-*` CSS 变量），输出 `figs.json`。
- `assemble.py`：把 figs.json 填进模板，写成 `content/blog/gpu-training-analysis/index.md`。
- `*.json`：图用到的实测数据。
- `rmsnorm_triton.py`：第 3 节折叠代码块的源（eager 和手写 Triton RMSNorm）。`# post:<name>` 到 `# post:end` 之间的代码由 assemble.py 按 `{{code:rmsnorm_triton.py:<name>}}` 贴进正文。直接运行是和 eager 对照的自检，没有 GPU 时用 `TRITON_INTERPRET=1`。

```bash
cd tools/gpu-training-analysis
python3 figs.py
python3 assemble.py post_v2.md figs.json ../../content/blog/gpu-training-analysis/index.md
```

不要直接改 index.md，下次 assemble 会覆盖。

## 系列第一篇（GPU 与 Triton 入门）

`post_intro.md` 是 `content/blog/gpu-triton-intro/index.md` 的模板，和主文章共用 `figs.py` / `figs.json`（图 1-1、1-2 的 id 是 `fig-h-1`、`fig-h-2`）：

```bash
python3 assemble.py post_intro.md figs.json ../../content/blog/gpu-triton-intro/index.md
```

Triton 例子的代码在 `triton_tutorial.py`，直接运行是和 PyTorch 对照的自检和计时。
