# gpu-training-analysis 的图和正文源

- `post_v2.md`：正文模板，图用 `{{fig-x-y}}` 占位。改正文改这里。
- `figs.py`：生成全部 inline SVG 图（Nilou 配色，走 `--fig-*` CSS 变量），输出 `figs.json`。
- `assemble.py`：把 figs.json 填进模板，写成 `content/blog/gpu-training-analysis/index.md`。
- `*.json`：图用到的实测数据。

```bash
cd tools/gpu-training-analysis
python3 figs.py
python3 assemble.py post_v2.md figs.json ../../content/blog/gpu-training-analysis/index.md
```

不要直接改 index.md，下次 assemble 会覆盖。
