# minicpm-o-4-5 的图和正文源

- `post.md`：正文模板，图用 `{{fig-x-y}}` 占位。改正文改这里。
- `figs.py`：生成全部 inline SVG 图（Nilou 配色，走 `--fig-*` CSS 变量），输出 `figs.json`。
- `assemble.py`：把 figs.json 填进模板，写成 `content/blog/minicpm-o-4-5/index.md`。

```bash
cd tools/minicpm-o-45
python3 figs.py
python3 assemble.py post.md figs.json ../../content/blog/minicpm-o-4-5/index.md
```

不要直接改 index.md，下次 assemble 会覆盖。
