"""Figures for the MiniCPM-o 4.5 post: inline SVG, Nilou palette via the --fig-* CSS variables in assets/css/custom.css.
Run: python3 figs.py  ->  figs.json   (then assemble.py fills {{fig-x-y}} in post.md)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CSS = """  <style>
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .tick { stroke: currentColor; stroke-opacity: .35; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>"""


class F:
    def __init__(self, fid, h, label, W=640):
        self.fid, self.h, self.label, self.W, self.b, self.mk = fid, h, label, W, [], {}

    def w(self, s):
        self.b.append(s)

    def marker(self, var):
        if var not in self.mk:
            mid = f"{self.fid}-m{len(self.mk)}"
            fill = f'style="fill: var({var})"' if var else 'fill="currentColor" fill-opacity=".65"'
            self.mk[var] = (mid, f'<marker id="{mid}" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" '
                                 f'markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" {fill}/></marker>')
        return self.mk[var][0]

    def path(self, d, var=None, dash=False, width=1.5, head=True):
        st = f'style="stroke: var({var})"' if var else 'stroke="currentColor" stroke-opacity=".6"'
        ds = ' stroke-dasharray="5 3"' if dash else ''
        m = f' marker-end="url(#{self.marker(var)})"' if head else ''
        self.w(f'<path d="{d}" fill="none" {st} stroke-width="{width}"{ds}{m}/>')

    def arrow(self, x1, y1, x2, y2, var=None, dash=False, width=1.5):
        self.path(f"M{x1},{y1} L{x2},{y2}", var, dash, width)

    def box(self, x, y, w, h, l1, l2=None, l3=None, var=None, bold=True):
        st = f'style="fill: color-mix(in srgb, var({var}) 16%, var(--fig-bg)); stroke: var({var})" stroke-width="1.6"' if var else 'class="op"'
        self.w(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" {st}/>')
        lines = [l for l in (l1, l2, l3) if l]
        y0 = y + h / 2 - (len(lines) - 1) * 7.5 + 4
        for k, l in enumerate(lines):
            cls = "tb" if (k == 0 and bold) else ("t" if k == 0 else "s")
            self.w(f'<text class="{cls}" x="{x + w / 2}" y="{y0 + 15 * k:.1f}" text-anchor="middle">{l}</text>')

    def text(self, x, y, s, cls="lab2", anchor="start"):
        self.w(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{s}</text>')

    def html(self, caption):
        defs = "".join(m for _, m in self.mk.values())
        minw = f' style="--fg-minw: {540 * self.W / 640:.0f}px"' if self.W != 640 else ''
        return (f'<figure id="{self.fid}" class="fg-fig">\n<svg class="fg" viewBox="0 0 {self.W} {self.h}" width="100%"{minw} '
                f'role="img" aria-label="{self.label}">\n' + CSS + (f"\n  <defs>{defs}</defs>" if defs else "") + "\n"
                + "\n".join("  " + x for x in self.b) + f"\n</svg>\n<figcaption>{caption}</figcaption>\n</figure>")


# ── 图 1-1 架构 ───────────────────────────────────────────────────────────────
def arch():
    f = F("fig-1-1", 330, "MiniCPM-o 4.5 架构：视觉和音频编码器把输入变成 LLM 的嵌入，Qwen3-8B 出文本和隐状态，0.3B 的语音 token 解码器把两者相加作为条件，出 25 Hz 的 S3 token，再由 flow matching 合成波形")
    # inputs and encoders
    f.box(8, 40, 74, 44, "视频帧", "1–5 FPS", bold=False)
    f.box(104, 34, 120, 56, "SigLIP ViT", "0.4B", "448², 1024 patch", var="--fig-2")
    f.box(246, 34, 110, 56, "Resampler", "64 个 query", var="--fig-2")
    f.box(8, 150, 74, 44, "音频", "16 kHz", bold=False)
    f.box(104, 144, 120, 56, "Whisper-medium", "0.3B", "mel 100 Hz → 50 Hz", var="--fig-3")
    f.box(246, 144, 110, 56, "MLP + 均值池化", "1024→4096→4096", "5 帧取平均", var="--fig-3")
    f.arrow(82, 62, 102, 62); f.arrow(224, 62, 244, 62)
    f.arrow(82, 172, 102, 172); f.arrow(224, 172, 244, 172)
    # LLM
    f.box(392, 34, 112, 166, "Qwen3-8B", "36 层，d = 4096", "每秒 3–4 个文本 token", var="--fig-1")
    f.arrow(356, 62, 390, 62); f.text(373, 56, "64/帧", "s", "middle")
    f.arrow(356, 172, 390, 172); f.text(373, 166, "10/s", "s", "middle")
    f.arrow(504, 80, 560, 80); f.text(532, 74, "文本", "s", "middle")
    f.text(566, 84, "文本输出", "t")
    # hidden -> projector -> decoder
    f.box(392, 232, 112, 44, "Projector", "4096→768→768，L2", var="--fig-hi")
    f.arrow(448, 200, 448, 230, "--fig-hi"); f.text(454, 220, "末层 h", "s")
    f.box(526, 222, 106, 64, "语音 token", "解码器 0.3B", "Llama 20 层，d 768", var="--fig-hi")
    f.arrow(504, 254, 524, 254, "--fig-hi")
    f.text(515, 244, "+", "tb", "middle")
    f.text(392, 300, "解码器输入 = 文本嵌入 + 投影后的 h", "s")
    # flow matching
    f.box(526, 120, 106, 64, "flow matching", "+ HiFT 声码器", "参考音频定音色", var="--fig-mute")
    f.arrow(579, 220, 579, 186); f.text(585, 206, "S3 25/s", "s")
    f.arrow(579, 118, 579, 100); f.text(585, 114, "24 kHz", "s")
    f.text(8, 18, "箭头上的数字是进入下一模块的 token 数", "lab2")
    return f


# ── 图 2-1 一个 unit 的 token 排布 ────────────────────────────────────────────
def unit():
    f = F("fig-2-1", 150, "全双工模式下一个 1 秒 unit 的 token 顺序：unit 开始标记、图像 64 个嵌入、10 个音频嵌入，然后 LLM 采样 listen 或 speak，speak 后跟文本 token，最后 chunk_eos 和 unit 结束标记")
    cells = [("&lt;unit&gt;", 50, None, "标记"), ("&lt;image&gt;", 54, None, "标记"), ("64 个视觉嵌入", 96, "--fig-2", "输入"),
             ("&lt;/image&gt;", 58, None, "标记"), ("10 个音频嵌入", 92, "--fig-3", "输入"),
             ("&lt;|speak|&gt;", 64, "--fig-hi", "决策"), ("文本 ≤ 19", 70, "--fig-1", "输出"),
             ("&lt;|chunk_eos|&gt;", 80, "--fig-hi", "决策"), ("&lt;/unit&gt;", 50, None, "标记")]
    x = 8
    for i, (lab, w, var, kind) in enumerate(cells):
        if var:
            f.w(f'<rect x="{x}" y="44" width="{w}" height="34" rx="4" style="fill: color-mix(in srgb, var({var}) 18%, var(--fig-bg)); stroke: var({var})" stroke-width="1.4"/>')
        else:
            f.w(f'<rect x="{x}" y="44" width="{w}" height="34" rx="4" class="op"/>')
        f.w(f'<text x="{x + w / 2}" y="65" text-anchor="middle" font-size="10.5" fill="currentColor">{lab}</text>')
        x += w + 2
    # brackets: prefill vs generated
    f.path("M8,92 L8,100 L364,100 L364,92", head=False)
    f.text(186, 116, "prefill：音视频按到达顺序写进 KV cache", "s", "middle")
    f.path("M366,92 L366,100 L632,100 L632,92", "--fig-hi", head=False)
    f.text(499, 116, "LLM 自回归生成，遇到终止 token 停", "s", "middle")
    f.text(8, 26, "不说话时 &lt;|speak|&gt; 处是 &lt;|listen|&gt;，后面没有文本；一轮话说完先出 &lt;|turn_eos|&gt; 才能 listen", "lab2")
    f.text(8, 140, "每个 unit 对应真实时间 1 秒；视觉帧数由调用方决定，图中按 1 帧画", "lab2")
    return f


# ── 图 2-2 时序 ───────────────────────────────────────────────────────────────
def timing():
    f = F("fig-2-2", 250, "全双工推理的时序：第 k 秒的音视频攒满后才 prefill，LLM 生成这一 unit 的文本，语音解码器紧接着出 25 个 S3 token，合成出的约 1 秒语音在下一秒播放")
    X0, S = 96, 150  # x of t=0, px per second
    lanes = [(34, "输入"), (92, "LLM"), (150, "语音解码器"), (208, "播放")]
    for y, name in lanes:
        f.w(f'<rect class="band" x="4" y="{y - 22}" width="632" height="44" rx="6"/>')
        f.text(14, y + 4, name, "ttl")
    for k in range(4):
        x = X0 + k * S
        f.w(f'<line class="tick" x1="{x}" y1="10" x2="{x}" y2="236" stroke-dasharray="3 3"/>')
        f.text(x, 246, f"{k} s", "s", "middle")
    for k in range(3):
        x = X0 + k * S
        f.box(x + 4, 20, S - 8, 28, f"第 {k + 1} 秒音视频", var="--fig-3", bold=False)
    for k, d in enumerate(["listen", "speak + 文本"]):
        x = X0 + (k + 1) * S
        f.box(x + 2, 78, 92, 28, d, var="--fig-1", bold=False)
        f.arrow(x - 6, 50, x + 20, 76, "--fig-3")
    # speech for unit 2 (speak) generated right after LLM, played in next second
    x2 = X0 + 2 * S
    f.box(x2 + 80, 136, 66, 28, "25 个 S3", var="--fig-hi", bold=False)
    f.arrow(x2 + 94, 106, x2 + 104, 134, "--fig-hi")
    f.box(x2 + 144, 194, S - 58, 28, "约 1 秒语音", var="--fig-mute", bold=False)
    f.arrow(x2 + 140, 164, x2 + 156, 192)
    f.text(X0 + 4, 146, "和 LLM 在同一次调用里先后执行，", "s")
    f.text(X0 + 4, 160, "等这一 unit 的文本出完才开始", "s")
    return f


# ── 图 3-1 TAIL ────────────────────────────────────────────────────────────────
def tail():
    f = F("fig-3-1", 232, "TAIL 的分组方式：每个文本 token 按它在语音里的开始时间归到对应的 1 秒 chunk；跨过边界的词，读音落在下一 chunk；每个 chunk 末尾几个 token 的语音推迟到下一 chunk 生成，给发音留一点下文")
    X0, S = 70, 180
    for k in range(4):
        x = X0 + k * S
        f.w(f'<line class="tick" x1="{x}" y1="22" x2="{x}" y2="206" stroke-dasharray="3 3"/>')
        f.text(x, 220, f"{k} s", "s", "middle")
    f.text(8, 16, "示意：一句话的读音沿时间铺开，竖线是 chunk 边界", "lab2")
    words = [("The", 0.05, 0.25), ("apple", 0.28, 0.72), ("on", 0.76, 0.92), ("the", 0.95, 1.12), ("table", 1.15, 1.6),
             ("looks", 1.66, 2.02), ("very", 2.06, 2.38), ("fresh", 2.42, 2.95)]
    f.text(8, 56, "语音", "ttl")
    for i, (w, a, b) in enumerate(words):
        f.box(X0 + a * S, 38, (b - a) * S - 2, 26, w, var="--fig-mute", bold=False)
    f.text(8, 112, "文本", "ttl")
    chunk_of = lambda a: int(a)
    for i, (w, a, b) in enumerate(words):
        k = chunk_of(a)
        idx = [j for j, (_, aa, _) in enumerate(words) if chunk_of(aa) == k].index(i)
        x = X0 + k * S + 6 + idx * 42
        last = i + 1 == len(words) or chunk_of(words[i + 1][1]) != k
        f.box(x, 94, 38, 26, w, var="--fig-hi" if last else "--fig-1", bold=False)
        f.path(f"M{x + 19},{93} L{X0 + (a + b) / 2 * S:.1f},{66}", "--fig-hi" if last else None, dash=True, width=1.1, head=False)
    f.text(8, 160, "规则", "ttl")
    f.text(X0, 150, "开始时间落在 [(k−1)t, kt) 的文本 token 归第 k 个 chunk（t = 1 s），对应的语音 token 一起放进这个 chunk", "s")
    f.text(X0, 166, "红色是 chunk 末尾的 token：它的语音推迟到下一 chunk 生成（look-ahead），读 the 时已经知道后面是 table", "s")
    f.text(X0, 182, "前面若已慢了，下一 chunk 少出几个文本 token，让语音追上；每 chunk 的文本数随累计进度变化", "s")
    return f


CAPTIONS = {
    "arch": '<strong>图 1-1</strong> MiniCPM-o 4.5 的三块：编码器、LLM、语音解码器。数字取自论文表 13 和 HuggingFace 上的推理代码；音频 50 Hz 到 10 Hz 那一步，代码里是先过 MLP 再做 5 帧均值池化。',
    "unit": '<strong>图 2-1</strong> 全双工模式下一个 unit（1 秒）的 token 顺序，取自推理代码 <code>MiniCPMODuplex.streaming_prefill</code> 和 <code>streaming_generate</code>。',
    "timing": '<strong>图 2-2</strong> 推理时序示意：第 1 秒 listen，第 2 秒开始说话。每秒的输入攒满后才进 LLM，语音在下一秒播出，所以从收到输入到播出语音至少隔一个 chunk。',
    "tail": '<strong>图 3-1</strong> TAIL 的训练数据怎么分组（示意，词和时间是编的）。规则来自论文 3.4 节；推理代码里找不到显式的推迟逻辑，我的理解是解码器从这种数据排布里学会了推迟。',
}


if __name__ == "__main__":
    figs = {k: fn().html(CAPTIONS[k]) for k, fn in [("arch", arch), ("unit", unit), ("timing", timing), ("tail", tail)]}
    json.dump(figs, open(os.path.join(HERE, "figs.json"), "w"), ensure_ascii=False)
    print(len(figs), "figures")
