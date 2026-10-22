"""All figures of the GPU-analysis post as theme-aware inline SVG, in the Nilou palette.

Colours come from CSS custom properties defined once in assets/css/custom.css:
  --fig-1 primary (navy; sky in dark)   W, matmul, forward
  --fig-2 steel                          A (activations)
  --fig-3 sky (deep steel in dark)       T (temporaries), baselines
  --fig-hi red-orange                    G, element-wise ops, S/P, memory traffic
  --fig-mute gray                        Adam state, totals, free tensors
  --fig-bg cream                         operator boxes
Text is always currentColor; hue only marks identity.
"""
import json, math

import os
HERE = os.path.dirname(os.path.abspath(__file__))
CSS = """  <style>
    .fg .grid { stroke: currentColor; stroke-opacity: .1; }
    .fg .axis { stroke: currentColor; stroke-opacity: .35; }
    .fg .ref { stroke: currentColor; stroke-opacity: .45; stroke-dasharray: 4 4; }
    .fg .tick { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .val { font-size: 11px; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    .fg .ring { stroke: var(--nv-bg, #fff); stroke-width: 2; }
    .fg .halo { fill: none; stroke: var(--nv-bg, #fff); stroke-width: 6px; stroke-linejoin: round; opacity: 1; }
    .fg g.m:hover > :not(title) { opacity: .85; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>"""

class F:
    def __init__(self, fid, h, label, W=640):
        self.fid, self.h, self.label, self.b, self.mk, self.W = fid, h, label, [], {}, W
    def w(self, s): self.b.append(s)
    def marker(self, var):
        if var not in self.mk:
            mid = f"{self.fid}-m{len(self.mk)}"
            fill = f'style="fill: var({var})"' if var else 'fill="currentColor" fill-opacity=".65"'
            self.mk[var] = (mid, f'<marker id="{mid}" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" {fill}/></marker>')
        return self.mk[var][0]
    def arrow(self, x1, y1, x2, y2, var=None, dash=False, width=1.6, head=True):
        st = f'style="stroke: var({var})"' if var else 'stroke="currentColor" stroke-opacity=".6"'
        d = ' stroke-dasharray="5 3"' if dash else ''
        m = f' marker-end="url(#{self.marker(var)})"' if head else ''
        self.w(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {st} stroke-width="{width}"{d}{m}/>')
    def path(self, d, var=None, dash=False, width=1.6, head=True):
        st = f'style="stroke: var({var})"' if var else 'stroke="currentColor" stroke-opacity=".6"'
        ds = ' stroke-dasharray="6 4"' if dash else ''
        m = f' marker-end="url(#{self.marker(var)})"' if head else ''
        self.w(f'<path d="{d}" fill="none" {st} stroke-width="{width}"{ds}{m}/>')
    def html(self, caption):
        defs = "".join(m for _, m in self.mk.values())
        minw = f' style="--fg-minw: {540 * self.W / 640:.0f}px"' if self.W != 640 else ''
        return (f'<figure id="{self.fid}" class="fg-fig">\n<svg class="fg" viewBox="0 0 {self.W} {self.h}" width="100%"{minw} role="img" aria-label="{self.label}">\n'
                + CSS + (f"\n  <defs>{defs}</defs>" if defs else "") + "\n" + "\n".join("  " + x for x in self.b)
                + f"\n</svg>\n<figcaption>{caption}</figcaption>\n</figure>")

def tint(var, pct=16, stroke=None, sw=1.4, dash=False):
    s = stroke or var
    d = ' stroke-dasharray="5 3"' if dash else ''
    return f'style="fill: color-mix(in srgb, var({var}) {pct}%, transparent); stroke: var({s})" stroke-width="{sw}"{d}'

def solid(var, stroke=None):
    return f'style="fill: var({var})' + (f'; stroke: var({stroke})" stroke-width="1"' if stroke else '"')

def legend(f, items, x, y, gap=22):
    for name, var, kind in items:
        if kind == "box":
            extra = '; stroke: var(--fig-2)' if var == "--fig-3" else ''
            f.w(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" style="fill: var({var}){extra}"/>')
        elif kind == "fill":
            f.w(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{var}"/>')
        elif kind == "line":
            f.w(f'<line x1="{x}" y1="{y - 4}" x2="{x + 12}" y2="{y - 4}" style="stroke: var({var})" stroke-width="2"/>')
        elif kind == "dash":
            f.w(f'<line x1="{x}" y1="{y - 4}" x2="{x + 12}" y2="{y - 4}" stroke="currentColor" stroke-width="1.6" stroke-dasharray="4 3"/>')
        elif kind == "x":
            f.w(f'<text x="{x + 2}" y="{y}" class="val" {"" if var is None else "opacity=.55"}>×</text>')
        f.w(f'<text class="lab" x="{x + 16}" y="{y}">{name}</text>')
        x += 16 + sum(12 if ord(c) > 0x2E80 else 6.6 for c in name) + gap

def rect(f, x, y, w, h, var, tip, rx=3, stroke=None):
    st = f'fill: var({var})' + (f'; stroke: var({stroke})' if stroke else '')
    f.w(f'<g class="m"><title>{tip}</title><rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}" style="{st}"/></g>')

def mt(s):
    """X_{L-1}, W_L, ^T -> tspans (italic X/W)."""
    out, cur, i = [], 0, 0
    while i < len(s):
        c = s[i]
        if c in "_^":
            i += 1
            if s[i] == "{":
                j = s.index("}", i); grp = s[i + 1:j]; i = j + 1
            else:
                grp = s[i]; i += 1
            tgt = 4 if c == "_" else -6
            out.append(f'<tspan dy="{tgt - cur}" font-size="10">{grp}</tspan>'); cur = tgt
            continue
        pre = f'<tspan dy="{-cur}">' if cur else ''
        post = '</tspan>' if cur else ''
        cur = 0
        out.append(pre + (f'<tspan font-style="italic">{c}</tspan>' if c in "XW" else c) + post)
        i += 1
    if cur:
        out.append(f'<tspan dy="{-cur}"> </tspan>')
    return "".join(out)

# ── 图 2-1 Linear 前向 / 反向（after the author's drawio sketch） ──────────────
def linear():
    f = F("fig-2-1", 444, "一个 Linear 的前向与反向：前向从输入 X_{L-1} 算出 X_L，进入深层；反向收到误差后，用留下来的 X_{L-1} 算参数梯度，用 W_L 算激活梯度，再传给浅层")
    def box(cx, cy, bw, bh, title, formula, style):
        f.w(f'<rect x="{cx - bw / 2:.0f}" y="{cy - bh / 2:.0f}" width="{bw}" height="{bh}" rx="8" {style}/>')
        if formula:
            f.w(f'<text class="s" x="{cx}" y="{cy - 9}" text-anchor="middle">{title}</text>')
            f.w(f'<text class="t" x="{cx}" y="{cy + 13}" text-anchor="middle">{mt(formula)}</text>')
        else:
            f.w(f'<text class="t" x="{cx}" y="{cy + 4}" text-anchor="middle">{title}</text>')
    A, W, G, T = tint("--fig-2", 22), tint("--fig-1", 14), tint("--fig-hi", 14), solid("--fig-3", "--fig-2")
    f.w('<text class="tb" x="176" y="24" text-anchor="middle">Forward Pass（前向）</text>')
    f.w('<text class="tb" x="482" y="24" text-anchor="middle">Backward Pass（反向）</text>')
    f.w('<line x1="320" y1="36" x2="320" y2="364" stroke="currentColor" stroke-opacity=".25" stroke-dasharray="5 4"/>')
    box(176, 76, 150, 50, "输入", "X_{L−1}", A)
    box(190, 182, 150, 54, "forward matmul", "X_L = X_{L−1} · W_L", A)
    box(44, 182, 68, 50, "权重", "W_L", W)
    box(176, 288, 150, 50, "输出", "X_L", A)
    f.arrow(176, 101, 176, 152); f.arrow(78, 182, 112, 182); f.arrow(176, 209, 176, 260)
    box(320, 398, 220, 44, "进入深层 Layer L+1，等误差传回", None, 'class="op"')
    f.path("M176,313 L176,398 L207,398"); f.path("M430,398 L482,398 L482,316")
    box(482, 288, 150, 50, "接收误差", "∇X_L", T)
    box(404, 182, 152, 54, "weight grad", "∇W_L = X_{L−1}^T · ∇X_L", G)
    box(562, 182, 152, 54, "input grad", "∇X_{L−1} = ∇X_L · W_L^T", T)
    box(562, 76, 152, 50, "进入浅层 Layer L−1", None, T)
    f.arrow(462, 263, 420, 212); f.arrow(502, 263, 540, 212); f.arrow(562, 155, 562, 104)
    f.path("M251,76 L404,76 L404,151", "--fig-2", dash=True, width=1.8)
    for c in ("s halo", "s"):
        f.w(f'<text class="{c}" x="262" y="66">{mt("X_{L−1}")} 留到反向（第 3 节的 A）</text>')
    f.path("M44,207 L44,236 L618,236 L618,212", "--fig-1", dash=True, width=1.8)
    for c in ("s halo", "s"):
        f.w(f'<text class="{c}" x="320" y="253" text-anchor="middle">{mt("W_L")} 是参数，本来就在显存里</text>')
    return f

# ── 图 2-2 一步的耗时构成 ──────────────────────────────────────────────────────
def step_time():
    rows = [("small", "0.13B", 17.2, 34.8, 3.7, 55.7, "27%"), ("medium", "0.42B", 51.1, 103.3, 12.9, 167.4, "31%"),
            ("large", "0.97B", 118.1, 227.8, 26.9, 372.8, "31%")]
    f = F("fig-2-2", 172, "三档模型一步训练的耗时构成：前向约 31%，反向约 62%，optimizer 约 7%；右侧是每步总耗时和 MFU")
    legend(f, [("前向", "--fig-1", "box"), ("反向", "--fig-2", "box"), ("optimizer", "--fig-3", "box")], 112, 20)
    X0, W = 112, 352
    for i, (name, n, fw, bw, o, tot, mfu) in enumerate(rows):
        y = 44 + i * 42
        f.w(f'<text class="lab" x="{X0 - 10}" y="{y + 9}" text-anchor="end">{name}</text>')
        f.w(f'<text class="lab2" x="{X0 - 10}" y="{y + 23}" text-anchor="end">{n}</text>')
        s, x = fw + bw + o, X0
        for part, var, on, lab in ((fw, "--fig-1", "--fig-on-1", "前向"), (bw, "--fig-2", "--fig-on-2", "反向"), (o, "--fig-3", "--fig-on-3", "optimizer")):
            wd = W * part / s
            rect(f, x, y, wd - 2, 26, var, f"{name} {lab}：{part} ms（{part / s:.0%}）", stroke="--fig-2" if var == "--fig-3" else None)
            if wd >= 30:
                f.w(f'<text x="{x + (wd - 2) / 2:.1f}" y="{y + 17}" text-anchor="middle" font-size="11" style="fill: var({on})">{part / s:.0%}</text>')
            x += wd
        f.w(f'<text class="val" x="{X0 + W + 10}" y="{y + 12}">{tot} ms / 步</text>')
        f.w(f'<text class="lab2" x="{X0 + W + 10}" y="{y + 26}">MFU {mfu}</text>')
    return f

# ── 图 2-3 roofline ───────────────────────────────────────────────────────────
ALL_ROOFS = [("fp32", 1.048e14, "fp32 π = 1.05e14", True), ("bf16", 2.095e14, "bf16 2.1e14", False),
             ("fp8", 4.19e14, "fp8 4.19e14", False), ("nvfp4", 1.676e15, "nvfp4 1.68e15", False)]

def roofline():
    return _roofline("fig-1-1", "RTX 5090 各精度的 roofline：带宽斜线 1.79e12 B/s 与 fp32、bf16、fp8、nvfp4 四条峰值平线分别交于 58、117、234、935 FLOPs/B", ALL_ROOFS, [], 15.5)

def roofline_ops():
    return _roofline("fig-3-5", "RTX 5090 fp32 的 roofline 和一层 attention 里实测的 op：attention 的 op 都在斜线上，只有作对照的 Linear 在平线下", [("fp32", 1.048e14, "峰值 π = 1.05e14 FLOPS", True)], None, 14.6)

def roofline_rms():
    n = 32 * 512 * 2560
    ops = [("eager", "RMSNorm eager（5 个 op）", 4 * n, 7 * n * 4, 0.8157, "MBU 80%"),
           ("融合后", "RMSNorm torch.compile（1 个 kernel）", 4 * n, 2 * n * 4, 0.2363, "MBU 79%")]
    return _roofline("fig-3-3", "RMSNorm 融合前后在 fp32 roofline 上的位置：eager 的 I = 0.14，融合后 0.5，都贴着带宽斜线，融合后实际 FLOPS 高 3.5 倍", [("fp32", 1.048e14, "峰值 π = 1.05e14 FLOPS", True)], ops, 14.6, link=True)

def _roofline(fid, label, ROOFS, OPS, YHI, link=False):
    B = 1.792e12
    MiB = 2 ** 20
    if OPS is None: OPS = [("Linear", "Linear（FFN w1）", 2 * 4096 * 1024 * 4096, 96 * MiB, 0.5, "MFU 64%"),
           ("QKᵀ", "S = QKᵀ", 64 * 2 * 1024 * 64 * 1024, 288 * MiB, 0.30, "MBU 57%"),
           ("PV", "O = PV", 64 * 2 * 1024 * 1024 * 64, 272 * MiB, 0.23, "MBU 70%"),
           ("softmax", "softmax（5 个 kernel）", 27 * 4 * 16 * 1024 * 1024, 2048 * MiB, 1.43, "MBU 84%"),
           ("S / √d", "S / √d", 4 * 16 * 1024 * 1024, 512 * MiB, 0.35, "MBU 86%")]
    X0, X1, XLO, XHI, Y0, Y1, YLO = 70, 620, -1.0, 4.0, 340, 20, 11.0
    xp = lambda i: X0 + (math.log10(i) - XLO) / (XHI - XLO) * (X1 - X0)
    yp = lambda v: Y0 - (math.log10(v) - YLO) / (YHI - YLO) * (Y0 - Y1)
    fmt = lambda v: (lambda m, e: f"{float(m):.2f}".rstrip("0").rstrip(".") + "e" + str(int(e)))(*f"{v:.2e}".split("e"))
    f = F(fid, 392, label)
    for k in range(-1, 5):
        x = xp(10 ** k)
        if k > -1: f.w(f'<line class="grid" x1="{x:.1f}" y1="{Y1}" x2="{x:.1f}" y2="{Y0}"/>')
        f.w(f'<text class="tick" x="{x:.1f}" y="{Y0 + 17}" text-anchor="middle">{["0.1", "1", "10", "100", "1000", "10000"][k + 1]}</text>')
    for k in range(11, int(YHI) + 1):
        y = yp(10 ** k)
        if k > 11: f.w(f'<line class="grid" x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}"/>')
        f.w(f'<text class="tick" x="{X0 - 8}" y="{y + 4:.1f}" text-anchor="end">1e{k}</text>')
    f.w(f'<line class="axis" x1="{X0}" y1="{Y0}" x2="{X1}" y2="{Y0}"/><line class="axis" x1="{X0}" y1="{Y1}" x2="{X0}" y2="{Y0}"/>')
    f.w(f'<text class="lab2" x="{(X0 + X1) / 2:.0f}" y="{Y0 + 40}" text-anchor="middle">算术强度 I（FLOPs/B，对数轴）</text>')
    f.w(f'<text class="lab2" transform="translate(16 {(Y0 + Y1) / 2:.0f}) rotate(-90)" text-anchor="middle">可达算力（FLOPS，对数轴）</text>')
    r32, rmax = ROOFS[0][1] / B, ROOFS[-1][1] / B
    f.w(f'<line x1="{xp(r32):.1f}" y1="{yp(ROOFS[0][1]):.1f}" x2="{xp(r32):.1f}" y2="{Y0}" stroke="currentColor" stroke-opacity=".25"/>')
    if len(ROOFS) > 1:
        f.w(f'<line x1="{xp(r32):.1f}" y1="{yp(ROOFS[0][1]):.1f}" x2="{xp(rmax):.1f}" y2="{yp(ROOFS[-1][1]):.1f}" style="stroke: var(--fig-2)" stroke-width="1.5" stroke-dasharray="6 4"/>')
        f.w(f'<text class="lab2" x="{X0 + 10}" y="{Y1 + 16}">拐点旁的数字是 ridge point I*（FLOPs/B）</text>')
    for name, peak, lab, main in ROOFS:
        r = peak / B
        st = 'style="stroke: var(--fig-1)" stroke-width="2.2"' if main else 'style="stroke: var(--fig-2)" stroke-width="1.5" stroke-dasharray="6 4"'
        f.w(f'<line x1="{xp(r):.1f}" y1="{yp(peak):.1f}" x2="{X1}" y2="{yp(peak):.1f}" {st}/>')
        f.w(f'<g class="m"><title>{name}：ridge point = {fmt(peak)} / 1.792e12 = {r:.0f} FLOPs/B</title><circle cx="{xp(r):.1f}" cy="{yp(peak):.1f}" r="3" style="fill: var(--fig-1)"/></g>')
        if len(ROOFS) == 1:
            f.w(f'<text class="lab" x="{xp(r):.1f}" y="{yp(peak) - 10:.1f}" text-anchor="middle">ridge point I* = {r:.0f}</text>')
        else:
            f.w(f'<text class="lab2" x="{xp(r) - 7:.1f}" y="{yp(peak) - 4:.1f}" text-anchor="end">{r:.0f}</text>')
        f.w(f'<text class="{"lab" if main else "lab2"}" x="{X1 - 2}" y="{yp(peak) + 14:.1f}" text-anchor="end">{lab}</text>')
    f.w(f'<line x1="{X0}" y1="{yp(0.1 * B):.1f}" x2="{xp(r32):.1f}" y2="{yp(ROOFS[0][1]):.1f}" style="stroke: var(--fig-1)" stroke-width="2.2"/>')
    ang = -math.degrees(math.atan((Y0 - Y1) / (YHI - YLO) / ((X1 - X0) / (XHI - XLO))))
    yb = yp(10 ** (XLO + (230 - X0) / (X1 - X0) * (XHI - XLO)) * B)
    f.w(f'<text class="lab" transform="translate(230 {yb - 8:.1f}) rotate({ang:.1f})" text-anchor="middle">带宽 β = 1.79e12 B/s</text>')
    f.w(f'<text class="lab2" x="{xp(8):.0f}" y="{Y0 - 14}" text-anchor="middle">memory-bound</text>')
    f.w(f'<text class="lab2" x="{xp(1500):.0f}" y="{Y0 - 14}" text-anchor="middle">compute-bound</text>')
    place = {"Linear": (-10, 18, "end"), "QKᵀ": (10, 14, "start"), "PV": (10, -2, "start"), "softmax": (10, 16, "start"), "S / √d": (10, 12, "start"),
             "eager": (10, 16, "start"), "融合后": (-10, -10, "end")}
    pts = [(xp(fl / by), yp(fl / (ms * 1e-3))) for _, _, fl, by, ms, _ in OPS]
    if link and len(pts) == 2:
        f.arrow(pts[0][0] + 7, pts[0][1] - 5, pts[1][0] - 8, pts[1][1] + 5, "--fig-hi", width=1.4)
    for key, full, fl, by, ms, util in OPS:
        i, v = fl / by, fl / (ms * 1e-3)
        x, y = xp(i), yp(v)
        f.w(f'<g class="m"><title>{full}：I = {i:.3g} FLOPs/B，实测 {fmt(v)} FLOPS，{util}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="transparent"/><circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="ring" style="fill: var(--fig-hi)"/></g>')
        dx, dy, anc = place[key]
        f.w(f'<text class="lab" x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anc}">{key}</text>')
    return f


# ── 图 4-1 attention 的前向、显存读写、saved tensors 和反向 ─────────────────────
def attn_flow():
    f = F("fig-3-4", 350, "eager attention 一层：前向 9 个 kernel 依次读写显存里 S 大小的张量，softmax 拆成 max、减 max、exp、求和、除；为反向新存了 e = exp(S−m) 和 P 两个 256 MiB 的张量，Q、K、V、mask 只是引用", W=800)
    YF, YM, YB = 70, 196, 290
    LANES = [(6, 112, "前向", "外框是 kernel 的 register，里面是计算单元；箭头是数据流"), (148, 240, "显存", ""), (252, 344, "反向", "")]
    rms_lanes(f, LANES, titles=False)
    X = [140 + 67 * i + (34 if i > 3 else 0) for i in range(9)]
    names = ["① QKᵀ", "② ÷√d", "③ +M", "max", "− m", "exp", "sum", "÷ Σ", "⑤ PV"]
    ops_f = [(x, 60, n, None) for x, n in zip(X, names)]
    ops_b = [(x, 60, n) for x, n in zip(X, names)]
    mid = [(X[k] + X[k + 1]) / 2 for k in range(8)]
    REF2, TMP = blk("--fig-2", "ref"), blk(None, "tmp")
    blocks = {"M": (18, 30, "M", "1 MiB", REF2, "--fig-2"),
              "Q": (56, 40, "Q", "16 MiB", REF2, "--fig-2"), "K": (96, 40, "K", "16 MiB", REF2, "--fig-2"),
              "S": (mid[0], 54, "S", "256 MiB", TMP, "--fig-mute"), "S2": (mid[1], 54, "S/√d", "256 MiB", TMP, "--fig-mute"),
              "S3": (mid[2], 54, "S+M", "256 MiB", TMP, "--fig-mute"), "idx": (X[3] + 22, 52, "下标", "+0.5 MiB", blk("--fig-3", "new"), "--fig-3"),
              "m": (X[4] - 16, 38, "m", "0.25 MiB", TMP, "--fig-mute"),
              "Sm": (mid[4], 54, "S−m", "256 MiB", TMP, "--fig-mute"), "e": (mid[5], 60, "e", "+256 MiB", blk("--fig-hi", "new"), "--fig-hi"),
              "Z": (mid[6], 44, "Σ", "+0.25 MiB", blk("--fig-3", "new"), "--fig-3"), "P": (mid[7], 60, "P", "+256 MiB", blk("--fig-1", "new"), "--fig-1"),
              "V": (774, 36, "V", "16 MiB", REF2, "--fig-2")}
    fe = [(0, "Q", "R"), (0, "K", "R"), (0, "S", "W"), (1, "S", "R"), (1, "S2", "W"), (2, "S2", "R"), (2, "M", "R", "over"), (2, "S3", "W"),
          (3, "S3", "R"), (3, "m", "W"), (3, "idx", "W"), (4, "S3", "R", "over"), (4, "m", "R"), (4, "Sm", "W"), (5, "Sm", "R"), (5, "e", "W"),
          (6, "e", "R"), (6, "Z", "W"), (7, "e", "R", "over"), (7, "Z", "R"), (7, "P", "W"), (8, "P", "R"), (8, "V", "R")]
    be = [(0, "Q"), (0, "K"), (2, "M", "under"), (3, "idx"), (5, "e"), (7, "e"), (7, "Z"), (8, "P"), (8, "V")]
    big, q = 256 << 20, 16 << 20
    flow(f, ops_f, ops_b, blocks, fe, be, YF=YF, YM=YM, YB=YB, route=400, cap=54,
         sizes={"Q": q, "K": q, "V": q, "M": 1 << 20, "S": big, "S2": big, "S3": big, "m": 256 << 10, "idx": 512 << 10,
                "Sm": big, "e": big, "Z": 256 << 10, "P": big})
    f.w(f'<line x1="{X[3] - 30}" y1="28" x2="{X[7] + 30}" y2="28" stroke="currentColor" stroke-opacity=".35"/>')
    f.w(f'<text class="lab2" x="{(X[3] + X[7]) / 2}" y="22" text-anchor="middle">④ softmax，5 个 kernel</text>')
    f.arrow(X[8] + 30, YF, 770, YF); f.w(f'<text class="t" x="778" y="{YF + 4}">O</text>')
    f.w(f'<text class="t" x="776" y="{YB + 4}">dO</text>'); f.arrow(770, YB, X[8] + 32, YB)
    for c0, c1 in zip(X, X[1:]):
        f.arrow(c1 - 30, YB, c0 + 32, YB)
    f.arrow(X[0] - 30, YB, 100, YB); f.w(f'<text class="t" x="96" y="{YB + 4}" text-anchor="end">dQ dK</text>')
    lane_titles(f, LANES)
    return f

# ── 图 2-4 eager softmax 的显存读写 ───────────────────────────────────────────
def softmax():
    f = F("fig-3-6", 262, "eager softmax 的 5 个 kernel 共读写显存里 S 大小的张量 8 次；融合成一个 kernel 后只读 S、写 P 两次")
    TY, KY, FY = 52, 150, 236
    def box(cx, cy, wd, lab, hbm):
        st = tint("--fig-1", 14, sw=1.5) if hbm else 'class="op"'
        f.w(f'<rect x="{cx - wd / 2:.0f}" y="{cy - 15}" width="{wd}" height="30" rx="6" {st}/>')
        f.w(f'<text class="t" x="{cx}" y="{cy + 4}" text-anchor="middle">{lab}</text>')
    for y, t1, t2 in ((TY, "显存", "(HBM)"), (KY, "kernel", "(片上算)")):
        f.w(f'<rect class="band" x="70" y="{y - 28}" width="562" height="56" rx="8"/>')
        f.w(f'<text class="lab" x="8" y="{y - 2}">{t1}</text><text class="lab2" x="8" y="{y + 14}">{t2}</text>')
    for cx, wd, lab in ((165, 64, "S"), (275, 76, "S − m"), (440, 104, "exp(S − m)"), (590, 56, "P")):
        box(cx, TY, wd, lab, True)
    for cx, wd, lab in ((112, 62, "max"), (218, 70, "减 max"), (330, 58, "exp"), (440, 62, "求和"), (548, 58, "除")):
        box(cx, KY, wd, lab, False)
    tb, kt = TY + 15, KY - 15
    for (x1, y1, x2, y2, lab, lx, ly) in ((152, tb, 122, kt - 2, "① 读", -38, 4), (178, tb, 208, kt - 2, "② 读", 6, 4),
                                          (232, kt, 262, tb + 2, "③ 写", 8, 6), (288, tb, 320, kt - 2, "④ 读", 7, 4),
                                          (342, kt, 412, tb + 2, "⑤ 写", 6, 10), (440, tb, 440, kt - 2, "⑥ 读", 6, 4),
                                          (468, tb, 536, kt - 2, "⑦ 读", 8, 4), (560, kt, 584, tb + 2, "⑧ 写", 6, 6)):
        f.arrow(x1, y1, x2, y2, "--fig-hi", width=1.8)
        f.w(f'<text class="lab" x="{(x1 + x2) / 2 + lx:.0f}" y="{(y1 + y2) / 2 + ly:.0f}">{lab}</text>')
    f.arrow(143, KY, 181, KY, width=1.2); f.w(f'<text class="lab2" x="162" y="{KY - 7}" text-anchor="middle">m</text>')
    f.arrow(471, KY, 517, KY, width=1.2); f.w(f'<text class="lab2" x="494" y="{KY - 7}" text-anchor="middle">Σ</text>')
    f.w(f'<text class="lab2" x="351" y="{KY + 42}" text-anchor="middle">eager：5 个 kernel，S 大小的张量进出显存 8 次；m、Σ 每行一个数，可忽略</text>')
    f.w(f'<text class="lab" x="8" y="{FY + 4}">融合后</text>')
    box(165, FY, 64, "S", True); box(330, FY, 150, "fused softmax", False); box(495, FY, 56, "P", True)
    f.arrow(197, FY, 253, FY, "--fig-hi", width=1.8); f.w(f'<text class="lab" x="209" y="{FY - 8}">① 读</text>')
    f.arrow(405, FY, 465, FY, "--fig-hi", width=1.8); f.w(f'<text class="lab" x="419" y="{FY - 8}">② 写</text>')
    return f

# ── 图 2-5 FLOPs vs 时间 ──────────────────────────────────────────────────────
def flops_vs_time():
    rows = [("QKᵀ", "--fig-1", 8.6e9, "8.6e9", 0.30), ("÷√d + mask", "--fig-hi", 6.7e7, "6.7e7", 0.75),
            ("softmax", "--fig-hi", 1.8e9, "1.8e9", 1.43), ("PV", "--fig-1", 8.6e9, "8.6e9", 0.23)]
    f = F("fig-3-6", 186, "medium、seq 1024 一层 attention 里各 op 的 FLOPs 与实测 GPU 时间：softmax 和 ÷√d、mask 的 FLOPs 很少，时间却最多")
    legend(f, [("矩阵乘", "--fig-1", "box"), ("逐元素", "--fig-hi", "box")], 112, 20)
    P1, P2, PW = 112, 392, 200
    f.w(f'<text class="ttl" x="{P1}" y="48">FLOPs</text><text class="ttl" x="{P2}" y="48">GPU 时间（ms）</text>')
    for i, (name, var, fl, fls, ms) in enumerate(rows):
        y = 62 + i * 30
        f.w(f'<text class="lab" x="{P1 - 10}" y="{y + 13}" text-anchor="end">{name}</text>')
        w1 = max(PW * fl / 9e9, 1.5)
        rect(f, P1, y, w1, 18, var, f"{name}：{fls} FLOPs（一层）")
        f.w(f'<text class="val" x="{P1 + w1 + 6:.1f}" y="{y + 13}">{fls}</text>')
        w2 = PW * ms / 1.5
        rect(f, P2, y, w2, 18, var, f"{name}：{ms} ms（一层）")
        f.w(f'<text class="val" x="{P2 + w2 + 6:.1f}" y="{y + 13}">{ms:.2f}</text>')
    for px in (P1, P2):
        f.w(f'<line class="axis" x1="{px}" y1="58" x2="{px}" y2="174"/>')
    return f

# ── 图 2-6 attention 占比 vs seq ─────────────────────────────────────────────
def share_vs_seq():
    seqs, X = [256, 512, 1024], {256: 110, 512: 290, 1024: 470}
    ser = [("合计", "--fig-mute", [10, 22, 46], [2.3, 10.2, 64.9], True), ("softmax", "--fig-hi", [4, 11, 24], [1.0, 5.0, 34.3], False),
           ("scores", "--fig-1", [4, 8, 18], [0.9, 3.6, 25.1], False), ("PV", "--fig-2", [2, 3, 4], [0.4, 1.6, 5.5], False)]
    f = F("fig-3-7", 262, "attention 三段占 forward 时间随 seq 的变化：softmax 从 4% 涨到 24%，scores 从 4% 涨到 18%，PV 只从 2% 到 4%，合计从 10% 到 46%")
    legend(f, [("softmax", "--fig-hi", "line"), ("scores（QKᵀ、÷√d、mask）", "--fig-1", "line"), ("PV", "--fig-2", "line"), ("attention 合计", None, "dash")], 70, 20)
    Y0, PY = 236, 3.6
    for v in range(0, 51, 10):
        y = Y0 - v * PY
        f.w(f'<line class="grid" x1="70" y1="{y:.1f}" x2="520" y2="{y:.1f}"/><text class="tick" x="62" y="{y + 4:.1f}" text-anchor="end">{v}%</text>')
    for s in seqs:
        f.w(f'<text class="tick" x="{X[s]}" y="{Y0 + 18}" text-anchor="middle">seq {s}</text>')
    f.w(f'<text class="lab2" x="70" y="{Y0 - 50 * PY - 10:.0f}">占 forward GPU 时间（medium，24 层）</text>')
    for name, var, pct, ms, dashed in ser:
        pts = " ".join(f"{X[s]},{Y0 - p * PY:.1f}" for s, p in zip(seqs, pct))
        st = 'stroke="currentColor" stroke-opacity=".55" stroke-dasharray="6 4"' if dashed else f'style="stroke: var({var})"'
        f.w(f'<polyline points="{pts}" fill="none" {st} stroke-width="2" stroke-linejoin="round"/>')
        fill = 'fill="currentColor" fill-opacity=".55"' if dashed else f'style="fill: var({var})"'
        for s, p, m in zip(seqs, pct, ms):
            f.w(f'<g class="m"><title>seq {s}：{name} {m} ms，占 {p}%</title><circle cx="{X[s]}" cy="{Y0 - p * PY:.1f}" r="4.5" class="ring" {fill}/></g>')
        f.w(f'<text class="val" x="{X[1024] + 12}" y="{Y0 - pct[-1] * PY + 4:.1f}">{name} {pct[-1]}%</text>')
    return f

# ── 图 3-1 纸面 vs 实测峰值显存 ───────────────────────────────────────────────
def peak_memory():
    data = [("small", "0.13B", 0.48, 3.50, 5.04), ("medium", "0.42B", 1.58, 8.91, 13.74), ("large", "0.97B", 3.61, 16.58, 27.51)]
    f = F("fig-2-3", 300, "三档模型 full step 的实测峰值显存，由权重、Adam 状态和 A、G 共用的一块组成：这块前向结束时装 activation，反向结束时装梯度，高度是两者中较大的 A")
    # A and G take turns in the same memory: diagonal stripes alternating the two colours
    f.w('<defs><pattern id="fig-2-3-ag" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect width="4" height="8" style="fill: var(--fig-2)"/><rect x="4" width="4" height="8" style="fill: var(--fig-hi)"/></pattern></defs>')
    legend(f, [("W 权重", "--fig-1", "box"), ("Adam m、v", "--fig-mute", "box"), ("A / G 共用", "url(#fig-2-3-ag)", "fill"), ("T 临时量", "--fig-3", "box")], 70, 20, gap=16)
    Y0, PY = 262, 6.6
    for v in range(0, 31, 10):
        y = Y0 - v * PY
        f.w(f'<line class="grid" x1="70" y1="{y:.1f}" x2="630" y2="{y:.1f}"/><text class="tick" x="62" y="{y + 4:.1f}" text-anchor="end">{v}</text>')
    f.w('<text class="lab2" x="70" y="40">GiB</text>')
    on = {"--fig-1": "--fig-on-1", "--fig-2": "--fig-on-2", "--fig-mute": "--fig-on-mute"}
    BW = 70
    for i, (name, n, W, A, meas) in enumerate(data):
        cx = 170 + i * 170
        segs = [("W", W, "--fig-1"), ("Adam", 2 * W, "--fig-mute"), ("A", A, "--fig-2"), ("T", meas - 3 * W - A, "--fig-3")]
        x, y = cx - BW / 2, Y0
        for lab, v, var in segs:
            h = v * PY
            if lab == "A":
                f.w(f'<g class="m"><title>{name}：A、G 共用的一块 {v:.2f} GiB。前向结束时装 A（{A:.2f} GiB），'
                    f'反向结束时 A 已释放、装 G（= W，{W:.2f} GiB）</title><rect x="{x:.1f}" y="{y - h + 1:.1f}" width="{BW}" '
                    f'height="{h - 2:.1f}" rx="2" fill="url(#fig-2-3-ag)"/></g>')
                for k, t in enumerate((f"A {A:.1f}", f"G {W:.1f}")):  # beside the bar: text on stripes is unreadable
                    f.w(f'<text class="val" x="{x + BW + 6:.1f}" y="{y - h / 2 - 2 + 14 * k:.1f}">{t}</text>')
                y -= h
                continue
            rect(f, x, y - h + 1, BW, h - 2 if h > 3 else h, var, f"{name}：{lab} {v:.2f} GiB", rx=2, stroke="--fig-2" if var == "--fig-3" else None)
            if h >= 16 and lab != "T":
                f.w(f'<text x="{cx}" y="{y - h / 2 + 4:.1f}" text-anchor="middle" font-size="11" style="fill: var({on[var]})">{lab} {v:.1f}</text>')
            y -= h
        f.w(f'<text class="val" x="{cx}" y="{y - 6:.1f}" text-anchor="middle">{meas:.2f} GiB</text>')
        f.w(f'<text class="lab" x="{cx}" y="{Y0 + 18}" text-anchor="middle">{name} {n}</text>')
    f.w(f'<line class="axis" x1="70" y1="{Y0}" x2="630" y2="{Y0}"/>')
    return f

# ── 图 3-2 一步 fwd_bwd 的显存 M(j)（replaces peak_moment.png） ─────────────────
def peak_moment():
    meas = {(r["size"], r["autocast"]): r for r in json.load(open(os.path.join(HERE, "peak_moment_measured.json")))}
    cases = [("xl", "seq 128（G > A）", 32, 12.7, 12.7, 5.3, 3.5 + 6.35, 0.2, 25.56, 25.55, 28),
             ("small", "seq 512（A > G）", 12, 0.48, 0.48, 3.41, 2.28 + 0.23, 0.17, 4.08, 3.18, 4.5)]
    f = F("fig-2-4", 300, "一步前向 + 反向的显存：按 M(j) 用实测的 W、A、G、T 堆叠；xl seq 128 时 G 大于 A，峰值在反向结束；small seq 512 时 A 大于 G，峰值在前向结束")
    legend(f, [("W", "--fig-1", "box"), ("A", "--fig-2", "box"), ("G", "--fig-hi", "box"), ("T", "--fig-3", "box"), ("逐层实测", None, "x")], 64, 18, gap=14)
    for p, (size, sub, L, W, G, A, A16, T, p32, p16, ymax) in enumerate(cases):
        X0, X1, Y0, Y1 = 64 + p * 300, 64 + p * 300 + 250, 250, 62
        xp = lambda k: X0 + (X1 - X0) * k / (2 * L)
        yp = lambda v: Y0 - (Y0 - Y1) * v / ymax
        f.w(f'<text class="ttl" x="{X0}" y="44">{size}，{sub}</text>')
        for v in [x * (5 if ymax > 10 else 1) for x in range(0, int(ymax // (5 if ymax > 10 else 1)) + 1)]:
            f.w(f'<line class="grid" x1="{X0}" y1="{yp(v):.1f}" x2="{X1}" y2="{yp(v):.1f}"/><text class="tick" x="{X0 - 6}" y="{yp(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
        ks = range(2 * L + 1)
        w = [W] * (2 * L + 1)
        a = [A * i / L for i in range(L + 1)] + [A * (L - j) / L for j in range(1, L + 1)]
        g = [0] * (L + 1) + [G * j / L for j in range(1, L + 1)]
        t = [0] * L + [T] * (L + 1)
        base = [0] * (2 * L + 1)
        for lab, ser, var in (("W", w, "--fig-1"), ("A", a, "--fig-2"), ("G", g, "--fig-hi"), ("T", t, "--fig-3")):
            top = [b + s for b, s in zip(base, ser)]
            d = " ".join(f"{xp(k):.1f},{yp(top[k]):.1f}" for k in ks) + " " + " ".join(f"{xp(k):.1f},{yp(base[k]):.1f}" for k in reversed(ks))
            extra = '; stroke: var(--fig-2)' if var == "--fig-3" else ''
            f.w(f'<g class="m"><title>{lab}</title><polygon points="{d}" style="fill: var({var}){extra}" stroke-width=".8"/></g>')
            base = top
        tot = base
        r = meas.get((size, False)) if p32 else None
        if r:
            pts = r["fwd"] + r["bwd"]
            for k, v in enumerate(pts, 1):
                f.w(f'<text x="{xp(k):.1f}" y="{yp(v) + 3.5:.1f}" text-anchor="middle" font-size="10" fill="currentColor">×</text>')
        kp = max(ks, key=lambda k: tot[k])
        right = kp > L
        f.w(f'<text class="val" x="{xp(kp) + (-6 if right else 6):.1f}" y="{yp(tot[kp]) - 8:.1f}" text-anchor="{"end" if right else "start"}">{f"峰值 {p32:.2f} GiB" if p32 else f"峰值约 {tot[kp]:.1f} GiB"}</text>')
        f.w(f'<line x1="{xp(L):.1f}" y1="{Y1}" x2="{xp(L):.1f}" y2="{Y0}" stroke="currentColor" stroke-opacity=".3" stroke-dasharray="2 3"/>')
        f.w(f'<line class="axis" x1="{X0}" y1="{Y0}" x2="{X1}" y2="{Y0}"/>')
        for k, lab, anc in ((0, "开始", "start"), (L, "前向结束", "middle"), (2 * L, "反向结束", "end")):
            f.w(f'<text class="tick" x="{xp(k):.1f}" y="{Y0 + 16}" text-anchor="{anc}">{lab}</text>')
        f.w(f'<text class="lab2" x="{X0}" y="{Y0 + 36}">W = G = {W}，A = {A} GiB</text>')
    f.w('<text class="lab2" x="14" y="160" transform="rotate(-90 14 160)" text-anchor="middle">GiB</text>')
    return f

# ── 图 3-3 / 3-4 RMSNorm ───────────────────────────────────────────────────────
def rms_lanes(f, ys, titles=True):
    for (y0, y1, t, s) in ys:
        f.w(f'<rect class="band" x="4" y="{y0}" width="{f.W - 8}" height="{y1 - y0}" rx="8"/>')
        if titles:
            f.w(f'<text class="ttl" x="16" y="{y0 + 18}">{t}</text><text class="lab2" x="{16 + 13 * len(t) + 8}" y="{y0 + 18}">{s}</text>')

def lane_titles(f, ys):
    for (y0, y1, t, s) in ys:
        for c in ("halo", ""):
            f.w(f'<text class="ttl {c}" x="16" y="{y0 + 18}">{t}</text><text class="lab2 {c}" x="{16 + 13 * len(t) + 8}" y="{y0 + 18}">{s}</text>')

def tbox(f, cx, cy, bw, bh, l1, l2, style, b1=False, b2=False, strike=None):
    f.w(f'<rect x="{cx - bw / 2:.1f}" y="{cy - bh / 2:.1f}" width="{bw}" height="{bh}" rx="6" {style}/>')
    f.w(f'<text class="{"tb" if b1 else "t"}" x="{cx}" y="{cy - 3}" text-anchor="middle">{l1}</text>')
    if l2:
        f.w(f'<text class="{"tb" if b2 else "s"}" x="{cx}" y="{cy + 13}" text-anchor="middle">{l2}</text>')
    if strike:
        f.w(f'<line x1="{cx - bw / 2 + 6:.1f}" y1="{cy + bh / 2 - 6:.1f}" x2="{cx + bw / 2 - 6:.1f}" y2="{cy - bh / 2 + 6:.1f}" style="stroke: var({strike})" stroke-width="1.4" stroke-opacity=".8"/>')

def chip(f, cx, cy, w, l1, l2, style, bold=False, strike=None):
    f.w(f'<rect x="{cx - w / 2:.1f}" y="{cy - 20}" width="{w}" height="40" rx="6" {style}/>')
    f.w(f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="{11.5 if w > 60 else 10.5}" fill="currentColor">{l1}</text>')
    w2 = 'font-weight="600"' if bold else 'opacity=".7"'
    f.w(f'<text x="{cx}" y="{cy + 12}" text-anchor="middle" font-size="10.5" {w2} fill="currentColor">{l2}</text>')
    if strike:
        f.w(f'<line x1="{cx - w / 2 + 6:.1f}" y1="{cy + 14}" x2="{cx + w / 2 - 6:.1f}" y2="{cy - 14}" style="stroke: var({strike})" stroke-width="1.4" stroke-opacity=".8"/>')

def bbox(f, cx, cy, w, h, label):
    f.w(f'<rect x="{cx - w / 2:.1f}" y="{cy - h / 2:.1f}" width="{w}" height="{h}" rx="6" class="op"/>')
    f.w(f'<text class="t" x="{cx}" y="{cy + 4.5:.1f}" text-anchor="middle">{label}</text>')

def passes(f, c, seq, label, YT, unit):
    n, bw, gap = len(seq), 11, 2
    x0 = c - (n * bw + (n - 1) * gap) / 2
    for k, ch in enumerate(seq):
        st = 'style="fill: var(--fig-hi)"' if ch in "Ww" else 'style="fill: color-mix(in srgb, var(--fig-hi) 12%, transparent); stroke: var(--fig-hi)" stroke-width="1.2"'
        h = 16 if ch in "RW" else 5
        f.w(f'<g class="m"><title>{"写" if ch in "Ww" else "读"} {unit if ch in "RW" else "8 KiB（每行一个数）"}</title><rect x="{x0 + k * (bw + gap):.1f}" y="{YT - 6 - h}" width="{bw}" height="{h}" rx="1.5" {st}/></g>')
    f.w(f'<text class="val" x="{c}" y="{YT + 8}" text-anchor="middle">{label}</text>')

def flow(f, ops_f, ops_b, blocks, fe, be, YF=56, YM=176, YB=296, route=170, sizes=None, cap=60):
    """Three lanes: forward ops / memory blocks / backward ops. fe: (op, block, 'R'|'W'); be: (op, block)."""
    if sizes:  # box width ∝ bytes (linear), thin bar below the minimum; labels go under the box
        big = max(sizes.values())
        bws = {b: max(4, cap * sizes[b] / big) for b in blocks}
    def attach(cx, w, items):
        w = max(w, 0)
        n = len(items)
        return {it: cx - w / 2 + w * (k + 1) / (n + 1) for k, it in enumerate(items)}
    # forward-side attachment points
    op_pts, blk_top, blk_bot, opb_pts = {}, {}, {}, {}
    for i, (cx, w, *_r) in enumerate(ops_f):
        es = sorted([e for e in fe if e[0] == i and len(e) == 3], key=lambda e: blocks[e[1]][0])
        op_pts.update({e: p for e, p in zip(es, attach(cx, w - 16, es).values())})
        op_pts.update({e: cx for e in fe if e[0] == i and len(e) > 3})
    for b, (cx, w, *_r) in blocks.items():
        if sizes: w = bws[b] + 8
        es = sorted([e for e in fe if e[1] == b and len(e) == 3], key=lambda e: ops_f[e[0]][0])
        blk_top.update({e: p for e, p in zip(es, attach(cx, w - 10, es).values())})
        blk_top.update({e: cx for e in fe if e[1] == b and len(e) > 3})
        es2 = sorted([e for e in be if e[1] == b and len(e) == 2], key=lambda e: ops_b[e[0]][0])
        blk_bot.update({e: p for e, p in zip(es2, attach(cx, w - 10, es2).values())})
        blk_bot.update({e: cx for e in be if e[1] == b and len(e) > 2})
    for i, (cx, w, *_r) in enumerate(ops_b):
        es = sorted([e for e in be if e[0] == i and len(e) == 2], key=lambda e: blocks[e[1]][0])
        opb_pts.update({e: p for e, p in zip(es, attach(cx, w - 16, es).values())})
        opb_pts.update({e: cx for e in be if e[0] == i and len(e) > 2})
    yo, yt, yb, yob = YF + 26, YM - 22, YM + (32 if sizes else 22), YB - 20
    otop, bbot = YF - 20, YB + 22                    # 箭头只到 register 外框，不碰里面的计算单元
    for e in fe:
        i, b, kind = e[:3]
        ox, bx, var = op_pts[e], blk_top[e], blocks[b][5]
        if len(e) > 3:  # over the top: leave the block straight up through a gap between op boxes
            gx, yc = blocks[b][0] if len(e) < 5 else e[4], otop - 14
            ox = ops_f[i][0] + (ops_f[i][1] / 2 - 10) * (1 if gx > ops_f[i][0] else -1)
            d = f"M{gx:.1f},{yt} L{gx:.1f},{yc} L{ox:.1f},{yc} L{ox:.1f},{otop - 2}" if kind == "R" else f"M{ox:.1f},{otop} L{ox:.1f},{yc} L{gx:.1f},{yc} L{gx:.1f},{yt - 2}"
            f.path(d, var, width=1.3)
        elif abs(ox - bx) > route:
            yc = yo + 14
            d = f"M{bx:.1f},{yt} L{bx:.1f},{yc} L{ox:.1f},{yc} L{ox:.1f},{yo + 2}" if kind == "R" else f"M{ox:.1f},{yo} L{ox:.1f},{yc} L{bx:.1f},{yc} L{bx:.1f},{yt - 2}"
            f.path(d, var, width=1.3)
        elif kind == "R":
            f.arrow(bx, yt, ox, yo + 2, var, width=1.3)
        else:
            f.arrow(ox, yo, bx, yt - 2, var, width=1.3)
    for e in be:
        i, b = e[:2]
        ox, bx, var = opb_pts[e], blk_bot[e], blocks[b][5]
        if len(e) > 2:  # under the backward row
            gx, yc = blocks[b][0] if len(e) < 4 else e[3], bbot + 14
            ox = ops_b[i][0] + (ops_b[i][1] / 2 - 10) * (1 if gx > ops_b[i][0] else -1)
            f.path(f"M{gx:.1f},{yb} L{gx:.1f},{yc} L{ox:.1f},{yc} L{ox:.1f},{bbot + 2}", var, dash=True, width=1.3)
        elif abs(ox - bx) > route:
            yc = yob - 14
            f.path(f"M{bx:.1f},{yb} L{bx:.1f},{yc} L{ox:.1f},{yc} L{ox:.1f},{yob - 2}", var, dash=True, width=1.3)
        else:
            f.arrow(bx, yb, ox, yob - 2, var, dash=True, width=1.3)
    for cx, w, t, sub in ops_f:                      # 外框是这个 kernel 的 register，里面是计算单元
        f.w(f'<rect x="{cx - w / 2:.1f}" y="{YF - 20}" width="{w}" height="48" rx="8" {tint("--fig-2", 22)}/>')
        tbox(f, cx, YF + 4, w - 10, 38, t, sub, 'class="op"', b1=True)
    for b, (cx, w, l1, l2, st, var) in blocks.items():
        if sizes:
            bw = bws[b]
            f.w(f'<rect x="{cx - bw / 2:.1f}" y="{YM - 22}" width="{bw:.1f}" height="24" rx="{min(4, bw / 2):.1f}" {st}/>')
            f.w(f'<text x="{cx}" y="{YM + 14}" text-anchor="middle" font-size="11" fill="currentColor">{l1}</text>')
            w2 = 'font-weight="600"' if l2.startswith("+") else 'opacity=".7"'
            f.w(f'<text x="{cx}" y="{YM + 27}" text-anchor="middle" font-size="10.5" {w2} fill="currentColor">{l2}</text>')
        else:
            chip(f, cx, YM, w, l1, l2, st, bold=l2.startswith("+"))
    for cx, w, t in ops_b:
        f.w(f'<rect x="{cx - w / 2:.1f}" y="{YB - 22}" width="{w}" height="44" rx="8" {tint("--fig-2", 22)}/>')
        bbox(f, cx, YB, w - 10, 34, t)

def blk(var, kind):
    """kind: new (thick solid, new memory), ref (thin solid, already there), tmp (gray dashed, freed after use)."""
    if kind == "tmp":
        return tint("--fig-mute", 6, dash=True)
    return tint(var, 16 if kind == "new" else 10, sw=2.2 if kind == "new" else 1.2)

def rms_eager():
    f = F("fig-3-1", 350, "RMSNorm eager：中间一排是显存里的每一块张量，前向箭头表示读写，反向虚线箭头表示读回存下的张量；x 被 ① 和 ④ 共用，r 被 ③ 和 ④ 共用")
    YF, YM, YB = 70, 196, 290
    LANES = [(6, 112, "前向", "外框是 kernel 的 register，里面是计算单元；箭头是数据流"), (148, 240, "显存", ""), (252, 344, "反向", "")]
    rms_lanes(f, LANES, titles=False)
    C, BW = [100, 210, 320, 430, 540], 84
    ops_f = [(C[0], BW, "① x²", "1 FLOP/元素"), (C[1], BW, "② mean", "1 FLOP/元素"), (C[2], BW, "③ rsqrt", "每行 2 FLOPs"),
             (C[3], BW, "④ x · r", "1 FLOP/元素"), (C[4], BW, "⑤ w ⊙ x̂", "1 FLOP/元素")]
    ops_b = [(c, BW, f"{n} bwd") for c, n in zip(C, "①②③④⑤")]
    blocks = {"x": (46, 80, "x …9040", "20 MiB，输入", blk("--fig-2", "ref"), "--fig-2"),
              "x2": (158, 76, "x²", "20 MiB，临时", blk(None, "tmp"), "--fig-mute"),
              "v": (264, 72, "v", "8 KiB，临时", blk(None, "tmp"), "--fig-mute"),
              "r": (377, 70, "r …6b00", "+8 KiB", blk("--fig-1", "new"), "--fig-1"),
              "xh": (491, 70, "x̂ …3c00", "+20 MiB", blk("--fig-hi", "new"), "--fig-hi"),
              "w": (600, 60, "w …1000", "参数", blk("--fig-3", "ref"), "--fig-3")}
    fe = [(0, "x", "R"), (0, "x2", "W"), (1, "x2", "R"), (1, "v", "W"), (2, "v", "R"), (2, "r", "W"),
          (3, "x", "R", "over", 11), (3, "r", "R"), (3, "xh", "W"), (4, "xh", "R"), (4, "w", "R")]
    be = [(0, "x"), (2, "r"), (3, "r"), (3, "x", "under", 11), (4, "xh"), (4, "w")]
    flow(f, ops_f, ops_b, blocks, fe, be, YF=YF, YM=YM, YB=YB, cap=72,
         sizes={"x": 20 << 20, "x2": 20 << 20, "v": 8 << 10, "r": 8 << 10, "xh": 20 << 20, "w": 10 << 10})
    f.arrow(C[4] + BW / 2, YF, 614, YF); f.w(f'<text class="t" x="620" y="{YF + 4}">y</text>')
    for c0, c1 in zip(C, C[1:]):
        f.arrow(c1 - BW / 2, YB, c0 + BW / 2 + 2, YB)
    f.w(f'<text class="t" x="620" y="{YB + 4}">dy</text>'); f.arrow(616, YB, C[4] + BW / 2 + 2, YB)
    f.arrow(C[0] - BW / 2, YB, 24, YB); f.w(f'<text class="t" x="30" y="{YB - 7}">dx</text>')
    lane_titles(f, LANES)
    return f

def rms_fused():
    f = F("fig-3-2", 336, "torch.compile 融合后的 RMSNorm：前向一个 kernel 只读 x、w，写 r 和 y；x̂ 不存；反向 3 个 kernel 读回 x、w、r，现场重算 x̂")
    LANES = [(6, 98, "前向", "外框是 kernel 的 register，里面是计算单元；箭头是数据流"), (128, 224, "显存", ""), (254, 330, "反向", "")]
    rms_lanes(f, LANES, titles=False)
    ops_f = [(330, 420, "fused fwd (①–⑤, one kernel)", "~4 FLOPs/元素")]
    ops_b = [(330, 420, "bwd: 3 kernels (dx ×1, dw ×2), recompute x̂ = x · r")]
    blocks = {"x": (180, 90, "x …3c80", "20 MiB，输入", blk("--fig-2", "ref"), "--fig-2"),
              "r": (330, 80, "r …f9c0", "+8 KiB", blk("--fig-1", "new"), "--fig-1"),
              "w": (480, 70, "w …b3c0", "参数", blk("--fig-3", "ref"), "--fig-3")}
    fe = [(0, "x", "R"), (0, "r", "W"), (0, "w", "R")]
    be = [(0, "x"), (0, "r"), (0, "w")]
    flow(f, ops_f, ops_b, blocks, fe, be, cap=90, sizes={"x": 20 << 20, "r": 8 << 10, "w": 10 << 10})
    f.w('<text class="t" x="12" y="60">x</text>'); f.arrow(24, 56, 116, 56)
    f.arrow(540, 56, 614, 56); f.w('<text class="t" x="620" y="60">y</text>')
    f.w('<text class="t" x="620" y="300">dy</text>'); f.arrow(616, 296, 542, 296)
    f.arrow(120, 296, 28, 296); f.w('<text class="t" x="8" y="300">dx</text>')
    lane_titles(f, LANES)
    return f

# ── 图 3-5 一层 saved tensors 的构成 ──────────────────────────────────────────
def layer_donut():
    sl = [("S、P", "[b, h, s, s]", 2048, "--fig-hi"), ("FFN 中间量", "[b, s, d_ff]", 960, "--fig-1"),
          ("[b, s, d] 级张量", "x、norm 输出、Q、K、V 等", 640, "--fig-2"), ("其他", "mask、RoPE、softmax 统计量", 7, "--fig-3")]
    tot = sum(s[2] for s in sl)
    f = F("fig-3-8", 246, "xl 一层为反向存的 3655 MiB：S、P 占 56%，FFN 中间量 26%，[b, s, d] 级张量 17.5%，其他 0.2%")
    cx, cy, R, r, a = 150, 122, 92, 58, -math.pi / 2
    for name, shape, v, var in sl:
        a2 = a + 2 * math.pi * v / tot
        lg = 1 if a2 - a > math.pi else 0
        p = [(cx + R * math.cos(a), cy + R * math.sin(a)), (cx + R * math.cos(a2), cy + R * math.sin(a2)),
             (cx + r * math.cos(a2), cy + r * math.sin(a2)), (cx + r * math.cos(a), cy + r * math.sin(a))]
        d = f"M{p[0][0]:.2f},{p[0][1]:.2f} A{R},{R} 0 {lg} 1 {p[1][0]:.2f},{p[1][1]:.2f} L{p[2][0]:.2f},{p[2][1]:.2f} A{r},{r} 0 {lg} 0 {p[3][0]:.2f},{p[3][1]:.2f} Z"
        f.w(f'<g class="m"><title>{name}（{shape}）：{v} MiB，{v / tot:.1%}</title><path d="{d}" class="ring" style="fill: var({var})"/></g>')
        a = a2
    f.w(f'<text class="ttl" x="{cx}" y="{cy - 2}" text-anchor="middle">3655 MiB</text><text class="lab2" x="{cx}" y="{cy + 15}" text-anchor="middle">xl 一层</text>')
    for i, (name, shape, v, var) in enumerate(sl):
        y = 62 + i * 40
        extra = '; stroke: var(--fig-2)' if var == "--fig-3" else ''
        f.w(f'<rect x="290" y="{y - 10}" width="12" height="12" rx="2" style="fill: var({var}){extra}"/>')
        f.w(f'<text class="lab" x="310" y="{y}">{name}</text><text class="lab2" x="310" y="{y + 16}">{shape}</text>')
        f.w(f'<text class="val" x="630" y="{y}" text-anchor="end">{v} MiB · {v / tot:.1%}</text>')
    return f

# ── 图 3-6 xl 显存时间线（replaces mem_xl_timelines.png） ─────────────────────
def timelines():
    tl = json.load(open(os.path.join(HERE, "timelines.json")))
    panels = [("seq128_forward", "seq 128 · 纯前向", None), ("seq2048_forward", "seq 2048 · 纯前向", None),
              ("seq128_full", "seq 128 · full step", "OOM"), ("seq2048_fwd_bwd", "seq 2048 · 前向 + 反向", "OOM")]
    names = {"forward": "前向", "backward": "反向", "optimizer": "optimizer"}
    f = F("fig-3-9", 420, "xl 一步的显存时间线：seq 128 纯前向是平的；seq 2048 纯前向每层冲出一个尖峰；seq 128 full step 前向和反向一路上升，到 optimizer 时 OOM；seq 2048 带反向时第 2 层 OOM")
    YMAX = 30
    for p, (key, title, oom) in enumerate(panels):
        d = tl[key]
        col, row = p % 2, p // 2
        X0, X1 = 52 + col * 300, 52 + col * 300 + 262
        Y1, Y0 = 46 + row * 196, 46 + row * 196 + 132
        xp = lambda k: X0 + (X1 - X0) * k / max(d["n"] - 1, 1)
        yp = lambda v: Y0 - (Y0 - Y1) * v / YMAX
        f.w(f'<text class="ttl" x="{X0}" y="{Y1 - 12}">{title}</text>')
        for v in (0, 10, 20, 30):
            f.w(f'<line class="grid" x1="{X0}" y1="{yp(v):.1f}" x2="{X1}" y2="{yp(v):.1f}"/><text class="tick" x="{X0 - 6}" y="{yp(v) + 4:.1f}" text-anchor="end">{v}</text>')
        pts = d["points"]
        line = " ".join(f"{xp(k):.1f},{yp(v):.1f}" for k, v in pts)
        f.w(f'<polygon points="{xp(pts[0][0]):.1f},{yp(0):.1f} {line} {xp(pts[-1][0]):.1f},{yp(0):.1f}" style="fill: var(--fig-3)" fill-opacity=".7"/>')
        f.w(f'<polyline points="{line}" fill="none" style="stroke: var(--fig-1)" stroke-width="1.2" stroke-linejoin="round"/>')
        f.w(f'<line x1="{X0}" y1="{yp(d["start"]):.1f}" x2="{X1}" y2="{yp(d["start"]):.1f}" stroke="currentColor" stroke-opacity=".5" stroke-dasharray="2 3"/>')
        f.w(f'<text class="lab2" x="{X1}" y="{yp(d["start"]) + 13:.1f}" text-anchor="end">权重 {d["start"]:.1f} GiB</text>')
        f.w(f'<g class="m"><title>峰值 {d["peak"]:.2f} GiB</title><text class="val" x="{X0 + 4}" y="{Y1 + 10}">峰值 {d["peak"]:.2f} GiB</text></g>')
        for k, ph in d["bounds"]:
            if k <= 1:
                continue
            f.w(f'<line x1="{xp(k):.1f}" y1="{Y1}" x2="{xp(k):.1f}" y2="{Y0}" style="stroke: var(--fig-hi)" stroke-width="1.2" stroke-dasharray="4 3"/>')
            near_end = xp(k) > X1 - 60
            f.w(f'<text class="lab2" x="{xp(k) + (-3 if near_end else 3):.1f}" y="{Y0 - 6}" text-anchor="{"end" if near_end else "start"}">{names.get(ph, ph)}</text>')
        if oom:
            k, v = pts[-1]
            f.w(f'<text x="{xp(k) - 3:.1f}" y="{yp(v) - 5:.1f}" text-anchor="end" font-size="11" font-weight="600" style="fill: var(--fig-hi)">OOM</text>')
        f.w(f'<line class="axis" x1="{X0}" y1="{Y0}" x2="{X1}" y2="{Y0}"/>')
        f.w(f'<text class="tick" x="{X1}" y="{Y0 + 15}" text-anchor="end">{d["n"]} 次分配 / 释放</text>')
    f.w('<text class="lab2" x="12" y="210" transform="rotate(-90 12 210)" text-anchor="middle">显存（GiB）</text>')
    return f

# ── 图 4-1 bf16 ───────────────────────────────────────────────────────────────
def bf16():
    sizes = ["small", "medium", "large"]
    sp = {"前向": [1.87, 2.05, 2.30], "反向": [1.69, 1.79, 1.87]}
    mem = [(4.08, 3.18, "−21%"), (10.58, 8.36, "−21%"), (20.28, 16.61, "−18%")]
    f = F("fig-4-1", 252, "bf16 autocast 相对 fp32：前向快 1.87 到 2.30 倍，反向快 1.69 到 1.87 倍；前向 + 反向的峰值显存少 18% 到 21%")
    legend(f, [("前向", "--fig-1", "box"), ("反向", "--fig-2", "box")], 64, 20)
    f.w('<text class="ttl" x="64" y="46">加速比（bf16 相对 fp32）</text>')
    Y0, PY = 226, 64
    for v in (0, 1, 2):
        f.w(f'<line class="{"ref" if v == 1 else "grid"}" x1="64" y1="{Y0 - v * PY}" x2="300" y2="{Y0 - v * PY}"/><text class="tick" x="56" y="{Y0 - v * PY + 4}" text-anchor="end">{v}×</text>')
    for i, s in enumerate(sizes):
        cx = 104 + i * 78
        for j, (k, var) in enumerate((("前向", "--fig-1"), ("反向", "--fig-2"))):
            v, x = sp[k][i], cx - 26 + j * 28
            rect(f, x, Y0 - v * PY, 24, v * PY, var, f"{s} {k}：{v:.2f}×")
            f.w(f'<text class="val" x="{x + 12}" y="{Y0 - v * PY - 5:.1f}" text-anchor="middle">{v:.2f}</text>')
        f.w(f'<text class="lab" x="{cx}" y="{Y0 + 17}" text-anchor="middle">{s}</text>')
    legend(f, [("fp32", "--fig-3", "box"), ("bf16", "--fig-1", "box")], 384, 20)
    f.w('<text class="ttl" x="384" y="46">前向 + 反向的峰值显存（GiB）</text>')
    PY2 = 7
    for v in (0, 10, 20):
        f.w(f'<line class="grid" x1="384" y1="{Y0 - v * PY2}" x2="630" y2="{Y0 - v * PY2}"/><text class="tick" x="376" y="{Y0 - v * PY2 + 4}" text-anchor="end">{v}</text>')
    for i, s in enumerate(sizes):
        cx, (f32, b16, dl) = 426 + i * 80, mem[i]
        for j, (v, var, k) in enumerate(((f32, "--fig-3", "fp32"), (b16, "--fig-1", "bf16"))):
            rect(f, cx - 26 + j * 28, Y0 - v * PY2, 24, v * PY2, var, f"{s} {k}：{v} GiB", stroke="--fig-2" if var == "--fig-3" else None)
        f.w(f'<text class="val" x="{cx + 14}" y="{Y0 - b16 * PY2 - 5:.1f}" text-anchor="middle">{dl}</text><text class="lab" x="{cx}" y="{Y0 + 17}" text-anchor="middle">{s}</text>')
    return f

# ── 图 4-2 checkpoint 示意 ────────────────────────────────────────────────────
def ckpt():
    f = F("fig-4-2", 340, "4 层 xl block：不 checkpoint 时每层留 3655 MiB 一起活到反向，峰值 14.6 GiB；每 2 层一个 checkpoint 时只留 entry x0、x2，反向时逐段重算，峰值 7.5 GiB")
    KEEP, RECOMP = tint("--fig-2", 22), solid("--fig-3", "--fig-1")
    rms_lanes(f, [(6, 156, "全部存下", "峰值 14.6 GiB"), (168, 334, "每 2 层一段", "峰值 7.5 GiB")])
    YL, YK, L = 62, 114, [140, 262, 384, 506]
    f.w(f'<text class="t" x="16" y="{YL + 4}">x0</text>'); f.arrow(36, YL, L[0] - 34, YL)
    for i, c in enumerate(L):
        tbox(f, c, YL + 4, 64, 30, f"L{i + 1}", None, 'class="op"', b1=True)
        if i: f.arrow(L[i - 1] + 32, YL, c - 34, YL)
        tbox(f, c, YK, 100, 40, "3655 MiB", f"含 x{i}", KEEP)
        f.arrow(c, YL + 19, c, YK - 20, "--fig-2", head=False, width=1.2)
    f.arrow(L[3] + 32, YL, 600, YL); f.w(f'<text class="t" x="606" y="{YL + 4}">y</text>')
    f.w(f'<text class="lab2" x="626" y="{YK + 34}" text-anchor="end">4 份同时留到反向</text>')
    YL2, YK2 = 222, 290
    for cx, lab in ((52, "x0"), (336, "x2")):
        tbox(f, cx, YL2 + 4, 56, 30, lab, None, KEEP)
        f.w(f'<text class="lab2" x="{cx}" y="{YL2 + 34}" text-anchor="middle">entry 80 MiB</text>')
    for cx, lab, src in ((196, "L1 L2", "x0"), (476, "L3 L4", "x2")):
        tbox(f, cx, YL2 + 4, 152, 34, f"checkpoint［{lab}］", None, 'class="op"', b1=True)
        tbox(f, cx, YK2, 176, 40, "2 × 3655 MiB", f"反向时用 {src} 重算，用完即丢", RECOMP.replace('stroke-width="1"', 'stroke-width="1.2" stroke-dasharray="5 3"'))
        f.arrow(cx, YL2 + 21, cx, YK2 - 20, "--fig-1", head=False, width=1.2)
    f.arrow(80, YL2, 118, YL2); f.arrow(272, YL2, 306, YL2); f.arrow(364, YL2, 398, YL2)
    f.arrow(552, YL2, 600, YL2); f.w(f'<text class="t" x="606" y="{YL2 + 4}">y</text>')
    f.w(f'<text class="lab2" x="626" y="{YK2 + 36}" text-anchor="end">反向时一次只重算一段</text>')
    return f

# ── 图 4-3 checkpoint 段长扫描（replaces checkpoint_large_sweep.png） ──────────
def sweep():
    rows = json.load(open(os.path.join(HERE, "checkpoint_large_b1_seq1024.json")))
    base = next(r for r in rows if r["checkpoint_every"] is None)
    pts = sorted((r for r in rows if r["checkpoint_every"]), key=lambda r: r["checkpoint_every"])
    es = [r["checkpoint_every"] for r in pts]
    f = F("fig-4-3", 250, f"checkpoint 段长扫描：step 时间都在 302 到 313 ms，高于不 checkpoint 的 {base['avg_ms']:.0f} ms；峰值显存随每段层数增加，每层一个 checkpoint 时最低 7.8 GiB")
    legend(f, [("checkpoint", "--fig-1", "line"), ("不 checkpoint", "--fig-hi", "line")], 64, 18)
    for p, (key, title, ymax, step, unit, bfmt) in enumerate((("avg_ms", "step 时间（ms）", 350, 100, "ms", "{:.0f} ms"),
                                                            ("peak_mem_gib", "峰值显存（GiB）", 16, 4, "GiB", "{:.1f} GiB"))):
        X0, X1, Y1, Y0 = 64 + p * 300, 64 + p * 300 + 250, 52, 206
        lx = lambda e: X0 + (X1 - X0) * math.log2(e) / math.log2(36)
        yp = lambda v: Y0 - (Y0 - Y1) * v / ymax
        f.w(f'<text class="ttl" x="{X0}" y="42">{title}</text>')
        for v in range(0, ymax + 1, step):
            f.w(f'<line class="grid" x1="{X0}" y1="{yp(v):.1f}" x2="{X1}" y2="{yp(v):.1f}"/><text class="tick" x="{X0 - 6}" y="{yp(v) + 4:.1f}" text-anchor="end">{v}</text>')
        bv = base[key]
        f.w(f'<line x1="{X0}" y1="{yp(bv):.1f}" x2="{X1}" y2="{yp(bv):.1f}" style="stroke: var(--fig-hi)" stroke-width="1.6" stroke-dasharray="5 4"/>')
        f.w(f'<text class="val" x="{X1}" y="{yp(bv) - 5:.1f}" text-anchor="end">不 checkpoint {bfmt.format(bv)}</text>')
        f.w(f'<polyline points="{" ".join(f"{lx(r["checkpoint_every"]):.1f},{yp(r[key]):.1f}" for r in pts)}" fill="none" style="stroke: var(--fig-1)" stroke-width="2"/>')
        for r in pts:
            f.w(f'<g class="m"><title>每段 {r["checkpoint_every"]} 层：{bfmt.format(r[key])}</title><circle cx="{lx(r["checkpoint_every"]):.1f}" cy="{yp(r[key]):.1f}" r="4" class="ring" style="fill: var(--fig-1)"/></g>')
        for e in es:
            f.w(f'<text class="tick" x="{lx(e):.1f}" y="{Y0 + 15}" text-anchor="middle">{e}</text>')
        f.w(f'<line class="axis" x1="{X0}" y1="{Y0}" x2="{X1}" y2="{Y0}"/>')
        f.w(f'<text class="lab2" x="{(X0 + X1) / 2:.0f}" y="{Y0 + 32}" text-anchor="middle">每段几层（对数轴）</text>')
    return f


# ── 图 1-2 / 1-3 GPU 硬件示意（RTX 5090，数据来自 NVIDIA RTX Blackwell 白皮书） ─────
def hw_hierarchy():
    f = F("fig-h-1", 300, "RTX 5090 的存储层级：每个 SM 里有 256 KB register 和 128 KB L1/shared memory，170 个 SM 共享芯片上的 96 MB L2，芯片外是 32 GB GDDR7 显存，带宽 1.79e12 B/s")
    f.w('<rect x="10" y="22" width="452" height="250" rx="12" style="fill: none; stroke: currentColor" stroke-opacity=".35" stroke-dasharray="6 4"/>')
    f.w('<text class="lab2" x="22" y="40">GPU 芯片（die），共 170 个 SM</text>')
    def sm(x):
        f.w(f'<rect x="{x}" y="52" width="92" height="108" rx="8" {tint("--fig-1", 10, sw=1.4)}/>')
        f.w(f'<text class="tb" x="{x + 46}" y="70" text-anchor="middle">SM</text>')
        f.w(f'<rect x="{x + 8}" y="80" width="76" height="32" rx="5" {tint("--fig-hi", 16)}/>')
        f.w(f'<text class="s" x="{x + 46}" y="94" text-anchor="middle">register</text><text class="s" x="{x + 46}" y="107" text-anchor="middle">256 KB</text>')
        f.w(f'<rect x="{x + 8}" y="118" width="76" height="34" rx="5" {tint("--fig-2", 22)}/>')
        f.w(f'<text class="s" x="{x + 46}" y="132" text-anchor="middle">L1 / shared</text><text class="s" x="{x + 46}" y="146" text-anchor="middle">128 KB</text>')
        f.arrow(x + 46, 160, x + 46, 194, None, width=1.2, head=False)
    for x in (22, 122, 222):
        sm(x - 2)
    f.w('<text class="t" x="339" y="110" text-anchor="middle">⋯</text>')
    sm(360)
    f.w(f'<rect x="24" y="194" width="426" height="40" rx="8" {tint("--fig-3", 45, stroke="--fig-2")}/>')
    f.w('<text class="t" x="237" y="219" text-anchor="middle">L2 cache 96 MB，所有 SM 共享</text>')
    f.w('<text class="lab2" x="237" y="258" text-anchor="middle">片上</text>')
    f.w(f'<rect x="504" y="70" width="124" height="150" rx="10" {tint("--fig-mute", 14, sw=1.4)}/>')
    f.w('<text class="tb" x="566" y="124" text-anchor="middle">GDDR7 显存</text><text class="t" x="566" y="144" text-anchor="middle">32 GB</text><text class="s" x="566" y="168" text-anchor="middle">带宽 β</text><text class="s" x="566" y="182" text-anchor="middle">1.79e12 B/s</text>')
    f.w('<text class="lab2" x="566" y="238" text-anchor="middle">片外</text>')
    f.arrow(452, 214, 500, 214, "--fig-mute", width=1.8); f.arrow(500, 200, 452, 200, "--fig-mute", width=1.8, head=True)
    f.w('<text class="lab2" x="10" y="292">越靠近 SM 越快、越小：register > L1 / shared > L2 > 显存</text>')
    return f

def hw_sm():
    f = F("fig-h-2", 392, "RTX 5090 的一个 SM：4 个 SMSP，每个有 1 个 warp 调度器、64 KB register、32 个 CUDA core、1 个 Tensor core 和 4 个 LD/ST；LD/ST 把请求交给 4 个 SMSP 共享的 128 KB L1/shared memory；TMA 接在 shared memory 和片上互联之间；请求经片上互联到 L2，再到显存")
    def bidir(x, y1, y2):                                     # 一来一回两条箭头，各有自己的杆
        f.arrow(x - 5, y1, x - 5, y2, None, width=1.4)
        f.arrow(x + 5, y2, x + 5, y1, None, width=1.4)
    f.w(f'<rect x="10" y="10" width="620" height="306" rx="12" {tint("--fig-1", 6, sw=1.4)}/>')
    f.w('<text class="tb" x="24" y="32">SM（每颗 5090 有 170 个）</text>')
    for k in range(4):
        x = 24 + k * 151
        f.w(f'<rect x="{x}" y="44" width="141" height="148" rx="8" class="op"/>')
        f.w(f'<text class="tb" x="{x + 70}" y="62" text-anchor="middle">SMSP {k}</text>')
        rows = [("warp 调度器", "--fig-1", 10), ("register 64 KB", "--fig-hi", 16), ("CUDA core ×32", "--fig-2", 22),
                ("Tensor core ×1", "--fig-1", 22), ("LD/ST ×4", "--fig-mute", 22)]
        for j, (t, var, pct) in enumerate(rows):
            y = 70 + j * 24
            f.w(f'<rect x="{x + 10}" y="{y}" width="121" height="20" rx="4" {tint(var, pct)}/>')
            f.w(f'<text class="s" x="{x + 70}" y="{y + 14}" text-anchor="middle">{t}</text>')
        bidir(x + 70, 192, 214)
    f.w(f'<rect x="24" y="214" width="594" height="38" rx="8" {tint("--fig-2", 22)}/>')
    f.w('<text class="t" x="321" y="238" text-anchor="middle">L1 cache / shared memory 128 KB（4 个 SMSP 共享）</text>')
    f.w(f'<rect x="447" y="274" width="171" height="34" rx="8" {tint("--fig-hi", 14)}/>')
    f.w('<text class="tb" x="532" y="289" text-anchor="middle">TMA</text><text class="s" x="532" y="302" text-anchor="middle">整块搬，自己算地址</text>')
    bidir(532, 252, 274)
    f.w(f'<rect x="24" y="346" width="594" height="30" rx="8" {tint("--fig-mute", 14)}/>')
    f.w('<text class="t" x="321" y="366" text-anchor="middle">片上互联（crossbar）→ L2（所有 SM 共享）→ 显存</text>')
    bidir(200, 252, 346)
    bidir(532, 308, 346)
    f.w('<text class="s" x="214" y="300">LD/ST 的请求</text><text class="s" x="434" y="330" text-anchor="end">TMA 的请求</text>')
    return f

def hw_move():
    f = F("fig-h-10", 150, "显存到 shared memory 的三种搬法：ld.global 加 st.shared 经过 L2 和 register；cp.async 和 TMA 经过 L2，直接写进 shared memory，不经过 register")
    COLS = [("显存", 200, 80, "--fig-mute", 14, None), ("L2", 310, 80, "--fig-3", 45, "--fig-2"),
            ("register", 420, 80, "--fig-hi", 16, None), ("shared memory", 548, 110, "--fig-2", 22, None)]
    for name, cx, *_ in COLS:
        f.w(f'<text class="lab2" x="{cx}" y="20" text-anchor="middle">{name}</text>')
    ROWS = [("ld.global + st.shared", "每个 thread 算地址", [0, 1, 2, 3]),
            ("cp.async", "每个 thread 算地址", [0, 1, 3]),
            ("TMA", "1 个 thread 发，整块搬", [0, 1, 3])]
    for r, (name, sub, cells) in enumerate(ROWS):
        y = 32 + r * 38
        f.w(f'<text class="lab" x="8" y="{y + 11}">{name}</text><text class="s" x="8" y="{y + 25}">{sub}</text>')
        for i in cells:
            n, cx, w, var, pct, st = COLS[i]
            f.w(f'<rect x="{cx - w / 2}" y="{y}" width="{w}" height="26" rx="4" {tint(var, pct, stroke=st)}/>')
        for a, b in zip(cells, cells[1:]):
            f.arrow(COLS[a][1] + COLS[a][2] / 2 + 2, y + 13, COLS[b][1] - COLS[b][2] / 2 - 2, y + 13, None, width=1.4)
    return f

def hw_smsp():
    f = F("fig-h-9", 322, "RTX 5090 一个 SMSP 的结构：L0 指令 cache、warp scheduler 和 dispatch，64 KB register file，32 个 CUDA core（FP32/INT32）和 1 个 Tensor core，4 个 LD/ST 和 SFU；下面接 SM 共享的 128 KB L1 / shared memory")
    f.w(f'<rect x="12" y="12" width="356" height="248" rx="12" {tint("--fig-1", 6, sw=1.4)}/>')
    f.w('<text class="tb" x="30" y="33">一个 SMSP（每个 SM 4 个）</text>')
    M = lambda x, y, n, anchor="start": f.w(f'<text class="s" x="{x}" y="{y}" text-anchor="{anchor}" style="fill: var(--fig-hi)">{n}</text>')
    def box(x, y, w, h, t, var, pct, cls="s"):
        f.w(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" {tint(var, pct)}/>')
        f.w(f'<text class="{cls}" x="{x + w / 2}" y="{y + h / 2 + 4}" text-anchor="middle">{t}</text>')
    box(30, 42, 96, 24, "L0 i-cache", "--fig-1", 10)
    box(132, 42, 112, 24, "① warp scheduler", "--fig-1", 18)
    box(250, 42, 104, 24, "dispatch", "--fig-1", 10)
    box(30, 72, 324, 24, "② register file：16,384 × 32-bit = 64 KB", "--fig-hi", 14)
    for r in range(4):                                # 32 个 CUDA core，4 × 8
        for c in range(8):
            f.w(f'<rect x="{30 + c * 25.5:.1f}" y="{102 + r * 26}" width="22.5" height="23" rx="3" {tint("--fig-2", 26, sw=1)}/>')
    f.w('<rect x="66" y="137" width="132" height="34" rx="5" style="fill: var(--fig-bg); stroke: var(--fig-2)"/>')
    f.w('<text class="tb" x="132" y="152" text-anchor="middle">③ CUDA core</text><text class="s" x="132" y="166" text-anchor="middle">FP32 / INT32 × 32</text>')
    f.w(f'<rect x="238" y="102" width="116" height="101" rx="4" {tint("--fig-1", 24)}/>')
    f.w('<text class="tb" x="296" y="148" text-anchor="middle">④ Tensor core</text><text class="s" x="296" y="165" text-anchor="middle">第 5 代，× 1</text>')
    M(16, 225, "⑤")
    for k in range(4):
        box(30 + k * 52.5, 210, 47, 22, "LD/ST", "--fig-mute", 22)
    box(250, 210, 104, 22, "⑥ SFU", "--fig-3", 45)
    f.w('<text class="lab2" x="190" y="251" text-anchor="middle">↓ 4 个 SMSP 共享 128 KB L1 / shared memory（图 1-1）</text>')
    notes = [("①", ["warp scheduler：每拍从驻留的", "warp 里挑一个就绪的；dispatch", "把它的一条指令发给 32 个 thread"]),
             ("②", ["register：按 thread 切给驻留的 thread，", "每个 thread 最多 255 个，只归自己用"]),
             ("③", ["CUDA core：一条加法、乘法或", "FMA，32 个 thread 各用一个；", "同一拍只能做 FP32 或 INT32"]),
             ("④", ["Tensor core：一条指令算一个", "小矩阵乘，tl.dot 默认走它"]),
             ("⑤", ["LD/ST：load / store，在 register", "和 L1 / shared memory、显存之间", "搬数据"]),
             ("⑥", ["SFU：exp、rsqrt、sin 这类特殊函数"])]
    y = 34
    for n, lines in notes:
        M(384, y, n)
        for k, line in enumerate(lines):
            f.w(f'<text class="lab2" x="400" y="{y + k * 15}">{line}</text>')
        y += 15 * len(lines) + 8
    return f

def hw_cores():
    f = F("fig-h-11", 476, "左：一条 FMA，每个 thread 用一个 CUDA core，从 register 读 a、b、c，先乘、乘积不舍入再加 c，最后舍入一次写回 d；warp 一条指令 64 FLOPs。右：一条 mma.m16n8k16，整个 warp 一起发，Tensor core 算 16×16 的 A 乘 16×8 的 B 再加 16×8 的 C；一条指令 4096 FLOPs。下：SFU 一条 MUFU.EX2，x 的高位查表取系数 C0、C1、C2，低位 Δ 代入 C0 + C1Δ + C2Δ² 得到 2 的 x 次方")
    for x, w in ((12, 304), (328, 300)):
        f.w(f'<rect x="{x}" y="10" width="{w}" height="292" rx="12" {tint("--fig-1", 6, sw=1.4)}/>')
    # 左：CUDA core 的 FMA
    f.w('<text class="tb" x="28" y="34">CUDA core：一条 FMA，d = a × b + c</text>')
    def reg(x, y, t):
        f.w(f'<rect x="{x}" y="{y}" width="40" height="24" rx="4" {tint("--fig-hi", 16)}/>')
        f.w(f'<text class="t" x="{x + 20}" y="{y + 17}" text-anchor="middle" font-style="italic">{t}</text>')
    def op(cx, cy, t):
        f.w(f'<circle cx="{cx}" cy="{cy}" r="16" class="op"/><text class="tb" x="{cx}" y="{cy + 5}" text-anchor="middle">{t}</text>')
    reg(44, 50, "a"); reg(104, 50, "b"); reg(200, 50, "c")
    op(94, 114, "×"); op(150, 164, "+")
    f.arrow(64, 74, 84, 99, None, width=1.4); f.arrow(124, 74, 104, 99, None, width=1.4)
    f.arrow(106, 125, 138, 152, None, width=1.4); f.arrow(220, 74, 162, 152, None, width=1.4)
    f.w('<text class="s" x="58" y="150" text-anchor="middle">乘积不舍入</text>')
    f.w(f'<rect x="110" y="196" width="80" height="22" rx="4" class="op"/><text class="s" x="150" y="211" text-anchor="middle">只舍入一次</text>')
    f.arrow(150, 180, 150, 194, None, width=1.4); f.arrow(150, 218, 150, 230, None, width=1.4)
    reg(130, 232, "d")
    f.w('<text class="s" x="246" y="66">register</text>')
    f.w('<text class="lab2" x="28" y="276">每个 thread 用一个 CUDA core，warp 32 个一起</text>')
    f.w('<text class="lab" x="28" y="292">一条指令：32 × 2 = 64 FLOPs</text>')
    # 右：Tensor core 的 mma
    f.w('<text class="tb" x="344" y="34">Tensor core：一条 mma，D = A × B + C</text>')
    def grid(x, y, cols, rows, var, pct):
        c = 4
        f.w(f'<rect x="{x}" y="{y}" width="{cols * c}" height="{rows * c}" {tint(var, pct, sw=1.2)}/>')
        for i in range(1, cols): f.w(f'<line x1="{x + i * c}" y1="{y}" x2="{x + i * c}" y2="{y + rows * c}" stroke="currentColor" stroke-opacity=".1"/>')
        for j in range(1, rows): f.w(f'<line x1="{x}" y1="{y + j * c}" x2="{x + cols * c}" y2="{y + j * c}" stroke="currentColor" stroke-opacity=".1"/>')
    Y = 72
    mats = [(350, 16, "A 16×16", "bf16", "--fig-2", 30), (436, 8, "B 16×8", "bf16", "--fig-2", 30),
            (494, 8, "C 16×8", "fp32", "--fig-3", 45), (566, 8, "D 16×8", "fp32", "--fig-3", 45)]
    for x, cols, l1, l2, var, pct in mats:
        grid(x, Y, cols, 16, var, pct)
        f.w(f'<text class="s" x="{x + cols * 2}" y="{Y + 82}" text-anchor="middle">{l1}</text><text class="s" x="{x + cols * 2}" y="{Y + 96}" text-anchor="middle">{l2}</text>')
    f.w(f'<text class="tb" x="425" y="{Y + 37}" text-anchor="middle">×</text><text class="tb" x="481" y="{Y + 37}" text-anchor="middle">+</text>')
    f.arrow(532, Y + 32, 560, Y + 32, None, width=1.4)
    f.w('<text class="lab2" x="344" y="212">A、B、C 分散在 32 个 thread 的 register 里</text>')
    f.w('<text class="lab2" x="344" y="230">输入 bf16 / fp16 / tf32 / fp8，fp32 累加</text>')
    f.w('<text class="lab2" x="344" y="276">整个 warp 发一条（mma.m16n8k16）</text>')
    f.w('<text class="lab" x="344" y="292">一条指令：2 × 16 × 8 × 16 = 4096 FLOPs</text>')
    # 下：SFU
    f.w(f'<rect x="12" y="314" width="616" height="156" rx="12" {tint("--fig-1", 6, sw=1.4)}/>')
    f.w('<text class="tb" x="28" y="338">SFU：一条 MUFU.EX2，近似 2<tspan dy="-5" font-size="9">x</tspan></text>')
    reg(28, 368, "x")
    f.w(f'<rect x="104" y="356" width="104" height="50" rx="4" class="op"/><line x1="104" y1="381" x2="208" y2="381" stroke="currentColor" stroke-opacity=".45"/>')
    f.w('<text class="s" x="156" y="373" text-anchor="middle">高位</text><text class="s" x="156" y="398" text-anchor="middle">低位 Δ</text>')
    f.arrow(68, 380, 102, 380, None, width=1.4)
    f.w(f'<rect x="248" y="350" width="112" height="26" rx="4" {tint("--fig-3", 45, stroke="--fig-2")}/><text class="s" x="304" y="367" text-anchor="middle">查表：C0、C1、C2</text>')
    f.w(f'<rect x="398" y="368" width="140" height="30" rx="4" class="op"/><text class="s" x="468" y="387" text-anchor="middle">C0 + C1·Δ + C2·Δ²</text>')
    f.arrow(208, 368, 246, 364, None, width=1.4); f.arrow(360, 366, 396, 378, None, width=1.4); f.arrow(208, 394, 396, 390, None, width=1.4)
    f.arrow(538, 383, 562, 383, None, width=1.4)
    f.w(f'<rect x="564" y="370" width="48" height="26" rx="4" {tint("--fig-hi", 16)}/><text class="t" x="588" y="388" text-anchor="middle">2<tspan dy="-5" font-size="9" font-style="italic">x</tspan></text>')
    f.w('<text class="lab2" x="28" y="436">tl.exp(x) 编出来是两条：FMUL 算 x × log₂e（CUDA core），MUFU.EX2（SFU）</text>')
    f.w('<text class="lab" x="28" y="456">每个 SM 每拍 16 个，FMA 是 128 个（实测吞吐比约 1 : 7）</text>')
    return f

# ── 3.6 节：matmul 的 BM、BN 扫描（mm_tile_sweep.json） ─────────────────────
def _tile_rows():
    rows = json.load(open(os.environ.get("MM_TILE_JSON", os.path.join(HERE, "mm_tile_sweep.json"))))
    for r in rows:
        r["acc"] = r["BM"] * r["BN"] // (32 * r["nw"]); r["wps"] = r["per_sm"] * r["nw"]
    return rows

def tile_res():
    rows = [r for r in _tile_rows() if (r["M"], r["N"], r["K"]) == (4096, 512, 2048)]
    L = json.load(open(os.path.join(HERE, "residency.json")))["limits"]
    best = min(rows, key=lambda r: r["us"])
    ref = next(r for r in rows if (r["BM"], r["BN"], r["BK"], r["nw"], r["ns"]) == (64, 64, 32, 4, 3))
    f = F("fig-d-1", 262, f"正文形状下最快的配置（BM = BN = {best['BM']}，{best['nw']} 个 warp）和正文用的 64 × 64、4 个 warp，在一个 SM 上各用了多少：最快的只驻留 {best['wps']} 个 warp，register 用掉 {best['reg_util']:.0%}")
    X0, BW = 156, 312
    BARS = (("warps", "warp 名额", L["warps"], f'{L["warps"]}'), ("blocks", "program 名额", L["blocks"], f'{L["blocks"]}'),
            ("regs", "register", L["regs"], f'{L["regs"]}'), ("shared", "shared memory", L["shared"], f'{L["shared"] // 1024} KB'))
    for i, (r, title) in enumerate(((best, "最快"), (ref, "正文用的"))):
        y0 = 10 + i * 112; n = r["per_sm"]
        use = {"warps": r["nw"], "blocks": 1, "regs": -(-r["regs"] // 8) * 8 * 32 * r["nw"], "shared": r["smem"] + L["reserved"]}
        mx = {k: L[k] // u for k, u in ((k, use[k]) for k in use)}
        bind = min(mx, key=lambda k: mx[k])
        f.w(f'<rect x="10" y="{y0}" width="620" height="100" rx="10" {tint("--fig-1", 5, sw=1)}/>')
        f.w(f'<text class="tb" x="22" y="{y0 + 18}">{title}：BM = {r["BM"]}，BN = {r["BN"]}，BK = {r["BK"]}，{r["nw"]} 个 warp</text>')
        f.w(f'<text class="lab2" x="22" y="{y0 + 34}">每个 thread {r["acc"]} 个 acc，{r["regs"]} 个 register；shared memory {r["smem"] // 1024} KB；{r["progs"]} 个 program</text>')
        f.w(f'<text class="tb" x="618" y="{y0 + 18}" text-anchor="end" style="fill: var(--fig-hi)">{r["us"]:.1f} µs</text>')
        for j, (bk, blab, C, cap) in enumerate(BARS):
            y = y0 + 44 + j * 14; u = use[bk]; sx = BW / C
            f.w(f'<text class="s" x="22" y="{y + 9}">{blab}</text><text class="s" x="{X0 - 6}" y="{y + 9}" text-anchor="end">{cap}</text>')
            f.w(f'<rect x="{X0}" y="{y}" width="{BW}" height="11" rx="2" fill="currentColor" fill-opacity=".06" stroke="currentColor" stroke-opacity=".3" stroke-width=".8"/>')
            for k in range(n):
                f.w(f'<rect x="{X0 + k * u * sx:.2f}" y="{y}" width="{u * sx:.2f}" height="11" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="{.8 if u * sx > 6 else .4}"/>')
            if bk == bind:
                f.w(f'<rect x="{X0 - 2}" y="{y - 2}" width="{BW + 4}" height="15" rx="3" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6"/>')
            used = n * u / C
            cls = 'class="s" style="fill: var(--fig-hi); opacity: 1; font-weight: 600"' if bk == bind else 'class="s"'
            f.w(f'<text {cls} x="{X0 + BW + 8}" y="{y + 9}">用了 {used:.0%}</text>')
    f.w(f'<rect x="22" y="{240}" width="22" height="10" style="fill: var(--fig-1)"/><text class="lab" x="52" y="249">一格是一个 program 占的量</text>')
    f.w(f'<rect x="230" y="{239}" width="22" height="12" rx="2" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6"/><text class="lab" x="260" y="249">先用完的那条</text>')
    return f

def tile_rules():
    rows = _tile_rows()
    SH = (((4096, 512, 2048), "--fig-1", "正文形状 4096 × 2048 × 512"), ((4096, 4096, 4096), "--fig-hi", "4096 × 4096 × 4096"))
    f = F("fig-d-2", 300, "按每个 thread 的 acc 个数、每个 SM 驻留的 warp 数分组，每组最快的配置比整体最快慢多少：acc 在 32 到 64 个、warp 在 8 到 16 个时最快，warp 驻留满 48 个反而慢一倍")
    YT, YB, LO, HI = 40, 230, 1.0, 2.4
    py = lambda v: YB - (min(v, HI) - LO) / (HI - LO) * (YB - YT)
    PAN = ((60, 300, "acc", [2, 4, 8, 16, 32, 64, 128], "每个 thread 的 acc 个数", (32, 64)),
           (380, 620, "wps", [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48], "每个 SM 驻留的 warp 数", (8, 16)))
    for x0, x1, key, xs, xlab, good in PAN:
        px = lambda v: x0 + xs.index(v) * (x1 - x0) / (len(xs) - 1)
        f.w(f'<rect x="{px(good[0]) - 6:.1f}" y="{YT}" width="{px(good[1]) - px(good[0]) + 12:.1f}" height="{YB - YT}" rx="4" {tint("--fig-2", 14, sw=0)}/>')
        for v in (1.0, 1.5, 2.0):
            f.w(f'<line class="grid" x1="{x0}" y1="{py(v):.1f}" x2="{x1}" y2="{py(v):.1f}"/><text class="tick" x="{x0 - 6}" y="{py(v) + 4:.1f}" text-anchor="end">{v:.1f}×</text>')
        for v in xs:
            if key == "wps" and v % 8 and v not in (4, 12): continue
            f.w(f'<text class="tick" x="{px(v):.1f}" y="{YB + 16}" text-anchor="middle">{v}</text>')
        f.w(f'<text class="lab2" x="{(x0 + x1) / 2:.1f}" y="{YB + 34}" text-anchor="middle">{xlab}</text>')
        for shape, var, lab in SH:
            rs = [r for r in rows if (r["M"], r["N"], r["K"]) == shape]
            top = min(r["us"] for r in rs)
            g = {}
            for r in rs:
                if r["spills"] == 0 and r[key] in xs:
                    g[r[key]] = min(g.get(r[key], 1e30), r["us"])
            pts = [(v, g[v] / top) for v in xs if v in g]
            f.w(f'<polyline points="{" ".join(f"{px(v):.1f},{py(t):.1f}" for v, t in pts)}" fill="none" style="stroke: var({var})" stroke-width="2" stroke-linejoin="round"/>')
            for v, t in pts:
                f.w(f'<g class="m"><title>{lab}，{xlab} {v}：最快 {g[v]:.1f} µs，是整体最快的 {t:.2f} 倍</title><circle cx="{px(v):.1f}" cy="{py(t):.1f}" r="3.5" class="ring" style="fill: var({var})"/></g>')
    f.w(f'<text class="lab2" x="60" y="22">比最快的配置慢几倍（每组取最快的，不算 spill 的配置）；底色是最快的区间</text>')
    lx = 60
    for shape, var, lab in SH:
        f.w(f'<line x1="{lx}" y1="{290 - 4}" x2="{lx + 22}" y2="{290 - 4}" style="stroke: var({var})" stroke-width="2"/><text class="lab" x="{lx + 30}" y="290">{lab}</text>')
        lx += 250
    return f

# ── 图 3-3 第 2 节四个 kernel 的 program 怎么放上一个 SM（residency.json） ─────────
def residency():
    R = json.load(open(os.path.join(HERE, "residency.json")))
    L = R["limits"]
    f = F("fig-t-8", 590, "第 2 节图里的一个 program 是一个 CUDA block：4 个 warp、128 个 thread。放上 SM 时占 4 个 warp 名额、1 个 program 名额、128 × 每个 thread 的 register、shared memory 加 1 KB；四个 kernel 各画四根条，哪条先用完就放几个：GELU 和 row sum 卡在 warp，softmax 卡在 register，matmul 卡在 shared memory")
    # 上：grid 的一格 → 一个 program = 一个 CUDA block → 上 SM 要占的四样
    f.w('<text class="ttl" x="10" y="16">第 2 节图里的 grid</text>')
    for k, lab in enumerate(("0", "1", "2")):
        st = tint("--fig-2", 45, stroke="--fig-1", sw=1.6) if k == 1 else tint("--fig-mute", 14, sw=1)
        f.w(f'<rect x="{10 + k * 34}" y="40" width="30" height="24" rx="4" {st}/>')
        f.w(f'<text class="t" x="{25 + k * 34}" y="56" text-anchor="middle">{lab}</text>')
    f.w('<text class="tb" x="122" y="56" text-anchor="middle">⋯</text>')
    f.w('<text class="s" x="10" y="82">一格一个 program</text>')
    f.arrow(136, 52, 160, 52, None, width=1.6)
    f.w(f'<rect x="164" y="24" width="258" height="94" rx="8" {tint("--fig-1", 8, sw=1.4)}/>')
    f.w('<text class="tb" x="176" y="42">一个 program = 一个 CUDA block</text>')
    for k in range(4):
        x = 176 + k * 59
        f.w(f'<rect x="{x}" y="52" width="54" height="36" rx="4" {tint("--fig-2", 26, sw=1)}/>')
        f.w(f'<text class="s" x="{x + 27}" y="67" text-anchor="middle">warp {k}</text><text class="s" x="{x + 27}" y="81" text-anchor="middle">32 thread</text>')
    f.w('<text class="s" x="176" y="108">共 128 个 thread，每个有自己的 register</text>')
    f.arrow(424, 71, 440, 71, None, width=1.6)
    f.w(f'<rect x="444" y="24" width="186" height="94" rx="8" class="op"/>')
    f.w('<text class="tb" x="456" y="42">放上 SM 时要占四样</text>')
    for k, lab in enumerate(("① 4 个 warp 名额", "② 1 个 program 名额", "③ register：128 × 每 thread 个数", "④ shared memory：用量 + 1 KB")):
        f.w(f'<text class="s" x="456" y="{60 + k * 15}">{lab}</text>')
    # 下：四个 kernel，各四根条
    X0, BW = 156, 312
    NAMES = (("gelu", "GELU", "fig-t-1", "2-2"), ("softmax", "softmax", "fig-t-2", "2-3"),
             ("row_sum", "row sum", "fig-t-3", "2-4"), ("matmul", "matmul", "fig-t-4", "2-5"))
    BARS = (("warps", "warp 名额", f'{L["warps"]}'), ("blocks", "program 名额", f'{L["blocks"]}'),
            ("regs", "register", f'{L["regs"]}'), ("shared", "shared memory", f'{L["shared"] // 1024} KB'))
    def amt(key, v):
        return (f"{v} B" if v < 2048 else f"{v // 1024} KB") if key == "shared" else f"{v}"
    for i, (key, name, fid, num) in enumerate(NAMES):
        r = R[key]; y0 = 136 + i * 104; n = r["per_sm"]
        alloc = r["use"]["regs"] // (32 * r["warps"])
        sh = "0 B" if r["shared"] == 0 else (f'{r["shared"]} B' if r["shared"] < 1024 else f'{r["shared"] // 1024} KB')
        bind = min(r["max"], key=lambda k: r["max"][k])
        f.w(f'<rect x="10" y="{y0}" width="620" height="96" rx="10" {tint("--fig-1", 5, sw=1)}/>')
        f.w(f'<text class="tb" x="22" y="{y0 + 18}">{name}（<a href="#{fid}">图 {num}</a>）</text>')
        f.w(f'<text class="lab2" x="{X0}" y="{y0 + 18}">每个 thread {r["regs"]} 个 register' + (f'（按 {alloc} 个分配）' if alloc != r["regs"] else '') + f'，shared memory {sh}</text>')
        f.w(f'<text class="tb" x="618" y="{y0 + 18}" text-anchor="end" style="fill: var(--fig-hi)">放 {n} 个</text>')
        for j, (bk, blab, cap) in enumerate(BARS):
            y = y0 + 28 + j * 16; u, C, m = r["use"][bk], L[bk], r["max"][bk]
            sx = BW / C
            f.w(f'<text class="s" x="22" y="{y + 10}">{blab}</text><text class="s" x="{X0 - 6}" y="{y + 10}" text-anchor="end">{cap}</text>')
            f.w(f'<rect x="{X0}" y="{y}" width="{BW}" height="12" rx="2" fill="currentColor" fill-opacity=".06" stroke="currentColor" stroke-opacity=".3" stroke-width=".8"/>')
            if m > n:
                f.w(f'<rect x="{X0 + n * u * sx:.1f}" y="{y + .5}" width="{(m - n) * u * sx:.1f}" height="11" rx="1.5" {tint("--fig-2", 14, sw=1, dash=True)}/>')
                if (m - n) * u * sx > 70:
                    f.w(f'<text class="s" x="{X0 + (n + m) / 2 * u * sx:.1f}" y="{y + 9.5}" text-anchor="middle" style="font-size: 9.5px">还能放 {m - n} 个</text>')
            for k in range(n):
                f.w(f'<rect x="{X0 + k * u * sx:.2f}" y="{y}" width="{u * sx:.2f}" height="12" style="fill: var(--fig-1); stroke: var(--nv-bg, #fff)" stroke-width="{.8 if u * sx > 6 else .4}"/>')
            hit = bk == bind
            if hit:
                f.w(f'<rect x="{X0 - 2}" y="{y - 2}" width="{BW + 4}" height="16" rx="3" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6"/>')
            cls = 'class="s" style="fill: var(--fig-hi); opacity: 1; font-weight: 600"' if hit else 'class="s"'
            f.w(f'<text {cls} x="{X0 + BW + 8}" y="{y + 10}">每个占 {amt(bk, u)}，最多 {m} 个</text>')
    yl = 576; x = 22
    f.w(f'<rect x="{x}" y="{yl - 9}" width="22" height="10" style="fill: var(--fig-1)"/><text class="lab" x="{x + 30}" y="{yl}">实际放进去的 program，一格一个</text>')
    x += 30 + 13 * 12 + 30
    f.w(f'<rect x="{x}" y="{yl - 9}" width="22" height="10" rx="1.5" {tint("--fig-2", 14, sw=1, dash=True)}/><text class="lab" x="{x + 30}" y="{yl}">只看这一条，还能再放</text>')
    x += 30 + 9 * 12 + 10 * 0 + 34
    f.w(f'<rect x="{x}" y="{yl - 10}" width="22" height="12" rx="2" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6"/><text class="lab" x="{x + 30}" y="{yl}">先用完的那条</text>')
    return f

def warp_slots():
    f = F("fig-h-6", 300, "一个 SM 上驻留 3 个 program、每个 4 个 warp 时的某一拍：每个 SMSP 有 3 个 warp，分别来自 3 个 program；每个调度器这一拍最多发射一个，SMSP 2 的 3 个都在等数据，调度器空转")
    f.w(f'<rect x="10" y="10" width="620" height="246" rx="12" {tint("--fig-1", 6, sw=1.4)}/>')
    f.w('<text class="tb" x="24" y="32">一个 SM 的驻留情况</text>')
    f.w('<text class="lab2" x="616" y="32" text-anchor="end">整个 SM 最多驻留 24 个 program、48 个 warp</text>')
    ST = {"I": 'style="fill: var(--fig-hi)"', "R": 'style="fill: var(--fig-2)"', "W": 'style="fill: var(--fig-1)" fill-opacity=".55"',
          "-": 'style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45"'}
    for k in range(4):
        x = 24 + k * 151
        f.w(f'<rect x="{x}" y="44" width="141" height="164" rx="8" class="op"/>')
        f.w(f'<text class="tb" x="{x + (58 if k == 2 else 70)}" y="62" text-anchor="middle">SMSP {k}</text>')
        if k == 2:
            f.w(f'<text class="s" x="{x + 104}" y="62" style="fill: var(--fig-hi)">空转</text>')
        f.w(f'<rect x="{x + 10}" y="70" width="121" height="22" rx="4" {tint("--fig-1", 10)}/>')
        f.w(f'<text class="s" x="{x + 70}" y="85" text-anchor="middle">warp 调度器</text>')
        f.w(f'<text class="s" x="{x + 70}" y="110" text-anchor="middle">最多驻留 12 个 warp</text>')
        state = ["IRW", "IWW", "WWW", "IRR"][k]   # 这一拍：I 正在发射，R 就绪但没被选中，W 在等数据，其余是空名额
        for q in range(12):
            sx, sy = x + 7 + (q % 4) * 33, 118 + (q // 4) * 28
            c = state[q] if q < len(state) else "-"
            f.w(f'<rect x="{sx}" y="{sy}" width="28" height="22" rx="3" {ST[c]}/>')
            if c != "-":
                f.w(f'<text class="s" x="{sx + 14}" y="{sy + 15}" text-anchor="middle">P{q}</text>')
    f.w(f'<rect x="24" y="218" width="594" height="28" rx="6" {tint("--fig-2", 22)}/>')
    f.w('<text class="s" x="321" y="236" text-anchor="middle">P0、P1、P2 各有 4 个 warp：warp 0 到 3 依次分到 SMSP 0 到 3</text>')
    lx = 96
    for c, lab in (("I", "正在发射"), ("R", "就绪，没被选中"), ("W", "等数据"), ("-", "空名额")):
        f.w(f'<rect x="{lx}" y="272" width="17" height="10" rx="2" {ST[c]}/><text class="lab2" x="{lx + 24}" y="281">{lab}</text>')
        lx += 24 + 13 * len(lab) + 28
    return f

# ── 图 B-1 / B-2 TPU（规格来自 How to Scale Your Model，TPU v5p） ─────
def tpu_core():
    f = F("fig-h-3", 300, "TPU v5p 的一个 TensorCore：Scalar unit 发指令，VMEM 是编译器管理的片上存储，VREG 供给 VPU 和 4 个 128×128 的 MXU；芯片外是 95 GB HBM，带宽 2.8e12 B/s")
    f.w(f'<rect x="10" y="22" width="452" height="250" rx="12" {tint("--fig-1", 6, sw=1.4)}/>')
    f.w('<text class="lab2" x="22" y="40">TensorCore（TPU v5p 每个芯片 2 个）</text>')
    f.w(f'<rect x="24" y="50" width="424" height="30" rx="6" {tint("--fig-1", 14)}/>')
    f.w('<text class="t" x="236" y="70" text-anchor="middle">Scalar unit：取指令，发给 VPU 和 MXU（只有这一路控制）</text>')
    f.w(f'<rect x="24" y="92" width="424" height="36" rx="6" {tint("--fig-2", 22)}/>')
    f.w('<text class="t" x="236" y="115" text-anchor="middle">VMEM：片上 scratchpad，由编译器显式搬入搬出</text>')
    f.w(f'<rect x="24" y="156" width="424" height="30" rx="6" {tint("--fig-hi", 16)}/>')
    f.w('<text class="t" x="236" y="176" text-anchor="middle">VREG 256 KB（vector register）</text>')
    f.arrow(200, 129, 200, 154, None, width=1.4); f.arrow(272, 155, 272, 130, None, width=1.4)
    f.w(f'<rect x="24" y="214" width="120" height="48" rx="6" {tint("--fig-2", 30)}/>')
    f.w('<text class="tb" x="84" y="235" text-anchor="middle">VPU</text><text class="s" x="84" y="252" text-anchor="middle">8 × 128 lanes</text>')
    f.arrow(84, 187, 84, 212, None, width=1.4)
    for k in range(4):
        x = 160 + k * 72
        f.w(f'<rect x="{x}" y="214" width="64" height="48" rx="6" {tint("--fig-1", 22)}/>')
        f.w(f'<text class="tb" x="{x + 32}" y="235" text-anchor="middle">MXU</text><text class="s" x="{x + 32}" y="252" text-anchor="middle">128 × 128</text>')
        f.arrow(x + 32, 187, x + 32, 212, None, width=1.4)
    f.w(f'<rect x="504" y="70" width="124" height="150" rx="10" {tint("--fig-mute", 14, sw=1.4)}/>')
    f.w('<text class="tb" x="566" y="124" text-anchor="middle">HBM</text><text class="t" x="566" y="144" text-anchor="middle">95 GB</text><text class="s" x="566" y="168" text-anchor="middle">带宽</text><text class="s" x="566" y="182" text-anchor="middle">2.8e12 B/s</text>')
    f.arrow(450, 104, 500, 104, "--fig-mute", width=1.8); f.arrow(500, 118, 450, 118, "--fig-mute", width=1.8)
    f.w('<text class="lab2" x="10" y="292">没有 warp，也没有硬件管理的 L1 / L2：HBM → VMEM → VREG → VPU / MXU 每一步都由编译器排好</text>')
    return f

def systolic():
    f = F("fig-h-4", 334, "GPU 的 CUDA core 每次乘加都从 register file 读两个数、写回一个；TPU 的 MXU 是 systolic array，权重固定在格子里，x 每拍向右流一格，部分和每拍向下流一格，只有边上读写")
    f.w('<text class="ttl" x="20" y="22">GPU：CUDA core 从 register file 取数</text>')
    f.w(f'<rect x="20" y="52" width="220" height="34" rx="6" {tint("--fig-hi", 16)}/>')
    f.w('<text class="t" x="130" y="74" text-anchor="middle">register file</text>')
    for k in range(4):
        x = 20 + k * 59; cx = x + 22
        f.w(f'<rect x="{x}" y="150" width="44" height="36" rx="5" {tint("--fig-2", 22)}/>')
        f.w(f'<text class="s" x="{cx}" y="172" text-anchor="middle">FMA</text>')
        f.arrow(cx - 12, 87, cx - 12, 148, None, width=1.3); f.arrow(cx, 87, cx, 148, None, width=1.3)
        f.arrow(cx + 12, 149, cx + 12, 89, "--fig-hi", width=1.3)
    f.w('<text class="lab" x="20" y="212">每次乘加读 2 个数、写回 1 个</text>')
    f.w('<text class="lab2" x="20" y="234">一个数要被用很多次，得靠 kernel 自己切 block：</text>')
    f.w('<text class="lab2" x="20" y="250">先搬进 shared memory / register，再反复读</text>')
    f.w('<text class="ttl" x="300" y="22">TPU：MXU 是 systolic array（示意 4 × 4，实际 128 × 128）</text>')
    f.w('<text class="lab2" x="300" y="40">W 先装进格子固定不动；x 每拍右移一格，部分和每拍下移一格</text>')
    X0, Y0, P, ST = 372, 52, 36, 66
    for i in range(4):
        for j in range(4):
            x, y = X0 + j * ST, Y0 + i * ST
            f.w(f'<rect x="{x}" y="{y}" width="{P}" height="{P}" rx="5" {tint("--fig-1", 22)}/>')
            f.w(f'<text class="s" x="{x + P / 2}" y="{y + 22}" text-anchor="middle">w<tspan baseline-shift="sub" font-size="8">{i}{j}</tspan></text>')
            if j < 3: f.arrow(x + P + 2, y + P / 2, x + ST - 3, y + P / 2, "--fig-2", width=1.4)
            if i < 3: f.arrow(x + P / 2, y + P + 2, x + P / 2, y + ST - 3, "--fig-hi", width=1.4)
        y = Y0 + i * ST + P / 2
        f.arrow(328, y, X0 - 3, y, "--fig-2", width=1.4)
        f.w(f'<text class="t" x="320" y="{y + 4}" text-anchor="end">x<tspan baseline-shift="sub" font-size="9">{i}</tspan></text>')
    for j in range(4):
        x = X0 + j * ST + P / 2
        f.arrow(x, Y0 + 3 * ST + P + 2, x, Y0 + 3 * ST + P + 24, "--fig-hi", width=1.4)
        f.w(f'<text class="t" x="{x}" y="{Y0 + 3 * ST + P + 40}" text-anchor="middle">y<tspan baseline-shift="sub" font-size="9">{j}</tspan></text>')
    return f

def cuda_model():
    f = F("fig-h-5", 352, "CUDA 的执行层级：grid 由 block 组成，block 由 warp 组成，warp 由 32 个 thread 组成；分别跑在整张 GPU、一个 SM、一个 SMSP 和一条 lane 上；在 Triton 里，grid 自己定，block 叫 program，warp 只能定个数，thread 写不到", W=820)
    f.w('<text class="ttl" x="20" y="18">软件：一次 kernel 启动</text><text class="ttl" x="432" y="18">硬件：跑在哪里（RTX 5090）</text><text class="ttl" x="648" y="18">Triton 里</text>')
    tri = [("grid", "kernel[grid](...) 自己定", "program 之间不能同步"), ("program", "tl.program_id 是它的编号", "shared memory 由编译器分配"),
           ("只给个数", "num_warps，默认 4", ""), ("写不到", "怎么分给 thread 由编译器定", "")]
    rows = [(30, 66, "--fig-mute", 12), (126, 66, "--fig-1", 12), (222, 50, "--fig-2", 16), (302, 42, "--fig-hi", 14)]
    hw = [("整张 GPU", "block 由硬件分到 170 个 SM"), ("一个 SM", "block 整个驻留，不跨 SM"),
          ("一个 SMSP", "调度器每拍发一条指令"), ("一条 lane", "CUDA core，自己的 register")]
    for (y, h, var, pct), (t1, t2) in zip(rows, hw):
        f.w(f'<rect x="20" y="{y}" width="350" height="{h}" rx="8" {tint(var, pct)}/>')
        f.w(f'<rect x="432" y="{y}" width="196" height="{h}" rx="8" {tint(var, pct)}/>')
        f.w(f'<text class="tb" x="530" y="{y + h / 2 - 3}" text-anchor="middle">{t1}</text><text class="s" x="530" y="{y + h / 2 + 12}" text-anchor="middle">{t2}</text>')
        f.arrow(374, y + h / 2, 428, y + h / 2, None, width=1.5)
    for (y, h, var, pct), (t1, t2, t3) in zip(rows, tri):
        f.w(f'<rect x="648" y="{y}" width="166" height="{h}" rx="8" style="fill: var(--fig-bg); stroke: var(--fig-hi)" stroke-opacity=".7" stroke-width="1.2"/>')
        lines = [x for x in (t2, t3) if x]
        y0 = y + h / 2 - 7 * len(lines) + 2
        f.w(f'<text class="tb" x="731" y="{y0}" text-anchor="middle">{t1}</text>')
        for i, l in enumerate(lines):
            f.w(f'<text class="s" x="731" y="{y0 + 15 + i * 14}" text-anchor="middle">{l}</text>')
    # grid
    f.w('<text class="lab" x="32" y="47">grid：所有 block，互相独立</text>')
    for k in range(7):
        x = 32 + k * 42
        f.w(f'<rect x="{x}" y="58" width="34" height="28" rx="4" {tint("--fig-1", 40 if k == 1 else 14)}/>')
        f.w(f'<text class="s" x="{x + 17}" y="76" text-anchor="middle">B{k}</text>')
    f.w('<text class="t" x="340" y="77">⋯</text>')
    zoom = 'stroke="currentColor" stroke-opacity=".3" stroke-dasharray="3 3" fill="none"'
    f.w(f'<path d="M74 86 L20 126 M108 86 L370 126" {zoom}/>')
    # block
    f.w('<text class="lab" x="32" y="143">block（Triton 的 program）</text>')
    f.w(f'<rect x="32" y="150" width="326" height="16" rx="3" {tint("--fig-3", 45, stroke="--fig-2")}/>')
    f.w('<text class="s" x="195" y="162" text-anchor="middle">shared memory：block 内共享，可以同步</text>')
    for k in range(4):
        x = 32 + k * 83
        f.w(f'<rect x="{x}" y="172" width="76" height="16" rx="3" {tint("--fig-2", 45 if k == 0 else 18)}/>')
        f.w(f'<text class="s" x="{x + 38}" y="184" text-anchor="middle">warp {k}</text>')
    f.w(f'<path d="M32 188 L20 222 M108 188 L370 222" {zoom}/>')
    # warp
    f.w('<text class="lab" x="32" y="239">warp：32 个 thread，同一时刻执行同一条指令</text>')
    for k in range(32):
        x = 32 + k * 10
        f.w(f'<rect x="{x}" y="248" width="8" height="16" rx="1.5" {tint("--fig-hi", 60 if k == 0 else 20)}/>')
    f.w(f'<path d="M32 264 L20 302 M40 264 L370 302" {zoom}/>')
    # thread
    f.w('<text class="lab" x="32" y="328">thread：自己的 register，按编号算出读哪个数</text>')
    return f

def occ_sweep():
    rows = json.load(open(os.path.join(HERE, "occupancy_sweep.json")))
    f = F("fig-h-7", 320, "每个 SMSP 驻留 1 到 12 个 warp 时，arithmetic intensity 从 0.25 到 128 的四个 kernel 的速度（占各自最快值的比例）：四条线形状几乎一样，都要 4 到 6 个 warp 才接近 100%")
    X0, X1, Y0, Y1 = 80, 610, 270, 40
    px = lambda m: X0 + (m - 1) * (X1 - X0) / 11
    py = lambda p: Y0 - p / 100 * (Y0 - Y1)
    for v in range(0, 101, 20):
        f.w(f'<line class="grid" x1="{X0}" y1="{py(v):.1f}" x2="{X1}" y2="{py(v):.1f}"/><text class="tick" x="{X0 - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{v}%</text>')
    for m in range(1, 13):
        f.w(f'<text class="tick" x="{px(m):.1f}" y="{Y0 + 16}" text-anchor="middle">{m}</text><text class="tick" x="{px(m):.1f}" y="{Y0 + 30}" text-anchor="middle" opacity=".7">{4 * m}</text>')
    f.w(f'<text class="lab2" x="{X0 - 8}" y="{Y0 + 16}" text-anchor="end">每 SMSP</text><text class="lab2" x="{X0 - 8}" y="{Y0 + 30}" text-anchor="end">每 SM</text>')
    f.w(f'<text class="lab2" x="{X0}" y="{Y1 - 16}">速度，占这条线最快值的比例；横轴是驻留的 warp 数</text>')
    ser = [(1, "--fig-3", "I = 0.25，显存瓶颈"), (32, "--fig-2", "I = 8，显存瓶颈"), (128, "--fig-1", "I = 32，显存瓶颈"), (512, "--fig-hi", "I = 128，算力瓶颈")]
    LX, LY = 360, 186
    for k, (K, var, lab) in enumerate(ser):
        r = [x for x in rows if x["K"] == K]
        best = min(x["us"] for x in r)
        pts = " ".join(f"{px(x['warps_per_smsp']):.1f},{py(best / x['us'] * 100):.1f}" for x in r)
        f.w(f'<polyline points="{pts}" fill="none" style="stroke: var({var})" stroke-width="2" stroke-linejoin="round"/>')
        for x in r:
            v = f'{x["flops"]:.2e} FLOPS' if K == 512 else f'{x["bw"]:.2e} B/s'
            f.w(f'<g class="m"><title>{lab}，每 SMSP {x["warps_per_smsp"]} 个 warp：{x["us"]} µs，{v}，最快值的 {best / x["us"] * 100:.0f}%</title><circle cx="{px(x["warps_per_smsp"]):.1f}" cy="{py(best / x["us"] * 100):.1f}" r="3.5" class="ring" style="fill: var({var})"/></g>')
        ly = LY + k * 20
        f.w(f'<line x1="{LX}" y1="{ly - 4}" x2="{LX + 22}" y2="{ly - 4}" style="stroke: var({var})" stroke-width="2"/><text class="lab" x="{LX + 30}" y="{ly}">{lab}</text>')
    return f

def tile_sweep():
    rows = json.load(open(os.path.join(HERE, "tile_sweep.json")))
    f = F("fig-h-8", 300, "row sum 的 TILE 从 128 到 16384 时的带宽：10880 行时都在 1.69e12 B/s 左右；170 行时随 TILE 起伏，128 最慢，4096 和 8192 也掉到 1.1e12 以下，512 和 16384 最快")
    X0, X1, Y0, Y1 = 110, 610, 236, 40
    Ts = sorted({r["TILE"] for r in rows})
    px = lambda T: X0 + (math.log2(T) - 7) * (X1 - X0) / 7
    py = lambda b: Y0 - b / 2e12 * (Y0 - Y1)
    for v in (0, 5e11, 1e12, 1.5e12, 2e12):
        f.w(f'<line class="grid" x1="{X0}" y1="{py(v):.1f}" x2="{X1}" y2="{py(v):.1f}"/><text class="tick" x="{X0 - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{v:.1e}</text>'.replace("0.0e+00", "0").replace("e+11", "e11").replace("e+12", "e12"))
    for T in Ts:
        f.w(f'<text class="tick" x="{px(T):.1f}" y="{Y0 + 16}" text-anchor="middle">{T}</text><text class="tick" x="{px(T):.1f}" y="{Y0 + 30}" text-anchor="middle" opacity=".7">{T // 128}</text>')
    f.w(f'<text class="lab2" x="{X0 - 28}" y="{Y0 + 16}" text-anchor="end">TILE</text><text class="lab2" x="{X0 - 28}" y="{Y0 + 30}" text-anchor="end">每线程</text>')
    f.w(f'<text class="lab2" x="{X0}" y="{Y1 - 16}">有效带宽（B/s），num_warps = 4；第二行是每个线程每轮读几个数</text>')
    f.w(f'<line class="ref" x1="{X0}" y1="{py(1.79e12):.1f}" x2="{X1}" y2="{py(1.79e12):.1f}"/><text class="lab2" x="{X1}" y="{py(1.79e12) - 6:.1f}" text-anchor="end">显存带宽 1.79e12</text>')
    f.w(f'<line class="ref" x1="{px(1024):.1f}" y1="{Y1}" x2="{px(1024):.1f}" y2="{Y0}"/><text class="lab2" x="{px(1024) + 6:.1f}" y="{Y0 - 8}">代码默认 1024</text>')
    ser = [(10880, "--fig-1", "10880 × 8192，每个 SM 分到 64 个 program"), (170, "--fig-hi", "170 × 524288，每个 SM 分到 1 个 program")]
    for k, (M, var, lab) in enumerate(ser):
        r = sorted((x for x in rows if x["M"] == M), key=lambda x: x["TILE"])
        f.w(f'<polyline points="{" ".join(f"{px(x["TILE"]):.1f},{py(x["bw"]):.1f}" for x in r)}" fill="none" style="stroke: var({var})" stroke-width="2" stroke-linejoin="round"/>')
        for x in r:
            f.w(f'<g class="m"><title>{lab}，TILE = {x["TILE"]}：{x["us"]} µs，{x["bw"]:.2e} B/s，每线程 {x["regs"]} 个 register</title><circle cx="{px(x["TILE"]):.1f}" cy="{py(x["bw"]):.1f}" r="3.5" class="ring" style="fill: var({var})"/></g>')
        ly = 170 + k * 20
        f.w(f'<line x1="{px(1024) + 24:.1f}" y1="{ly - 4}" x2="{px(1024) + 46:.1f}" y2="{ly - 4}" style="stroke: var({var})" stroke-width="2"/><text class="lab" x="{px(1024) + 54:.1f}" y="{ly}">{lab}</text>')
    return f

# ── 图 3-5 GELU 的 8192 个 program 怎么分批跑（实测 trace） ──────────────────────
def wave_trace():
    d = json.load(open(os.path.join(HERE, "wave_trace.json")))
    tr, cap = d["trace"], 170 * 12
    T = max(b for _, b, _ in tr) / 1000                              # µs
    W5 = [.14, .32, .5, .68, .9]                                     # 第 1 到 5 批：越晚越深
    nw = (len(tr) + cap - 1) // cap
    f = F("fig-t-5", 404, f"GELU 的 {len(tr)} 个 program 在 170 个 SM 上的实测时间线：上面是全卡同时在跑的 program 数，按批次堆叠，最多 {cap} 个；下面是一个 SM 的 12 个名额，前一个 program 结束就换下一个进来，批次之间没有清楚的分界，最后一批只有 {len(tr) - (nw - 1) * cap} 个")
    X0, X1 = 74, 620
    px = lambda t: X0 + t / T * (X1 - X0)
    # 上：全卡同时在跑的 program 数，按批次堆叠
    Y0, Y1 = 200, 52
    py = lambda c: Y0 - c / cap * (Y0 - Y1)
    for v in (0, 500, 1000, 1500, 2040):
        f.w(f'<line class="grid" x1="{X0}" y1="{py(v):.1f}" x2="{X1}" y2="{py(v):.1f}"/><text class="tick" x="{X0 - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{v}</text>')
    f.w(f'<text class="lab2" x="{X0}" y="{Y1 - 24}">某一时刻全卡已开始、未结束的 program 数，按批次堆叠；虚线是 170 × 12 = {cap} 个名额</text>')
    f.w(f'<text class="lab2" x="16" y="{(Y0 + Y1) / 2:.1f}" text-anchor="middle" transform="rotate(-90 16 {(Y0 + Y1) / 2:.1f})">同时在跑的 program 数</text>')
    f.w(f'<line class="ref" x1="{X0}" y1="{py(cap):.1f}" x2="{X1}" y2="{py(cap):.1f}"/>')
    N = 230
    ts = [T * k / N for k in range(N + 1)]
    cnt = [[0] * (N + 1) for _ in range(nw)]
    for pid, (a, b, _) in enumerate(tr):
        a, b, w = a / 1000, b / 1000, pid // cap
        for k, t in enumerate(ts):
            if a <= t < b: cnt[w][k] += 1
    base = [0] * (N + 1)
    for w in range(nw):
        top = [base[k] + cnt[w][k] for k in range(N + 1)]
        pts = [f"{px(t):.1f},{py(c):.1f}" for t, c in zip(ts, top)] + [f"{px(t):.1f},{py(c):.1f}" for t, c in zip(ts[::-1], base[::-1])]
        lo, hi = w * cap, min(len(tr), (w + 1) * cap) - 1
        f.w(f'<g class="m"><title>第 {w + 1} 批：program {lo} 到 {hi}，共 {hi - lo + 1} 个</title><polygon points="{" ".join(pts)}" style="fill: var(--fig-1)" fill-opacity="{W5[w]}"/></g>')
        base = top
    # bubble：第一个 program 结束到最后一个 program 开始之间，曲线和名额上限之间的空档；之后是 tail
    t1, t2 = min(b for _, b, _ in tr) / 1000, max(a for a, _, _ in tr) / 1000
    seg = [(t, c) for t, c in zip(ts, base) if t1 <= t <= t2]
    pts = [f"{px(t):.1f},{py(c):.1f}" for t, c in seg] + [f"{px(t2):.1f},{py(cap):.1f}", f"{px(t1):.1f},{py(cap):.1f}"]
    f.w(f'<polygon points="{" ".join(pts)}" style="fill: var(--fig-hi)" fill-opacity=".28"/>')
    tb, cb = max(seg, key=lambda x: cap - x[1])                       # 空档最大的地方放标注
    f.w(f'<text class="s halo" x="{px(tb):.1f}" y="{(py(cap) + py(cb)) / 2 + 4:.1f}" text-anchor="middle">bubble</text><text class="s" x="{px(tb):.1f}" y="{(py(cap) + py(cb)) / 2 + 4:.1f}" text-anchor="middle" style="fill: var(--fig-hi)">bubble</text>')
    f.path(f"M{px(t2):.1f},{Y1 - 4} L{px(t2):.1f},{Y1 - 9} L{px(T):.1f},{Y1 - 9} L{px(T):.1f},{Y1 - 4}", "--fig-hi", width=1.2, head=False)
    f.w(f'<text class="s" x="{(px(t2) + px(T)) / 2:.1f}" y="{Y1 - 12}" text-anchor="middle" style="fill: var(--fig-hi)">tail</text>')
    # 下：一个 SM 的 12 个名额（选跑了最后一批的 SM）
    last = (nw - 1) * cap
    sm = tr[last][2]
    mine = sorted((a / 1000, b / 1000, pid) for pid, (a, b, s) in enumerate(tr) if s == sm)
    slots = [[] for _ in range(12)]
    for a, b, pid in mine:
        free = [j for j in range(12) if not slots[j] or slots[j][-1][1] <= a]          # 能用的名额里挑最晚空出来的，空档按最短算
        k = max(free, key=lambda j: slots[j][-1][1] if slots[j] else -1) if free else min(range(12), key=lambda j: slots[j][-1][1])
        slots[k].append((a, b, pid))
    G0, RH = 246, 9
    f.w(f'<text class="lab2" x="{X0}" y="{G0 - 12}">SM {sm} 的 12 个名额，一条是一个 program 从第一条指令到最后一条 store 发出；这个 SM 一共跑了 {len(mine)} 个</text>')
    for j in range(12):
        y = G0 + j * RH
        if j % 3 == 0: f.w(f'<text class="tick" x="{X0 - 8}" y="{y + RH - 1}" text-anchor="end">{j}</text>')
        for a, b, pid in slots[j]:
            f.w(f'<g class="m"><title>program {pid}（第 {pid // cap + 1} 批）：{a:.2f} 到 {b:.2f} µs</title><rect x="{px(a):.1f}" y="{y + 1}" width="{max(px(b) - px(a) - .6, .8):.1f}" height="{RH - 2}" rx="1.5" style="fill: var(--fig-1)" fill-opacity="{W5[pid // cap]}"/></g>')
    gap = max(((slots[j][i][1], slots[j][i + 1][0], j) for j in range(12) for i in range(len(slots[j]) - 1)), key=lambda g: g[1] - g[0])
    ga, gb, gj = gap
    f.w(f'<rect x="{px(ga) - 2:.1f}" y="{G0 + gj * RH - 1}" width="{px(gb) - px(ga) + 4:.1f}" height="{RH + 2}" rx="3" style="fill: none; stroke: var(--fig-hi)" stroke-width="1.6"/>')
    f.w(f'<text class="s" x="{(px(ga) + px(gb)) / 2:.1f}" y="{G0 + gj * RH + RH - 1}" text-anchor="middle" style="fill: var(--fig-hi)">bubble</text>')
    GB = G0 + 12 * RH
    for t in range(0, int(T) + 1, 5):
        f.w(f'<text class="tick" x="{px(t):.1f}" y="{GB + 15}" text-anchor="middle">{t}</text>')
        f.w(f'<line class="grid" x1="{px(t):.1f}" y1="{Y1}" x2="{px(t):.1f}" y2="{Y0}"/>')
    f.w(f'<text class="lab2" x="{X0 - 8}" y="{GB + 15}" text-anchor="end">µs</text>')
    f.w(f'<text class="lab2" x="{X0 - 44}" y="{G0 + 6 * RH + 4}" text-anchor="middle" transform="rotate(-90 {X0 - 44} {G0 + 6 * RH})">名额</text>')
    LX, LY = X0, GB + 40
    f.w(f'<text class="lab2" x="{LX}" y="{LY}">批次</text>')
    for w in range(nw):
        lo, hi = w * cap, min(len(tr), (w + 1) * cap) - 1
        x = LX + 36 + w * 108
        f.w(f'<rect x="{x}" y="{LY - 9}" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="{W5[w]}"/><text class="lab" x="{x + 17}" y="{LY}">{lo}–{hi}</text>')
    return f

# ── 图 2-1 到 2-4 Triton kernel 示意：左边是 grid 和一个 SM，右边是一个 program 的步骤、片上和显存 ─────
def lanes_fig(fid, label, kind, prog_title, grid, grid_note, ops, chip_lab, notes, mem_y1=268, smem=None, smem_x=204):
    """左边 grid；右边四条泳道：计算单元 / register / shared memory / 显存。
    grid(f) 画左边的 grid 并返回高亮 program 的 y；ops 是步骤，"⋯" 表示省略的几轮。
    返回 (f, cx, put)，put 用来往片上 / 显存泳道放块、画读写箭头。"""
    gy = 28 + 4 * 32 + 12 + len(grid_note) * 15 + 6
    f = F(fid, max(mem_y1 + 18 + 15 * len(notes), gy), label)
    f.w('<text class="ttl" x="10" y="16">grid</text>')
    f.w(f'<text class="ttl" x="162" y="16">{prog_title}</text>')
    f.w(f'<text class="tb" x="630" y="16" text-anchor="end" style="fill: var(--fig-hi)">{kind}</text>')
    hy = grid(f)
    hx = 130
    if isinstance(hy, tuple):
        hy, hx = hy
    for k, line in enumerate(grid_note):
        f.w(f'<text class="lab2" x="10" y="{28 + 4 * 32 + 12 + k * 15}">{line}</text>')
    # 四层：步骤 / register / shared memory / 显存；matmul 的 register 层窄、shared memory 层宽
    MID = [(100, 152, "register"), (160, 236, "shared\nmemory")] if chip_lab == "matmul" else [(100, 176, "register"), (184, 216, "shared\nmemory")]
    MT = MID[-1][1] + 8
    LN = [(26, 92, "计算\n单元")] + MID + [(MT, mem_y1, "显存")]
    for y0, y1, t in LN:
        f.w(f'<rect class="band" x="146" y="{y0}" width="490" height="{y1 - y0}" rx="8"/>')
        for k, line in enumerate(t.split("\n")):
            f.w(f'<text class="lab2" x="152" y="{(y0 + y1) / 2 + 4 + (k - (t.count(chr(10))) / 2) * 13:.1f}">{line}</text>')
        if t == "显存":                                  # 芯片外的 GDDR7；中间的 L2、L1 cache 由硬件管，不画
            f.w(f'<text class="s" x="152" y="{(y0 + y1) / 2 + 18}">GDDR7</text>')
        if t.startswith("shared") and smem is not None:    # 这个 program 用了多少 shared memory（residency.json）
            B = json.load(open(os.path.join(HERE, "residency.json")))[smem]["shared"]
            txt = f"用 {B // 1024} KB" if B >= 1024 else f"用 {B} B"
            xy = (152, (y0 + y1) / 2 + 30) if chip_lab == "matmul" else (smem_x, (y0 + y1) / 2 + 4)   # matmul 泳道高，写在标题下面
            f.w(f'<text class="s" x="{xy[0]}" y="{xy[1]:.1f}" style="fill: var(--fig-1); opacity: 1; font-weight: 600">{txt}</text>')
    f.arrow(hx, hy, 144, hy, None, width=1.5)
    TR = lambda o: o != "⋯" and o[0] == "@"           # 搬运列：load / store 不经过计算单元，只画成箭头
    n, nd, nt = len(ops), sum(o == "⋯" for o in ops), sum(TR(o) for o in ops)
    x0, x1, G, DW, TW = 212, 628, 10, 22, 40
    BW = min(84, (x1 - x0 - DW * nd - TW * nt - G * (n - 1)) / (n - nd - nt))
    cx, x = [], x0
    for o in ops:
        w = DW if o == "⋯" else TW if TR(o) else BW
        cx.append(x + w / 2); x += w + G
    if x - G < x1 - 1:                               # 步骤少时整体居中
        sh = (x1 - (x - G)) / 2; cx = [c + sh for c in cx]
    OY0, OY1 = 46, 84
    for k, o in enumerate(ops):
        if o == "⋯":
            f.w(f'<text class="tb" x="{cx[k]:.1f}" y="{(OY0 + OY1) / 2 + 4}" text-anchor="middle">⋯</text>')
            continue
        if TR(o):
            continue
        t1, t2 = o
        f.w(f'<rect class="op" x="{cx[k] - BW / 2:.1f}" y="{OY0}" width="{BW:.1f}" height="{OY1 - OY0}" rx="6"/>')
        f.w(f'<text class="tb" x="{cx[k]:.1f}" y="{OY0 + 16}" text-anchor="middle">{t1}</text><text class="s" x="{cx[k]:.1f}" y="{OY0 + 31}" text-anchor="middle">{t2}</text>')
    for k, line in enumerate(notes):
        f.w(f'<text class="lab2" x="152" y="{mem_y1 + 16 + k * 15}">{line}</text>')

    class Put:
        ROW = {0: (110, 128), 1: (140, 158), "m": (MT + 28, MT + 56), "s": (MID[1][0] + 7, MID[1][0] + 25)}
        def chip(self, name, x, row=0, w=50, var="--fig-2", size=None):
            y0, y1 = self.ROW[row]
            if size:                                     # 第二行写这块数据多大
                y1 = y0 + 28
            f.w(f'<rect x="{x - w / 2:.1f}" y="{y0}" width="{w}" height="{y1 - y0}" rx="4" {tint(var, 26)}/>')
            f.w(f'<text class="s" x="{x:.1f}" y="{y0 + (12 if size else 13)}" text-anchor="middle">{name}</text>')
            if size:
                f.w(f'<text class="s" x="{x:.1f}" y="{y0 + 24}" text-anchor="middle" style="font-size: 9.5px; fill: var(--fig-1); opacity: 1">{size}</text>')
        def mem(self, name, xa, xb, hls=(), part=None, px=None):
            y0, y1 = self.ROW["m"]
            if part:                                     # 这个 program 读写多少，写在箭头左上
                f.w(f'<text class="s" x="{px:.1f}" y="{y0 - 5}" text-anchor="end" style="fill: var(--fig-1); opacity: 1">{part}</text>')
            f.w(f'<rect x="{xa:.1f}" y="{y0}" width="{xb - xa:.1f}" height="{y1 - y0}" rx="4" {tint("--fig-1", 10)}/>')
            for hx0, hx1, fy, fh in hls:                 # 这个 program 读写的部分
                f.w(f'<rect x="{hx0:.1f}" y="{y0 + fy * (y1 - y0):.1f}" width="{hx1 - hx0:.1f}" height="{fh * (y1 - y0):.1f}" style="fill: var(--fig-1)" fill-opacity=".55"/>')
            f.w(f'<text class="s" x="{(xa + xb) / 2:.1f}" y="{y1 + 13}" text-anchor="middle">{name}</text>')
        def down(self, x, y, xo=None, var="--fig-2"):     # 步骤写到 (xo, y)
            f.arrow(x, OY1 + 2, x if xo is None else xo, y - 2, var, width=1.4)
        def xfer(self, x, y_from, y_to, txt, left=False, ly=None, xo=None):   # load / store：显存和 register、shared memory 之间搬（中间经过的 L2 不画）
            f.arrow(x, y_from, x if xo is None else xo, y_to, "--fig-1", width=1.6)
            if txt:
                ly = (y_from + y_to) / 2 + 4 if ly is None else ly
                f.w(f'<text class="s" x="{x - 4 if left else x + 4:.1f}" y="{ly:.1f}" text-anchor="{"end" if left else "start"}" style="fill: var(--fig-1)">{txt}</text>')
        def up(self, x, y, xo=None, var="--fig-2"):       # 从 (x, y) 读到步骤
            f.arrow(x, y - 2, x if xo is None else xo, OY1 + 2, var, width=1.4)
        def carry(self, xa, xb, txt):                      # 步骤上方的红线：acc 留在 register 里，带过省略的几轮
            f.path(f"M{xa:.1f},{OY0 - 2} L{xa:.1f},{OY0 - 16} L{xb:.1f},{OY0 - 16} L{xb:.1f},{OY0 - 3}", "--fig-hi", width=1.5)   # 竖段留出箭杆
            f.w(f'<text class="s halo" x="{(xa + xb) / 2:.1f}" y="{OY0 - 12}" text-anchor="middle">{txt}</text>')
            f.w(f'<text class="s" x="{(xa + xb) / 2:.1f}" y="{OY0 - 12}" text-anchor="middle" style="fill: var(--fig-hi)">{txt}</text>')
    P = Put(); P.MT, P.SH = MT, MID[1]
    return f, cx, P

def membar(f, x0, x1, y0, h, var="--fig-2", pct=14):
    f.w(f'<rect x="{x0:.1f}" y="{y0}" width="{x1 - x0:.1f}" height="{h}" rx="5" {tint(var, pct, sw=1.2)}/>')

def over(f, x1, y1, x2, y2, var="--fig-2", width=1.4):          # 从别的存储块上面穿过的箭头
    f.w(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke: var(--nv-bg, #fff)" stroke-width="5"/>')
    f.arrow(x1, y1, x2, y2, var, width=width)

def grid_rows(f):
    for i, (t, hi) in enumerate([("行 0", False), ("行 1 ← pid=1", True), ("行 2", False), ("⋯", False)]):
        y = 28 + i * 32
        f.w(f'<rect x="10" y="{y}" width="118" height="26" rx="5" {tint("--fig-1" if hi else "--fig-mute", 30 if hi else 12)}/>')
        f.w(f'<text class="t" x="69" y="{y + 18}" text-anchor="middle">{t}</text>')
    return 73

def grid_strip(f):
    f.w('<text class="lab2" x="10" y="48">x 展平成一维</text>')
    y, sw = 60, 118 / 4
    for k, t in enumerate(["0", "1", "2", "⋯"]):
        x, hi = 10 + k * sw, k == 1
        f.w(f'<rect x="{x:.1f}" y="{y}" width="{sw - 2:.1f}" height="26" rx="3" {tint("--fig-1" if hi else "--fig-mute", 30 if hi else 12)}/>')
        f.w(f'<text class="s" x="{x + (sw - 2) / 2:.1f}" y="{y + 17}" text-anchor="middle">{t}</text>')
    f.w(f'<text class="s" x="{10 + 1.5 * sw - 1:.1f}" y="{y + 40}" text-anchor="middle">pid=1</text>')
    return 73

def grid_tiles42(f):
    for r in range(4):
        for c in range(2):
            x, y, hi = 10 + c * 40, 28 + r * 27, (r, c) == (1, 1)
            f.w(f'<rect x="{x}" y="{y}" width="36" height="23" rx="4" {tint("--fig-1" if hi else "--fig-mute", 30 if hi else 12)}/>')
            if hi:
                f.w(f'<text class="s" x="{x + 18}" y="{y + 15}" text-anchor="middle">(1,1)</text>')
    f.w('<text class="lab2" x="10" y="148">每个 C block 一个 program</text>')
    return 66, 90

def grid_tiles(f):
    for r in range(3):
        for c in range(3):
            x, y, hi = 10 + c * 40, 28 + r * 34, (r, c) == (1, 2)
            f.w(f'<rect x="{x}" y="{y}" width="36" height="30" rx="4" {tint("--fig-1" if hi else "--fig-mute", 30 if hi else 12)}/>')
            if hi:
                f.w(f'<text class="s" x="{x + 18}" y="{y + 19}" text-anchor="middle">(1,2)</text>')
    f.w('<text class="lab2" x="10" y="146">每个 C block 一个 program</text>')
    return 77

# ── 表 2-x 改成的图：四个例子的数据怎么分给 program（简化版的 grid 示意） ─────────────
def op_split():
    f = F("fig-o-1", 452, "四个例子的数据怎么分给 program：GELU 把 x 展平按 BLOCK 切段，一段一个 program，不循环；softmax 一行一个 program，整行一次读进来；row sum 一行一个 program，沿行一个 block 一个 block 读，累加器存部分和；matmul 每个 C block 一个 program，沿 K 循环，读 A 的一个 row block 和 B 的一个 column block")
    C = 14                                                     # 一格的边长
    def cell(x, y, var, pct, w=C, h=C):
        f.w(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w - 1.5:.1f}" height="{h - 1.5:.1f}" rx="1.5" {tint(var, pct, sw=1)}/>')
    def mat(x, y, rows, cols, hi=lambda r, c: None, w=C, h=C):
        for r in range(rows):
            for c in range(cols):
                v = hi(r, c)
                cell(x + c * w, y + r * h, *(v if v else ("--fig-mute", 10)), w=w, h=h)
    f.w('<defs><pattern id="fig-o-1-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-1)" stroke-opacity=".55" stroke-width="1.3"/></pattern></defs>')
    def hatch(x, y, w=C, h=C):                                 # 之后几轮才读的 block
        f.w(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w - 1.5:.1f}" height="{h - 1.5:.1f}" rx="1.5" style="fill: url(#fig-o-1-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1"/>')
    def panel(x0, y0, title, kind, notes):
        f.w(f'<rect x="{x0}" y="{y0}" width="306" height="206" rx="10" class="band"/>')
        f.w(f'<text class="tb" x="{x0 + 12}" y="{y0 + 22}">{title}</text><text class="s" x="{x0 + 294}" y="{y0 + 22}" text-anchor="end" style="fill: var(--fig-hi)">{kind}</text>')
        for i, n in enumerate(notes):
            f.w(f'<text class="s" x="{x0 + 12}" y="{y0 + 174 + i * 15}">{n}</text>')
    def arrow_r(x1, x2, y):
        f.arrow(x1, y, x2, y, None, width=1.4)
    # ── GELU
    x0, y0 = 10, 8
    panel(x0, y0, "GELU", "逐元素", ["x 展平，按 BLOCK 切段，一段一个 program", "不循环：一段一次读进 register"])
    f.w(f'<text class="lab2" x="{x0 + 12}" y="{y0 + 58}">x</text><text class="lab2" x="{x0 + 12}" y="{y0 + 128}">y</text>')
    for k in range(16):
        seg = k // 4
        hi = ("--fig-1", 32) if seg == 1 else None
        cell(x0 + 30 + k * C, y0 + 46, *(hi or ("--fig-mute", 10)))
        cell(x0 + 30 + k * C, y0 + 116, *(("--fig-hi", 30) if seg == 1 else ("--fig-mute", 10)))
    for s in range(4):
        f.w(f'<text class="s" x="{x0 + 30 + (s * 4 + 2) * C:.1f}" y="{y0 + 76}" text-anchor="middle">{"pid=1" if s == 1 else f"段 {s}"}</text>')
    f.arrow(x0 + 30 + 6 * C, y0 + 82, x0 + 30 + 6 * C, y0 + 112, None, width=1.4)
    # ── softmax
    x0, y0 = 324, 8
    panel(x0, y0, "softmax", "按行归约", ["一行一个 program", "不循环：整行（max 和 sum 都要用）一次读进来"])
    mat(x0 + 14, y0 + 40, 5, 7, lambda r, c: ("--fig-1", 32) if r == 1 else None)
    f.w(f'<text class="lab2" x="{x0 + 14}" y="{y0 + 126}">x</text>')
    arrow_r(x0 + 14 + 7 * C + 6, x0 + 14 + 7 * C + 38, y0 + 40 + 1.5 * C)
    mat(x0 + 14 + 7 * C + 44, y0 + 40, 5, 7, lambda r, c: ("--fig-hi", 30) if r == 1 else None)
    f.w(f'<text class="lab2" x="{x0 + 14 + 7 * C + 44}" y="{y0 + 126}">y</text>')
    f.w(f'<text class="s" x="{x0 + 14 + 7 * C + 44 + 7 * C + 4}" y="{y0 + 40 + 1.5 * C + 4}">pid=1</text>')
    # ── row sum
    x0, y0 = 10, 222
    panel(x0, y0, "row sum", "按行归约", ["一行一个 program", "沿行循环：一轮读一个 block，累加器存部分和"])
    mat(x0 + 14, y0 + 40, 5, 12, lambda r, c: ("--fig-1", 45) if (r, c // 3) == (1, 0) else None)
    for c in range(3, 12):
        hatch(x0 + 14 + c * C, y0 + 40 + C)
    f.w(f'<text class="lab2" x="{x0 + 14}" y="{y0 + 126}">x</text>')
    arrow_r(x0 + 14 + 12 * C + 6, x0 + 14 + 12 * C + 40, y0 + 40 + 1.5 * C)
    mat(x0 + 14 + 12 * C + 46, y0 + 40, 5, 1, lambda r, c: ("--fig-hi", 30) if r == 1 else None)
    f.w(f'<text class="lab2" x="{x0 + 14 + 12 * C + 46}" y="{y0 + 126}">s</text>')
    f.w(f'<text class="s" x="{x0 + 14 + 12 * C + 46 + C + 6}" y="{y0 + 40 + 1.5 * C + 4}">pid=1</text>')
    # ── matmul：C = A @ B，一格是一个 block
    x0, y0 = 324, 222
    panel(x0, y0, "matmul", "block 乘加", ["C 的一个 block 一个 program，grid 是二维", "沿 K 循环：一轮读 A、B 各一个 block，累加器存 C 的 block"])
    b, n4 = 16, 4
    gx = [x0 + 18, x0 + 18 + n4 * b + 30, x0 + 18 + 2 * (n4 * b + 30)]
    gy = y0 + 66
    for x, name in zip(gx, ("C", "A", "B")):
        f.w(f'<text class="tb" x="{x + n4 * b / 2:.1f}" y="{gy - 10}" text-anchor="middle">{name}</text>')
    for k in range(2):
        f.w(f'<text class="t" x="{gx[k] + n4 * b + 15:.1f}" y="{gy + n4 * b / 2 + 5:.1f}" text-anchor="middle">{"=" if k == 0 else "@"}</text>')
    for r in range(n4):
        for c in range(n4):
            cell(gx[0] + c * b, gy + r * b, *(("--fig-hi", 40) if (r, c) == (1, 1) else ("--fig-mute", 8)), w=b, h=b)
            if r == 1:
                (cell(gx[1] + c * b, gy + r * b, "--fig-1", 45, w=b, h=b) if c == 0 else hatch(gx[1] + c * b, gy + r * b, w=b, h=b))
            else:
                cell(gx[1] + c * b, gy + r * b, "--fig-mute", 8, w=b, h=b)
            if c == 1:
                (cell(gx[2] + c * b, gy + r * b, "--fig-1", 45, w=b, h=b) if r == 0 else hatch(gx[2] + c * b, gy + r * b, w=b, h=b))
            else:
                cell(gx[2] + c * b, gy + r * b, "--fig-mute", 8, w=b, h=b)
    f.w(f'<text class="s" x="{gx[0] + n4 * b / 2:.1f}" y="{gy + n4 * b + 16}" text-anchor="middle">pid = (1, 1)</text>')
    f.w(f'<text class="s" x="{(gx[1] + gx[2] + n4 * b) / 2:.1f}" y="{gy + n4 * b + 16}" text-anchor="middle">实心是第 0 轮读的 block，斜线是之后几轮</text>')
    return f

# ── 2.4 的 tiling 示意：C = A @ B，一格是一个 block（画法参考用户给的 Medium 示意图） ─────
def mm_tiling():
    f = F("fig-o-2", 236, "matmul 切 block：C 的 block (1, 1) 由 A 的第 1 个 row block 和 B 的第 1 个 column block 算出；沿 K 一轮读一对 block，实心是第 0 轮，斜线是第 1 到 3 轮")
    f.w('<defs><pattern id="fig-o-2-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<line x1="0" y1="0" x2="0" y2="6" style="stroke: var(--fig-1)" stroke-opacity=".55" stroke-width="1.6"/></pattern></defs>')
    b, n = 32, 4
    gx = [40, 40 + n * b + 54, 40 + 2 * (n * b + 54)]
    gy = 48
    def box(x, y, style):
        f.w(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{b - 2}" height="{b - 2}" rx="3" {style}/>')
    solid = lambda var, pct: tint(var, pct, sw=1.4)
    plain = tint("--fig-mute", 8, sw=1)
    hatch = 'style="fill: url(#fig-o-2-hatch); stroke: var(--fig-1)" stroke-opacity=".8" stroke-width="1.4"'
    names = [("C", "M × N"), ("A", "M × K"), ("B", "K × N")]
    for x, (nm, shape) in zip(gx, names):
        f.w(f'<text class="tb" x="{x + n * b / 2}" y="{gy - 22}" text-anchor="middle">{nm}</text><text class="s" x="{x + n * b / 2}" y="{gy - 8}" text-anchor="middle">{shape}</text>')
    for k, op in enumerate(("=", "@")):
        f.w(f'<text class="tb" x="{gx[k] + n * b + 27}" y="{gy + n * b / 2 + 5}" text-anchor="middle" style="font-size: 18px">{op}</text>')
    for r in range(n):
        for c in range(n):
            box(gx[0] + c * b, gy + r * b, solid("--fig-hi", 40) if (r, c) == (1, 1) else plain)
            box(gx[1] + c * b, gy + r * b, (solid("--fig-1", 45) if c == 0 else hatch) if r == 1 else plain)
            box(gx[2] + c * b, gy + r * b, (solid("--fig-1", 45) if r == 0 else hatch) if c == 1 else plain)
    f.w(f'<text class="s" x="{gx[0] + 1.5 * b}" y="{gy + 1.5 * b + 4}" text-anchor="middle">(1, 1)</text>')
    for c in range(n):
        f.w(f'<text class="s" x="{gx[1] + c * b + b / 2}" y="{gy + n * b + 16}" text-anchor="middle">T={c}</text>')
    for r in range(n):
        f.w(f'<text class="s" x="{gx[2] + n * b + 6}" y="{gy + r * b + b / 2 + 4}">T={r}</text>')
    f.w(f'<text class="s" x="{gx[0] + n * b / 2}" y="{gy + n * b + 16}" text-anchor="middle">一个 program 负责的 C block</text>')
    ly = gy + n * b + 44
    f.w(f'<rect x="40" y="{ly - 10}" width="14" height="12" rx="2" {solid("--fig-1", 45)}/><text class="s" x="60" y="{ly}">第 0 轮读的一对 block</text>')
    f.w(f'<rect x="200" y="{ly - 10}" width="14" height="12" rx="2" {hatch}/><text class="s" x="220" y="{ly}">之后几轮（沿 K）</text>')
    f.w(f'<rect x="340" y="{ly - 10}" width="14" height="12" rx="2" {solid("--fig-hi", 40)}/><text class="s" x="360" y="{ly}">累加器：C block 一直留在 register 里</text>')
    return f

# ── 单卡优化方法一文的示意图（画法参考 CS336 lecture 5） ─────────────────────────
def opt_divergence():
    f = F("fig-o-3", 446, "control divergence：if (lane < 4) { A; B } else { X; Y }; Z。Pascal 及以前分支两边串行，先 A、B 再 X、Y，执行一边时另一半 lane 被屏蔽，最后一起执行 Z，共 5 段；Volta 起两边可以交替执行，A、X、B、Y 轮流，Z 也可能分两次执行，共 6 段，同一时刻仍只有一半 lane 在执行；分支按 warp 一致时，每个 warp 只走自己那条路")
    X0, U = 96, 44
    on_st = tint("--fig-1", 40, sw=1)
    off_st = 'style="fill: none; stroke: currentColor" stroke-opacity=".3" stroke-dasharray="3 2"'
    def panel(y0, title, sched, note):
        f.w(f'<text class="tb" x="8" y="{y0}">{title}</text>')
        for i, (lab, who) in enumerate(sched):
            f.w(f'<text class="s" x="{X0 + (i + 0.5) * U}" y="{y0 + 18}" text-anchor="middle">{lab}</text>')
        for k in range(8):
            y = y0 + 24 + k * 15
            f.w(f'<text class="s" x="{X0 - 8}" y="{y + 10}" text-anchor="end">lane {k}</text>')
            for i, (lab, who) in enumerate(sched):
                act = who == "all" or (who == "lo") == (k < 4)
                f.w(f'<rect x="{X0 + i * U + 1}" y="{y}" width="{U - 2}" height="12" rx="2" {on_st if act else off_st}/>')
        f.w(f'<text class="s" x="{X0 + len(sched) * U + 10}" y="{y0 + 24 + 4 * 15 + 3}">{note}</text>')
    panel(22, "Pascal 及以前：分支两边串行，结束后在 Z 合起来",
          [("A", "lo"), ("B", "lo"), ("X", "hi"), ("Y", "hi"), ("Z", "all")], "5 段，前 4 段都有一半 lane 空着")
    panel(196, "Volta 起：两边可以交替执行，Z 不一定合起来",
          [("A", "lo"), ("X", "hi"), ("B", "lo"), ("Y", "hi"), ("Z", "lo"), ("Z", "hi")], "6 段，同一时刻仍只有一半 lane 在执行")
    y2 = 196 + 24 + 8 * 15 + 34
    f.w(f'<text class="tb" x="8" y="{y2}">分支按 warp 一致：warp 0 都走 if，warp 1 都走 else</text>')
    for w, segs in ((0, ("A", "B", "Z")), (1, ("X", "Y", "Z"))):
        y = y2 + 12 + w * 20
        f.w(f'<text class="s" x="{X0 - 8}" y="{y + 10}" text-anchor="end">warp {w}</text>')
        for i, lab in enumerate(segs):
            f.w(f'<rect x="{X0 + i * U + 1}" y="{y}" width="{U - 2}" height="14" rx="2" {on_st}/><text class="s" x="{X0 + (i + 0.5) * U}" y="{y + 11}" text-anchor="middle">{lab}</text>')
    f.w(f'<text class="s" x="{X0 + 3 * U + 10}" y="{y2 + 12 + 20 + 11}">每个 warp 3 段，没有空着的 lane</text>')
    ya = y2 + 12 + 40 + 14
    f.arrow(X0, ya, X0 + 6 * U, ya, None, width=1.2)
    f.w(f'<text class="s" x="{X0 + 6 * U + 6}" y="{ya + 4}">时间</text>')
    return f

def opt_precision():
    f = F("fig-o-4", 292, "几种浮点格式的位：符号位、指数位、尾数位各几位，能表示的最大值，以及一条 128 B 的 cache line 装几个数。fp32 是 1、8、23；bf16 是 1、8、7；fp16 是 1、5、10；fp8 E4M3 是 1、4、3；fp8 E5M2 是 1、5、2；fp4 E2M1 是 1、2、1")
    rows = [("fp32", 1, 8, 23, "3.4e38", 32), ("bf16", 1, 8, 7, "3.4e38", 64), ("fp16", 1, 5, 10, "65504", 64),
            ("fp8 E4M3", 1, 4, 3, "448", 128), ("fp8 E5M2", 1, 5, 2, "57344", 128), ("fp4 E2M1", 1, 2, 1, "6", 256)]
    X0, cw = 92, 11.5
    f.w(f'<text class="s" x="{X0 + 33 * cw + 14}" y="22">最大值</text><text class="s" x="{X0 + 33 * cw + 72}" y="22">128 B 装几个</text>')
    for i, (nm, s, e, m, mx, n) in enumerate(rows):
        y = 32 + i * 34
        f.w(f'<text class="lab" x="8" y="{y + 15}">{nm}</text>')
        x = X0
        for var, cnt, pct in (("--fig-hi", s, 45), ("--fig-1", e, 45), ("--fig-3", m, 55)):
            for k in range(cnt):
                f.w(f'<rect x="{x:.1f}" y="{y}" width="{cw - 1.5:.1f}" height="20" rx="1.5" {tint(var, pct, sw=0.8)}/>')
                x += cw
        f.w(f'<text class="s" x="{X0 + 33 * cw + 14}" y="{y + 15}">{mx}</text><text class="s" x="{X0 + 33 * cw + 72}" y="{y + 15}">{n} 个</text>')
    ly = 32 + 6 * 34 + 20
    x = 8
    for var, pct, lab in (("--fig-hi", 45, "符号位"), ("--fig-1", 45, "指数位：决定范围"), ("--fig-3", 55, "尾数位：决定精度")):
        f.w(f'<rect x="{x}" y="{ly - 10}" width="14" height="12" rx="2" {tint(var, pct, sw=0.8)}/><text class="s" x="{x + 20}" y="{ly}">{lab}</text>')
        x += 40 + sum(10.5 if ord(c) > 0x2E80 else 5.8 for c in lab)
    return f

def opt_mx():
    f = F("fig-o-9", 290, "block scaling 和转置：W 是 [N, K]，前向 Y = X Wᵀ 沿 K 归约，每行 4 个数一个 block、一个 scale；反向 dX = dY W 沿 N 归约，block 要竖着分，原来的 scale 对不上，只能从高精度的 W 再量化一份")
    n, c = 8, 20
    def grid(x0, y0, along_rows, title, note):
        f.w(f'<text class="tb" x="{x0}" y="22">{title}</text>')
        for r in range(n):
            for k in range(n):
                blk = (k // 4) if along_rows else (r // 4)
                key = (r, blk) if along_rows else (k, blk)
                pct = 16 + 22 * ((key[0] + key[1]) % 2)
                f.w(f'<rect x="{x0 + k * c + 0.5}" y="{y0 + r * c + 0.5}" width="{c - 1}" height="{c - 1}" rx="2" {tint("--fig-1", pct, sw=0.8)}/>')
        if along_rows:
            for r in range(n):
                for b in range(2):
                    f.w(f'<rect x="{x0 + n * c + 8 + b * 14}" y="{y0 + r * c + 3}" width="12" height="{c - 6}" rx="2" {tint("--fig-hi", 45, sw=0.8)}/>')
            f.w(f'<text class="s" x="{x0 + n * c + 8}" y="{y0 - 6}">scale</text>')
            f.w(f'<text class="s" x="{x0}" y="{y0 - 6}">K →</text>')
        else:
            for k in range(n):
                for b in range(2):
                    f.w(f'<rect x="{x0 + k * c + 3}" y="{y0 + n * c + 8 + b * 14}" width="{c - 6}" height="12" rx="2" {tint("--fig-hi", 45, sw=0.8)}/>')
            f.w(f'<text class="s" x="{x0 + n * c + 6}" y="{y0 + n * c + 22}">scale</text>')
            f.w(f'<text class="s" x="{x0 - 22}" y="{y0 + 12}">N</text><text class="s" x="{x0 - 22}" y="{y0 + 26}">↓</text>')
        for i, l in enumerate(note):
            f.w(f'<text class="s" x="{x0}" y="{y0 + n * c + (56 if not along_rows else 22) + i * 15}">{l}</text>')
    grid(40, 40, True, "前向 Y = X Wᵀ：沿 K 分块", ["W 是 [N, K]，每行 2 个 block，各一个 scale"])
    grid(380, 40, False, "反向 dX = dY W：沿 N 分块", ["同一个 W，每列 2 个 block，scale 和左边对不上", "只能从高精度的 W 再量化一份"])
    f.arrow(300, 120, 360, 120, None, width=1.4)
    f.w('<text class="s" x="330" y="112" text-anchor="middle">换个方向分块</text>')
    return f

def opt_fusion():
    f = F("fig-o-5", 262, "fusion：eager 每个 op 是一个 kernel，中间结果都写回显存、下一个再读回来；fused 一个 kernel 读一次 x、写一次 y，中间结果留在 register 里")
    def col(x, title):
        f.w(f'<text class="tb" x="{x}" y="22">{title}</text>')
    # eager
    col(8, "eager：每个 op 一个 kernel")
    mx, cx = 20, 190
    f.w(f'<rect x="{mx}" y="34" width="70" height="214" rx="6" {tint("--fig-mute", 14)}/><text class="s" x="{mx + 35}" y="50" text-anchor="middle">显存</text>')
    mem = [("x", 72), ("t1", 124), ("t2", 176), ("y", 228)]
    for nm, y in mem:
        f.w(f'<rect x="{mx + 14}" y="{y - 12}" width="42" height="18" rx="4" {tint("--fig-2", 22, sw=1)}/><text class="s" x="{mx + 35}" y="{y + 1}" text-anchor="middle">{nm}</text>')
    for i, op in enumerate(("op 1", "op 2", "op 3")):
        y = 85 + i * 52
        f.w(f'<rect x="{cx}" y="{y}" width="74" height="26" rx="6" class="op"/><text class="t" x="{cx + 37}" y="{y + 18}" text-anchor="middle">{op}</text>')
        f.arrow(mx + 58, mem[i][1] - 3, cx - 4, y + 8, "--fig-hi", width=1.4)        # 读上一个
        f.arrow(cx - 4, y + 20, mx + 58, mem[i + 1][1] - 3, "--fig-hi", width=1.4)    # 写下一个
    f.w(f'<text class="s" x="{cx + 37}" y="{246}" text-anchor="middle">读 3 次、写 3 次</text>')
    # fused
    col(340, "fused：一个 kernel")
    mx2, cx2 = 352, 520
    f.w(f'<rect x="{mx2}" y="34" width="70" height="214" rx="6" {tint("--fig-mute", 14)}/><text class="s" x="{mx2 + 35}" y="50" text-anchor="middle">显存</text>')
    for nm, y in (("x", 72), ("y", 228)):
        f.w(f'<rect x="{mx2 + 14}" y="{y - 12}" width="42" height="18" rx="4" {tint("--fig-2", 22, sw=1)}/><text class="s" x="{mx2 + 35}" y="{y + 1}" text-anchor="middle">{nm}</text>')
    f.w(f'<rect x="{cx2 - 10}" y="60" width="94" height="152" rx="8" {tint("--fig-hi", 10)}/><text class="s" x="{cx2 + 37}" y="74" text-anchor="middle">register</text>')
    for i, op in enumerate(("op 1", "op 2", "op 3")):
        y = 84 + i * 40
        f.w(f'<rect x="{cx2}" y="{y}" width="74" height="26" rx="6" class="op"/><text class="t" x="{cx2 + 37}" y="{y + 18}" text-anchor="middle">{op}</text>')
        if i < 2:
            f.arrow(cx2 + 37, y + 27, cx2 + 37, y + 39, None, width=1.2)
    f.arrow(mx2 + 58, 69, cx2 - 12, 92, "--fig-hi", width=1.4)
    f.arrow(cx2 - 12, 200, mx2 + 58, 225, "--fig-hi", width=1.4)
    f.w(f'<text class="s" x="{cx2 + 37}" y="{246}" text-anchor="middle">读 1 次、写 1 次</text>')
    return f

def opt_recompute():
    f = F("fig-o-6", 250, "三层 sigmoid 的前向和反向。左：前向把 s1、s2 写回显存给反向用，前向 1 读 3 写，反向 3 读 1 写，一共 8 次；右：前向只写 out，反向读 x 重算 s1、s2，前向 1 读 1 写，反向 2 读 1 写，一共 5 次")
    def stack(x0, title, saved):
        f.w(f'<text class="tb" x="{x0}" y="22">{title}</text>')
        f.w(f'<text class="s" x="{x0 + 50}" y="44" text-anchor="middle">x</text>')
        for i in range(3):
            y = 54 + i * 42
            f.w(f'<rect x="{x0 + 14}" y="{y}" width="72" height="24" rx="5" class="op"/><text class="t" x="{x0 + 50}" y="{y + 16}" text-anchor="middle">sigmoid</text>')
            f.arrow(x0 + 50, y - 6 if i == 0 else y - 17, x0 + 50, y - 1, None, width=1.2)
            if saved and i < 2:
                f.arrow(x0 + 88, y + 18, x0 + 128, y + 26, "--fig-hi", width=1.3)
                f.w(f'<text class="s" x="{x0 + 132}" y="{y + 30}">s{i + 1} 写回显存</text>')
        f.arrow(x0 + 50, 54 + 2 * 42 + 25, x0 + 50, 54 + 2 * 42 + 40, None, width=1.2)
        f.w(f'<text class="s" x="{x0 + 50}" y="{54 + 2 * 42 + 52}" text-anchor="middle">out</text>')
    stack(8, "存下中间结果", True)
    stack(330, "反向时重算", False)
    rows = [("前向", "读 x，写 s1、s2、out：1 读 3 写", "读 x，写 out：1 读 1 写"),
            ("反向", "读 s1、s2、dout，写 dx：3 读 1 写", "读 x、dout 重算 s1、s2，写 dx：2 读 1 写"),
            ("合计", "8 次显存读写", "5 次，少了 3/8")]
    for i, (a, b, c) in enumerate(rows):
        y = 212 + i * 16
        f.w(f'<text class="s" x="8" y="{y}">{a}</text><text class="s" x="44" y="{y}">{b}</text><text class="s" x="330" y="{y}">{c}</text>')
    return f

def opt_coalescing():
    f = F("fig-o-7", 250, "memory coalescing：8 个 thread 各读一个 4 B 的数。上：相邻 thread 读相邻地址，32 B 正好是一个 sector，取 1 个 sector；下：相邻 thread 隔 32 B，每个 thread 落在不同的 sector，要取 8 个 sector，每个只用到 4 B")
    X0, cw = 60, 9
    def row(y, title, addr, note):
        f.w(f'<text class="tb" x="8" y="{y - 40}">{title}</text>')
        for s in range(8):                                   # 8 个 32 B sector，每个 8 个 4 B 的格子
            used = any(a // 8 == s for a in addr)
            for k in range(8):
                a = s * 8 + k
                st = tint("--fig-hi", 45, sw=0.8) if a in addr else (tint("--fig-2", 18, sw=0.8) if used else tint("--fig-mute", 8, sw=0.8))
                f.w(f'<rect x="{X0 + a * cw:.1f}" y="{y}" width="{cw - 1:.1f}" height="16" {st}/>')
            f.w(f'<rect x="{X0 + s * 8 * cw - 0.5:.1f}" y="{y - 2}" width="{8 * cw:.1f}" height="20" rx="2" style="fill: none; stroke: currentColor" stroke-opacity="{.6 if used else .15}" stroke-width="1"/>')
        for t_, a in enumerate(addr):
            x = X0 + a * cw + cw / 2 - 0.5
            f.arrow(x, y - 20, x, y - 3, None, width=1)
            f.w(f'<text class="s" x="{x:.1f}" y="{y - 22}" text-anchor="middle" style="font-size: 9px">T{t_}</text>') if t_ in (0, 7) or len(set(range(8)) & {t_}) and addr[1] - addr[0] > 1 else None
        f.w(f'<text class="s" x="{X0}" y="{y + 32}">{note}</text>')
    row(74, "相邻 thread 读相邻的 4 B：合并", list(range(8)), "取 1 个 sector（32 B），每个字节都用上")
    row(190, "相邻 thread 隔 32 B（比如按列读行优先的矩阵）：不合并", [s * 8 for s in range(8)], "取 8 个 sector（256 B），只用到 32 B")
    f.w(f'<text class="s" x="{X0 + 64 * cw + 12}" y="88">一格 4 B</text><text class="s" x="{X0 + 64 * cw + 12}" y="102">一个框 32 B</text>')
    return f

def opt_bank():
    f = F("fig-o-8", 264, "bank conflict（按 8 个 bank 画）：8 × 8 的 fp32 矩阵按行存进 shared memory，一列的 8 个数都落在 bank 0，8 个 thread 同时读这一列要排 8 拍；每行多留一个数（padding），或者把第 r 行第 c 列放进第 (c xor r) 个 bank（swizzle），一列的数都错开到 8 个不同的 bank，一拍读完")
    n, cw = 8, 23
    def panel(x0, title, where, note, pad=False):
        f.w(f'<text class="tb" x="{x0}" y="22">{title}</text>')
        for b in range(8):
            f.w(f'<text class="s" x="{x0 + b * cw + cw / 2}" y="44" text-anchor="middle">{b}</text>')
        for r in range(n):
            for c in range(n):
                bank, slot = where(r, c)
                st = tint("--fig-hi", 45, sw=0.8) if c == 0 else tint("--fig-1", 10, sw=0.8)
                f.w(f'<rect x="{x0 + bank * cw + 1}" y="{52 + slot * 19}" width="{cw - 2}" height="17" rx="2" {st}/>')
                f.w(f'<text class="s" x="{x0 + bank * cw + cw / 2}" y="{52 + slot * 19 + 12}" text-anchor="middle" style="font-size: 8.5px">{r},{c}</text>')
            if pad:
                a = r * (n + 1) + n
                f.w(f'<rect x="{x0 + (a % 8) * cw + 1}" y="{52 + (a // 8) * 19}" width="{cw - 2}" height="17" rx="2" style="fill: none; stroke: currentColor" stroke-opacity=".35" stroke-dasharray="3 2"/>')
        f.w(f'<text class="s" x="{x0}" y="{52 + 9 * 19 + 22}">{note}</text>')
    f.w('<text class="s" x="8" y="40">bank</text>')
    panel(36, "按行存", lambda r, c: (c, r), "第 0 列全在 bank 0：排 8 拍")
    panel(240, "每行补 1 个空位（padding）", lambda r, c: ((r * 9 + c) % 8, (r * 9 + c) // 8), "错开到 8 个 bank：一拍", pad=True)
    panel(444, "swizzle：bank = c xor r", lambda r, c: (c ^ r, r), "错开到 8 个 bank：一拍")
    return f

def tri_gelu():
    f, cx, p = lanes_fig("fig-t-1", "GELU kernel：数组展平成一维按 BLOCK 切段，一段一个 program；tl.load 把一段从显存读进 register，计算单元算 GELU，tl.store 从 register 直接写回显存",
        "逐元素：y[i] 只依赖 x[i]", "一个 program（pid = 1）", grid_strip,
        ["grid = (cdiv(n, BLOCK),)", "n = 8388608", "8192 个 program"],
        [("offsets", "tl.arange"), ("@", "tl.load"), ("gelu", "tl.exp"), ("@", "tl.store")], "reg",
        ["每个 program 1024 个元素，每个 thread 8 个，都在 register 里，不用 shared memory；", "offsets 是 load、store 的地址。显存只读 x 一次、写 y 一次。"], mem_y1=308, smem="gelu")
    MY = p.ROW["m"][0]
    p.chip("offsets", cx[0], size="4 KB"); p.down(cx[0], 110)                       # 算下标 → offsets
    p.chip("x 段", cx[1], w=44, size="4 KB"); p.xfer(cx[1], MY - 2, 140, "tl.load")
    p.up(cx[1] + 14, 110, cx[2] - 16)                                  # x 段 → GELU
    p.chip("y 段", cx[3], w=44, size="4 KB"); p.down(cx[2] + 16, 110, cx[3] - 14)   # GELU → y 段
    p.xfer(cx[3], 140, MY - 2, "tl.store")
    p.mem("整个 x：32 MB", cx[1] - 40, cx[1] + 40, [(cx[1] - 10, cx[1] + 10, 0, 1)], "读 4 KB", cx[1] - 5)
    p.mem("整个 y：32 MB", cx[3] - 40, cx[3] + 40, [(cx[3] - 10, cx[3] + 10, 0, 1)], "写 4 KB", cx[3] - 5)
    return f

def tri_softmax():
    f, cx, p = lanes_fig("fig-t-2", "softmax kernel：一行一个 program；tl.load 把一整行从显存读进 register，计算单元减最大值、exp、求和、归一化，中间结果都在 register 里，跨 warp 的归约经 shared memory，最后 tl.store 写回",
        "按行归约：y[i, j] 依赖整行 x[i, :]", "一个 program（pid = 1）", grid_rows,
        ["grid = (M,)", "M = 4096 个 program"],
        [("@", "tl.load"), ("sub max", "tl.max"), ("exp", "tl.exp"), ("div sum", "tl.sum"), ("@", "tl.store")], "reg",
        ["x、x − m、e、y 是这一行在不同时刻的值，都在同一块 register 里，每个 thread 存 32 个数；",
         "tl.max、tl.sum 先在 warp 内归约，各 warp 的部分结果经 shared memory 交换（同一块 16 B）；",
         "显存只读 x 一次、写 y 一次。"], mem_y1=308)
    MY = p.ROW["m"][0]
    BY0, BY1 = 112, 144                                                # 这一行的 register：一块，从 load 活到 store
    membar(f, cx[0] - 30, cx[4] + 30, BY0, BY1 - BY0)
    for k in (1, 2, 3):                                                # 每一步之后，这块里存的值换了
        f.w(f'<line x1="{cx[k]:.1f}" y1="{BY0 + 3}" x2="{cx[k]:.1f}" y2="{BY0 + 17}" style="stroke: var(--fig-2)" stroke-dasharray="2 2"/>')
    for t, a, b in (("x", 0, 1), ("x − m", 1, 2), ("e", 2, 3), ("y", 3, 4)):
        f.w(f'<text class="s" x="{(cx[a] + cx[b]) / 2:.1f}" y="{BY0 + 13}" text-anchor="middle">{t}</text>')
    f.w(f'<text class="s" x="{cx[2]:.1f}" y="{BY1 - 5}" text-anchor="middle" style="font-size: 9.5px; fill: var(--fig-1); opacity: 1">整行，16 KB</text>')
    p.xfer(cx[0], MY - 2, BY1 + 2, "tl.load", left=True)
    for k in (1, 2, 3):                                                # 读左边的旧值，写回右边
        f.arrow(cx[k] - 22, BY0 - 2, cx[k] - 22, 86, "--fig-2", width=1.4)
        f.arrow(cx[k] + 22, 86, cx[k] + 22, BY0 - 2, "--fig-2", width=1.4)
    p.xfer(cx[4], BY1 + 2, MY - 2, "tl.store")
    for nm, k, part in [("整个 x：64 MB", 0, "读 1 行，16 KB"), ("整个 y：64 MB", 4, "写 1 行，16 KB")]:
        p.mem(nm, cx[k] - 36, cx[k] + 36, [(cx[k] - 36, cx[k] + 36, .3, .25)], part, cx[k] - 5)
    for k, t in [(1, "warp max"), (3, "warp sum")]:             # 跨 warp 归约经 shared memory
        y0 = p.ROW["s"][0]
        f.w(f'<rect x="{cx[k] - 40:.1f}" y="{y0}" width="80" height="18" rx="4" {tint("--fig-3", 40)}/><text class="s" x="{cx[k]:.1f}" y="{y0 + 13}" text-anchor="middle">{t}，16 B</text>')
        over(f, cx[k] - 5, 86, cx[k] - 5, y0 - 2, width=1.3); over(f, cx[k] + 5, y0 - 2, cx[k] + 5, 86, width=1.3)   # 归约这一步自己经 shared memory 交换各 warp 的部分结果
    return f

def tri_rowsum():
    f, cx, p = lanes_fig("fig-t-3", "row sum kernel：一行一个 program；tl.load 一轮把一个 block 从显存读进 register，计算单元把它加到 acc 上，颜色深浅是 for 循环的顺序；acc 一直留在 register 里，最后归约成一个数，tl.store 写回",
        "按行归约，按 block 累加", "一个 program（pid = 1）", grid_rows,
        ["grid = (M,)", "M = 4096 个 program"],
        [("@", "tl.load"), ("add", "acc += x"), "⋯", ("@", "tl.load"), ("add", "acc += x"), ("reduce", "tl.sum"), ("@", "tl.store")], "reg",
        ["图里 TILE = 4（代码里 1024，标的大小按 1024 算），一行 16 个数分 4 个 block；",
         "x、acc 各是一块 register，横向是时间：每轮 tl.load 把新的 block 覆盖进 x，acc 原地累加；",
         "最后的 tl.sum 跨 4 个 warp，经 shared memory 交换。"], mem_y1=332)
    tiles = [[1, 2, 0, 3], [4, 1, 2, 0], [2, 2, 1, 1], [0, 3, 1, 2]]
    acc0, accN = tiles[0], [sum(c) for c in zip(*tiles)]                      # [7, 8, 4, 6]，和是 25
    OP = [.12, .34, .56, .78]                                                # 第 0 到 3 个 block 的颜色：越晚读越深
    TX = lambda o: 'style="fill: var(--fig-on-1)"' if o > .5 else 'style="opacity: .9"'   # 深色格子里用反色字
    gid = "fig-t-3-acc"
    f.w(f'<defs><linearGradient id="{gid}" x1="0" x2="1" y1="0" y2="0">'
        + "".join(f'<stop offset="{k / 3:.2f}" style="stop-color: var(--fig-1)" stop-opacity="{o}"/>' for k, o in enumerate(OP))
        + '</linearGradient></defs>')
    def cells(x0, y0, vals, fill, w=11, h=16):                                # 一排数，fill 是 opacity 或渐变
        if fill == "grad":
            f.w(f'<rect x="{x0:.1f}" y="{y0}" width="{w * len(vals)}" height="{h}" fill="url(#{gid})"/>')
        for j, v in enumerate(vals):
            if fill != "grad":
                f.w(f'<rect x="{x0 + j * w:.1f}" y="{y0}" width="{w}" height="{h}" style="fill: var(--fig-1)" fill-opacity="{fill}"/>')
            f.w(f'<rect x="{x0 + j * w:.1f}" y="{y0}" width="{w}" height="{h}" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>')
            f.w(f'<text class="s" x="{x0 + j * w + w / 2:.1f}" y="{y0 + h - 4.5}" text-anchor="middle" {TX(OP[j] if fill == "grad" else fill)}>{v}</text>')
    w = (cx[3] - cx[0]) / 3; cw = w / 4; xa = cx[0] - w / 2; Y = p.MT + 14; RH = 16
    # register 里两块：acc（上，从第 0 轮的 add 活到 tl.sum）和 x（下，每轮 tl.load 覆盖）；格子是这块在那个时刻存的值
    AY0, XY0, BH = 102, 140, 32
    SZ = 'style="font-size: 9.5px; fill: var(--fig-1); opacity: 1"'
    membar(f, cx[1] - 18, cx[5] - 12, AY0, BH); membar(f, cx[0] - 34, cx[3] + 34, XY0, BH)
    f.w(f'<text class="s" x="{cx[2] - 12:.1f}" y="{AY0 + 11}" {SZ}>acc，4 KB</text><text class="s" x="{cx[0] + 28:.1f}" y="{XY0 + 11}" {SZ}>x，4 KB</text>')
    for k, c in [(0, cx[0]), (3, cx[3])]:
        cells(c - 22, XY0 + 14, tiles[k], OP[k])
        p.xfer(c, Y + RH, XY0 + BH + 2, "tl.load")
    f.w(f'<text class="tb" x="{cx[2]:.1f}" y="{XY0 + 27}" text-anchor="middle">⋯</text><text class="tb" x="{cx[2] + 6:.1f}" y="{AY0 + 27}" text-anchor="middle">⋯</text>')
    f.arrow(cx[0] + 6, XY0 - 2, cx[1] - 16, 86, "--fig-2", width=1.4)                  # 第 0 轮：x → add（acc 还没开始，不穿过它）
    over(f, cx[3] + 14, XY0 - 2, cx[4] - 16, 86)                                     # 最后一轮：x → add，从 acc 上面穿过
    A1, A2 = cx[1] + 22, cx[4] + 22
    cells(A1 - 22, AY0 + 14, acc0, OP[0]); cells(A2 - 22, AY0 + 14, accN, "grad")
    p.down(cx[1] + 8, AY0, A1 - 10); p.down(cx[4] + 8, AY0, A2 - 10); p.up(cx[5] - 22, AY0, cx[5] - 22)
    p.carry(cx[1], cx[4], "acc 带到下一轮")
    f.w(f'<rect x="{cx[6] - 12:.1f}" y="112" width="24" height="16" style="fill: var(--fig-1)" fill-opacity=".78"/><text class="s" x="{cx[6]:.1f}" y="123.5" text-anchor="middle" style="fill: var(--fig-on-1)">25</text>')
    p.down(cx[5] + 22, 110, cx[6] - 10)                                       # 归约 → 25
    y0 = p.ROW["s"][0]                                                       # tl.sum 跨 4 个 warp：reduce 把各 warp 的部分和经 shared memory 交换
    f.w(f'<rect x="{cx[5] - 40:.1f}" y="{y0}" width="80" height="18" rx="4" {tint("--fig-3", 40)}/><text class="s" x="{cx[5]:.1f}" y="{y0 + 13}" text-anchor="middle">warp sum，16 B</text>')
    f.arrow(cx[5] - 2, 86, cx[5] - 2, y0 - 2, "--fig-2", width=1.3); f.arrow(cx[5] + 8, y0 - 2, cx[5] + 8, 86, "--fig-2", width=1.3)
    # 显存：x 画 3 行，第 1 行 16 个数按 block 着色；out 第 1 位写 25
    for r in range(3):
        for k in range(4):
            for j in range(4):
                x = xa + k * w + j * cw
                if r == 1:
                    f.w(f'<rect x="{x:.1f}" y="{Y + r * RH}" width="{cw:.1f}" height="{RH}" style="fill: var(--fig-1)" fill-opacity="{OP[k]}"/>')
                    f.w(f'<text class="s" x="{x + cw / 2:.1f}" y="{Y + r * RH + RH - 4.5}" text-anchor="middle" {TX(OP[k])}>{tiles[k][j]}</text>')
                f.w(f'<rect x="{x:.1f}" y="{Y + r * RH}" width="{cw:.1f}" height="{RH}" style="fill: none; stroke: var(--fig-1)" stroke-opacity="{.45 if r == 1 else .18}" stroke-width=".6"/>')
    f.w(f'<rect x="{xa:.1f}" y="{Y}" width="{4 * w:.1f}" height="{3 * RH}" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>')
    for k in range(1, 4):                                                    # block 之间的粗分隔线
        f.w(f'<line x1="{xa + k * w:.1f}" y1="{Y}" x2="{xa + k * w:.1f}" y2="{Y + 3 * RH}" style="stroke: var(--fig-1)" stroke-width="1.2"/>')
    f.w(f'<text class="s" x="{xa + 4 * w + 4:.1f}" y="{Y + RH + 11.5}">行 1</text>')
    for k in range(4):
        f.w(f'<text class="s" x="{xa + k * w + w / 2:.1f}" y="{Y + 3 * RH + 12}" text-anchor="middle">block {k}</text>')
    f.w(f'<text class="s" x="{xa + 2 * w:.1f}" y="{Y + 3 * RH + 26}" text-anchor="middle">整个 x [4096, 4096]：64 MB；program 1 只读行 1，16 KB</text>')
    ox = cx[6] - 8
    for r in range(3):
        st = 'style="fill: var(--fig-1)" fill-opacity=".78"' if r == 1 else 'fill="none"'
        f.w(f'<rect x="{ox:.1f}" y="{Y + r * RH}" width="16" height="{RH}" {st}/><rect x="{ox:.1f}" y="{Y + r * RH}" width="16" height="{RH}" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".45" stroke-width=".6"/>')
    f.w(f'<text class="s" x="{ox + 8:.1f}" y="{Y + RH + 11.5}" text-anchor="middle" style="fill: var(--fig-on-1)">25</text>')
    f.w(f'<rect x="{ox:.1f}" y="{Y}" width="16" height="{3 * RH}" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>')
    f.w(f'<text class="s" x="{ox + 8:.1f}" y="{Y + 3 * RH + 13}" text-anchor="middle">out</text>')
    p.xfer(cx[6], 130, Y + RH - 1, "tl.store", left=True, ly=152)
    return f

def tri_matmul():
    f, cx, p = lanes_fig("fig-t-4", "matmul kernel（A 4096 × 2048，B 2048 × 512）：C 的每个 block 一个 program；program (1, 1) 沿 K 一轮用 cp.async 把 A、B 各一个 block 从显存经 L2 搬进 shared memory，tl.dot 从 shared memory 读一小片进 register 做乘加，acc 留在 register 里，中间几轮用 ⋯ 省略，最后 ReLU 后 tl.store 写回 C 的 block (1, 1)",
        "block 乘加", "一个 program（pid_m = 1，pid_n = 1）", grid_tiles42,
        ["grid = (M / BM, N / BN)", "64 × 8 = 512 个 program"],
        [("@", "cp.async"), ("dot", "tl.dot"), "⋯", ("@", "cp.async"), ("dot", "tl.dot"), ("relu", "tl.maximum"), ("@", "tl.store")], "matmul",
        ["cp.async 把 A、B 的 block 从显存经 L2 搬进 shared memory，tl.dot 从中读 a、b 到 register 做 FMA；",
         "K 按 BK = 32 切成 64 轮，只画 T = 0 和 63；两个 slot 轮流用，算第 T 轮时第 T + 1 轮在搬；",
         "tl.store 先把 C block 经 shared memory 换一次排布（A、B 已用完，复用同一块），再写回显存。"], mem_y1=384, smem="matmul")
    RAMP = [.12 + .66 * k / 7 for k in range(8)]     # 沿 K 的 for 循环顺序：越晚读越深
    TX = lambda o: 'style="fill: var(--fig-on-1)"' if o > .5 else ''
    gid = "fig-t-4-acc"
    f.w(f'<defs><linearGradient id="{gid}" x1="0" x2="1" y1="0" y2="0">'
        + "".join(f'<stop offset="{k / 7:.2f}" style="stop-color: var(--fig-1)" stop-opacity="{o:.2f}"/>' for k, o in enumerate(RAMP))
        + '</linearGradient></defs>')
    SZ = 'style="font-size: 9.5px; fill: var(--fig-1); opacity: 1"'
    # register 里两块：acc（上，ReLU 之后就是 C block）和 a、b（下，每轮从 shared memory 覆盖）
    AY0, BY0, RH_ = 103, 128, 21
    membar(f, cx[1] - 12, cx[6] + 22, AY0, RH_); membar(f, cx[0] - 6, cx[4] - 12, BY0, RH_)
    f.w(f'<rect x="{cx[5]:.1f}" y="{AY0}" width="{cx[6] + 22 - cx[5]:.1f}" height="{RH_}" rx="5" {tint("--fig-2", 45, sw=1.2)}/>')   # ReLU 写出的 C block：同一块 register，换了新内容
    f.w(f'<text class="s" x="{cx[2]:.1f}" y="{AY0 + 14}" text-anchor="middle">acc <tspan {SZ}>16 KB</tspan></text>')
    f.w(f'<text class="s" x="{(cx[5] + cx[6] + 22) / 2:.1f}" y="{AY0 + 14}" text-anchor="middle">C block <tspan {SZ}>16 KB</tspan></text>')
    f.w(f'<text class="s" x="{cx[2] - 8:.1f}" y="{BY0 + 14}" text-anchor="middle">a、b <tspan {SZ}>各 8 KB</tspan></text>')
    f.arrow(cx[1] - 22, BY0 - 2, cx[1] - 22, 86, "--fig-2", width=1.4)        # 第 0 轮：a、b → dot（acc 这时还没开始）
    f.arrow(cx[1] + 14, 86, cx[1] + 14, AY0 - 2, "--fig-2", width=1.4)        # dot → acc
    over(f, cx[4] - 22, BY0 - 2, cx[4] - 22, 86)                               # 最后一轮：a、b → dot，从 acc 上面穿过
    f.arrow(cx[4] + 14, 86, cx[4] + 14, AY0 - 2, "--fig-2", width=1.4)
    f.arrow(cx[5] - 14, AY0 - 2, cx[5] - 14, 86, "--fig-2", width=1.4); f.arrow(cx[5] + 14, 86, cx[5] + 14, AY0 - 2, "--fig-2", width=1.4)   # ReLU：读 acc，写 C block
    p.carry(cx[1], cx[4], "acc 带到下一轮")
    f.w(f'<text class="s" x="{cx[1] - 8:.1f}" y="34" text-anchor="end" style="fill: var(--fig-hi)">T = 0</text><text class="s" x="{cx[4] + 10:.1f}" y="34" style="fill: var(--fig-hi)">T = 63</text>')
    # shared memory 一块 32 KB，分两个 slot 轮流用：slot 0 放偶数轮、slot 1 放奇数轮；循环结束后 C block 换排布也用这一块
    SY0, SY1, CW = 166, 232, 34
    S1, S0 = (SY0 + 4, SY0 + 30), (SY0 + 36, SY0 + 62)               # slot 1 在上、slot 0 在下
    membar(f, cx[0] - 22, cx[6] - 2, SY0, SY1 - SY0)
    def ab(x0, row, o, dash=False):                                  # 一个 slot 里的 A、B block，各 8 KB
        y0, y1 = row
        for k, nm in enumerate("AB"):
            x = x0 + k * CW
            st = f'style="fill: var(--fig-1)" fill-opacity="{o:.2f}"' if not dash else 'fill="none"'
            d = ' stroke-dasharray="3 2"' if dash else ''
            f.w(f'<rect x="{x:.1f}" y="{y0}" width="{CW}" height="{y1 - y0}" rx="3" {st}/><rect x="{x:.1f}" y="{y0}" width="{CW}" height="{y1 - y0}" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7"{d}/>')
            tx = TX(o) if not dash else ''
            f.w(f'<text class="s" x="{x + CW / 2:.1f}" y="{y0 + 11}" text-anchor="middle" {tx}>{nm}</text><text class="s" x="{x + CW / 2:.1f}" y="{y0 + 22}" text-anchor="middle" style="font-size: 9.5px" {tx}>8 KB</text>')
    T0X, T1X, T63X = cx[0] - 16, cx[0] + 62, cx[3] - 22
    ab(T0X, S0, RAMP[0]); ab(T63X, S1, RAMP[-1])
    f.w(f'<rect x="{T1X:.1f}" y="{S1[0]}" width="{2 * CW}" height="{S1[1] - S1[0]}" rx="3" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".7" stroke-dasharray="3 2"/>')   # 第 1 轮的 A、B 正在搬进 slot 1
    f.w(f'<text class="s" x="{T1X + CW:.1f}" y="{S1[0] + 11}" text-anchor="middle" style="font-size: 9.5px">T = 1，16 KB</text><text class="s" x="{T1X + CW:.1f}" y="{S1[0] + 22}" text-anchor="middle" style="font-size: 9.5px">正在搬</text>')
    f.w(f'<text class="s" x="{cx[4] + 4:.1f}" y="{S1[0] + 16}">slot 1</text><text class="s" x="{cx[4] + 4:.1f}" y="{S0[0] + 16}">slot 0</text>')
    over(f, T0X + CW, S0[0] - 2, T0X + CW, BY0 + RH_ + 2)               # 第 0 轮：slot 0 → a、b，从 slot 1 上面穿过
    f.arrow(T63X + CW, S1[0] - 2, T63X + CW, BY0 + RH_ + 2, "--fig-2", width=1.4)   # 最后一轮：slot 1 → a、b
    CR = (cx[5] - 4, cx[6] - 30)                                       # tl.store 先给 C block 换排布（TTGIR 的 convert_layout 记在 tl.store 那一行）：用的还是这一块
    f.w(f'<rect x="{CR[0]:.1f}" y="{S1[0]}" width="{CR[1] - CR[0]:.1f}" height="{S0[1] - S1[0]}" rx="3" style="fill: var(--fig-1)" fill-opacity="{RAMP[-1]:.2f}"/>')
    f.w(f'<text class="s" x="{(CR[0] + CR[1]) / 2:.1f}" y="{S1[0] + 24}" text-anchor="middle" {TX(RAMP[-1])}>tl.store</text><text class="s" x="{(CR[0] + CR[1]) / 2:.1f}" y="{S1[0] + 38}" text-anchor="middle" {TX(RAMP[-1])}>换排布</text>')
    f.arrow(CR[1] - 22, AY0 + RH_ + 2, CR[1] - 22, S1[0] - 2, "--fig-2", width=1.4); f.arrow(CR[1] - 10, S1[0] - 2, CR[1] - 10, AY0 + RH_ + 2, "--fig-2", width=1.4)
    S = 6                                            # 显存里的矩阵画全：一格 256 × 256 个元素，形状比例不变
    def mat(x0, y0, rows, cols, cells, labels, anchor="middle", lx=None, ly=None):
        for r in range(rows):
            for c in range(cols):
                op = cells.get((r, c), 0)
                fill = f'style="fill: var(--fig-1)" fill-opacity="{op}"' if op else 'fill="none"'
                f.w(f'<rect x="{x0 + c * S:.1f}" y="{y0 + r * S}" width="{S}" height="{S}" {fill}/>')
                f.w(f'<rect x="{x0 + c * S:.1f}" y="{y0 + r * S}" width="{S}" height="{S}" style="fill: none; stroke: var(--fig-1)" stroke-opacity=".22" stroke-width=".5"/>')
        f.w(f'<rect x="{x0:.1f}" y="{y0}" width="{cols * S}" height="{rows * S}" style="fill: none; stroke: var(--fig-1)" stroke-width="1.2"/>')
        lx = x0 + cols * S / 2 if lx is None else lx; ly = y0 + rows * S + 13 if ly is None else ly
        for k, t in enumerate(labels):
            f.w(f'<text class="s" x="{lx:.1f}" y="{ly + k * 13:.1f}" text-anchor="{anchor}">{t}</text>')
    AX, AY = cx[0] - 30, 256                         # A [4096, 2048]：16 × 8 格，第 1 行，K 从左往右
    mat(AX, AY, 16, 8, {(1, c): RAMP[c] for c in range(8)}, ["A [4096, 2048]", "整个 32 MB"])
    BX, BY = cx[0] + 26, 256                         # B [2048, 512]：8 × 2 格，第 1 列，K 从上往下
    mat(BX, BY, 8, 2, {(r, 1): RAMP[r] for r in range(8)}, ["B [2048, 512]", "整个 4 MB"], "start", BX + 18, BY + 14)
    p.xfer(AX + 3, AY + S, S0[1] + 2, "", xo=T0X + CW / 2)                   # cp.async：显存 → shared memory
    p.xfer(BX + 9, BY, S0[1] + 2, "cp.async", xo=T0X + CW * 1.5, ly=248)
    CX, CY = cx[6] - 9, 256                          # C [4096, 512]：16 × 2 格，写回 (1, 1)
    mat(CX, CY, 16, 2, {(1, 1): RAMP[-1]}, ["C [4096, 512]", "整个 8 MB"], "end", 630)
    p.xfer(cx[6] + 10, AY0 + RH_ + 2, CY + S - 1, "", xo=cx[6])
    f.w(f'<text class="s" x="{cx[6] - 2:.1f}" y="248" text-anchor="end" style="fill: var(--fig-1)">tl.store：写 16 KB</text>')
    lx, ly = BX + 18, BY + 58                         # 图例：渐变是 for 循环沿 K 读 block 的顺序
    f.w(f'<text class="s" x="{lx}" y="{ly - 6}">沿 K 第 T 轮读的 block：T = 0 … 63</text>')
    for k, o in enumerate(RAMP):
        f.w(f'<rect x="{lx + k * 12}" y="{ly}" width="12" height="10" style="fill: var(--fig-1); stroke: var(--fig-1)" fill-opacity="{o:.2f}" stroke-opacity=".4" stroke-width=".6"/>')
    f.w(f'<text class="s" x="{lx}" y="{ly + 23}">T = 0</text><text class="s" x="{lx + 96}" y="{ly + 23}" text-anchor="end">T = 63</text>')
    f.w(f'<text class="s" x="{lx}" y="{ly + 40}">每轮读 A 的 64 × 32、B 的 32 × 64，各 8 KB；</text><text class="s" x="{lx}" y="{ly + 54}">图里一格 256 列，是 8 轮</text>')
    return f

# ── 图 3-6 matmul 的 512 个 program 怎么分批跑（实测 trace） ─────────────────────
def mm_wave():
    runs = {r["M"]: r for r in json.load(open(os.path.join(HERE, "mm_wave_trace.json")))}
    cap, NB = 170 * 3, 8                                             # 名额；grid 第二维（N / BN）
    lid = lambda i: i // NB + (i % NB) * (4096 // 64)               # 存的是 pid_m · 8 + pid_n，换成发射顺序 pid_m + pid_n · 64
    tr = [(a / 1000, b / 1000, sm, lid(i)) for i, (a, b, sm) in enumerate(runs[4096]["trace"])]
    small = [(a / 1000, b / 1000) for a, b, _ in runs[4032]["trace"]]
    T = max(b for b, *_ in [(x[1],) for x in tr])
    nw = (len(tr) + cap - 1) // cap
    W2 = [.3, .9]                                                    # 第 1、2 批：越晚越深，和图 2-1a 一样
    late = sorted(x for x in tr if x[3] >= cap)
    w1end = max(b for a, b, _, l in tr if l < cap)
    f = F("fig-t-7", 372, f"matmul 的 512 个 program 在 170 个 SM 上的实测时间线：上面是全卡同时在跑的 program 数，按批次堆叠，最多 510 个；第 1 批 510 个一起开始，第 2 批只有 2 个，等到 {late[0][0]:.0f} µs 才开始，{w1end:.0f} µs 以后全卡只有这 2 个在跑。虚线是 A 4032 × 2048 时 504 个 program 一批跑完的对照。下面是跑了第 2 批的两个 SM 的 3 个名额")
    X0, X1 = 74, 620
    px = lambda t: X0 + t / 210 * (X1 - X0)
    Y0, Y1 = 186, 52
    py = lambda c: Y0 - c / cap * (Y0 - Y1)
    for v in (0, 100, 200, 300, 400, 510):
        f.w(f'<line class="grid" x1="{X0}" y1="{py(v):.1f}" x2="{X1}" y2="{py(v):.1f}"/><text class="tick" x="{X0 - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{v}</text>')
    f.w(f'<text class="lab2" x="{X0}" y="{Y1 - 24}">某一时刻全卡已开始、未结束的 program 数，按批次堆叠；虚线是 170 × 3 = {cap} 个名额</text>')
    f.w(f'<text class="lab2" x="16" y="{(Y0 + Y1) / 2:.1f}" text-anchor="middle" transform="rotate(-90 16 {(Y0 + Y1) / 2:.1f})">同时在跑的 program 数</text>')
    f.w(f'<line class="ref" x1="{X0}" y1="{py(cap):.1f}" x2="{X1}" y2="{py(cap):.1f}"/>')
    N = 420
    ts = [T * k / N for k in range(N + 1)]
    cnt = [[0] * (N + 1) for _ in range(nw)]
    for a, b, _, l in tr:
        for k, t in enumerate(ts):
            if a <= t < b: cnt[l // cap][k] += 1
    base = [0] * (N + 1)
    for w in range(nw):
        top = [base[k] + cnt[w][k] for k in range(N + 1)]
        pts = [f"{px(t):.1f},{py(c):.1f}" for t, c in zip(ts, top)] + [f"{px(t):.1f},{py(c):.1f}" for t, c in zip(ts[::-1], base[::-1])]
        lo, hi = w * cap, min(len(tr), (w + 1) * cap) - 1
        f.w(f'<g class="m"><title>第 {w + 1} 批：program {lo} 到 {hi}，共 {hi - lo + 1} 个</title><polygon points="{" ".join(pts)}" style="fill: var(--fig-1)" fill-opacity="{W2[w]}"/></g>')
        base = top
    sp = [f"{px(0):.1f},{py(0):.1f}"]                                # 对照：A 4032 × 2048，504 个 program
    for t in sorted({a for a, _ in small} | {b for _, b in small}):
        sp.append(f"{px(t):.1f},{py(sum(1 for a, b in small if a <= t < b)):.1f}")
    f.w(f'<polyline points="{" ".join(sp)}" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6" stroke-dasharray="5 3"/>')
    # 第 2 批只有 2 个，按比例只有 1 px 高：沿底边画一条粗线代表它，再用红框圈出来，标注引到框上
    b0, y0 = late[0][0], py(0)
    f.w(f'<line x1="{px(b0):.1f}" y1="{y0 - 2:.1f}" x2="{px(T):.1f}" y2="{y0 - 2:.1f}" style="stroke: var(--fig-1)" stroke-opacity="{W2[1]}" stroke-width="4"/>')
    f.w(f'<rect x="{px(b0) - 4:.1f}" y="{y0 - 10:.1f}" width="{px(T) - px(b0) + 8:.1f}" height="16" rx="5" style="fill: none; stroke: var(--fig-hi)" stroke-width="1.8"/>')
    xm = (px(w1end) + px(T)) / 2
    f.w(f'<line x1="{xm:.1f}" y1="{y0 - 40:.1f}" x2="{xm:.1f}" y2="{y0 - 11:.1f}" style="stroke: var(--fig-hi)" stroke-width="1.2"/>')
    f.w(f'<text class="s" x="{xm:.1f}" y="{y0 - 58:.1f}" text-anchor="middle" style="fill: var(--fig-hi)">第 2 批（510–511）：2 个 program</text><text class="s" x="{xm:.1f}" y="{y0 - 44:.1f}" text-anchor="middle" style="fill: var(--fig-hi)">{b0:.0f} µs 开始，跑到 {T:.0f} µs</text>')
    # 下：跑了第 2 批的两个 SM，各 3 个名额
    G0, RH = 232, 11
    f.w(f'<text class="lab2" x="{X0}" y="{G0 - 10}">跑了第 2 批的两个 SM，每个 3 个名额；一条是一个 program 从第一条指令到写完 C</text>')
    y = G0
    for a0, b0, sm, l0 in late:
        mine = sorted((a, b, l) for a, b, s, l in tr if s == sm)
        slots = [[] for _ in range(3)]
        for a, b, l in mine:
            k = min(range(3), key=lambda j: slots[j][-1][1] if slots[j] else -1)
            slots[k].append((a, b, l))
        f.w(f'<text class="lab" x="{X0 - 8}" y="{y + 2 * RH - 2}" text-anchor="end">SM {sm}</text>')
        for j in range(3):
            for a, b, l in slots[j]:
                f.w(f'<g class="m"><title>program {l}（第 {l // cap + 1} 批）：{a:.1f} 到 {b:.1f} µs</title><rect x="{px(a):.1f}" y="{y + j * RH + 1}" width="{px(b) - px(a) - .6:.1f}" height="{RH - 2}" rx="1.5" style="fill: var(--fig-1)" fill-opacity="{W2[l // cap]}"/></g>')
        y += 3 * RH + 6
    for t in range(0, 211, 25):
        f.w(f'<text class="tick" x="{px(t):.1f}" y="{y + 10}" text-anchor="middle">{t}</text>')
        f.w(f'<line class="grid" x1="{px(t):.1f}" y1="{Y1}" x2="{px(t):.1f}" y2="{Y0}"/>')
    f.w(f'<text class="lab2" x="{X0 - 8}" y="{y + 10}" text-anchor="end">µs</text>')
    LY = y + 36
    f.w(f'<text class="lab2" x="{X0}" y="{LY}">批次</text>')
    for w in range(nw):
        lo, hi = w * cap, min(len(tr), (w + 1) * cap) - 1
        x = X0 + 36 + w * 100
        f.w(f'<rect x="{x}" y="{LY - 9}" width="12" height="10" rx="2" style="fill: var(--fig-1)" fill-opacity="{W2[w]}"/><text class="lab" x="{x + 17}" y="{LY}">{lo}–{hi}</text>')
    f.w(f'<line x1="{X0 + 250}" y1="{LY - 4}" x2="{X0 + 272}" y2="{LY - 4}" style="stroke: var(--fig-hi)" stroke-width="1.6" stroke-dasharray="5 3"/><text class="lab" x="{X0 + 280}" y="{LY}">A 4032 × 2048：504 个，一批</text>')
    return f

# ── 图 2-4a matmul 的流水线：一个 warp 每轮等数据和计算的拍数（Nsight Compute 实测的平均值） ──
def mm_pipe_ncu():
    d = json.load(open(os.path.join(HERE, "ncu_matmul.json")))
    run = {r["num_stages"]: r for r in d["runs"] if r["M"] == 4032}
    wait = lambda r: (r["stall"]["long_scoreboard"] + r["stall"]["lg_throttle"]) * r["inst_per_round"]
    P1, W1 = run[1]["cycles_per_inst"] * run[1]["inst_per_round"], wait(run[1])     # 不预取：每轮拍数、其中等数据
    P3, W3 = run[3]["cycles_per_inst"] * run[3]["inst_per_round"], wait(run[3])     # 预取
    L = W1                                                                           # load 的延迟：按不预取时每轮等数据的拍数画
    N = 5
    f = F("fig-t-6", 340, f"matmul 一个 warp 的流水线，条长是 Nsight Compute 测的每轮平均拍数：不预取时每轮 {P1:.0f} 拍，先等数据 {W1:.0f} 拍再计算，一轮接一轮；num_stages = 3 时第 k 个 block 的 load 在第 k − 2 轮算完后发出，{L:.0f} 拍后就到了，到用的时候只等 {W3:.0f} 拍，每轮 {P3:.0f} 拍")
    X0, X1, TMAX = 112, 626, 30000
    px = lambda t: X0 + t / TMAX * (X1 - X0)
    RH = 15
    def bar(x0, x1, y, var, op=1, dash=False, tip=""):
        st = f'style="fill: none; stroke: var({var})" stroke-opacity=".8" stroke-dasharray="3 2"' if dash else f'style="fill: var({var})" fill-opacity="{op}"'
        f.w(f'<g class="m"><title>{tip}</title><rect x="{px(x0):.1f}" y="{y + 2}" width="{max(px(x1) - px(x0), .8):.1f}" height="{RH - 4}" rx="2" {st}/></g>')
    def panel(y0, title, sub):
        f.w(f'<text class="tb" x="8" y="{y0 - 8}">{title}</text><text class="lab2" x="{X0}" y="{y0 - 8}">{sub}</text>')
        for k in range(N):
            f.w(f'<text class="lab" x="8" y="{y0 + k * RH + 11}">block {k}</text>')
        for t in range(0, TMAX, 5000):
            f.w(f'<line class="grid" x1="{px(t):.1f}" y1="{y0}" x2="{px(t):.1f}" y2="{y0 + N * RH}"/>')
    Y1 = 44                                                                          # 不预取
    panel(Y1, "num_stages = 1", f"不预取：每轮 {P1:.0f} 拍，其中等数据 {W1:.0f} 拍")
    for k in range(N):
        t = k * P1
        bar(t, t + W1, Y1 + k * RH, "--fig-hi", tip=f"block {k}：load 后等数据 {W1:.0f} 拍")
        bar(t + W1, t + P1, Y1 + k * RH, "--fig-1", .55, tip=f"block {k}：ld.shared 和乘加 {P1 - W1:.0f} 拍")
    Y3 = Y1 + N * RH + 44                                                            # 预取
    panel(Y3, "num_stages = 3", f"预取：每轮 {P3:.0f} 拍，其中等数据 {W3:.0f} 拍")
    for k in range(N):
        y = Y3 + k * RH
        issue = 0 if k < 2 else L + (k - 1) * P3                                     # 第 k 轮的 load 在第 k − 2 轮末尾发出
        start = L + k * P3
        bar(issue, issue + L, y, "--fig-1", dash=True, tip=f"block {k}：load 在路上，约 {L:.0f} 拍")
        if issue + L < start:
            bar(issue + L, start, y, "--fig-2", .25, tip=f"block {k}：已在 shared memory，等着被用")
        bar(start, start + W3, y, "--fig-hi", tip=f"block {k}：等数据 {W3:.0f} 拍")
        bar(start + W3, start + P3, y, "--fig-1", .55, tip=f"block {k}：ld.shared 和乘加 {P3 - W3:.0f} 拍")
    yb = Y3 + N * RH
    for t in range(0, TMAX, 5000):
        f.w(f'<text class="tick" x="{px(t):.1f}" y="{yb + 14}" text-anchor="middle">{t}</text>')
    f.w(f'<text class="lab2" x="{X0 - 8}" y="{yb + 14}" text-anchor="end">拍</text>')
    ly = yb + 40
    items = [("--fig-hi", 1, False, "等数据（ncu 的 long scoreboard）"), ("--fig-1", .55, False, "ld.shared 和乘加"),
             ("--fig-1", 1, True, f"load 在路上（约 {L:.0f} 拍）"), ("--fig-2", .25, False, "已到 shared memory，等着被用")]
    for k, (var, op, dash, lab) in enumerate(items):
        x, y = 8 + (k % 2) * 300, ly + (k // 2) * 20
        st = f'style="fill: none; stroke: var({var})" stroke-dasharray="3 2"' if dash else f'style="fill: var({var})" fill-opacity="{op}"'
        f.w(f'<rect x="{x}" y="{y - 9}" width="22" height="11" rx="2" {st}/><text class="s" x="{x + 30}" y="{y}">{lab}</text>')
    return f

# ── 图 1-4 asynchronous barrier vs asynchronous transaction barrier（示意，画法参考 H100 白皮书 Figure 20） ──
def tx_barrier():
    f = F("fig-b-1", 440, "两种 barrier 的甘特图。上下搬的是同样的 16 KB，放行时刻一样。上：asynchronous barrier 只数 thread 的 arrive，4 个 thread 各自搬完自己那份才 arrive，几乎没空做别的。下：asynchronous transaction barrier 还数字节，thread 0 发起 TMA 并登记 16 KB，4 个 thread 一开始就 arrive、去做别的活，TMA 分 4 段写进 shared memory，字节数到 16 KB 时放行")
    X0, X1 = 92, 628
    px = lambda u: X0 + u / 100 * (X1 - X0)
    RH = 20
    def bar(u0, u1, y, var, pct=None, dash=False, lab=""):
        if u1 <= u0: return
        st = (f'style="fill: none; stroke: var({var})" stroke-dasharray="3 2"' if dash
              else f'style="fill: var({var})" fill-opacity="{pct}"')
        f.w(f'<rect x="{px(u0):.1f}" y="{y + 3}" width="{px(u1) - px(u0):.1f}" height="{RH - 6}" rx="2" {st}/>')
        if lab and px(u1) - px(u0) > sum(11 if ord(c) > 0x2E80 else 6 for c in lab) + 16:
            f.w(f'<text class="s" x="{(px(u0) + px(u1)) / 2:.1f}" y="{y + RH / 2 + 4}" text-anchor="middle">{lab}</text>')
    def arrive(u, y):
        x = px(u)
        f.w(f'<path d="M{x:.1f} {y + 2} L{x + 5:.1f} {y + RH / 2} L{x:.1f} {y + RH - 2} L{x - 5:.1f} {y + RH / 2} Z" style="fill: var(--fig-1)"/>')
    def release(u, y0, y1, lab):
        x = px(u)
        f.w(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="currentColor" stroke-opacity=".6" stroke-dasharray="4 3"/>')
        f.w(f'<text class="s" x="{x + 6:.1f}" y="{y1 + 2}">{lab}</text>')
    # 上：asynchronous barrier
    Y = 40
    f.w(f'<text class="tb" x="8" y="{Y - 12}">asynchronous barrier（Ampere 起）：每个 thread 搬完自己那份再 arrive，只数 arrive</text>')
    prod, work = [30, 40, 26, 34], [6, 0, 8, 4]
    R = max(prod)
    for k in range(4):
        y = Y + k * RH
        f.w(f'<text class="lab" x="8" y="{y + 14}">thread {k}</text>')
        bar(0, prod[k], y, "--fig-2", .55, lab="搬自己那份")
        bar(prod[k], prod[k] + work[k], y, "--fig-3", .9, lab="别的活")
        bar(prod[k] + work[k], R, y, "--fig-hi", dash=True)
        s = max(R, prod[k] + work[k])
        bar(s, s + 22, y, "--fig-1", .6, lab="用数据")
        arrive(prod[k], y)
    release(R, Y - 2, Y + 4 * RH + 10, "最后一个 thread arrive，放行")
    # 下：asynchronous transaction barrier
    Y2 = Y + 4 * RH + 60
    f.w(f'<text class="tb" x="8" y="{Y2 - 12}">asynchronous transaction barrier（Hopper 起）：同样 16 KB 交给 TMA 搬，还要数字节</text>')
    chunks = [(4, 13), (13, 22), (22, 31), (31, 40)]
    R2 = chunks[-1][1]
    work2 = [36, 32, 40, 30]
    for k in range(4):
        y = Y2 + k * RH
        f.w(f'<text class="lab" x="8" y="{y + 14}">thread {k}</text>')
        a = 4 if k == 0 else 1
        if k == 0:
            bar(0, 4, y, "--fig-hi", .7)
        bar(a, work2[k], y, "--fig-3", .9, lab="别的活")
        bar(work2[k], R2, y, "--fig-hi", dash=True)
        bar(R2, R2 + 22, y, "--fig-1", .6, lab="用数据")
        arrive(a, y)
    yt = Y2 + 4 * RH
    f.w(f'<text class="lab" x="8" y="{yt + 14}">TMA</text><text class="s" x="{px(chunks[-1][1]) + 6:.1f}" y="{yt + 14}">写进 shared memory，共 16 KB，分 4 段</text>')
    for i, (u0, u1) in enumerate(chunks):
        bar(u0, u1, yt, "--fig-hi", .35, lab="写 4 KB")
    yb = yt + RH
    f.w(f'<text class="lab" x="8" y="{yb + 14}">字节计数</text>')
    for i, (u0, u1) in enumerate(chunks):
        f.w(f'<text class="s" x="{px(u1) - 3:.1f}" y="{yb + 14}" text-anchor="end">{4 * (i + 1)} KB</text>')
    release(R2, Y2 - 2, yb + RH + 10, "thread 都 arrive 过、字节到齐，放行")
    # 图例（两行）
    ly = yb + RH + 38
    rows = [[("--fig-2", .55, False, "thread 自己搬数据"), ("--fig-3", .9, False, "不依赖这批数据的活"), ("--fig-1", .6, False, "用数据")],
            [("--fig-hi", None, True, "等（A100 空转，H100 可以睡眠）"), ("--fig-hi", .7, False, "发 TMA、登记 16 KB")]]
    for r, items in enumerate(rows):
        x, y = 8, ly + r * 20
        for var, op, dash, lab in items:
            st = f'style="fill: none; stroke: var({var})" stroke-dasharray="3 2"' if dash else f'style="fill: var({var})" fill-opacity="{op}"'
            f.w(f'<rect x="{x}" y="{y - 9}" width="18" height="11" rx="2" {st}/><text class="s" x="{x + 24}" y="{y}">{lab}</text>')
            x += 32 + sum(10.5 if ord(c) > 0x2E80 else 5.8 for c in lab) + 14
    y = ly + 40
    f.w(f'<path d="M13 {y - 9} L18 {y - 3} L13 {y + 3} L8 {y - 3} Z" style="fill: var(--fig-1)"/><text class="s" x="32" y="{y}">arrive：告诉 barrier 自己这份做完了，不阻塞</text>')
    return f

def _ncu_rounds():
    d = json.load(open(os.path.join(HERE, "ncu_matmul.json")))
    run = {r["num_stages"]: r for r in d["runs"] if r["M"] == 4032}
    per = lambda r: r["cycles_per_inst"] * r["inst_per_round"]
    wait = lambda r: (r["stall"]["long_scoreboard"] + r["stall"]["lg_throttle"]) * r["inst_per_round"]
    nsel = lambda r: r["stall"]["not_selected"] * r["inst_per_round"]           # 自己就绪、但这一拍调度器发了别的 warp
    return {s: (per(run[s]), wait(run[s]), nsel(run[s])) for s in (1, 3, 4)}

# ── 图 3-4 num_stages：按 shared memory slot 看 matmul 的循环（ncu_matmul.json） ───────────
def pipe_slots():
    R = _ncu_rounds()
    P1, W1, N1 = R[1]
    L = W1                                                                           # load 的延迟，按不预取时每轮等数据的拍数画
    NR = 4                                                                           # 每栏都只画前 4 轮，按同一根时间轴对齐
    f = F("fig-t-9", 470, f"matmul 循环按 shared memory slot 画的甘特图，三栏共用一根时间轴，都只画前 {NR} 轮。num_stages = 1 只有 1 个 slot：每轮先搬数据、等 {W1:.0f} 拍，再从 slot 里读出来算。num_stages = 3 有 2 个 slot：第 k 轮的 load 在第 k − 2 轮算完后发出，算第 k 轮时，第 k + 1 轮已经在另一个 slot 里。num_stages = 4 有 3 个 slot，load 提前两整轮发出，每轮更短，但一个 program 占 49 KB，每个 SM 只放得下 2 个，504 个 program 要分两批")
    X0, X1, TMAX = 120, 600, 23000
    px = lambda t: X0 + min(t, TMAX) / TMAX * (X1 - X0)
    RH = 22
    def bar(t0, t1, y, var, op=1, dash=False, tip="", lab=""):
        if t0 >= TMAX: return
        st = (f'style="fill: none; stroke: var({var})" stroke-opacity=".85" stroke-dasharray="3 2"' if dash
              else f'style="fill: var({var})" fill-opacity="{op}"')
        w = max(px(t1) - px(t0), .8)
        f.w(f'<g class="m"><title>{tip}</title><rect x="{px(t0):.1f}" y="{y + 3}" width="{w:.1f}" height="{RH - 6}" rx="2" {st}/></g>')
        if lab and w > 40:
            f.w(f'<text class="s" x="{px(t0) + w / 2:.1f}" y="{y + RH / 2 + 4}" text-anchor="middle">{lab}</text>')
    def rows(y0, names):
        for i, n in enumerate(names):
            f.w(f'<text class="lab" x="8" y="{y0 + i * RH + RH / 2 + 4}">{n}</text>')
    def finish(t_end, y0, y1):                                                       # 这一栏 4 轮算完的时刻
        f.w(f'<line x1="{px(t_end):.1f}" y1="{y0 - 2}" x2="{px(t_end):.1f}" y2="{y1 + 2}" style="stroke: var(--fig-hi)" stroke-width="1.6"/>')
        f.w(f'<text class="s" x="{px(t_end) + 5:.1f}" y="{y1 - 6}" style="fill: var(--fig-hi)">{NR} 轮算完</text>')
        f.w(f'<text class="s" x="{px(t_end) + 5:.1f}" y="{y1 + 8}" style="fill: var(--fig-hi)">{t_end:.0f} 拍</text>')
    ends = {}
    # num_stages = 1
    Y1 = 40
    f.w(f'<text class="tb" x="8" y="{Y1 - 10}">num_stages = 1</text><text class="lab2" x="{X0 + 64}" y="{Y1 - 10}">1 个 slot，不预取：每轮 {P1:.0f} 拍，其中等显存 {W1:.0f}、让给别的 warp {N1:.0f}</text>')
    rows(Y1, ["slot (shmem)", "warp"])
    for k in range(NR):
        t = k * P1
        bar(t, t + W1, Y1, "--fig-1", dash=True, tip=f"第 {k} 轮：load 在路上，{W1:.0f} 拍")
        bar(t + W1, t + P1, Y1, "--fig-1", .22, tip=f"第 {k} 轮：在 slot 里被读", lab=f"第 {k} 轮")
        bar(t, t + W1, Y1 + RH, "--fig-hi", tip=f"第 {k} 轮：等数据 {W1:.0f} 拍")
        bar(t + W1, t + P1 - N1, Y1 + RH, "--fig-1", .6, tip=f"第 {k} 轮：自己在跑 {P1 - W1 - N1:.0f} 拍（发射、等 shared memory、barrier 等）", lab=f"算第 {k} 轮")
        bar(t + P1 - N1, t + P1, Y1 + RH, "--fig-2", .75, tip=f"第 {k} 轮：让给别的 warp {N1:.0f} 拍")
    ends[1] = (NR * P1, Y1, Y1 + 2 * RH)
    def staged(Y, S, title, sub):
        P, W, N = R[S]
        nslot = S - 1
        f.w(f'<text class="tb" x="8" y="{Y - 10}">{title}</text><text class="lab2" x="{X0 + 64}" y="{Y - 10}">{sub.format(P=P, W=W, N=N)}</text>')
        rows(Y, [f"slot {i} (shmem)" for i in range(nslot)] + ["warp"])
        s = lambda k: L + k * P                                                      # 第 k 轮开始算
        for k in range(NR):
            y = Y + (k % nslot) * RH
            issue = 0 if k < nslot else s(k - nslot + 1)                             # 第 k − S + 1 轮算完时发出
            bar(issue, issue + L, y, "--fig-1", dash=True, tip=f"第 {k} 轮：load 在路上，约 {L:.0f} 拍")
            bar(issue + L, s(k), y, "--fig-2", .28, tip=f"第 {k} 轮：已到 slot 里，等着被读")
            bar(s(k), s(k) + P, y, "--fig-1", .22, tip=f"第 {k} 轮：在 slot 里被读", lab=f"第 {k} 轮")
        yw = Y + nslot * RH
        bar(0, L, yw, "--fig-hi", tip=f"第 0 轮：等数据 {L:.0f} 拍")
        for k in range(NR):
            bar(s(k), s(k) + W, yw, "--fig-hi", tip=f"第 {k} 轮：等数据 {W:.0f} 拍")
            bar(s(k) + W, s(k) + P - N, yw, "--fig-1", .6, tip=f"第 {k} 轮：自己在跑 {P - W - N:.0f} 拍（发射、等 shared memory、barrier 等）", lab=f"算第 {k} 轮")
            bar(s(k) + P - N, s(k) + P, yw, "--fig-2", .75, tip=f"第 {k} 轮：让给别的 warp {N:.0f} 拍")
        ends[S] = (s(NR), Y, yw + RH)
        return s, yw
    # num_stages = 3
    Y3 = Y1 + 2 * RH + 50
    s3, yw3 = staged(Y3, 3, "num_stages = 3", "2 个 slot，提前一轮发 load：每轮 {P:.0f} 拍，其中等显存 {W:.0f}、让给别的 warp {N:.0f}")
    tm = s3(2) + R[3][0] * .4                                                        # 标一个时刻
    f.w(f'<line x1="{px(tm):.1f}" y1="{Y3 - 4}" x2="{px(tm):.1f}" y2="{yw3 + RH + 4}" stroke="currentColor" stroke-opacity=".55" stroke-dasharray="4 3"/>')
    f.w(f'<text class="s" x="{X0}" y="{yw3 + RH + 18}">竖虚线这一刻：第 2 轮在 slot 0 里被读、在算；第 3 轮已经到了 slot 1</text>')
    # num_stages = 4
    Y4 = yw3 + RH + 62
    s4, yw4 = staged(Y4, 4, "num_stages = 4", "3 个 slot，提前两轮发 load：每轮 {P:.0f} 拍，其中等显存 {W:.0f}、让给别的 warp {N:.0f}")
    f.w(f'<text class="s" x="{X0}" y="{yw4 + RH + 18}">每轮更短，主要短在让给别的 warp：每个 SM 只剩 2 个 program、8 个 warp</text>')
    yb = yw4 + RH + 24
    for t in range(0, TMAX + 1, 5000):                                               # 网格线贯穿三栏，三栏共用一根时间轴
        f.w(f'<line class="grid" x1="{px(t):.1f}" y1="{Y1}" x2="{px(t):.1f}" y2="{yb}"/>')
        f.w(f'<text class="tick" x="{px(t):.1f}" y="{yb + 14}" text-anchor="middle">{t}</text>')
    f.w(f'<text class="lab2" x="{X0 - 8}" y="{yb + 14}" text-anchor="end">拍</text>')
    for S in (1, 3, 4):
        finish(*ends[S])
    ly = yb + 36
    items = [("--fig-1", 1, True, "load 在路上"), ("--fig-2", .28, False, "已到 slot 里，等着被读"),
             ("--fig-1", .22, False, "在 slot 里被读"), ("--fig-1", .6, False, "自己在跑"), ("--fig-2", .75, False, "让给别的 warp"), ("--fig-hi", 1, False, "等显存")]
    x = 8
    for i, (var, op, dash, lab) in enumerate(items):
        if i == 3: x, ly = 8, ly + 20                                                # 图例分两行
        st = f'style="fill: none; stroke: var({var})" stroke-dasharray="3 2"' if dash else f'style="fill: var({var})" fill-opacity="{op}"'
        f.w(f'<rect x="{x}" y="{ly - 9}" width="20" height="11" rx="2" {st}/><text class="s" x="{x + 26}" y="{ly}">{lab}</text>')
        x += 34 + sum(11 if ord(c) > 0x2E80 else 6 for c in lab) + 14
    return f

# ── 图 3-5 num_stages = 3 和 4：一个 SM 放几个 program，整个 kernel 跑几批（num_stages.json、ncu_matmul.json） ──
def stages_sm():
    ns = json.load(open(os.path.join(HERE, "num_stages.json")))["matmul_4032"]
    d = json.load(open(os.path.join(HERE, "ncu_matmul.json")))
    run = {r["num_stages"]: r for r in d["runs"] if r["M"] == 4032}
    per = {s: run[s]["cycles_per_inst"] * run[s]["inst_per_round"] for s in (3, 4)}
    tot = {s: run[s]["us_ncu"] * 1e-6 * run[s]["sm_hz"] for s in (3, 4)}         # 整个 kernel 的拍数
    NP, SMS, RES = 504, 170, 1                                                    # program 数、SM 数、每个 program 保留的 KB
    f = F("fig-s-1", 350, "num_stages = 3 和 4 时一个 SM 的 shared memory 和整个 kernel 的时间。3 级每个 program 占 33 KB，100 KB 放得下 3 个，504 个 program 一批跑完；4 级每个 program 占 49 KB，只放得下 2 个，要分两批，第二批每个 SM 最多 1 个")
    X0, X1 = 130, 600
    # 上：shared memory
    f.w('<text class="tb" x="8" y="20">一个 SM 的 shared memory（100 KB）</text>')
    kx = lambda kb: X0 + kb / 100 * (X1 - X0)
    for i, s in enumerate((3, 4)):
        y = 44 + i * 44
        kb = ns[str(s)]["shared"] / 1024 + RES
        n = ns[str(s)]["per_sm"]
        f.w(f'<text class="lab" x="8" y="{y + 17}">num_stages = {s}</text>')
        f.w(f'<rect x="{kx(0):.1f}" y="{y + 2}" width="{kx(100) - kx(0):.1f}" height="22" rx="2" fill="none" stroke="currentColor" stroke-opacity=".35"/>')
        for p_ in range(n):
            x = kx(p_ * kb)
            for k in range(s - 1):                                                # A、B 各 S − 1 个 slot，一个 slot 16 KB
                x0 = x + kx(k * 16) - kx(0)
                f.w(f'<g class="m"><title>P{p_} 的 slot {k}：A、B 各 8 KB</title><rect x="{x0 + .5:.1f}" y="{y + 3}" width="{kx(16) - kx(0) - 1:.1f}" height="20" rx="2" style="fill: var(--fig-1)" fill-opacity="{.55 if p_ % 2 == 0 else .35}"/></g>')
                f.w(f'<text class="s" x="{x0 + (kx(16) - kx(0)) / 2:.1f}" y="{y + 17}" text-anchor="middle" style="fill: #fff; opacity: 1">slot {k}</text>')
            f.w(f'<rect x="{x + kx((s - 1) * 16) - kx(0):.1f}" y="{y + 3}" width="{kx(RES) - kx(0):.1f}" height="20" style="fill: var(--fig-mute)"/>')
            f.w(f'<text class="s" x="{x + (kx(kb) - kx(0)) / 2:.1f}" y="{y - 1}" text-anchor="middle">P{p_}：{kb:.0f} KB</text>')
        free = 100 - n * kb
        f.w(f'<text class="s" x="{X1 + 6}" y="{y + 11}">放 {n} 个</text><text class="s" x="{X1 + 6}" y="{y + 24}">剩 {free:.0f} KB</text>')
    # 下：整个 kernel，一个 SM 上同时驻留的 program
    Yk = 158
    TMAX = 520000
    tx = lambda c: X0 + c / TMAX * (X1 - X0)
    gi = len(f.b)                                                                 # 网格线插在条的下面
    f.w(f'<text class="tb" x="8" y="{Yk - 12}">整个 kernel：一个 SM 上同时跑的 program（{NP} 个 program，{SMS} 个 SM）</text>')
    RH = 18
    def kbar(c0, c1, y, op, lab, tip):
        f.w(f'<g class="m"><title>{tip}</title><rect x="{tx(c0):.1f}" y="{y + 2}" width="{tx(c1) - tx(c0):.1f}" height="{RH - 4}" rx="2" style="fill: var(--fig-1)" fill-opacity="{op}"/></g>')
        f.w(f'<text class="s" x="{tx(c0) + 6:.1f}" y="{y + RH / 2 + 4}" style="fill: #fff; opacity: 1">{lab}</text>')
    # 3 级：3 行，一批
    y3 = Yk
    f.w(f'<text class="lab" x="8" y="{y3 + RH * 1.5 + 4}">num_stages = 3</text>')
    for r in range(3):
        kbar(0, tot[3], y3 + r * RH, .55, f"第 1 批 P{r}：64 轮 × {per[3]:.0f} 拍", f"每个 SM 3 个 program，{NP} 个一批跑完")
    # 4 级：2 行，两批
    y4 = y3 + 3 * RH + 22
    f.w(f'<text class="lab" x="8" y="{y4 + RH + 4}">num_stages = 4</text>')
    b1 = 64 * per[4]
    for r in range(2):
        kbar(0, b1, y4 + r * RH, .55, f"第 1 批 P{r}：64 轮 × {per[4]:.0f} 拍", f"第 1 批 {2 * SMS} 个 program，每个 SM 2 个")
    kbar(b1, tot[4], y4, .35, "第 2 批：64 轮", f"第 2 批 {NP - 2 * SMS} 个 program，每个 SM 最多 1 个")
    f.w(f'<rect x="{tx(b1):.1f}" y="{y4 + RH + 2}" width="{tx(tot[4]) - tx(b1):.1f}" height="{RH - 4}" rx="2" fill="none" stroke="currentColor" stroke-opacity=".45" stroke-dasharray="3 2"/>')
    f.w(f'<text class="s" x="{tx(b1) + 6:.1f}" y="{y4 + RH * 1.5 + 4}">空着</text>')
    # 对齐线：两种配置整个 kernel 结束的时刻
    yb = y4 + 2 * RH + 8
    for s, (ya, lab_y) in ((3, (y3, y3 + 3 * RH)), (4, (y4, y4 + 2 * RH))):
        f.w(f'<line x1="{tx(tot[s]):.1f}" y1="{Yk - 4}" x2="{tx(tot[s]):.1f}" y2="{yb}" style="stroke: var(--fig-hi)" stroke-width="1.4"{"" if s == 4 else " stroke-dasharray=\"4 3\""}/>')
    f.w(f'<text class="s" x="{tx(tot[3]) + 4:.1f}" y="{y3 + 3 * RH + 12}" style="fill: var(--fig-hi); opacity: 1">3 级结束：{tot[3]:.0f} 拍</text>')
    f.w(f'<text class="s" x="{tx(tot[4]) - 4:.1f}" y="{yb + 12}" text-anchor="end" style="fill: var(--fig-hi); opacity: 1">4 级结束：{tot[4]:.0f} 拍（{tot[4] / tot[3]:.2f} 倍）</text>')
    ya = yb + 30
    grid = [f'<line class="grid" x1="{X0}" y1="{ya - 12}" x2="{X1}" y2="{ya - 12}"/>']
    for c in range(0, TMAX + 1, 100000):
        grid.append(f'<line class="grid" x1="{tx(c):.1f}" y1="{Yk - 4}" x2="{tx(c):.1f}" y2="{ya - 12}"/>')
        f.w(f'<text class="tick" x="{tx(c):.1f}" y="{ya + 2}" text-anchor="middle">{f"{c / 1e5:.0f}e5" if c else "0"}</text>')
    f.b[gi:gi] = grid
    f.w(f'<text class="lab2" x="{X0 - 8}" y="{ya + 2}" text-anchor="end">拍</text>')
    return f

CAPS = {
 "hw_hierarchy": '<strong>图 1-5</strong> RTX 5090 的存储层级示意。规格来自 <a href="https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf">NVIDIA RTX Blackwell 白皮书</a>，官方的整芯片和 SM 结构图也在白皮书里。',
 "hw_move": '<strong>图 C-1</strong> 显存到 shared memory 的三种搬法走的路径。',
 "hw_sm": '<strong>图 1-1</strong> RTX 5090 一个 SM 的结构示意（SMSP 即 SM sub-partition）。每个 SMSP 有自己的 warp 调度器、register、32 个 CUDA core 和 1 个 Tensor core，4 个 SMSP 共享 L1 / shared memory。普通的 load / store 由 SMSP 里的 LD/ST 单元按每个 thread 的地址发出，经 L1 / shared memory、片上互联到 L2。TMA 是专门整块搬数据的单元，和 LD/ST 走同一条路，见 Appendix C。',
 "residency": '<strong>图 3-2</strong> 第 2 节四个 kernel 的 program 怎么放上一个 SM（RTX 5090，形状同正文，每个 program 4 个 warp）。上面：第 2 节图里 grid 的一格是一个 program，也就是一个 CUDA block，放上 SM 时要占四样东西。下面每个 kernel 四根条，对应<a href="#tab-3-1">表 3-1</a> 的四条上限，条长是这条上限的总量；深色一格是一个实际放进去的 program 占的量，虚线框是只看这一条还能再放的，红框是先用完的那条，它决定了一个 SM 放几个。register 和 shared memory 读自编译出的 kernel；register 以 warp 为单位按 256 个分配，摊到每个 thread 是 8 的倍数，shared memory 按 128 B 取整。四条取最小的结果和 <code>cuOccupancyMaxActiveBlocksPerMultiprocessor</code> 算出的一致。softmax 和 row sum 的 16 B shared memory 用来让 4 个 warp 交换归约的部分结果。',
 "tile_res": '<strong>图 3-6</strong> 正文形状（A 4096 × 2048，B 2048 × 512，fp32）下最快的配置和正文用的配置，在一个 SM 上各用了多少。四根条对应<a href="#tab-3-1">表 3-1</a> 的四条上限，条长是这条上限的总量。',
 "tile_rules": '<strong>图 3-7</strong> 两个形状各扫 198 组配置（RTX 5090，fp32，<code>input_precision="ieee"</code>）。左：按每个 thread 存的 acc 个数（BM × BN / (32 × num_warps)）分组；右：按每个 SM 驻留的 warp 数分组。每组取最快的一个，纵轴是它比这个形状下整体最快的配置慢几倍；有 spill 的配置不算在内。悬停可看时间。',
 "warp_slots": '<strong>图 3-1</strong> 一个 SM 上驻留 3 个 program（P0、P1、P2，每个 4 个 warp）时的某一拍。每个 program 的 4 个 warp 分到 4 个 SMSP，格子里写的是这个 warp 属于哪个 program；每个 SMSP 有 12 个格子，整个 SM 共 48 个，就是<a href="#tab-3-1">表 3-1</a> 的 warp 名额。每个调度器每拍最多发射一个 warp，SMSP 2 的 3 个 warp 都在等数据，这一拍空转。',
 "hw_smsp": '<strong>图 1-2</strong> RTX 5090 一个 SMSP 的结构示意，按 <a href="https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf">NVIDIA RTX Blackwell 白皮书</a>的图 5 画。Blackwell 的 CUDA core 既做 FP32 也做 INT32，同一拍只能做其中一种。',
 "hw_cores": '<strong>图 1-3</strong> CUDA core、Tensor core 和 SFU 各算什么（示意）。左：一条 FMA，warp 的每个 thread 用一个 CUDA core 算 a × b + c，乘积不舍入，最后只舍入一次。右：一条 mma，以 bf16 的 m16n8k16 为例，整个 warp 一起发，同样一条指令，算的量是 FMA 的 64 倍。下：SFU 用查表加二次插值近似 exp2、rcp、rsqrt、sin 这类函数（Oberman, Siu 2005），SASS 读自 5090 上编译出的 kernel，吞吐比是实测。',
 "cuda_model": '<strong>图 1-6</strong> CUDA 的执行层级。左边从上到下逐层放大：grid 里的一个 block、block 里的一个 warp、warp 里的一个 thread；中间是每一层跑在哪个硬件上，对应<a href="#fig-h-2">图 1-1</a>；右边是这一层在 Triton 里怎么写。',
 "op_split": '<strong>图 2-1</strong> 四个例子的数据怎么分给 program。深蓝是 pid = 1 的 program 要读的数据，红色是它写回的输出；row sum 和 matmul 要循环，实心是第 0 轮读的 block，斜线是之后几轮要读的。matmul 那格一格是一个 block。每个例子的完整数据流图在后面各小节。',
 "mm_tiling": '<strong>图 7-1</strong> matmul 怎么切 block，一格是一个 block。C 的 block (1, 1) 只用到 A 的第 1 个 row block 和 B 的第 1 个 column block；沿 K 一轮读一对 block，乘完加到累加器上，实心是第 0 轮，斜线是之后几轮。A block 里的每个数被 C block 里同一行的 BN 个输出用到，B block 里的每个数被同一列的 BM 个输出用到。',
 "opt_divergence": '<strong>图 1-1</strong> control divergence（按 8 条 lane 画，实际一个 warp 32 条），代码是 <code>if (lane &lt; 4) { A; B } else { X; Y }; Z</code>。虚线是这一段被屏蔽、没在执行的 lane。上：Pascal 及以前一个 warp 只有一个程序计数器，分支两边串行执行。中：Volta 起每个 thread 有自己的程序计数器，两边的语句可以交替执行，调度器也不保证在 Z 合起来；但每一拍仍然只执行一条指令，空着的 lane 一样多。下：分支按 warp 一致时没有空着的 lane。上、中两层的画法参考 <a href="https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf">Volta 白皮书</a> Figure 20、22 和 <a href="https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf">CS336 lecture 5</a>。',
 "opt_precision": '<strong>图 2-1</strong> 几种浮点格式的位分配。bf16 和 fp32 的指数位一样多，范围相同，只是尾数短；fp16 范围小得多，最大只到 65504；fp8 两种格式在范围和精度之间取舍，E4M3 精度高、E5M2 范围大。显存、L2 都按 128 B 的 cache line 搬数据，位数减半，同样的搬运量就多一倍的数。画法参考 CS336 lecture 5。',
 "opt_mx": '<strong>图 2-2</strong> block scaling 为什么转置以后要再量化一份（示意，一个 block 按 4 个数画，MX 格式是 32 个）。颜色深浅区分不同的 block，红色是每个 block 的 scale。W 按 PyTorch 的 Linear 存成 [N, K]。同一个 block 里的数共用一个 scale，所以 block 必须沿矩阵乘的归约方向连续：前向沿 K，反向算 dX 时沿 N，原来的 block 和 scale 用不上。',
 "opt_fusion": '<strong>图 3-1</strong> fusion 省掉中间结果的往返（示意，按 3 个 op 画）。eager 下每个 op 是一个 kernel，t1、t2 都要写回显存、下一个 kernel 再读回来；合成一个 kernel 以后，中间结果留在 register 里。画法参考 CS336 lecture 5。',
 "opt_recompute": '<strong>图 4-1</strong> 三层 sigmoid，存中间结果和反向重算各要几次显存读写。例子来自 <a href="https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467">PyTorch dev-discuss 的 min-cut recomputation 一文</a>，CS336 lecture 5 也用了它。sigmoid 的导数只要 sigmoid 自己的值，所以反向读 x 就能重算出 s1、s2。',
 "opt_coalescing": '<strong>图 5-1</strong> memory coalescing（按 8 个 thread 画，实际一个 warp 32 个）。红色是被读的数，浅蓝是同一个 sector 里顺带取上来、没用到的字节。',
 "opt_bank": '<strong>图 6-1</strong> bank conflict 的两种解法（按 8 个 bank、8 × 8 的矩阵画，实际是 32 个 bank）。一格是一个 4 B 的数，标的是它在矩阵里的 (行, 列)；同一列的格子在同一个 bank，同一个 bank 一拍只能读一个地址。红色是 8 个 thread 同时要读的第 0 列。padding 多占一点空间；swizzle 不多占空间，按行读、按列读都不冲突。',
 "tri_gelu": '<strong>图 2-2</strong> GELU kernel（n = 8388608，fp32）。左边是 grid，数组按 BLOCK = 1024 切段，一段一个 program。右边是 program 1 的数据流，泳道和箭头的含义见正文。显存泳道里的整条是整个 x、y（各 8388608 个 fp32，32 MB），深色一段是 program 1 读写的 1024 个（4 KB）；按比例只有整条的 1/8192，图里放大了。',
 "wave_trace": '<strong>图 3-2</strong> GELU 的 8192 个 program 实际怎么分批跑（RTX 5090，n = 8388608，fp32，每个 program 4 个 warp）。每个 program 开头读一次 <code>%globaltimer</code> 和 <code>%smid</code>，store 发出后再读一次时间，记录前先清空 L2。颜色由淡到浓是 program 编号所在的批次，一批 2040 个。浅红色是 bubble，最后一个 program 开始以后是 tail。下面的名额是按开始、结束时间排出来的，红框是这个 SM 上最长的一段空档。打点版 46.9 µs，不打点 45.2 µs。悬停可看 program 编号和时间。',
 "tri_softmax": '<strong>图 2-3</strong> softmax kernel（4096 × 4096，fp32），画法同<a href="#fig-t-1">图 2-2</a>。一行一个 program，中间结果都留在 register 里，显存只读写各一次。显存泳道里的方块是整个 x、y（各 64 MB），深色一条是 program 1 读写的那一行（4096 个 fp32，16 KB），按比例只有 1/4096，图里放大了。',
 "tri_rowsum": '<strong>图 2-4</strong> row sum kernel（4096 × 4096，fp32），画法同<a href="#fig-t-1">图 2-2</a>。图里 TILE = 4，一行 16 个数分 4 个 block，代码里 TILE = 1024。颜色由淡到浓是 for 循环读 block 的顺序：第 0 轮后 acc 只有 block 0 的数，最后的 acc 是 4 个 block 逐位相加 [7, 8, 4, 6]，铺满整条渐变；中间两轮用 ⋯ 省略，最后 <code>tl.sum</code> 加成 25 写回。',
 "tri_matmul": '<strong>图 2-5</strong> matmul kernel（A 4096 × 2048，B 2048 × 512，fp32，BM = BN = 64，BK = 32），画法同<a href="#fig-t-1">图 2-2</a>。显存里的矩阵按比例画全，一格是 256 × 256 个元素。program (1, 1) 读 A 的第 1 个 row block 和 B 的第 1 个 column block，颜色由淡到浓是沿 K 读 block 的顺序。acc 就是 C 的 block (1, 1)，64 轮 dot 在它上面一轮轮累加，ReLU 之后就是要写回的值，图里 ReLU 后的那段叫 C block。A、B 的 block 由 cp.async 从显存经 L2 搬进 shared memory，不经过 register。shared memory 里 A、B 各有 2 个 slot，因为默认 <code>num_stages=3</code>：第 T 轮的 block 在第 T − 2 轮算完时开始搬，第 T − 1 轮期间在路上，第 T 轮被读出来算；算完 slot 就空出来，接着搬第 T + 2 轮。所以偶数轮用 slot 0、奇数轮用 slot 1，算第 T 轮时另一个 slot 里是第 T + 1 轮，见 <a href="#stages">3.3 节</a>。',
 "mm_wave": '<strong>图 3-3</strong> matmul kernel 的 512 个 program 实际怎么分批跑（RTX 5090，fp32，A 4096 × 2048，B 2048 × 512，每个 SM 3 个 program），画法同<a href="#fig-t-5">图 3-2</a>。每个 program 开头读一次 <code>%globaltimer</code> 和 <code>%smid</code>，写回 C 之后再读一次时间，记录前先清空 L2。颜色由淡到浓是 program 的发射顺序所在的批次，一批 510 个。第 2 批只有 2 个，按比例只有 1 像素高，图里画成底边上的一条粗线并用红框圈出。打点版 210.8 µs，不打点 203.3 µs。悬停可看 program 编号和时间。',
 "tx_barrier": '<strong>图 C-2</strong> 两种 barrier 怎么等数据（示意，横轴是时间，长度不按比例；画法参考 <a href="https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf">H100 白皮书</a> Figure 20）。上下搬的是同样的 16 KB，显存带宽一样，所以放行得一样早；区别在 thread 的时间花在哪。上：asynchronous barrier 只数 thread 的 arrive，每个 thread 要自己搬完自己那份才 arrive，几乎没有空做别的。下：asynchronous transaction barrier 还数字节，发起 TMA 的 thread 先登记这次要等 16 KB，TMA 每写进 shared memory 一段就把字节数加到 barrier 上；thread 一开始就 arrive，搬运期间都在做别的活，thread 都 arrive 过、字节数也到齐才放行。白皮书说 A100 上等着的 thread 在 barrier 上空转，H100 上可以睡眠。',
 "stages_sm": '<strong>图 3-5</strong> num_stages = 3 和 4 时，一个 SM 放几个 program、整个 kernel 跑几批（RTX 5090，形状同<a href="#tab-3-2">表 3-2</a>）。上：一个 slot 存一轮的 A、B block，各 8 KB，共 16 KB；灰色是每个 program 保留的 1 KB。下：每轮的拍数和整个 kernel 的拍数都按 Nsight Compute 实测（<code>--clock-control none</code>），整个 kernel 的拍数 = kernel 耗时 × SM 时钟频率；4 级第 1 批按 64 轮 × 每轮拍数画，第 2 批画到 kernel 结束。',
 "pipe_slots": '<strong>图 3-4</strong> matmul 的循环按 shared memory slot 看（RTX 5090，fp32，A 4032 × 2048，B 2048 × 512，BM = BN = 64，BK = 32，4 个 warp）。1 级和 3 级时 504 个 program 一批跑完，4 级时每个 SM 只放得下 2 个，要分两批，这里画的是第一批里的一个 program。条长按 Nsight Compute 实测（<code>--clock-control none</code>）画，都是 64 轮的平均值：每轮的拍数 = 一个 warp 每条指令平均隔几拍 × 每轮的指令数；把这几拍按 warp 当时的状态拆开，等 L2 / 显存的记作「等显存」，自己已经就绪、但调度器这一拍发了别的 warp 的记作「让给别的 warp」，其余（发射、等 shared memory、barrier 等）记作「自己在跑」。三种状态在一轮里是穿插的，这里按比例拼在一起画。load 的延迟没法直接测，按不预取时每轮等数据的拍数画。三栏共用一根时间轴，都只画前 4 轮，红线是 4 轮算完的时刻。',
 "mm_pipe_ncu": '<strong>图 3-4</strong> matmul kernel 一个 warp 的流水线（RTX 5090，fp32，A 4032 × 2048，504 个 program 一批跑完；Nsight Compute 实测，<code>--clock-control none</code>）。每轮的拍数 = 每条指令平均隔几拍 × 每轮的指令数，等数据的拍数 = 每条指令平均等 L2 / 显存几拍 × 每轮的指令数，都是 64 轮的平均值。load 的延迟没法直接测，按不预取时每轮等数据的拍数画。只画前 5 轮。',
 "occ_sweep": '<strong>图 2-5</strong> 驻留的 warp 数和速度（RTX 5090，256 MiB 的 fp32 数组，每个元素读 4 B、写 4 B，中间串 K 次相互依赖的 FMA，I = 2K / 8 FLOPs/B）。grid 正好是 170 × m 个 program、每个 4 个 warp，全部同时驻留，每个 SMSP 就是 m 个 warp。最快值：前三条是显存带宽 1.4e12 B/s，I = 128 是 8.5e13 FLOPS（fp32 峰值的 81%）。悬停可看数值。',
 "tile_sweep": '<strong>图 2-6</strong> row sum 的 TILE 和带宽（RTX 5090，fp32，两种形状都是 340 MiB，<code>num_warps=4</code>，每点取 8 次计时的最快值）。悬停可看耗时和每线程的 register 数。',
 "tpu_core": '<strong>图 B-1</strong> TPU v5p 一个 TensorCore 的结构示意，画法对应<a href="#fig-h-1">图 1-5</a>。规格来自 <a href="https://jax-ml.github.io/scaling-book/tpus/">How to Scale Your Model</a>，原书也有一张 TensorCore 的结构图。',
 "systolic": '<strong>图 B-2</strong> 乘加单元怎么拿到数据。左边是 GPU 的 CUDA core，右边是 TPU 的 MXU；x 按行错开一拍进入，y 是每列的输出。动画演示见 <a href="https://jax-ml.github.io/scaling-book/tpus/#appendix-b-how-does-a-systolic-array-work">How to Scale Your Model 的 Appendix B</a>。',
 "linear": '<strong>图 2-1</strong> 一个 Linear 的前向与反向：前向从左边往下，误差从右边传回；虚线是反向要从前向拿的东西。',
 "step_time": '<strong>图 2-2</strong> 一步训练里前向、反向、optimizer 的耗时占比（fp32，batch 4，seq 512），右侧是每步耗时和 MFU。',
 "roofline": '<strong>图 1-1</strong> RTX 5090 各精度的 roofline：斜线是带宽，平线是峰值算力（dense，boost clock 2407 MHz，来自 NVIDIA RTX Blackwell 白皮书；Tensor core 按 fp32 累加）。',
 "roofline_ops": '<strong>图 3-5</strong> RTX 5090 fp32 的 roofline，以及 medium、seq 1024 时一层 attention 里实测的 op（另放一个 Linear 作对照）。causal mask 没有 FLOPs，不在图上；悬停可看数值。',
 "roofline_rms": '<strong>图 3-3</strong> RMSNorm 融合前后在 fp32 roofline 上的位置（前向，<code>x: [32, 512, 2560]</code>，160 MiB，比 L2 大，避免数据留在缓存里）。读写字节按 eager 每元素 28 B、融合后 8 B 算，悬停可看数值。',
 "attn_flow": '<strong>图 3-4</strong> eager attention 一层（medium，seq 1024，画法同<a href="#fig-3-1">图 3-1</a>）。框下是每块的大小：S、S/√d、S+M、S−m、e、P 都是 [b, h, seq, seq]，m 和 Σ 每行一个数。max 同时写出每行最大值的下标（int64），反向只用它，m 用完即释放。粗实线框是为反向新存下的，细实线框是本来就在、只被引用的 Q、K、V、mask，灰色虚线框是用完即释放的临时量。',
 "softmax": '<strong>图 3-5</strong> eager softmax 的显存读写：每条编号箭头是一次完整的读或写，共 8 次；融合后只剩 2 次。',
 "flops_vs_time": '<strong>图 3-6</strong> 一层 attention 里各 op 的 FLOPs 与实测 GPU 时间（medium，seq 1024）。',
 "share_vs_seq": '<strong>图 3-7</strong> attention 三段占 forward GPU 时间的比例随 seq 变化（medium）。',
 "peak_memory": '<strong>图 2-3</strong> full step 的实测峰值显存（batch 4，seq 512）。斜纹块是 A 和 G 轮流用的同一块显存：前向结束时装 A，反向时 A 一层层释放、G 一层层分配；G 和 W 一样大，比 A 小，所以这块的高度是 A。A = 带梯度的前向峰值 − W，顶上 ~0.1 GiB 是 T。',
 "peak_moment": '<strong>图 2-4</strong> 一步前向 + 反向的显存（fp32）：色带按 M(j) 用实测的 W、A、G、T 堆叠，× 是逐层实测值。',
 "rms_eager": '<strong>图 3-1</strong> RMSNorm（eager，x 是 20 MiB）：中间一排是显存里的张量，框宽按实际字节数线性画（KiB 级的只剩一条细线），每块只画一次，同色是同一块内存（ptr 相同）；粗实线框新占显存，细实线框本来就在、只被引用（x 是上一层的输出，w 是参数，都会一直留着），灰色虚线框是用完即释放的临时量。每个 kernel 画成两层框：外框是它的 register，里面是计算单元；实线箭头是数据流，从显存指向 register 是 load，从 register 指向显存是 store；反向沿虚线箭头把存下的张量读回 register。',
 "rms_fused": '<strong>图 3-2</strong> 融合后的 RMSNorm（画法同<a href="#fig-3-1">图 3-1</a>）：前向只读 x、w，写 r 和 y，x²、v、x̂ 都不进显存；反向读回 x、w、r，现场重算 x̂。',
 "layer_donut": '<strong>图 3-8</strong> xl 一层为反向存的张量（batch 4，seq 2048，16 头，<code>torch.compile</code> 后用 <code>saved_tensors_hooks</code> 实测）。',
 "timelines": '<strong>图 3-9</strong> xl（batch 4，32 头）一步的显存时间线，横轴是分配 / 释放的次序。',
 "bf16": '<strong>图 4-1</strong> bf16 autocast 相对 fp32（前向 + 反向，batch 4，seq 512）：左边是加速比，右边是峰值显存。',
 "ckpt": '<strong>图 4-2</strong> 4 层 xl block 有无 checkpoint：钢蓝框一直占到反向，浅蓝虚线框在反向时用 entry 重算、用完即丢。',
 "sweep": '<strong>图 4-3</strong> checkpoint 段长扫描（large，batch 1，seq 1024，前向 + 反向，fp32 eager）。',
}
OUT = {k: fn().html(CAPS[k]) for k, fn in (("hw_hierarchy", hw_hierarchy), ("hw_move", hw_move), ("hw_sm", hw_sm), ("hw_smsp", hw_smsp), ("hw_cores", hw_cores), ("residency", residency), ("tile_res", tile_res), ("tile_rules", tile_rules), ("warp_slots", warp_slots), ("occ_sweep", occ_sweep), ("tile_sweep", tile_sweep), ("wave_trace", wave_trace), ("op_split", op_split), ("mm_tiling", mm_tiling), ("opt_divergence", opt_divergence), ("opt_precision", opt_precision), ("opt_mx", opt_mx), ("opt_fusion", opt_fusion), ("opt_recompute", opt_recompute), ("opt_coalescing", opt_coalescing), ("opt_bank", opt_bank), ("tri_gelu", tri_gelu), ("tri_softmax", tri_softmax), ("tri_rowsum", tri_rowsum), ("tri_matmul", tri_matmul), ("mm_wave", mm_wave), ("mm_pipe_ncu", mm_pipe_ncu), ("pipe_slots", pipe_slots), ("stages_sm", stages_sm), ("tx_barrier", tx_barrier), ("cuda_model", cuda_model), ("tpu_core", tpu_core), ("systolic", systolic), ("linear", linear), ("step_time", step_time), ("roofline", roofline), ("roofline_ops", roofline_ops), ("roofline_rms", roofline_rms), 
       ("flops_vs_time", flops_vs_time), ("share_vs_seq", share_vs_seq), ("peak_memory", peak_memory), ("peak_moment", peak_moment),
       ("rms_eager", rms_eager), ("rms_fused", rms_fused), ("attn_flow", attn_flow), ("layer_donut", layer_donut), ("timelines", timelines),
       ("bf16", bf16), ("ckpt", ckpt), ("sweep", sweep))}
json.dump(OUT, open(os.path.join(HERE, "figs.json"), "w"), ensure_ascii=False)
print(len(OUT), "figures")
