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
    box(190, 182, 150, 54, "计算", "X_L = X_{L−1} · W_L", A)
    box(44, 182, 68, 50, "权重", "W_L", W)
    box(176, 288, 150, 50, "输出", "X_L", A)
    f.arrow(176, 101, 176, 152); f.arrow(78, 182, 112, 182); f.arrow(176, 209, 176, 260)
    box(320, 398, 220, 44, "进入深层 Layer L+1，等误差传回", None, 'class="op"')
    f.path("M176,313 L176,398 L207,398"); f.path("M430,398 L482,398 L482,316")
    box(482, 288, 150, 50, "接收误差", "∇X_L", T)
    box(404, 182, 152, 54, "计算参数梯度", "∇W_L = X_{L−1}^T · ∇X_L", G)
    box(562, 182, 152, 54, "计算激活梯度", "∇X_{L−1} = ∇X_L · W_L^T", T)
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
    return _roofline("fig-4-2", "RTX 5090 fp32 的 roofline 和一层 attention 里实测的 op：attention 的 op 都在斜线上，只有作对照的 Linear 在平线下", [("fp32", 1.048e14, "峰值 π = 1.05e14 FLOPS", True)], None, 14.6)

def _roofline(fid, label, ROOFS, OPS, YHI):
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
    place = {"Linear": (-10, 18, "end"), "QKᵀ": (10, 14, "start"), "PV": (10, -2, "start"), "softmax": (10, 16, "start"), "S / √d": (10, 12, "start")}
    for key, full, fl, by, ms, util in OPS:
        i, v = fl / by, fl / (ms * 1e-3)
        x, y = xp(i), yp(v)
        f.w(f'<g class="m"><title>{full}：I = {i:.3g} FLOPs/B，实测 {fmt(v)} FLOPS，{util}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="transparent"/><circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="ring" style="fill: var(--fig-hi)"/></g>')
        dx, dy, anc = place[key]
        f.w(f'<text class="lab" x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anc}">{key}</text>')
    return f


# ── 图 4-1 attention 的前向、显存读写、saved tensors 和反向 ─────────────────────
def attn_flow():
    f = F("fig-4-1", 350, "eager attention 一层：前向 9 个 kernel 依次读写显存里 S 大小的张量，softmax 拆成 max、减 max、exp、求和、除；为反向新存了 e = exp(S−m) 和 P 两个 256 MiB 的张量，Q、K、V、mask 只是引用", W=800)
    YF, YM, YB = 70, 196, 290
    LANES = [(6, 112, "前向", "箭头指向 op 是读，指向显存是写"), (148, 240, "显存", ""), (252, 344, "反向", "")]
    rms_lanes(f, LANES, titles=False)
    X = [140 + 67 * i + (34 if i > 3 else 0) for i in range(9)]
    names = ["① QKᵀ", "② ÷√d", "③ +M", "max", "− m", "exp", "求和", "÷ Σ", "⑤ PV"]
    ops_f = [(x, 60, n, None) for x, n in zip(X, names)]
    ops_b = [(x, 60, n) for x, n in zip(X, names)]
    mid = [(X[k] + X[k + 1]) / 2 for k in range(8)]
    REF2, TMP = blk("--fig-2", "ref"), blk(None, "tmp")
    blocks = {"M": (18, 30, "M", "引用", REF2, "--fig-2"),
              "Q": (54, 40, "Q", "引用", REF2, "--fig-2"), "K": (90, 40, "K", "引用", REF2, "--fig-2"),
              "S": (mid[0], 54, "S", "临时", TMP, "--fig-mute"), "S2": (mid[1], 54, "S/√d", "临时", TMP, "--fig-mute"),
              "S3": (mid[2], 54, "S+M", "临时", TMP, "--fig-mute"), "idx": (X[3] + 6, 52, "下标", "+0.5 MiB", blk("--fig-3", "new"), "--fig-3"),
              "m": (X[4] - 30, 38, "m", "临时", TMP, "--fig-mute"),
              "Sm": (mid[4], 54, "S−m", "临时", TMP, "--fig-mute"), "e": (mid[5], 60, "e", "+256 MiB", blk("--fig-hi", "new"), "--fig-hi"),
              "Z": (mid[6], 44, "Σ", "每行 1 个", blk("--fig-3", "new"), "--fig-3"), "P": (mid[7], 60, "P", "+256 MiB", blk("--fig-1", "new"), "--fig-1"),
              "V": (774, 36, "V", "引用", REF2, "--fig-2")}
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
    f = F("fig-4-3", 262, "eager softmax 的 5 个 kernel 共读写显存里 S 大小的张量 8 次；融合成一个 kernel 后只读 S、写 P 两次")
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
    f = F("fig-4-3", 186, "medium、seq 1024 一层 attention 里各 op 的 FLOPs 与实测 GPU 时间：softmax 和 ÷√d、mask 的 FLOPs 很少，时间却最多")
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
    f = F("fig-4-4", 262, "attention 三段占 forward 时间随 seq 的变化：softmax 从 4% 涨到 24%，scores 从 4% 涨到 18%，PV 只从 2% 到 4%，合计从 10% 到 46%")
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
    f = F("fig-2-3", 300, "三档模型 full step 的实测峰值显存，由权重、Adam 状态和 activation 组成，里面没有梯度")
    legend(f, [("W 权重", "--fig-1", "box"), ("Adam m、v", "--fig-mute", "box"), ("A activation", "--fig-2", "box"), ("T 临时量", "--fig-3", "box")], 70, 20, gap=16)
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
    yo, yt, yb, yob = YF + 20, YM - 22, YM + (32 if sizes else 22), YB - 18
    otop, bbot = YF - 16, YB + 18
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
    for cx, w, t, sub in ops_f:
        tbox(f, cx, YF + 4, w, 40, t, sub, 'class="op"', b1=True)
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
        bbox(f, cx, YB, w, 36, t)

def blk(var, kind):
    """kind: new (thick solid, new memory), ref (thin solid, already there), tmp (gray dashed, freed after use)."""
    if kind == "tmp":
        return tint("--fig-mute", 6, dash=True)
    return tint(var, 16 if kind == "new" else 10, sw=2.2 if kind == "new" else 1.2)

def rms_eager():
    f = F("fig-3-1", 350, "RMSNorm eager：中间一排是显存里的每一块张量，前向箭头表示读写，反向虚线箭头表示读回存下的张量；x 被 ① 和 ④ 共用，r 被 ③ 和 ④ 共用")
    YF, YM, YB = 70, 196, 290
    LANES = [(6, 112, "前向", "箭头指向 op 是读，指向显存是写"), (148, 240, "显存", ""), (252, 344, "反向", "")]
    rms_lanes(f, LANES, titles=False)
    C, BW = [100, 210, 320, 430, 540], 84
    ops_f = [(C[0], BW, "① x²", "1 FLOP/元素"), (C[1], BW, "② mean", "1 FLOP/元素"), (C[2], BW, "③ rsqrt", "每行 2 FLOPs"),
             (C[3], BW, "④ x · r", "1 FLOP/元素"), (C[4], BW, "⑤ w ⊙ x̂", "1 FLOP/元素")]
    ops_b = [(c, BW, f"{n} 反向") for c, n in zip(C, "①②③④⑤")]
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
    LANES = [(6, 98, "前向", "箭头指向 op 是读，指向显存是写"), (128, 224, "显存", ""), (254, 330, "反向", "")]
    rms_lanes(f, LANES, titles=False)
    ops_f = [(330, 420, "fused forward：①–⑤ 一个 kernel", "~4 FLOPs/元素")]
    ops_b = [(330, 420, "反向 3 个 kernel（dx 1 个，dw 2 个），x̂ = x · r 现场重算")]
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
    f = F("fig-4-5", 246, "xl 一层为反向存的 3655 MiB：S、P 占 56%，FFN 中间量 26%，[b, s, d] 级张量 17.5%，其他 0.2%")
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
    f = F("fig-4-6", 420, "xl 一步的显存时间线：seq 128 纯前向是平的；seq 2048 纯前向每层冲出一个尖峰；seq 128 full step 前向和反向一路上升，到 optimizer 时 OOM；seq 2048 带反向时第 2 层 OOM")
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
    f = F("fig-5-1", 252, "bf16 autocast 相对 fp32：前向快 1.87 到 2.30 倍，反向快 1.69 到 1.87 倍；前向 + 反向的峰值显存少 18% 到 21%")
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
    f = F("fig-5-2", 340, "4 层 xl block：不 checkpoint 时每层留 3655 MiB 一起活到反向，峰值 14.6 GiB；每 2 层一个 checkpoint 时只留 entry x0、x2，反向时逐段重算，峰值 7.5 GiB")
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
    f = F("fig-5-3", 250, f"checkpoint 段长扫描：step 时间都在 302 到 313 ms，高于不 checkpoint 的 {base['avg_ms']:.0f} ms；峰值显存随每段层数增加，每层一个 checkpoint 时最低 7.8 GiB")
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

CAPS = {
 "linear": '<strong>图 2-1</strong> 一个 Linear 的前向与反向：前向从左边往下，误差从右边传回；虚线是反向要从前向拿的东西。',
 "step_time": '<strong>图 2-2</strong> 一步训练里前向、反向、optimizer 的耗时占比（fp32，batch 4，seq 512），右侧是每步耗时和 MFU。',
 "roofline": '<strong>图 1-1</strong> RTX 5090 各精度的 roofline：斜线是带宽，平线是峰值算力（dense，boost clock 2407 MHz，来自 NVIDIA RTX Blackwell 白皮书；Tensor core 按 fp32 累加）。',
 "roofline_ops": '<strong>图 4-2</strong> RTX 5090 fp32 的 roofline，以及 medium、seq 1024 时一层 attention 里实测的 op（另放一个 Linear 作对照）。causal mask 没有 FLOPs，不在图上；悬停可看数值。',
 "attn_flow": '<strong>图 4-1</strong> eager attention 一层（medium，seq 1024，画法同<a href="#fig-3-1">图 3-1</a>）。S、S/√d、S+M、S−m、e、P 都是 [b, h, seq, seq]，各 256 MiB；Q、K、V 各 16 MiB，mask 1 MiB，m 和 Σ 每行一个数（256 KiB）。max 同时写出每行最大值的下标（int64，0.5 MiB），反向只用它，m 用完即释放。粗实线框是为反向新存下的，细实线框是本来就在、只被引用的 Q、K、V、mask，灰色虚线框是用完即释放的临时量。',
 "softmax": '<strong>图 4-2</strong> eager softmax 的显存读写：每条编号箭头是一次完整的读或写，共 8 次；融合后只剩 2 次。',
 "flops_vs_time": '<strong>图 4-3</strong> 一层 attention 里各 op 的 FLOPs 与实测 GPU 时间（medium，seq 1024）。',
 "share_vs_seq": '<strong>图 4-4</strong> attention 三段占 forward GPU 时间的比例随 seq 变化（medium）。',
 "peak_memory": '<strong>图 2-3</strong> full step 的实测峰值显存（batch 4，seq 512）。A = 带梯度的前向峰值 − W，顶上 ~0.1 GiB 是 T。',
 "peak_moment": '<strong>图 2-4</strong> 一步前向 + 反向的显存（fp32）：色带按 M(j) 用实测的 W、A、G、T 堆叠，× 是逐层实测值。',
 "rms_eager": '<strong>图 3-1</strong> RMSNorm（eager，x 是 20 MiB）：中间一排是显存里的张量，框宽按实际字节数线性画（KiB 级的只剩一条细线），每块只画一次，同色是同一块内存（ptr 相同）；粗实线框新占显存，细实线框本来就在、只被引用（x 是上一层的输出，w 是参数，都会一直留着），灰色虚线框是用完即释放的临时量。前向的实线箭头指向 op 是读、指向显存是写；反向沿虚线箭头读回存下的张量。',
 "rms_fused": '<strong>图 3-2</strong> 融合后的 RMSNorm（画法同<a href="#fig-3-1">图 3-1</a>）：前向只读 x、w，写 r 和 y，x²、v、x̂ 都不进显存；反向读回 x、w、r，现场重算 x̂。',
 "layer_donut": '<strong>图 4-5</strong> xl 一层为反向存的张量（batch 4，seq 2048，16 头，<code>torch.compile</code> 后用 <code>saved_tensors_hooks</code> 实测）。',
 "timelines": '<strong>图 4-6</strong> xl（batch 4，32 头）一步的显存时间线，横轴是分配 / 释放的次序。',
 "bf16": '<strong>图 5-1</strong> bf16 autocast 相对 fp32（前向 + 反向，batch 4，seq 512）：左边是加速比，右边是峰值显存。',
 "ckpt": '<strong>图 5-2</strong> 4 层 xl block 有无 checkpoint：钢蓝框一直占到反向，浅蓝虚线框在反向时用 entry 重算、用完即丢。',
 "sweep": '<strong>图 5-3</strong> checkpoint 段长扫描（large，batch 1，seq 1024，前向 + 反向，fp32 eager）。',
}
OUT = {k: fn().html(CAPS[k]) for k, fn in (("linear", linear), ("step_time", step_time), ("roofline", roofline), ("roofline_ops", roofline_ops), 
       ("flops_vs_time", flops_vs_time), ("share_vs_seq", share_vs_seq), ("peak_memory", peak_memory), ("peak_moment", peak_moment),
       ("rms_eager", rms_eager), ("rms_fused", rms_fused), ("attn_flow", attn_flow), ("layer_donut", layer_donut), ("timelines", timelines),
       ("bf16", bf16), ("ckpt", ckpt), ("sweep", sweep))}
json.dump(OUT, open(os.path.join(HERE, "figs.json"), "w"), ensure_ascii=False)
print(len(OUT), "figures")
