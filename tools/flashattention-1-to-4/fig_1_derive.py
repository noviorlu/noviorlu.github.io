"""1.1–1.2 推导用的示意图：图 1-2、1-3、1-4、1-6、1-7、1-8。每张图对应正文里的一个公式块。
配色和图 1-1 一致：深蓝 = 这一轮新读的 block，灰 = 前面的 block 留下的，浅蓝 = 这一轮算出来的，红 = 输出；红框标同一行（一个 query）。
用法：python3 tools/flashattention-1-to-4/fig_1_derive.py content/blog/flashattention-1-to-4/index.md
会把文中对应 <figure id="..."> 里的 <svg> 换掉，figcaption 不动。
"""
import re, sys

# ---- 文字：_{下标}、^{上标}，上标里可以再套 _{} ----
OFF = {'n': 0, 'sub': 3.5, 'sup': -5, 'supsub': -2}
FS = {'sub': '72%', 'sup': '72%', 'supsub': '58%'}
def parse(t):
    out, buf, i = [], '', 0
    def add(x, l):
        if x: out.append((x, l))
    while i < len(t):
        if t[i] in '_^' and i + 1 < len(t) and t[i + 1] == '{':
            add(buf, 'n'); buf = ''
            j, d = i + 2, 1
            while d:
                d += {'{': 1, '}': -1}.get(t[j], 0); j += 1
            inner = t[i + 2:j - 1]
            if t[i] == '_':
                add(inner, 'sub')
            else:
                k, b = 0, ''
                while k < len(inner):
                    if inner[k] == '_' and inner[k + 1] == '{':
                        add(b, 'sup'); b = ''
                        e = inner.index('}', k); add(inner[k + 2:e], 'supsub'); k = e + 1
                    else:
                        b += inner[k]; k += 1
                add(b, 'sup')
            i = j
        else:
            buf += t[i]; i += 1
    add(buf, 'n')
    return out

STYLE = {
    'old': 'fill: color-mix(in srgb, var(--fig-mute) 24%, transparent); stroke: var(--fig-mute)',
    'new': 'fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)',
    'res': 'fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)',
    'out': 'fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)',
    'skip': 'fill: none; stroke: var(--fig-mute)',
}
CSS = ('  <style>\n    .os2 { font-size: 11px; fill: currentColor; }\n    .os2-b { font-size: 12px; font-weight: 600; fill: currentColor; }\n'
       '    .os2-s { font-size: 10.5px; fill: currentColor; opacity: .72; }\n    .os2-m { font-size: 13px; fill: currentColor; }\n'
       '    .os2-hi { fill: var(--fig-hi); }\n  </style>\n')

class Fig:
    def __init__(self, fid):
        self.fid, self.B = fid, []
    def w(self, s): self.B.append(s)
    def T(self, cls, x, y, t, anchor='start', style=''):
        cur, h = 0, ''
        for k, (s, l) in enumerate(parse(t)):
            s = s.replace('&', '&amp;').replace('<', '&lt;')
            if k == 0 and l == 'n':
                h += s; continue
            dy = OFF[l] - cur; cur = OFF[l]
            a = f' dy="{dy:g}"' if dy else ''
            f = f' font-size="{FS[l]}"' if l in FS else ''
            h += f'<tspan{a}{f}>{s}</tspan>'
        st = f' style="{style}"' if style else ''
        an = f' text-anchor="{anchor}"' if anchor != 'start' else ''
        self.w(f'<text class="{cls}" x="{x:g}" y="{y:g}"{an}{st}>{h}</text>')
    def rect(self, x, y, w, h, kind, label=None, cls='os2', rx=1.5):
        d = ' stroke-dasharray="3 2"' if kind == 'skip' else ''
        self.w(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}" style="{STYLE[kind]}" stroke-width="1"{d}/>')
        if label: self.T(cls, x + w / 2, y + h / 2 + 4, label, 'middle')
    def mat(self, x, y, r, c, kind, cs, name=None, hl=None):
        for i in range(r):
            for j in range(c):
                self.rect(x + j * cs, y + i * cs, cs, cs, kind)
        if hl is not None:
            self.w(f'<rect x="{x - 1.5:g}" y="{y + hl * cs - 1.5:g}" width="{c * cs + 3:g}" height="{cs + 3:g}" rx="2" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6"/>')
        if name: self.T('os2', x + c * cs / 2, y + r * cs + 15, name, 'middle')
        return x + c * cs
    def arrow(self, x1, y1, x2, y2, hi=False):
        st = 'style="stroke: var(--fig-hi)"' if hi else 'stroke="currentColor" stroke-opacity=".55"'
        m = f'{self.fid}-arh' if hi else f'{self.fid}-ar'
        self.w(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" {st} stroke-width="1.4" marker-end="url(#{m})"/>')
    def bracket(self, x1, x2, y, up=False, hi=False):
        v = 5 if up else -5
        st = 'style="stroke: var(--fig-hi)"' if hi else 'stroke="currentColor" stroke-opacity=".5"'
        self.w(f'<path d="M{x1:g},{y + v:g} V{y:g} H{x2:g} V{y + v:g}" fill="none" {st} stroke-width="1.2"/>')
    def svg(self, H, label, W=680):
        f = self.fid
        defs = (f'<defs><marker id="{f}-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="currentColor" fill-opacity=".6"/></marker>'
                f'<marker id="{f}-arh" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" style="fill: var(--fig-hi)"/></marker></defs>\n')
        return (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{label}">\n' + CSS + defs
                + '\n'.join(self.B) + '\n</svg>')

def legend(F, x, y, items):
    for k, (kind, txt) in enumerate(items):
        if kind == 'row':
            F.w(f'<rect x="{x}" y="{y - 9}" width="12" height="12" rx="2" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6"/>')
        else:
            F.rect(x, y - 9, 12, 12, kind)
        F.T('os2-s', x + 17, y + 1, txt)
        t = txt.replace('_{', '').replace('}', '')
        x += 17 + sum(10.5 if ord(ch) > 0x2E80 else 4.6 for ch in t) + 20

FIGS = {}

# ===================== 图 1-2：一行要看完 =====================
F = Fig('fig-1-2')
labs = ['1', '⋯', 'j − 1', 'j', 'j + 1', '⋯', 'T_{k}']
CW, CH, BG = 20, 24, 10
X0, Y0 = 104, 64
bx = lambda b: X0 + b * (3 * CW + BG)
for b in range(7):
    kind = 'old' if b < 3 else ('new' if b == 3 else 'skip')
    for e in range(3):
        F.rect(bx(b) + e * CW, Y0, CW, CH, kind)
    F.T('os2', bx(b) + 1.5 * CW, Y0 + CH + 15, labs[b], 'middle')
F.T('os2-s', X0 - 10, Y0 + CH + 15, 'j =', 'end')
F.T('os2-b', X0 - 10, Y0 + 16, 'S 的一行', 'end')
XE = bx(6) + 3 * CW
F.bracket(X0, XE, Y0 - 8)
F.T('os2', (X0 + XE) / 2, Y0 - 18, 'm = max_{k} s_{k}，ℓ = Σ_{k} e^{s_{k} − m}：要看完这一行的全部 N_{k} 个 score', 'middle')
F.T('os2-s', (bx(0) + bx(3) - BG) / 2, Y0 + CH + 32, '已经读过', 'middle')
F.T('os2-s', (bx(4) + XE) / 2, Y0 + CH + 32, '还没读，也还没算', 'middle')
cx = bx(3) + 1.5 * CW
F.arrow(cx, Y0 + CH + 40, cx, Y0 + CH + 66, hi=True)
F.T('os2', cx, Y0 + CH + 82, '这一轮想算 p_{k} = e^{s_{k} − m} / ℓ，但 m、ℓ 都要等最后一个 block 读完才知道', 'middle', 'fill: var(--fig-hi)')
Y1 = Y0 + CH + 116
F.T('os2-b', 16, Y1, '按定义算要扫三遍')
for k, t in enumerate(['求 m', '求 ℓ，要用 m', '算 p_{k}，乘 V 加进 O，要用 m 和 ℓ']):
    y = Y1 + 28 + k * 30
    F.T('os2-s', X0 - 10, y + 4, f'第 {k + 1} 遍', 'end')
    F.arrow(X0, y, XE, y)
    F.T('os2', (X0 + XE) / 2, y - 5, t, 'middle')
FIGS['fig-1-2'] = F.svg(Y1 + 28 + 2 * 30 + 16,
    'S 的一行分在 T_k 个 block 里：前面几个已经读过，第 j 个是这一轮读的，后面的还没读。m 和 ℓ 要看完整行，所以这一轮算不出 p_k。按定义要扫三遍：求 m、求 ℓ、算 p_k 并乘 V')

# ===================== 图 1-3：两段合并 =====================
F = Fig('fig-1-3')
CW, CH, G = 58, 26, 16
X0 = (680 - (8 * CW + G)) / 2
cxk = lambda k: X0 + k * CW + (G if k >= 4 else 0)
A0, A1, B0, B1 = cxk(0), cxk(3) + CW, cxk(4), cxk(7) + CW
YS, YE, YC = 40, 92, 220
for k in range(8):
    kind = 'old' if k < 4 else 'new'
    seg = 'A' if k < 4 else 'B'
    F.rect(cxk(k), YS, CW, CH, kind, f's_{{{k + 1}}}')
    F.rect(cxk(k), YE, CW, CH, kind, f'e^{{s_{{{k + 1}}} − m_{{{seg}}}}}')
    F.rect(cxk(k), YC, CW, CH, 'res', f'e^{{s_{{{k + 1}}} − m}}')
F.T('os2-b', (A0 + A1) / 2, YS - 8, 'A 段', 'middle'); F.T('os2-b', (B0 + B1) / 2, YS - 8, 'B 段', 'middle')
F.T('os2-s', X0 - 8, YS + 17, 'score', 'end')
F.T('os2-s', X0 - 8, YE + 12, '各按', 'end'); F.T('os2-s', X0 - 8, YE + 25, '各的 m', 'end')
F.T('os2-s', X0 - 8, YC + 12, '换到', 'end'); F.T('os2-s', X0 - 8, YC + 25, '同一个 m', 'end')
for a, b, s in ((A0, A1, 'A'), (B0, B1, 'B')):
    F.bracket(a, b, YE + CH + 8, up=False)
    F.w(f'<path d="M{a:g},{YE + CH + 3:g} V{YE + CH + 8:g} H{b:g} V{YE + CH + 3:g}" fill="none" stroke="currentColor" stroke-opacity=".5" stroke-width="1.2"/>')
    F.T('os2', (a + b) / 2, YE + CH + 24, f'm_{{{s}}} = {s} 段的最大值', 'middle')
    F.T('os2', (a + b) / 2, YE + CH + 40, f'ℓ_{{{s}}} = 这四项之和', 'middle')
    F.arrow((a + b) / 2, YE + CH + 48, (a + b) / 2, YC - 4, hi=True)
    F.T('os2', (a + b) / 2 + 8, YC - 22, f'× e^{{m_{{{s}}} − m}}', 'start', 'fill: var(--fig-hi)')
F.T('os2-b', (A1 + B0) / 2, YC - 36, 'm = max(m_{A}, m_{B})', 'middle')
F.w(f'<path d="M{A0:g},{YC + CH + 3:g} V{YC + CH + 8:g} H{B1:g} V{YC + CH + 3:g}" fill="none" stroke="currentColor" stroke-opacity=".5" stroke-width="1.2"/>')
F.T('os2', (A0 + B1) / 2, YC + CH + 26, 'ℓ = e^{m_{A} − m} ℓ_{A} + e^{m_{B} − m} ℓ_{B} = Σ_{k} e^{s_{k} − m}：就是按定义算的整行分母', 'middle')
FIGS['fig-1-3'] = F.svg(YC + CH + 40,
    '一行 8 个 score 分成 A、B 两段。各段先按自己的最大值 m_A、m_B 算指数并求和，得到 ℓ_A、ℓ_B。再各乘修正因子 e 的 m_A 减 m 次方和 e 的 m_B 减 m 次方，换到同一个最大值 m，相加就是整行的分母')

# ===================== 图 1-4：第 j 轮的 m、ℓ =====================
F = Fig('fig-1-4')
CS, R = 16, 1
legend(F, 24, 18, [('old', '前 j − 1 个 block 留下的'), ('new', '这一轮的 S_{ij}'), ('res', '这一轮算出来的'), ('row', '同一行（一个 query）')])
def line(y, parts, note):
    x = 24
    for p in parts:
        if isinstance(p, str):
            F.T('os2-m', x + 2, y + 2 * CS + 5, p); x += len(p) * 7.2 + 8 if p.isascii() else len(p) * 13 + 8
        else:
            kind, r, c, name = p
            x = F.mat(x, y, r, c, kind, CS, name, hl=R) + 10
    for k, t in enumerate(note):
        F.T('os2' if k == 0 else 'os2-s', 360, y + 18 + k * 17, t)
line(48, [('res', 4, 1, 'm_{j}'), '=', 'max(', ('old', 4, 1, 'm_{j−1}'), ',', 'rowmax(', ('new', 4, 4, 'S_{ij}'), '))'],
     ['m_{j}：前 j 个 block 里每一行的最大值', '第 r 行只看 S_{ij} 的第 r 行'])
line(148, [('res', 4, 4, 'P̃_{ij}'), '=', 'exp(', ('new', 4, 4, 'S_{ij}'), '−', ('res', 4, 1, 'm_{j}'), ')'],
     ['m_{j} 的第 r 个数从 S_{ij} 第 r 行的每个元素里减掉', 'P̃_{ij}：还没除以分母的 softmax'])
line(248, [('res', 4, 1, 'ℓ_{j}'), '=', ('res', 4, 1, 'e^{m_{j−1} − m_{j}}'), '·', ('old', 4, 1, 'ℓ_{j−1}'), '+', 'rowsum(', ('res', 4, 4, 'P̃_{ij}'), ')'],
     ['第一项：旧的和每行乘自己的修正因子，换到新的最大值', '第二项：新 block 每行加起来'])
FIGS['fig-1-4'] = F.svg(248 + 4 * CS + 28,
    '第 j 轮 m 和 ℓ 的更新，B_q 等于 4。m_j 等于 m_{j−1} 和 S_ij 每行最大值中大的那个；P̃_ij 等于 S_ij 每行减去 m_j 的对应数再取 exp；ℓ_j 等于修正因子乘 ℓ_{j−1} 加上 P̃_ij 每行之和。红框标出同一行，每一行只和自己的那个数运算')

# ===================== 图 1-6：把分母提出来 =====================
F = Fig('fig-1-6')
WC, HC, VS = 68, 26, 20
def row_eq(y, pre, cells, kind_w, post, res_kind, res_name, wname):
    x = 24
    if pre:
        F.T('os2-m', x, y + 2 * VS + 5, pre); x += 46
    wy = y + 2 * VS - HC / 2
    for k, t in enumerate(cells):
        F.rect(x + k * WC, wy, WC, HC, kind_w, t)
    F.T('os2', x + 2 * WC, wy + HC + 15, wname, 'middle')
    x += 4 * WC + 12
    F.T('os2-m', x, y + 2 * VS + 5, '×'); x += 20
    F.mat(x, y, 4, 3, 'new', VS, 'V')
    for k in range(4): F.T('os2-s', x + 3 * VS + 5, y + k * VS + 14, f'v_{{{k + 1}}}')
    x += 3 * VS + 28
    F.T('os2-m', x, y + 2 * VS + 5, '='); x += 20
    if post:
        F.T('os2-m', x, y + 2 * VS + 5, post); x += 46
    F.mat(x, y + 2 * VS - VS / 2, 1, 3, res_kind, VS, res_name)
row_eq(20, '', [f'e^{{s_{{{k}}} − m}} / ℓ' for k in range(1, 5)], 'res', '', 'out', 'o', 'P 的一行：p_{1} … p_{4}')
F.arrow(150, 124, 150, 150, hi=True)
F.T('os2', 160, 141, 'ℓ 对每一项都一样，提到外面', 'start', 'fill: var(--fig-hi)')
row_eq(156, '1/ℓ ×', [f'e^{{s_{{{k}}} − m}}' for k in range(1, 5)], 'res', '1/ℓ ×', 'res', 'õ', '没除以 ℓ 的权重')
F.T('os2', 24, 270, 'õ = Σ_{k} e^{s_{k} − m} v_{k}：没除以 ℓ 的输出。õ 和 ℓ 分开算，最后 o = õ / ℓ')
FIGS['fig-1-6'] = F.svg(284,
    '上：o 等于 P 的一行乘 V，每个权重是 e 的 s_k 减 m 次方除以 ℓ。下：ℓ 对每一项都一样，提到外面，o 等于 1/ℓ 乘 õ，õ 是没除以 ℓ 的权重乘 V')

# ===================== 图 1-7：ℓ 和 Õ 并排 =====================
F = Fig('fig-1-7')
CS = 14
legend(F, 24, 18, [('old', '前 j − 1 个 block 留下的'), ('new', '这一轮读的 V_{j}'), ('res', '这一轮算出来的'), ('row', '同一行')])
def eq(y, parts):
    x = 24; pos = {}
    for p in parts:
        if isinstance(p, str):
            F.T('os2-m', x + 2, y + 2 * CS + 5, p); x += len(p) * 7.2 + 8
        else:
            key, kind, r, c, name = p
            pos[key] = (x, x + c * CS)
            x = F.mat(x, y, r, c, kind, CS, name, hl=R) + 10
    return pos
p1 = eq(48, [('l', 'res', 4, 1, 'ℓ_{j}'), '=', ('f', 'res', 4, 1, 'e^{m_{j−1} − m_{j}}'), '·', ('o', 'old', 4, 1, 'ℓ_{j−1}'), '+', 'rowsum(', ('p', 'res', 4, 4, 'P̃_{ij}'), ')'])
p2 = eq(150, [('l', 'res', 4, 3, 'Õ_{j}'), '=', 'diag(', ('f', 'res', 4, 1, 'e^{m_{j−1} − m_{j}}'), ')', ('o', 'old', 4, 3, 'Õ_{j−1}'), '+', ('p', 'res', 4, 4, 'P̃_{ij}'), ('v', 'new', 4, 3, 'V_{j}')])
F.T('os2', 410, 66, '旧的和：每行乘自己的修正因子', 'start')
F.T('os2-s', 410, 83, '新 block：P̃_{ij} 每行的元素加起来', 'start')
F.T('os2', 410, 168, '旧的和：每行乘同一个修正因子，和 ℓ 用的一样', 'start')
F.T('os2-s', 410, 185, '新 block：P̃_{ij} 每行当权重，组合 V_{j} 的各行', 'start')
F.T('os2-s', 410, 200, '（图 1-1 (b) 的线性组合）', 'start')
FIGS['fig-1-7'] = F.svg(150 + 4 * CS + 30,
    '第 j 轮 ℓ 和 Õ 的更新放在一起。ℓ_j 等于修正因子乘 ℓ_{j−1} 加上 P̃_ij 每行之和；Õ_j 等于同一个修正因子按行乘 Õ_{j−1}，加上 P̃_ij 乘 V_j。两式只差新 block 那一项')

# ===================== 图 1-8：两个 block 展开 =====================
F = Fig('fig-1-8')
BH, X0, BW = 30, 92, 200
def bar(x, y, kind, t):
    F.rect(x, y, BW, BH, kind, t, 'os2-m', rx=6)
F.T('os2-b', 24, 40 + 20, 'j = 1')
bar(X0, 40, 'old', 'e^{S_{i1} − m_{1}} V_{1}')
F.T('os2', X0 + BW + 18, 52, '= Õ_{1}：第 1 个 block 按它自己的最大值 m_{1} 算')
F.T('os2-s', X0 + BW + 18, 67, '这时还不知道后面有没有更大的数')
F.arrow(X0 + BW / 2, 40 + BH + 4, X0 + BW / 2, 124, hi=True)
F.T('os2', X0 + BW / 2 + 8, 103, '× e^{m_{1} − m}：换到整行的最大值 m = m_{2}', 'start', 'fill: var(--fig-hi)')
F.T('os2-b', 24, 128 + 20, 'j = 2')
bar(X0, 128, 'res', 'e^{S_{i1} − m} V_{1}')
F.T('os2-m', X0 + BW + 8, 128 + 20, '+')
bar(X0 + BW + 28, 128, 'new', 'e^{S_{i2} − m} V_{2}')
F.T('os2-m', X0 + 2 * BW + 40, 128 + 20, '= Õ_{2}')
F.w(f'<path d="M{X0:g},{128 + BH + 4:g} V{128 + BH + 9:g} H{X0 + 2 * BW + 28:g} V{128 + BH + 4:g}" fill="none" stroke="currentColor" stroke-opacity=".5" stroke-width="1.2"/>')
F.T('os2', X0 + BW + 14, 128 + BH + 27, '每一项都按整行的 m 算，就是 õ 的定义；再每行除以 ℓ 就是 O_{i}', 'middle')
FIGS['fig-1-8'] = F.svg(128 + BH + 42,
    '两个 block 时 Õ 的展开。第 1 轮 Õ_1 等于 e 的 S_i1 减 m_1 次方乘 V_1。第 2 轮先乘 e 的 m_1 减 m 次方，变成 e 的 S_i1 减 m 次方乘 V_1，再加上 e 的 S_i2 减 m 次方乘 V_2，每一项都按整行的最大值 m 算')

if __name__ == '__main__':
    p = sys.argv[1]
    s = open(p).read()
    for fid, svg in FIGS.items():
        s, n = re.subn(r'(<figure id="' + fid + r'"[^>]*>\n)<svg.*?</svg>', lambda m: m.group(1) + svg, s, count=1, flags=re.S)
        assert n == 1, fid
    open(p, 'w').write(s)
    print('ok', list(FIGS))
