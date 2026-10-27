"""1.1–1.2 推导用的示意图：图 1-2、1-3、1-4、1-6、1-7、1-8。每张图对应正文里的一个公式块，
一行一个操作，分三栏：左边是数学公式，中间是这个操作的示意，右边是解释。
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
    t = t.replace('P̃', 'P\x01')
    while i < len(t):
        if t[i] == '\x01':                     # P̃ 的波浪号单独画：网站字体不渲染组合字符
            add(buf, 'n'); buf = ''; add('~', 'tilde'); i += 1; continue
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
        cur, h, pend = 0, '', ''
        for k, (s, l) in enumerate(parse(t)):
            s = s.replace('&', '&amp;').replace('<', '&lt;')
            if k == 0 and l == 'n':
                h += s; continue
            if l == 'tilde':
                h += '<tspan dx="-5.6" dy="-7" font-size="80%">~</tspan>'
                cur -= 7; pend = ' dx="1.2"'
                continue
            dy = OFF[l] - cur; cur = OFF[l]
            a = (f' dy="{dy:g}"' if dy else '') + pend; pend = ''
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
    def svg(self, H, label):
        f = self.fid
        defs = (f'<defs><marker id="{f}-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="currentColor" fill-opacity=".6"/></marker>'
                f'<marker id="{f}-arh" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" style="fill: var(--fig-hi)"/></marker></defs>\n')
        return (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{label}">\n' + CSS + defs
                + '\n'.join(self.B) + '\n</svg>')

# ---- 三栏：公式 | 示意 | 解释 ----
W, FX, DX, EX = 860, 16, 262, 660           # 三栏的左边界
def rows(fid, spec, label, leg=None):
    """spec：[(行高, [公式...], 画示意的函数 draw(F, x0, y0), [解释...]), ...]"""
    F = Fig(fid)
    F.T('os2-s', FX, 18, '公式'); F.T('os2-s', DX, 18, '示意'); F.T('os2-s', EX, 18, '解释')
    y = 28
    for k, (h, fm, draw, ex) in enumerate(spec):
        F.w(f'<line x1="8" y1="{y:g}" x2="{W - 8}" y2="{y:g}" stroke="currentColor" stroke-opacity=".14"/>')
        y += 12
        for n, t in enumerate(fm):
            F.T('os2-m', FX, y + h / 2 - (len(fm) - 1) * 11 + n * 22 + 5, t)
        draw(F, DX, y)
        for n, t in enumerate(ex):
            F.T('os2', EX, y + h / 2 - (len(ex) - 1) * 9 + n * 18 + 4, t)
        y += h + 12
    if leg:
        F.w(f'<line x1="8" y1="{y:g}" x2="{W - 8}" y2="{y:g}" stroke="currentColor" stroke-opacity=".14"/>')
        legend(F, DX, y + 20, leg); y += 30
    return F.svg(y + 4, label)

def legend(F, x, y, items):
    for k, (kind, txt) in enumerate(items):
        if kind == 'row':
            F.w(f'<rect x="{x}" y="{y - 9}" width="12" height="12" rx="2" fill="none" style="stroke: var(--fig-hi)" stroke-width="1.6"/>')
        else:
            F.rect(x, y - 9, 12, 12, kind)
        F.T('os2-s', x + 17, y + 1, txt)
        t = txt.replace('_{', '').replace('}', '')
        x += 17 + sum(10.5 if ord(ch) > 0x2E80 else 4.6 for ch in t) + 20

def seq(F, x, y, parts, cs, hl=None):
    """把公式画成一排形状：字符串是运算符，(kind, 行, 列, 名字) 是矩阵。"""
    for p in parts:
        if isinstance(p, str):
            F.T('os2-m', x + 2, y + 2 * cs + 5, p); x += len(p) * 7.2 + 8
        else:
            kind, r, c, name = p
            x = F.mat(x, y, r, c, kind, cs, name, hl=hl) + 10
    return x

FIGS = {}
LEG = [('old', '前 j − 1 个 block 留下的'), ('new', '这一轮读的'), ('res', '这一轮算出来的'), ('row', '同一行（一个 query）')]

# ===================== 图 1-2：softmax 要看完全部 block =====================
BW, CW = 36, 12
labs = ['1', '⋯', 'j − 1', 'j', 'j + 1', '⋯', 'T_{k}']
def strip(F, x, y):
    for b in range(7):
        kind = 'old' if b < 3 else ('new' if b == 3 else 'skip')
        for e in range(3):
            F.rect(x + b * (BW + 6) + e * CW, y + 14, CW, 20, kind)
        F.T('os2', x + b * (BW + 6) + BW / 2, y + 48, labs[b], 'middle')
    F.T('os2-s', x - 6, y + 48, 'j =', 'end')
    F.T('os2', x + 3 * (BW + 6) + BW / 2, y + 8, '这一轮', 'middle', 'fill: var(--fig-hi)')
    F.T('os2-s', x + (3 * BW + 12) / 2, y + 64, '已经读过', 'middle')
    F.T('os2-s', x + 4 * (BW + 6) + (3 * BW + 12) / 2, y + 64, '还没读', 'middle')
def passes(F, x, y):
    for k, t in enumerate(['求 m', '求 ℓ', '算 P_{ij}，乘 V_{j} 加进 O_{i}']):
        F.arrow(x, y + 16 + k * 28, x + 7 * BW + 36, y + 16 + k * 28)
        F.T('os2-s', x + (7 * BW + 36) / 2, y + 11 + k * 28, t, 'middle')
FIGS['fig-1-2'] = rows('fig-1-2', [
    (68, ['P_{ij} = e^{S_{ij} − m} / ℓ', 'm = max_{j} rowmax(S_{ij})', 'ℓ = Σ_{j} rowsum(e^{S_{ij} − m})'], strip,
     ['读到第 j 个 block 时后面还没读，', 'm、ℓ 未知，P_{ij} 算不出来。']),
    (76, ['第 1 遍：求 m', '第 2 遍：求 ℓ', '第 3 遍：O_{i} = Σ_{j} P_{ij} V_{j}'], passes,
     ['按定义要把 K、V 扫三遍。']),
], '第一行：softmax 的公式；第 i 个 query block 的 S 分在 T_k 个 block 里，读到第 j 个时后面的还没读，m 和 ℓ 不知道，P_ij 算不出来。第二行：按定义要扫三遍，求 m、求 ℓ、算 P_ij 乘 V_j')

# ===================== 图 1-3：两段合并 =====================
SC, SG = 40, 14
sx = lambda x, k: x + k * SC + (SG if k >= 4 else 0)
def seg_own(F, x, y):
    F.T('os2-b', sx(x, 0) + 2 * SC, y + 8, 'A 段', 'middle'); F.T('os2-b', sx(x, 4) + 2 * SC, y + 8, 'B 段', 'middle')
    for k in range(8):
        kind, s = ('old', 'A') if k < 4 else ('new', 'B')
        F.rect(sx(x, k), y + 14, SC, 20, kind, 's', 'os2-s')
        F.rect(sx(x, k), y + 40, SC, 22, kind, f'e^{{s − m_{{{s}}}}}', 'os2-s')
    for a, s in ((0, 'A'), (4, 'B')):
        F.bracket(sx(x, a), sx(x, a + 3) + SC, y + 70)
        F.T('os2', sx(x, a) + 2 * SC, y + 86, f'加起来 = ℓ_{{{s}}}', 'middle')
def seg_merge(F, x, y):
    for a, s in ((0, 'A'), (4, 'B')):
        cx = sx(x, a) + 2 * SC
        F.arrow(cx, y, cx, y + 34, hi=True)
        F.T('os2', cx + 6, y + 21, f'× e^{{m_{{{s}}} − m}}', 'start', 'fill: var(--fig-hi)')
    for k in range(8):
        F.rect(sx(x, k), y + 38, SC, 22, 'res', 'e^{s − m}', 'os2-s')
    F.bracket(sx(x, 0), sx(x, 7) + SC, y + 68)
    F.T('os2', sx(x, 0) + 4 * SC + SG / 2, y + 84, '加起来 = ℓ（整行的分母）', 'middle')
FIGS['fig-1-3'] = rows('fig-1-3', [
    (90, ['m_{A} = rowmax(S_{A})', 'ℓ_{A} = rowsum(e^{S_{A} − m_{A}})', 'B 段同理'], seg_own,
     ['各段按自己的最大值求指数和。']),
    (88, ['m = max(m_{A}, m_{B})', 'ℓ = e^{m_{A} − m} ℓ_{A} + e^{m_{B} − m} ℓ_{B}'], seg_merge,
     ['乘修正因子换到同一个 m，再相加。']),
], '两段合并。第一行：A、B 两段各按自己的最大值 m_A、m_B 算指数，加起来是 ℓ_A、ℓ_B。第二行：各乘修正因子换到同一个最大值 m，加起来就是整行的分母 ℓ')

# ===================== 图 1-4：第 j 轮的 m、ℓ =====================
CS = 16
FIGS['fig-1-4'] = rows('fig-1-4', [
    (82, ['m_{j} = max(m_{j−1}, rowmax(S_{ij}))'],
     lambda F, x, y: seq(F, x, y, [('res', 4, 1, 'm_{j}'), '=', 'max(', ('old', 4, 1, 'm_{j−1}'), ',', 'rowmax(', ('new', 4, 4, 'S_{ij}'), '))'], CS, 1),
     ['每行取旧最大值和新 block 最大值', '中大的那个。']),
    (82, ['P̃_{ij} = exp(S_{ij} − m_{j})'],
     lambda F, x, y: seq(F, x, y, [('res', 4, 4, 'P̃_{ij}'), '=', 'exp(', ('new', 4, 4, 'S_{ij}'), '−', ('res', 4, 1, 'm_{j}'), ')'], CS, 1),
     ['每行减去自己的 m_{j} 再取 exp。']),
    (82, ['ℓ_{j} = e^{m_{j−1} − m_{j}} ℓ_{j−1}', '        + rowsum(P̃_{ij})'],
     lambda F, x, y: seq(F, x, y, [('res', 4, 1, 'ℓ_{j}'), '=', ('res', 4, 1, 'e^{m_{j−1} − m_{j}}'), '·', ('old', 4, 1, 'ℓ_{j−1}'), '+', 'rowsum(', ('res', 4, 4, 'P̃_{ij}'), ')'], CS, 1),
     ['旧的和换到新最大值，', '加上新 block 每行的和。']),
], '第 j 轮 m 和 ℓ 的更新，B_q 等于 4。m_j 等于 m_{j−1} 和 S_ij 每行最大值中大的那个；P̃_ij 等于 S_ij 每行减去 m_j 的对应数再取 exp；ℓ_j 等于修正因子乘 ℓ_{j−1} 加上 P̃_ij 每行之和。红框标出同一行', LEG)

# ===================== 图 1-6：把分母提出来 =====================
WC, VS = 40, 16
def weights_v(F, x, y, pre, labels, res_pre, res_kind, res_name):
    if pre:
        F.T('os2-m', x, y + 2 * VS + 5, pre); x += 32
    for k, t in enumerate(labels):
        F.rect(x + k * WC, y + 2 * VS - 11, WC, 22, 'res', t, 'os2-s')
    x += 4 * WC + 8
    F.T('os2-m', x, y + 2 * VS + 5, '×'); x += 16
    F.mat(x, y, 4, 3, 'new', VS, 'V')
    for k in range(4): F.T('os2-s', x + 3 * VS + 3, y + k * VS + 12, f'V_{{{k + 1}}}')
    x += 3 * VS + 22
    F.T('os2-m', x, y + 2 * VS + 5, '='); x += 16
    if res_pre:
        F.T('os2-m', x, y + 2 * VS + 5, res_pre); x += 32
    F.mat(x, y + 2 * VS - VS / 2, 1, 3, res_kind, VS, res_name)
FIGS['fig-1-6'] = rows('fig-1-6', [
    (82, ['O_{i} = Σ_{j} P_{ij} V_{j}', 'P_{ij} = diag(ℓ)^{−1} e^{S_{ij} − m}'],
     lambda F, x, y: weights_v(F, x, y, '', [f'P_{{i{k}}}' for k in range(1, 5)], '', 'out', 'O_{i}'),
     ['O_{i} 是 V 各行的加权和，', '权重的分母都是 ℓ。']),
    (82, ['O_{i} = diag(ℓ)^{−1} Σ_{j} e^{S_{ij} − m} V_{j}', '    = diag(ℓ)^{−1} Õ'],
     lambda F, x, y: weights_v(F, x, y, '1/ℓ ×', [f'e^{{S_{{i{k}}} − m}}' for k in range(1, 5)], '1/ℓ ×', 'res', 'Õ'),
     ['把 ℓ 提到外面，Õ 和 ℓ 分开累加。']),
], '把分母提出来。第一行：O_i 等于 P 的第 i 个 block 行乘 V，每个 P_ij 是 e 的 S_ij 减 m 次方除以 ℓ。第二行：ℓ 对每一项都一样，提到外面，O_i 等于 1/ℓ 乘 Õ')

# ===================== 图 1-7：ℓ 和 Õ 并排 =====================
CS = 14
FIGS['fig-1-7'] = rows('fig-1-7', [
    (76, ['ℓ_{j} = e^{m_{j−1} − m_{j}} ℓ_{j−1}', '        + rowsum(P̃_{ij})'],
     lambda F, x, y: seq(F, x, y, [('res', 4, 1, 'ℓ_{j}'), '=', ('res', 4, 1, 'e^{m_{j−1} − m_{j}}'), '·', ('old', 4, 1, 'ℓ_{j−1}'), '+', 'rowsum(', ('res', 4, 4, 'P̃_{ij}'), ')'], CS, 1),
     ['旧的和乘修正因子，', '加上新 block 每行的和。']),
    (76, ['Õ_{j} = diag(e^{m_{j−1} − m_{j}}) Õ_{j−1}', '        + P̃_{ij} V_{j}'],
     lambda F, x, y: seq(F, x, y, [('res', 4, 3, 'Õ_{j}'), '=', 'diag(', ('res', 4, 1, 'e^{m_{j−1} − m_{j}}'), ')', ('old', 4, 3, 'Õ_{j−1}'), '+', ('res', 4, 4, 'P̃_{ij}'), ('new', 4, 3, 'V_{j}')], CS, 1),
     ['同一个修正因子按行乘旧的 Õ，', '加上新 block 的 P̃_{ij}V_{j}。']),
], '第 j 轮 ℓ 和 Õ 的更新放在一起。ℓ_j 等于修正因子乘 ℓ_{j−1} 加上 P̃_ij 每行之和；Õ_j 等于同一个修正因子按行乘 Õ_{j−1}，加上 P̃_ij 乘 V_j。两式只差新 block 那一项', LEG)

# ===================== 图 1-8：两个 block 展开 =====================
BH, BL = 28, 150
def bar(F, x, y, kind, t):
    F.rect(x, y, BL, BH, kind, t, 'os2', rx=6)
def two1(F, x, y):
    bar(F, x, y + 4, 'old', 'e^{S_{i1} − m_{1}} V_{1}')
def two2(F, x, y):
    F.arrow(x + BL / 2, y - 12, x + BL / 2, y + 26, hi=True)
    F.T('os2', x + BL / 2 + 8, y + 12, '× e^{m_{1} − m}', 'start', 'fill: var(--fig-hi)')
    bar(F, x, y + 30, 'res', 'e^{S_{i1} − m} V_{1}')
    F.T('os2-m', x + BL + 7, y + 30 + 19, '+')
    bar(F, x + BL + 24, y + 30, 'new', 'e^{S_{i2} − m} V_{2}')
FIGS['fig-1-8'] = rows('fig-1-8', [
    (36, ['Õ_{1} = e^{S_{i1} − m_{1}} V_{1}'], two1,
     ['第 1 个 block 按 m_{1} 算。']),
    (58, ['Õ_{2} = e^{m_{1} − m} Õ_{1} + e^{S_{i2} − m} V_{2}', '    = e^{S_{i1} − m} V_{1} + e^{S_{i2} − m} V_{2}'], two2,
     ['乘 e^{m_{1} − m} 以后，', '每一项都按整行的 m 算。']),
], '两个 block 时 Õ 的展开。第一行：Õ_1 等于 e 的 S_i1 减 m_1 次方乘 V_1。第二行：先乘 e 的 m_1 减 m 次方，变成 e 的 S_i1 减 m 次方乘 V_1，再加上 e 的 S_i2 减 m 次方乘 V_2，每一项都按整行的最大值 m 算', [('old', '按旧的最大值算的'), ('new', '这一轮读的'), ('res', '换到新的最大值以后')])

if __name__ == '__main__':
    p = sys.argv[1]
    s = open(p).read()
    for fid, svg in FIGS.items():
        s, n = re.subn(r'(<figure id="' + fid + r'"[^>]*>\n)<svg.*?</svg>', lambda m: m.group(1) + svg, s, count=1, flags=re.S)
        assert n == 1, fid
    open(p, 'w').write(s)
    print('ok', list(FIGS))
