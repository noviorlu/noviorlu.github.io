"""推导用的示意图：图 1-2、1-3、1-4、1-6、1-7、1-9。每张图对应正文里的一个公式块，
一行一个操作，分三栏：左边是数学公式，中间是这个操作的示意，右边是解释。
配色和图 1-1 一致：深蓝 = 这一轮新读的 block，灰 = 前面的 block 留下的，浅蓝 = 这一轮算出来的，红 = 输出；红框标同一行（一个 query）。
用法：python3 tools/flashattention-1-to-4/fig_1_derive.py content/blog/flashattention-1-to-4/index.md
会把文中对应 <figure id="..."> 里的 <svg> 换掉，figcaption 不动。
"""
import re, sys

# ---- 文字：_{下标}、^{上标}，上标里可以再套 _{} ----
OFF = {'n': 0, 'sub': 3.5, 'sup': -5, 'supsub': -2, 'subsub': 5.5}
FS = {'sub': '72%', 'sup': '72%', 'supsub': '58%', 'subsub': '58%'}
def parse(t):
    out, buf, i = [], '', 0
    hl = [False]                              # «…» 包住的是修正因子，单独着色
    def add(x, l):
        if x: out.append((x, l, hl[0]))
    t = t.replace('P̃', 'P\x01')
    while i < len(t):
        if t[i] in '«»':
            add(buf, 'n'); buf = ''; hl[0] = t[i] == '«'; i += 1; continue
        if t[i] == '\x01':                     # P̃ 的波浪号单独画：网站字体不渲染组合字符
            add(buf, 'n'); buf = ''; add('~', 'tilde'); i += 1; continue
        if t[i] in '_^' and i + 1 < len(t) and t[i + 1] == '{':
            add(buf, 'n'); buf = ''
            j, d = i + 2, 1
            while d:
                d += {'{': 1, '}': -1}.get(t[j], 0); j += 1
            inner = t[i + 2:j - 1]
            if t[i] == '_':
                k, b = 0, ''                          # 下标里可以再套 _{}，例如 Q_{T_{q}}
                while k < len(inner):
                    if inner[k] == '_' and inner[k + 1] == '{':
                        add(b, 'sub'); b = ''
                        e = inner.index('}', k); add(inner[k + 2:e], 'subsub'); k = e + 1
                    else:
                        b += inner[k]; k += 1
                add(b, 'sub')
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
    'new': 'fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)',
    'res': 'fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)',
    'out': 'fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)',
    'skip': 'fill: none; stroke: var(--fig-mute)',
    'fac': 'fill: color-mix(in srgb, var(--fig-hi) 22%, transparent); stroke: var(--fig-hi)',
    'state': 'fill: color-mix(in srgb, var(--fig-1) 8%, transparent); stroke: var(--fig-1)',     # 跨 tile 保留的状态
    'tmp': 'fill: color-mix(in srgb, var(--fig-3) 45%, transparent); stroke: var(--fig-2)',       # 这一轮的临时量
    # Algorithm 1 第 1、2 行：每个变量一个颜色，和算法框里的色块一致
    'vQ': 'fill: color-mix(in srgb, var(--fig-1) 45%, transparent); stroke: var(--fig-1)',
    'vK': 'fill: color-mix(in srgb, var(--fig-2) 45%, transparent); stroke: var(--fig-2)',
    'vV': 'fill: color-mix(in srgb, var(--fig-3) 70%, transparent); stroke: var(--fig-2)',
    'vO': 'fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)',
    'vL': 'fill: color-mix(in srgb, var(--fig-mute) 20%, transparent); stroke: var(--fig-mute)',
}
DASH = {'skip', 'tmp', 'fac'}                 # 虚线框 = 用完就丢
CSS = ('  <style>\n    .os2 { font-size: 11px; fill: currentColor; }\n    .os2-b { font-size: 12px; font-weight: 600; fill: currentColor; }\n'
       '    .os2-s { font-size: 10.5px; fill: currentColor; opacity: .72; }\n    .os2-m { font-size: 13px; fill: currentColor; }\n'
       '    .os2-hi { fill: var(--fig-hi); }\n  </style>\n')

class Fig:
    def __init__(self, fid):
        self.fid, self.B = fid, []
    def w(self, s): self.B.append(s)
    def T(self, cls, x, y, t, anchor='start', style=''):
        cur, h, pend = 0, '', ''
        for k, (s, l, hl) in enumerate(parse(t)):
            s = s.replace('&', '&amp;').replace('<', '&lt;')
            c = ' style="fill: var(--fig-hi); font-weight: 600"' if hl else ''
            if k == 0 and l == 'n' and not hl:
                h += s; continue
            if l == 'tilde':
                h += f'<tspan dx="-5.6" dy="-7" font-size="80%"{c}>~</tspan>'
                cur -= 7; pend = ' dx="1.2"'
                continue
            dy = OFF[l] - cur; cur = OFF[l]
            a = (f' dy="{dy:g}"' if dy else '') + pend; pend = ''
            f = f' font-size="{FS[l]}"' if l in FS else ''
            h += f'<tspan{a}{f}{c}>{s}</tspan>'
        st = f' style="{style}"' if style else ''
        an = f' text-anchor="{anchor}"' if anchor != 'start' else ''
        self.w(f'<text class="{cls}" x="{x:g}" y="{y:g}"{an}{st}>{h}</text>')
    def rect(self, x, y, w, h, kind, label=None, cls='os2', rx=1.5):
        d = ' stroke-dasharray="3 2"' if kind in DASH else ''
        sw = 1.7 if kind == 'state' else 1
        self.w(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}" style="{STYLE[kind]}" stroke-width="{sw}"{d}/>')
        if label: self.T(cls, x + w / 2, y + h / 2 + 4, label, 'middle')
    def mat(self, x, y, r, c, kind, cs, name=None, hl=None):
        for i in range(r):
            for j in range(c):
                self.rect(x + j * cs, y + i * cs, cs, cs, kind)
        if hl is not None:
            self.w(f'<rect x="{x - 1.5:g}" y="{y + hl * cs - 1.5:g}" width="{c * cs + 3:g}" height="{cs + 3:g}" rx="2" fill="currentColor" fill-opacity=".2" stroke="none"/>')
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
            if isinstance(t, tuple): F.T(t[2] if len(t) > 2 else 'os2-m', FX, y + t[0], t[1])
            else: F.T('os2-m', FX, y + h / 2 - (len(fm) - 1) * 11 + n * 22 + 5, t)
        draw(F, DX, y)
        for n, t in enumerate(ex):
            if isinstance(t, tuple): F.T('os2', EX, y + t[0], t[1])
            else: F.T('os2', EX, y + h / 2 - (len(ex) - 1) * 9 + n * 18 + 4, t)
        y += h + 12
    if leg:
        F.w(f'<line x1="8" y1="{y:g}" x2="{W - 8}" y2="{y:g}" stroke="currentColor" stroke-opacity=".14"/>')
        legend(F, DX, y + 20, leg); y += 30
    return F.svg(y + 4, label)

def legend(F, x, y, items):
    for k, (kind, txt) in enumerate(items):
        if kind == 'row':
            F.w(f'<rect x="{x}" y="{y - 9}" width="12" height="12" rx="2" fill="currentColor" fill-opacity=".25"/>')
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
        elif p[0] == 'diag':
            _, n, name = p
            for a in range(n):
                for b in range(n):
                    F.rect(x + b * cs, y + a * cs, cs, cs, 'fac' if a == b else 'skip', None)
            for b in range(n):
                for a in range(n):
                    if a != b: F.T('os2-s', x + b * cs + cs / 2, y + a * cs + cs / 2 + 3.5, '0', 'middle')
            F.T('os2', x + n * cs / 2, y + n * cs + 15, name, 'middle')
            for a in range(n):                      # 第 a 个对角元指向下一个矩阵的第 a 行
                yc = y + a * cs + cs / 2
                F.arrow(x + n * cs + 2, yc, x + n * cs + 17, yc)
            x = x + n * cs + 18
        else:
            kind, r, c, name = p[:4]
            x1 = F.mat(x, y, r, c, kind, cs, name, hl=hl)
            if len(p) > 4:                    # 名字下面再标一行：上一轮留下的 / 留到下一轮
                F.T('os2-s', (x + x1) / 2, y + r * cs + 32, p[4], 'middle', 'opacity: 1; fill: var(--fig-1); font-weight: 600' if p[4].startswith('→') else '')
            x = x1 + 10
    return x

FIGS = {}
LEG = [('state', '状态：留给下一个 block 用'), ('tmp', '临时：这个 block 用完就丢'), ('fac', '修正因子（临时）'), ('new', '这一轮从 HBM 读的'), ('row', '同一行（一个 query）')]

# ===================== 图 1-2：按定义算 softmax（不分 block） =====================
NC, CW2 = 8, 36
def naive(lab, out, op):
    def draw(F, x, y):
        for j in range(NC):
            F.rect(x + j * CW2, y + 6, CW2, 22, 'tmp', lab.format(j=j + 1), 'os2-s')
        F.arrow(x, y + 38, x + NC * CW2, y + 38)           # 扫一遍整行
        if out:
            F.arrow(x + NC * CW2 + 6, y + 17, x + NC * CW2 + 32, y + 17)
            F.T('os2-s', x + NC * CW2 + 19, y + 11, op, 'middle')
            F.rect(x + NC * CW2 + 36, y + 6, 26, 22, 'state', out)
    return draw
FIGS['fig-1-2'] = rows('fig-1-2', [
    (40, ['m = max_{j} s_{j}'], naive('s_{{{j}}}', 'm', 'max'), ['第 1 遍：扫整行求最大值。']),
    (40, ['ℓ = Σ_{j} e^{s_{j} − m}'], naive('e^{{s_{{{j}}} − m}}', 'ℓ', 'Σ'), ['第 2 遍：再扫一遍求分母，要用 m。']),
    (40, ['p_{j} = e^{s_{j} − m} / ℓ'], naive('p_{{{j}}}', None, ''), ['第 3 遍：再扫一遍算 p_{j}，', '要用 m 和 ℓ。']),
], '按定义算 S 的一行 softmax，不分 block：第 1 遍扫整行求最大值 m，第 2 遍求分母 ℓ，第 3 遍算每个 p_j。m 和 ℓ 都要看完整行')

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
        F.T('os2', cx + 6, y + 21, f'× «e^{{m_{{{s}}} − m}}»', 'start')
    for k in range(8):
        F.rect(sx(x, k), y + 38, SC, 22, 'res', 'e^{s − m}', 'os2-s')
    F.bracket(sx(x, 0), sx(x, 7) + SC, y + 68)
    F.T('os2', sx(x, 0) + 4 * SC + SG / 2, y + 84, '加起来 = ℓ（整行的分母）', 'middle')
FIGS['fig-1-3'] = rows('fig-1-3', [
    (90, ['m_{A} = rowmax(S_{A})', 'ℓ_{A} = rowsum(e^{S_{A} − m_{A}})', 'B 段同理'], seg_own,
     ['各段按自己的最大值求指数和。']),
    (88, ['m = max(m_{A}, m_{B})', 'ℓ = «e^{m_{A} − m}» ℓ_{A} + «e^{m_{B} − m}» ℓ_{B}'], seg_merge,
     ['乘«修正因子»换到同一个 m，再相加。']),
], '两段合并。第一行：A、B 两段各按自己的最大值 m_A、m_B 算指数，加起来是 ℓ_A、ℓ_B。第二行：各乘修正因子换到同一个最大值 m，加起来就是整行的分母 ℓ')

# ===================== 图 1-4：第 j 轮的 m、ℓ =====================
CS = 16
FIGS['fig-1-4'] = rows('fig-1-4', [
    (92, ['m_{j} = max(m_{j−1}, rowmax(S_{ij}))'],
     lambda F, x, y: seq(F, x, y, [('state', 4, 1, 'm_{j}', '→ 给 j + 1 用'), '=', 'max(', ('state', 4, 1, 'm_{j−1}', '来自 j − 1'), ',', 'rowmax(', ('tmp', 4, 4, 'S_{ij}'), '))'], CS, 1),
     ['每行取旧最大值和新 block 最大值', '中大的那个。']),
    (82, ['P̃_{ij} = exp(S_{ij} − m_{j})'],
     lambda F, x, y: seq(F, x, y, [('tmp', 4, 4, 'P̃_{ij}'), '=', 'exp(', ('tmp', 4, 4, 'S_{ij}'), '−', ('state', 4, 1, 'm_{j}'), ')'], CS, 1),
     ['每行减去自己的 m_{j} 再取 exp。']),
    (92, ['ℓ_{j} = «e^{m_{j−1} − m_{j}}» ℓ_{j−1}', '        + rowsum(P̃_{ij})'],
     lambda F, x, y: seq(F, x, y, [('state', 4, 1, 'ℓ_{j}', '→ 给 j + 1 用'), '=', ('fac', 4, 1, '«e^{m_{j−1} − m_{j}}»'), '·', ('state', 4, 1, 'ℓ_{j−1}', '来自 j − 1'), '+', 'rowsum(', ('tmp', 4, 4, 'P̃_{ij}'), ')'], CS, 1),
     ['旧的和乘«修正因子»换到新最大值，', '加上新 block 每行的和。']),
], '第 j 轮 m 和 ℓ 的更新，B_q 等于 4。m_j 等于 m_{j−1} 和 S_ij 每行最大值中大的那个；P̃_ij 等于 S_ij 每行减去 m_j 的对应数再取 exp；ℓ_j 等于修正因子乘 ℓ_{j−1} 加上 P̃_ij 每行之和。红框标出同一行', LEG)

# ===================== 图 1-6：把分母提出来（画法同图 1-1 (b)） =====================
PW6, C6 = 50, 24
def pv(lab, res_name):
    def draw(F, x, y):
        py, vx, vy = y + 52, x + 4 * PW6 + 50, y + 16
        for j in range(4):
            F.rect(x + j * PW6, py, PW6, C6, 'tmp', lab.format(j=j + 1), 'os2-s')
            F.rect(vx, vy + j * C6, C6, C6, 'new', f'V_{{{j + 1}}}')
            yc, cx = vy + j * C6 + 12, x + j * PW6 + PW6 / 2
            xj, ty = vx - 8 - 6 * j, y + 2 + 4 * j          # 先往左，再往上，再往左到第 j 格上方，互不交叉
            F.w(f'<path d="M{vx:g},{yc:g} H{xj + 4:g} Q{xj:g},{yc:g} {xj:g},{yc - 4:g} V{ty + 4:g} Q{xj:g},{ty:g} {xj - 4:g},{ty:g} '
                f'H{cx + 4:g} Q{cx:g},{ty:g} {cx:g},{ty + 4:g} V{py - 1:g}" fill="none" stroke="currentColor" stroke-opacity=".55" '
                f'stroke-width="1.4" marker-end="url(#{F.fid}-ar)"/>')
        F.T('os2-b', x + 4 * PW6 + 12, py + 17, '×', 'middle')
        F.T('os2-b', vx + C6 + 14, py + 17, '=', 'middle')
        F.rect(vx + C6 + 28, py, C6, C6, 'res', res_name)
        F.T('os2', x + 2 * PW6, py + C6 + 15, 'P 的第 i 个 block 行' if 'P' in lab else '没除以 ℓ 的权重', 'middle')
        F.T('os2', vx + C6 / 2, vy + 4 * C6 + 15, 'V', 'middle')
    return draw
FIGS['fig-1-6'] = rows('fig-1-6', [
    (128, ['O_{i} = Σ_{j} P_{ij} V_{j}', 'P_{ij} = e^{S_{ij} − m} / ℓ'], pv('P_{{i{j}}}', 'O_{i}'),
     ['V_{j} 乘 P 的第 j 个 block，', '加起来是 O_{i}（图 1-1 (b)）。']),
    (128, ['Õ = Σ_{j} e^{S_{ij} − m} V_{j}'], pv('e^{{S_{{i{j}}} − m}}', 'Õ'),
     ['权重换成没除以 ℓ 的 e^{S_{ij} − m}，', '结果是 Õ。']),
    (82, ['O_{i} = Õ / ℓ'],
     lambda F, x, y: seq(F, x, y, [('res', 4, 3, 'O_{i}'), '=', ('res', 4, 3, 'Õ'), '÷', ('state', 4, 1, 'ℓ')], 16, 1),
     ['Õ 的第 r 行除以 ℓ 的第 r 个数', '（图 1-1 (c)），就是 O_{i}。']),
], '把分母提出来，画法同图 1-1 (b)。第一行：V 的每个 block V_j 乘 P 第 i 个 block 行的第 j 个 block，加起来是 O_i。第二行：权重换成没除以 ℓ 的 e 的 S_ij 减 m 次方，结果是 Õ。第三行：Õ 的第 r 行除以 ℓ 的第 r 个数，得到 O_i')

# ===================== 图 1-7：Õ 一个 block 一个 block 地累加（画法同图 1-6） =====================
def acc(cells, under, res_name):
    """cells：每格 (kind, 标签)，kind 为 'skip' 的格子表示还没读；under：每格下面的小字（«» 标红）。"""
    def draw(F, x, y):
        py, vx, vy = y + 52, x + 4 * PW6 + 50, y + 16
        for j, (kind, lab) in enumerate(cells):
            F.rect(x + j * PW6, py, PW6, C6, kind, lab, 'os2-s')
            F.rect(vx, vy + j * C6, C6, C6, 'skip' if kind == 'skip' else 'new', f'V_{{{j + 1}}}')
            if under[j]: F.T('os2-s', x + j * PW6 + PW6 / 2, py + C6 + 15, under[j], 'middle', 'opacity: 1')
            if kind == 'skip': continue
            yc, cx = vy + j * C6 + 12, x + j * PW6 + PW6 / 2
            xj, ty = vx - 8 - 6 * j, y + 2 + 4 * j
            F.w(f'<path d="M{vx:g},{yc:g} H{xj + 4:g} Q{xj:g},{yc:g} {xj:g},{yc - 4:g} V{ty + 4:g} Q{xj:g},{ty:g} {xj - 4:g},{ty:g} '
                f'H{cx + 4:g} Q{cx:g},{ty:g} {cx:g},{ty + 4:g} V{py - 1:g}" fill="none" stroke="currentColor" stroke-opacity=".55" '
                f'stroke-width="1.4" marker-end="url(#{F.fid}-ar)"/>')
        F.T('os2-b', x + 4 * PW6 + 12, py + 17, '×', 'middle')
        F.T('os2-b', vx + C6 + 14, py + 17, '=', 'middle')
        F.rect(vx + C6 + 28, py, C6 + 8, C6, 'state', res_name)
    return draw
SK = ('skip', '')
GEN7 = [('state', 4, 3, 'Õ_{2}'), '=', ('diag', 4, 'diag(«e^{m_{1} − m_{2}}»)'), ('state', 4, 3, 'Õ_{1}'), '+', ('tmp', 4, 4, 'P̃_{i2}'), ('new', 4, 3, 'V_{2}')]
def step2(F, x, y):
    acc([('res', 'e^{S_{i1} − m_{2}}'), ('tmp', 'e^{S_{i2} − m_{2}}'), SK, SK], ['«× e^{m_{1} − m_{2}}»', '', '还没读', '还没读'], 'Õ_{2}')(F, x, y)
    F.arrow(x + 150, y + 112, x + 150, y + 140)              # 具体的 j = 2 → 一般的第 j 轮
    F.T('os2-s', x + 158, y + 130, '写成矩阵，B_{q} = 4 行', 'start', 'opacity: 1')
    seq(F, x, y + 148, GEN7, 14, 1)
FIGS['fig-1-7'] = rows('fig-1-7', [
    (128, ['Õ_{1} = e^{S_{i1} − m_{1}} V_{1}'],
     acc([('tmp', 'e^{S_{i1} − m_{1}}'), SK, SK, SK], ['', '还没读', '还没读', '还没读'], 'Õ_{1}'),
     ['只读了第 1 个 block，', '按它自己的最大值 m_{1} 算。']),
    (238, [(60, 'Õ_{2} = diag(«e^{m_{1} − m_{2}}») Õ_{1}'), (82, '        + e^{S_{i2} − m_{2}} V_{2}'),
           (130, '⇓ 写成矩阵', 'os2-s'),
           (172, 'Õ_{2} = diag(«e^{m_{1} − m_{2}}») Õ_{1}'), (194, '        + P̃_{i2} V_{2}'),
           (220, '一般的第 j 轮：把 1、2 换成 j − 1、j', 'os2-s')],
     step2,
     [(60, '已经累加的部分每行乘自己的«修正因子»，'), (78, '换成按 m_{2} 算，再加上第 2 个 block。'),
      (178, '同一个更新：Õ_{1} 第 r 行乘第 r 个'), (196, '«修正因子»（图 1-1 (c)），再加 P̃_{i2} V_{2}。')]),
    (128, ['扫完：Σ_{j} e^{S_{ij} − m} V_{j} = Õ'],
     acc([('res', f'e^{{S_{{i{j}}} − m}}') for j in range(1, 5)], ['', '', '', ''], 'Õ'),
     ['扫完以后每一格都按整行的 m 算，', '就是图 1-6 第二行的 Õ。']),
], 'Õ 一个 block 一个 block 地累加，画法同图 1-6。第 1 轮只有第 1 个 block，按 m_1 算。第 2 轮旧的一项乘 e 的 m_1 减 m_2 次方，换成按 m_2 算，再加第 2 个 block；下面是同一个更新写成矩阵形式，一般的第 j 轮把 1、2 换成 j − 1、j。扫完以后每一项都按整行的最大值 m 算，就是图 1-6 的 Õ', LEG)

# ===================== 图 1-9：为什么 mask 只能在算完 S 之后做（对角线上的 3 × 3 tile） =====================
F = Fig('fig-1-9')
C9, QX9, SX9, SY9 = 34, 60, 110, 70
MK9 = 'fill: color-mix(in srgb, var(--fig-mute) 80%, transparent); stroke: var(--fig-mute)'
for r in range(3):
    F.rect(QX9, SY9 + r * C9, 30, C9 - 2, 'new', f'q_{{{r + 1}}}')
    F.rect(SX9 + r * C9, SY9 - 40, C9 - 2, 30, 'new', f'k_{{{r + 1}}}')
    for c in range(3):
        x, y = SX9 + c * C9, SY9 + r * C9
        if c <= r: F.rect(x, y, C9 - 2, C9 - 2, 'res', f's_{{{r + 1}{c + 1}}}')
        else:
            F.w(f'<rect x="{x}" y="{y}" width="{C9 - 2}" height="{C9 - 2}" rx="1.5" style="{MK9}" stroke-width="1"/>')
            F.T('os2', x + (C9 - 2) / 2, y + C9 / 2 + 3, '−∞', 'middle')
F.T('os2-b', QX9 - 8, SY9 + 1.5 * C9 + 4, 'Q_{i}', 'end'); F.T('os2-b', SX9 + 1.5 * C9, SY9 - 48, 'K_{j}^{ᵀ}', 'middle')
F.T('os2-b', SX9 + 1.5 * C9, SY9 + 3 * C9 + 16, 'S_{ij}（对角线上的 tile）', 'middle')
# 第 2 列（k_2）加框
F.w(f'<rect x="{SX9 + C9 - 3}" y="{SY9 - 43}" width="{C9 + 4}" height="{3 * C9 + 45}" rx="4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-dasharray="4 3"/>')
tx = SX9 + 3 * C9 + 30
F.T('os2-b', tx, SY9 - 22, 'k_{2} 这一列：')
F.T('os2', tx, SY9 - 4, 'q_{1} 要 mask（s_{12} = −∞），q_{2}、q_{3} 要保留')
F.T('os2', tx, SY9 + 14, '所以 load K_{j} 时没法决定留不留 k_{2}')
F.T('os2-b', tx, SY9 + 46, '把 k_{2} 填成 0 也不行：')
F.T('os2', tx, SY9 + 64, 's_{12} = q_{1} · 0 = 0，e^{0 − m} 不是 0，还会加进 ℓ')
F.T('os2-b', tx, SY9 + 96, '只能先算完 S_{ij}，再按 (r, c) 设成 −∞：')
F.T('os2', tx, SY9 + 114, 'jB_{k} + c > iB_{q} + r 的位置是 −∞，exp 以后是 0')
FIGS['fig-1-9'] = F.svg(SY9 + 3 * C9 + 26,
    '对角线上的 3 乘 3 tile：q_1 到 q_3 是 Q_i 的行，k_1 到 k_3 是 K_j 的行，S_ij 的右上三角是 −∞。k_2 这一列对 q_1 要 mask，对 q_2、q_3 要保留，所以不能在 load K_j 时处理；把 k_2 填 0 得到的 score 是 0 不是 −∞。只能先算完 S_ij，再按位置设成 −∞')

# ===================== Algorithm 1 第 1、2 行：切 block =====================
# 块的大小按在显存里的形状画（B_q、B_k、d 各取一个长度），符号写在块外面
UBQ, UBK, UD, UV, GAP = 26, 22, 18, 6, 3
def split_col(F, x, name, labs, kind, w, annot):
    """竖着一摞：Q、O（每块 B_q × d）、L（每段长 B_q）。标签写在左边。"""
    F.T('os2-b', x + w / 2, 12, name, 'middle')
    for k, t in enumerate(labs):
        y = 18 + k * (UBQ + GAP)
        F.rect(x, y, w, UBQ, 'skip' if t == '⋯' else kind)
        F.T('os2-s', x - 4, y + UBQ / 2 + 4, t, 'end', 'opacity: 1')
    for n, t in enumerate(annot):
        F.T('os2' if n == 0 else 'os2-s', x + w + 10, 48 + n * 18, t)
def split_row(F, x, name, labs, kind, annot):
    """横着一排：K、V（每块 B_k × d，横放）。标签写在下面。"""
    F.T('os2-b', x - 8, 18 + UD / 2 + 4, name, 'end')
    for k, t in enumerate(labs):
        bx = x + k * (UBK + GAP)
        F.rect(bx, 18, UBK, UD, 'skip' if t == '⋯' else kind)
        F.T('os2-s', bx + UBK / 2, 18 + UD + 13, t, 'middle', 'opacity: 1')
    for n, t in enumerate(annot):
        F.T('os2' if n == 0 else 'os2-s', x, 74 + n * 18, t)
def algo_svg(fid, cols, W=640, H=136):
    F = Fig(fid)
    for c in cols:
        if c[0] == 'row': split_row(F, *c[1:])     # 横着放：K、V
        else: split_col(F, *c)                     # 竖着放：Q、O、L
    return (f'<svg id="{fid}" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{fid}" style="max-width: {W}px; display: block">\n'
            + CSS + '\n'.join(F.B) + '\n</svg>')
QL = lambda n, T: [f'{n}_{{1}}', f'{n}_{{2}}', '⋯', f'{n}_{{{T}}}']
ALGO = {
    'algo-l1': algo_svg('algo-l1', [
        (44, 'Q', QL('Q', 'T_{q}'), 'vQ', UD, ['T_{q} = ⌈N / B_{q}⌉ 个 block', '每个 B_{q} × d']),
        ('row', 250, 'K', QL('K', 'T_{k}'), 'vK', ['T_{k} = ⌈N / B_{k}⌉ 个 block', '每个 B_{k} × d（横放）']),
        ('row', 470, 'V', QL('V', 'T_{k}'), 'vV', ['和 K 一样切', '每个 B_{k} × d（横放）'])]),
    'algo-l2': algo_svg('algo-l2', [
        (44, 'O', QL('O', 'T_{q}'), 'vO', UD, ['输出，和 Q 一样切', '每个 B_{q} × d']),
        (290, 'L', QL('L', 'T_{q}'), 'vL', UV, ['L = m + log ℓ（logsumexp），T_{q} 段', '每段长 B_{q}；第 14 行算出，给反向用'])]),
}

if __name__ == '__main__':
    p = sys.argv[1]
    s = open(p).read()
    for fid, svg in FIGS.items():
        s, n = re.subn(r'(<figure id="' + fid + r'"[^>]*>\n)<svg.*?</svg>', lambda m: m.group(1) + svg, s, count=1, flags=re.S)
        assert n == 1, fid
    for fid, svg in ALGO.items():
        s, n = re.subn(r'<svg id="' + fid + r'".*?</svg>', lambda m: svg, s, count=1, flags=re.S)
        assert n == 1, fid
    open(p, 'w').write(s)
    print('ok', list(FIGS))
