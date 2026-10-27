"""图 1-8（FlashAttention 的 tiling，按 Triton program 看）。上栏是数据怎么分给 program，下栏是 causal 时每个 program 算哪些 block。
画法和配色沿用 gpu-triton-intro 的图 2-1、2-5：深蓝 = 这个 program 读的数据，红 = 它写回的输出，斜线 = 之后几轮才读的 block，灰 = 别的 program 的。
原来还有一栏「一轮的数据流」（B[SPLIT2:]），现在不输出。
用法：python3 tools/flashattention-1-to-4/fig_1_2.py content/blog/flashattention-1-to-4/index.md
会把文中 <figure id="fig-1-8" ...> 里的 <svg> 换掉，figcaption 不动。
"""
import re, sys

FID = "fig-1-2"
CSS = """  <style>
    .fg .lab { font-size: 12px; fill: currentColor; }
    .fg .lab2 { font-size: 11px; fill: currentColor; opacity: .65; }
    .fg .ttl { font-size: 12px; font-weight: 600; fill: currentColor; }
    .fg .t { font-size: 12.5px; fill: currentColor; }
    .fg .tb { font-size: 12.5px; font-weight: 600; fill: currentColor; }
    .fg .s { font-size: 10.5px; fill: currentColor; opacity: .7; }
    .fg .op { fill: var(--fig-bg); stroke: currentColor; stroke-opacity: .45; stroke-width: 1.2; }
    .fg .band { fill: currentColor; fill-opacity: .045; }
    @media (max-width: 640px) { .fg-fig { overflow-x: auto; } .fg-fig > svg { min-width: var(--fg-minw, 540px); } }
  </style>"""

B = []
def w(s): B.append(s)

# ---- 文字：_{下标}、^{上标}，上标里可以再套 _{} ----
OFF = {'n': 0, 'sub': 3.5, 'sup': -5, 'supsub': -2}
FS = {'sub': '72%', 'sup': '72%', 'supsub': '58%'}
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
def T(cls, x, y, t, anchor='start', style='', extra=''):
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
    w(f'<text class="{cls}" x="{x:g}" y="{y:g}"{an}{st}{extra}>{h}</text>')

# ---- 格子 ----
def cell(x, y, cw, ch, kind, rx=1.5):
    st = {
        'mute': 'fill: color-mix(in srgb, var(--fig-mute) 10%, transparent); stroke: var(--fig-mute)',
        'read': 'fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)',
        'later': f'fill: url(#{FID}-hatch); stroke: var(--fig-1)',
        'write': 'fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)',
        'chip': 'fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)',
        'skip': 'fill: none; stroke: var(--fig-mute)',
    }[kind]
    dash = ' stroke-dasharray="3 2"' if kind == 'skip' else ''
    w(f'<rect x="{x:g}" y="{y:g}" width="{cw:g}" height="{ch:g}" rx="{rx}" style="{st}" stroke-width="1"{dash}/>')

MK = {}
def marker(var):
    if var not in MK:
        mid = f"{FID}-m{len(MK)}"
        fill = f'style="fill: var({var})"' if var else 'fill="currentColor" fill-opacity=".65"'
        MK[var] = (mid, f'<marker id="{mid}" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" {fill}/></marker>')
    return MK[var][0]
def arrow(x1, y1, x2, y2, var=None, dash=False, width=1.5):
    st = f'style="stroke: var({var})"' if var else 'stroke="currentColor" stroke-opacity=".6"'
    d = ' stroke-dasharray="5 3"' if dash else ''
    w(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" {st} stroke-width="{width}"{d} marker-end="url(#{marker(var)})"/>')
def path(d, var=None, width=1.5, head=True):
    st = f'style="stroke: var({var})"' if var else 'stroke="currentColor" stroke-opacity=".6"'
    m = f' marker-end="url(#{marker(var)})"' if head else ''
    w(f'<path d="{d}" fill="none" {st} stroke-width="{width}"{m}/>')

# ===================== 上半：数据怎么分给 program（同图 2-1） =====================
TQ, TK, I = 4, 8, 2            # B_q = 2 B_k；画 pid_m = 2 的 program
NJ = 2 * (I + 1)               # causal：这个 program 要读的 K/V block 数（j = 0 … NJ-1）
CW, RH, DW = 22, 44, 44        # 一个 key block 宽、一个 query block 高、d 方向宽
SX, SY = 150, 116              # S 左上角
QX, KY = SX - 24 - DW, SY - 14 - DW
VX = SX + TK * CW + 44
OX = VX + DW + 40
SW, SH = TK * CW, TQ * RH

w(f'<rect class="band" x="4" y="4" width="632" height="{SY + SH + 34}" rx="10"/>')
T('s', 624, 26, 'causal', 'end', 'fill: var(--fig-hi)')
T('t', 16, 26, 'S_{ij} = Q_{i} K_{j}^{ᵀ}，   O_{i} = Σ_{j} P_{ij} V_{j}')

# Kᵀ（d × N_k），在 S 上面
for j in range(TK):
    kind = 'read' if j == 0 else ('later' if j < NJ else 'skip')
    cell(SX + j * CW, KY, CW - 2, DW, kind)
T('tb', SX - 8, KY + 12, 'K^{ᵀ}', 'end')
T('s', SX - 8, KY + 25, 'd × N_{k}', 'end')
arrow(SX, KY - 8, SX + NJ * CW - 2, KY - 8)
T('s', SX + NJ * CW / 2, KY - 14, 'j = 0 … 5', 'middle')
# Q（N_q × d）
for i in range(TQ):
    cell(QX, SY + i * RH, DW, RH - 2, 'read' if i == I else 'mute')
T('tb', QX + DW / 2, SY - 20, 'Q', 'middle')
arrow(24, SY, 24, SY + SH - 2)                                   # 竖着：i = 0 … 3，每个 i 一个 program
T('s', 0, 0, 'i = 0 … 3', 'middle', '', f' transform="translate(16 {SY + SH / 2}) rotate(-90)"')
T('s', QX + DW / 2, SY - 7, 'N_{q} × d', 'middle')
# S：虚线框，只在片上
w(f'<rect x="{SX - 3}" y="{SY - 3}" width="{SW + 4}" height="{SH + 4}" rx="4" fill="none" stroke="currentColor" stroke-opacity=".4" stroke-dasharray="4 3"/>')
for i in range(TQ):
    for j in range(TK):
        full_masked = j * CW >= (i + 1) * RH / 1 and False
        if i == I:
            kind = 'chip' if j < NJ else 'skip'
        else:
            kind = 'skip' if j * CW >= (i + 1) * RH else 'mute'
        cell(SX + j * CW, SY + i * RH, CW - 2, RH - 2, kind)
w(f'<line x1="{SX}" y1="{SY}" x2="{SX + SW}" y2="{SY + SH}" stroke="currentColor" stroke-opacity=".45" stroke-dasharray="4 3"/>')
arrow(SX - 20, SY + I * RH + RH / 2 - 1, SX - 4, SY + I * RH + RH / 2 - 1)
# V（N_k × d）
for j in range(TK):
    kind = 'read' if j == 0 else ('later' if j < NJ else 'skip')
    cell(VX, SY + j * CW, DW, CW - 2, kind)
T('tb', VX + DW / 2, SY - 20, 'V', 'middle')
T('s', VX + DW / 2, SY - 7, 'N_{k} × d', 'middle')
# O（N_q × d）
for i in range(TQ):
    cell(OX, SY + i * RH, DW, RH - 2, 'write' if i == I else 'mute')
T('tb', OX + DW / 2, SY - 20, 'O', 'middle')
T('s', OX + DW / 2, SY - 7, 'N_{q} × d', 'middle')
arrow(SX + SW + 6, SY + I * RH + RH / 2 - 1, VX - 4, SY + I * RH + RH / 2 - 1)
T('s', (SX + SW + VX) / 2, SY + I * RH + RH / 2 - 6, '×', 'middle')
arrow(VX + DW + 4, SY + I * RH + RH / 2 - 1, OX - 4, SY + I * RH + RH / 2 - 1)
T('s', (VX + DW + OX) / 2, SY + I * RH + RH / 2 - 6, '=', 'middle')
# 图例
LX, LY = OX + DW + 18, SY + 4
for k, (kind, txt) in enumerate([('read', '这一轮读的（j = 0）'), ('later', '之后几轮读的'), ('write', '写回的输出'),
                                  ('chip', '片上算的 S 块'), ('skip', 'causal 跳过'), ('mute', '别的 program 的')]):
    cell(LX, LY + k * 20, 12, 12, kind)
    T('s', LX + 18, LY + k * 20 + 10, txt)

SPLIT = len(B)                 # 这之前是图 1-8，之后是图 1-9

# ===================== 中间：每个 program 干什么，以及部分 mask 的 block =====================
Y1 = SY + SH + 46
HB = 182
w(f'<rect class="band" x="4" y="{Y1}" width="632" height="{HB}" rx="10"/>')
T('tb', 16, Y1 + 22, '每个 program 干什么（causal，B_{q} = 2B_{k}）')
FULL = 'fill: color-mix(in srgb, var(--fig-2) 70%, transparent); stroke: var(--fig-1)'          # 整块都算
MASKED = 'fill: color-mix(in srgb, var(--fig-mute) 75%, transparent)'                             # 被 mask 成 −∞
def full(x, y, sz):
    w(f'<rect x="{x}" y="{y}" width="{sz}" height="{sz}" rx="1.5" style="{FULL}" stroke-width="1"/>')
def part(x, y, sz):                        # 左下三角要算，右上三角被 mask
    w(f'<rect x="{x}" y="{y}" width="{sz}" height="{sz}" rx="1.5" style="fill: none; stroke: var(--fig-1)" stroke-width="1"/>')
    w(f'<path d="M{x},{y} L{x},{y + sz} L{x + sz},{y + sz} Z" style="{FULL}" stroke-width="0"/>')
    w(f'<path d="M{x},{y} L{x + sz},{y} L{x + sz},{y + sz} Z" style="{MASKED}"/>')
    w(f'<line x1="{x}" y1="{y}" x2="{x + sz}" y2="{y + sz}" stroke="currentColor" stroke-opacity=".7" stroke-width="1"/>')
def skip(x, y, sz):                        # 整块跳过：空框加 ×
    cell(x, y, sz, sz, 'skip')
    w(f'<path d="M{x + 3},{y + 3} L{x + sz - 3},{y + sz - 3} M{x + sz - 3},{y + 3} L{x + 3},{y + sz - 3}" stroke="currentColor" stroke-opacity=".45" stroke-width="1"/>')
for i in range(TQ):
    y = Y1 + 40 + i * 28
    T('s', 16, y + 13, f'pid_i = {i}', 'start', 'opacity: 1; font-weight: 600' if i == I else '')
    for j in range(TK):
        x = 74 + j * 18
        if j < 2 * i: full(x, y, 16)
        elif j < 2 * i + 2: part(x, y, 16)
        else: skip(x, y, 16)
if True:
    lx = 74
    for k, (fn, txt) in enumerate([(full, '整块都算'), (part, '部分 mask：右上三角设成 −∞'), (skip, '整块跳过')]):
        fn(lx + [0, 90, 290][k], Y1 + 156, 12); T('s', lx + [0, 90, 290][k] + 18, Y1 + 166, txt)
# 放大 pid_m = 2 的两个部分 mask 的 block
ZX, ZY, ZC = 454, Y1 + 50, 10
for b, j in enumerate((2 * I, 2 * I + 1)):
    x0 = ZX + b * 62
    T('s', x0 + 20, ZY - 8, 'S_{' + f'{I},{j}' + '}', 'middle', 'opacity: 1; font-weight: 600')
    for r in range(8):
        for c in range(4):
            keep = (c + 4 * (j - 2 * I)) <= r
            st = FULL if keep else MASKED + '; stroke: var(--fig-mute)'
            w(f'<rect x="{x0 + c * ZC}" y="{ZY + r * ZC}" width="{ZC - 1}" height="{ZC - 1}" rx="1" style="{st}" stroke-width=".8"/>')
w(f'<path d="M{ZX + 62 + 42},{ZY} h4 v{4 * ZC - 1} h-4" fill="none" stroke="currentColor" stroke-opacity=".6"/>')
T('s', ZX + 62 + 50, ZY + 17, '整行', 'start'); T('s', ZX + 62 + 50, ZY + 30, '都是 −∞', 'start')
T('s', ZX - 6, ZY + 4, 'query', 'end'); T('s', ZX - 6, ZY + 16, '↓', 'end')
T('s', ZX + 20, ZY + 8 * ZC + 13, 'key →', 'middle')

SPLIT2 = len(B)                # 图 1-9 只要中栏（causal mask），下半的一轮数据流不画

# ===================== 下半：pid_m = 2 这个 program 的一轮（泳道同图 2-5，不画 grid） =====================
Y0 = Y1 + HB + 16
T('ttl', 10, Y0 + 16, 'pid_i = 2 这个 program，j = 4 这一轮')
MK['__both'] = (f'{FID}-mb', f'<marker id="{FID}-mb" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor" fill-opacity=".65"/></marker>')
def both(x, y1, y2):   # 双向箭头：读出来、写回去
    w(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="currentColor" stroke-opacity=".6" stroke-width="1.5" marker-start="url(#{FID}-mb)" marker-end="url(#{FID}-mb)"/>')

LX0, LW = 10, 620
lanes = [('计算', '单元', Y0 + 26, 82), ('register', '', Y0 + 116, 50), ('shared', 'memory', Y0 + 174, 58), ('显存', '', Y0 + 240, 92)]
for a, b, y, h in lanes:
    w(f'<rect class="band" x="{LX0}" y="{y}" width="{LW}" height="{h}" rx="8"/>')
    T('lab2', 18, y + 18, a)
    if b: T('lab2', 18, y + 31, b)
yC, yR, yS, yM = [l[2] for l in lanes]

OW, OS, OX0 = 92, 108, 92
def op(k, a, b, hi=False, dash=False):
    x = OX0 + k * OS
    st = ' style="stroke: var(--fig-hi); stroke-opacity: 1"' if hi else ''
    d = ' stroke-dasharray="4 3"' if dash else ''
    w(f'<rect class="op" x="{x}" y="{yC + 32}" width="{OW}" height="40" rx="6"{st}{d}/>')
    T('tb', x + OW / 2, yC + 49, a, 'middle'); T('s', x + OW / 2, yC + 64, b, 'middle')
oy, ob = yC + 32, yC + 72
op(0, 'dot', 'Q_{i} K_{j}^{ᵀ}')
op(1, 'mask', 'tl.where', True)
op(2, '① ② ③', 'max·exp2·sum', True)
op(3, 'dot  ④', 'P̃_{ij} V_{j}')
op(4, '÷ ℓ', 'O_{i} / ℓ', dash=True)
for k in range(3):
    arrow(OX0 + k * OS + OW + 1, oy + 20, OX0 + (k + 1) * OS - 2, oy + 20)
cx = lambda k: OX0 + k * OS + OW / 2
path(f'M{cx(0)},{oy - 2} V{yC + 18} H{cx(3)} V{oy - 2}', var='--fig-hi', head=False)
T('s', (cx(0) + cx(3)) / 2, yC + 14, '第 j 轮', 'middle', 'fill: var(--fig-hi); opacity: 1')

def bar(x0, x1, txt, kind):
    st = {'keep': 'fill: color-mix(in srgb, var(--fig-1) 18%, transparent); stroke: var(--fig-1)',
          'tmp': 'fill: color-mix(in srgb, var(--fig-3) 45%, transparent); stroke: var(--fig-2)'}[kind]
    w(f'<rect x="{x0}" y="{yR + 13}" width="{x1 - x0}" height="24" rx="5" style="{st}" stroke-width="1.2"/>')
    T('s', (x0 + x1) / 2, yR + 29, txt, 'middle', 'opacity: 1')
rt, rb = yR + 13, yR + 37
QX, KX, VX2 = 122, 166, 426
bar(96, 148, 'Q_{i}', 'keep')
bar(OX0 + OS, OX0 + OS + OW, 'S_{ij}、P̃_{ij}', 'tmp')
bar(OX0 + 2 * OS, OX0 + 2 * OS + OW, 'm、ℓ', 'keep')
bar(438, OX0 + 3 * OS + OW, 'O_{i}', 'keep')
arrow(QX, rt, QX, ob + 1)                                      # q → dot
both(cx(1), rt - 1, ob + 1)                                    # mask ↕ s、p̃
both(cx(2), rt - 1, ob + 1)                                    # ①②③ ↕ m、ℓ
both((438 + OX0 + 3 * OS + OW) / 2, rt - 1, ob + 1)            # dot ④ ↕ acc

def blk(c, n):
    w(f'<rect x="{c - 22}" y="{yS + 9}" width="44" height="40" rx="5" style="fill: var(--fig-1); stroke: var(--fig-1)"/>')
    T('tb', c, yS + 33, n + '_{j}', 'middle', 'fill: var(--fig-on-1)')
blk(KX, 'K'); blk(VX2, 'V')
w(f'<rect x="{252}" y="{yS + 9}" width="92" height="40" rx="5" fill="none" stroke="currentColor" stroke-opacity=".5" stroke-dasharray="4 3"/>')
T('s', 298, yS + 33, 'K_{j+1}、V_{j+1}', 'middle')
arrow(KX, yS + 9, KX, ob + 1)                                  # K_j → dot
arrow(VX2, yS + 9, VX2, ob + 1)                                # V_j → dot ④

def col(c, y, n, ch, kinds, name):
    for k in range(n):
        cell(c - 13, y + k * ch, 26, ch - 1.5, kinds[k], rx=1)
    T('s', c, y + n * ch + 13, name, 'middle', 'opacity: 1; font-weight: 600')
my = yM + 12
kv = ['read'] + ['later'] * (NJ - 1) + ['skip'] * (TK - NJ)
col(QX, my, TQ, 14, ['read' if i == I else 'mute' for i in range(TQ)], 'Q')
for k in range(TK):                                              # Kᵀ 横着放，第 0 个 block 对着 K_j
    cell(KX - 5.5 + k * 11, my, 10, 26, kv[k], rx=1)
T('s', KX - 5.5 + TK * 11 / 2, my + 69, 'K^{ᵀ}', 'middle', 'opacity: 1; font-weight: 600')
col(VX2, my, TK, 7, kv, 'V')
col(cx(4), my, TQ, 14, ['write' if i == I else 'mute' for i in range(TQ)], 'O')
arrow(QX, my - 2, QX, rb + 1)
arrow(KX, my - 2, KX, yS + 50)
arrow(VX2, my - 2, VX2, yS + 50)
path(f'M{cx(4)},{ob + 1} V{my - 2}', var='--fig-hi')
T('s', cx(4) + 6, yS + 33, 'tl.store', 'start', 'fill: var(--fig-hi); opacity: 1')

# 底下的公式：两列
yT = yM + 92 + 16
w(f'<rect class="band" x="10" y="{yT}" width="620" height="96" rx="8"/>')
T('lab2', 18, yT + 18, '第 j 轮')
T('s', 18, yT + 36, 'Algorithm 1'); T('s', 18, yT + 50, '第 9–10 行')
cols = [['① m_{new} = max(m, rowmax(S_{ij}))',
         '② P̃_{ij} = exp(S_{ij} − m_{new})',
         '③ ℓ ← «exp(m − m_{new})»·ℓ + rowsum(P̃_{ij})'],
        ['④ O_{i} ← «exp(m − m_{new})»·O_{i} + P̃_{ij}V_{j}',
         '⑤ m ← m_{new}',
         '循环后：O_{i} ← O_{i} / ℓ']]
for c, rows in enumerate(cols):
    for k, t in enumerate(rows):
        T('lab', 92 + c * 272, yT + 30 + k * 24, t)
H = yT + 104

LABEL_TOP = ("FlashAttention 的 tiling，按 Triton program 看：i = 2 的 program 读 Q 的第 2 个 block，沿 j 依次读 K、V 的 block，"
             "S 只在片上算，乘 V 以后累加进 O 的第 2 个 block，最后写回；causal 时对角线右上的 block 跳过。")
LABEL_BOT = ("causal 时每个 Q block 一个 program，各自算的 block 数不同：对角线右上方的 block 整块跳过，对角线上的 block 部分 mask。"
             "右边放大了 pid_i = 2 的两个部分 mask 的 block。")
defs = ''.join(m for _, m in MK.values())
DEFS = (f'\n  <defs>{defs}<pattern id="{FID}-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        f'<line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-1)" stroke-opacity=".55" stroke-width="1.3"/></pattern></defs>\n')
HT = SY + SH + 42
Y1T = Y1 - 4
svg_top = (f'<svg class="fg" viewBox="0 0 640 {Y1 + HB + 4}" width="100%" role="img" aria-label="{LABEL_TOP}{LABEL_BOT}">\n' + CSS + DEFS
           + '\n'.join('  ' + x for x in B[:SPLIT2]) + '\n</svg>').replace(f'{FID}-', 'fig-1-8-')   # 上栏 tiling + 中栏 causal mask
svg_bot = (f'<svg class="fg" viewBox="0 0 640 {HB + 8}" width="100%" role="img" aria-label="{LABEL_BOT}">\n' + CSS + DEFS
           + f'  <g transform="translate(0 {-Y1T})">\n' + '\n'.join('  ' + x for x in B[SPLIT:SPLIT2]) + '\n  </g>\n</svg>').replace(f'{FID}-', 'fig-1-9-')

if __name__ == '__main__':
    p = sys.argv[1]
    s = open(p).read()
    for fid, svg in (('fig-1-8', svg_top),):
        s, n = re.subn(r'(<figure id="' + fid + r'"[^>]*>\n)<svg.*?</svg>', lambda m: m.group(1) + svg, s, count=1, flags=re.S)
        assert n == 1, fid
    open(p, 'w').write(s)
    print('H =', Y1 + HB + 4)
