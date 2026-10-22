"""图 1-2（FlashAttention 的 tiling，按 Triton program 看）。画法和配色沿用 gpu-triton-intro 的图 2-1、2-5：
深蓝 = 这个 program 读的数据，红 = 它写回的输出，斜线 = 之后几轮才读的 block，灰 = 别的 program 的。
用法：python3 tools/flashattention-1-to-4/fig_1_2.py content/blog/flashattention-1-to-4/index.md
会把文中 <figure id="fig-1-2" ...> ... </svg> 整段换掉，figcaption 不动。
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
def T(cls, x, y, t, anchor='start', style='', extra=''):
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

w(f'<rect class="band" x="4" y="4" width="632" height="{SY + SH + 68}" rx="10"/>')
T('tb', 16, 26, '数据怎么分给 program')
T('s', 624, 26, '以 pid_m = 2 为例，causal', 'end', 'fill: var(--fig-hi)')

# Kᵀ（d × N_k），在 S 上面
for j in range(TK):
    kind = 'read' if j == 0 else ('later' if j < NJ else 'skip')
    cell(SX + j * CW, KY, CW - 2, DW, kind)
T('tb', SX - 8, KY + 12, 'K^{ᵀ}', 'end')
T('s', SX - 8, KY + 25, 'd × N_{k}', 'end')
arrow(SX, KY - 8, SX + NJ * CW - 2, KY - 8)
T('s', SX + NJ * CW / 2, KY - 14, 'j = 0 … 5，在 program 里串行走', 'middle')
# Q（N_q × d）
for i in range(TQ):
    cell(QX, SY + i * RH, DW, RH - 2, 'read' if i == I else 'mute')
T('tb', QX + DW / 2, SY - 20, 'Q', 'middle')
T('s', QX + DW / 2, SY - 7, 'N_{q} × d', 'middle')
T('s', QX - 6, SY + I * RH + RH / 2 + 4, 'pid_m = 2', 'end', 'fill: var(--fig-1); opacity: 1; font-weight: 600')
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
T('s', SX + SW / 2, SY + SH + 18, 'S = QK^{ᵀ}/√d：只在片上算，不写回显存', 'middle')
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
T('s', 16, SY + SH + 40, '一个 Q block 一个 program：Q_{i} 读一次，沿 j 每轮读一对 K_{j}、V_{j}，S 和 P 都只在片上，最后写回 O_{i}。')
T('s', 16, SY + SH + 55, 'softmax 按行做，一行正好落在一个 program 里，所以不需要和别的 program 通信。')

# ===================== 中间：每个 program 干什么，以及部分 mask 的 block =====================
Y1 = SY + SH + 84
HB = 182
w(f'<rect class="band" x="4" y="{Y1}" width="632" height="{HB}" rx="10"/>')
T('tb', 16, Y1 + 22, '每个 program 干什么（causal，B_{q} = 2B_{k}）')
T('s', 624, Y1 + 22, '一行一个 program，互相独立、并行', 'end', 'fill: var(--fig-hi)')
def part(x, y, sz):
    cell(x, y, sz, sz, 'chip')
    w(f'<path d="M{x},{y} L{x + sz},{y} L{x + sz},{y + sz} Z" style="fill: color-mix(in srgb, var(--fig-mute) 45%, transparent)"/>')
for i in range(TQ):
    y = Y1 + 40 + i * 28
    T('s', 16, y + 13, f'pid_m = {i}', 'start', 'opacity: 1; font-weight: 600' if i == I else '')
    for j in range(TK):
        x = 74 + j * 18
        if j < 2 * i: cell(x, y, 16, 16, 'chip')
        elif j < 2 * i + 2: part(x, y, 16)
        else: cell(x, y, 16, 16, 'skip')
    T('s', 74 + TK * 18 + 6, y + 13, f'算 {2 * i + 2} 块，j = {2 * i}、{2 * i + 1} 部分 mask')
if True:
    lx = 74
    for k, (fn, txt) in enumerate([(lambda x, y: cell(x, y, 12, 12, 'chip'), '整块都算'), (lambda x, y: part(x, y, 12), '部分 mask'), (lambda x, y: cell(x, y, 12, 12, 'skip'), '整块跳过')]):
        fn(lx + k * 90, Y1 + 156); T('s', lx + k * 90 + 18, Y1 + 166, txt)
# 放大 pid_m = 2 的两个部分 mask 的 block
ZX, ZY, ZC = 454, Y1 + 50, 10
for b, j in enumerate((2 * I, 2 * I + 1)):
    x0 = ZX + b * 62
    T('s', x0 + 20, ZY - 8, 'S_{' + f'{I},{j}' + '}', 'middle', 'opacity: 1; font-weight: 600')
    for r in range(8):
        for c in range(4):
            keep = (c + 4 * (j - 2 * I)) <= r
            st = 'fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)' if keep else 'fill: color-mix(in srgb, var(--fig-mute) 40%, transparent); stroke: var(--fig-mute)'
            w(f'<rect x="{x0 + c * ZC}" y="{ZY + r * ZC}" width="{ZC - 1}" height="{ZC - 1}" rx="1" style="{st}" stroke-width=".8"/>')
w(f'<path d="M{ZX + 62 + 42},{ZY} h4 v{4 * ZC - 1} h-4" fill="none" stroke="currentColor" stroke-opacity=".6"/>')
T('s', ZX + 62 + 50, ZY + 17, '整行', 'start'); T('s', ZX + 62 + 50, ZY + 30, '都是 −∞', 'start')
T('s', ZX - 6, ZY + 4, 'query', 'end'); T('s', ZX - 6, ZY + 16, '↓', 'end')
T('s', ZX + 20, ZY + 8 * ZC + 13, 'key →', 'middle')

# ===================== 下半：一个 program 里面（同图 2-5） =====================
Y0 = Y1 + HB + 16
T('ttl', 10, Y0 + 16, 'grid')
T('ttl', 162, Y0 + 16, '一个 program（pid_m = 2）的 j = 4 这一轮')
T('tb', 630, Y0 + 16, 'block 乘加 + softmax', 'end', 'fill: var(--fig-hi)')
for r in range(TQ):
    for c in range(3):
        on = (r == I and c == 0)
        st = 'fill: color-mix(in srgb, var(--fig-1) 30%, transparent); stroke: var(--fig-1)' if on else 'fill: color-mix(in srgb, var(--fig-mute) 12%, transparent); stroke: var(--fig-mute)'
        x, y = 10 + c * 42, Y0 + 28 + r * 27
        w(f'<rect x="{x}" y="{y}" width="38" height="23" rx="4" style="{st}" stroke-width="1.4"/>')
        if on: T('s', x + 19, y + 15, '(2,0)', 'middle')
for k, t in enumerate(['每个 Q block', '一个 program', 'grid =', '(N_{q} / BLOCK_M, B·H)', '例：N = 4096、', 'BLOCK_M = 128、', 'B·H = 32', '→ 1024 个 program']):
    T('lab2', 10, Y0 + 150 + k * 15 + (6 if k >= 2 else 0) + (6 if k >= 4 else 0), t)
T('lab2', 10, Y0 + 290, '行：pid_m')
T('lab2', 10, Y0 + 305, '列：pid_bh')
arrow(136, Y0 + 28 + I * 27 + 11, 144, Y0 + 28 + I * 27 + 11)

LX0, LW = 146, 490
lanes = [('计算', '单元', Y0 + 26, 70), ('register', '', Y0 + 104, 76), ('shared', 'memory', Y0 + 188, 62), ('显存', 'GDDR / HBM', Y0 + 258, 112)]
for a, b, y, h in lanes:
    w(f'<rect class="band" x="{LX0}" y="{y}" width="{LW}" height="{h}" rx="8"/>')
    T('lab2', 152, y + 16, a)
    if b: T('lab2', 152, y + 29, b)
yC, yR, yS, yM = [l[2] for l in lanes]

def op(x, y, wd, a, b, hi=False):
    st = ' style="stroke: var(--fig-hi); stroke-opacity: 1"' if hi else ''
    w(f'<rect class="op" x="{x}" y="{y}" width="{wd}" height="38" rx="6"{st}/>')
    T('tb', x + wd / 2, y + 16, a, 'middle'); T('s', x + wd / 2, y + 31, b, 'middle')
oy = yC + 24
op(210, oy, 60, 'dot', 'S = q·k^{ᵀ}')
op(276, oy, 66, 'mask', 'tl.where', True)
op(348, oy, 84, '① ② ③', 'max·exp2·sum', True)
op(438, oy, 124, 'dot  ④', 'exp(m−m_{new})·acc + p̃·v')
op(568, oy, 62, '÷ ℓ', '最后一次')
arrow(271, oy + 19, 274, oy + 19); arrow(343, oy + 19, 346, oy + 19); arrow(433, oy + 19, 436, oy + 19)
path(f'M{240},{oy - 2} V{yC + 14} H{500} V{oy - 2}', var='--fig-hi', head=False)
T('s', 358, yC + 10, '第 j 轮；对角线上的 block 多一步 mask', 'middle', 'fill: var(--fig-hi); opacity: 1')
T('s', 599, yC + 16, '循环结束后', 'middle')

def bar(x, y, wd, txt, kind):
    st = {'keep': 'fill: color-mix(in srgb, var(--fig-1) 18%, transparent); stroke: var(--fig-1)',
          'tmp': 'fill: color-mix(in srgb, var(--fig-3) 45%, transparent); stroke: var(--fig-2)'}[kind]
    w(f'<rect x="{x}" y="{y}" width="{wd}" height="22" rx="5" style="{st}" stroke-width="1.2"/>')
    T('s', x + wd / 2, y + 15, txt, 'middle', 'opacity: 1')
r1, r2 = yR + 10, yR + 42
bar(196, r1, 56, 'q', 'keep')
bar(262, r1, 188, 's、p̃（每轮重算，用完就丢）', 'tmp')
bar(460, r1, 60, 'acc', 'keep')
bar(330, r2, 120, 'm、ℓ（每行一个）', 'keep')
T('s', 224, r2 + 9, 'B_{q} × d', 'middle')
T('s', 490, r2 + 9, 'B_{q} × d', 'middle')
arrow(224, r1, 224, oy + 39)                       # q → dot
arrow(266, oy + 39, 266, r1)                       # dot → s
arrow(320, r1, 320, oy + 39)                       # s → mask → softmax
arrow(400, oy + 39, 400, r1)                       # softmax → p̃
arrow(446, r1, 446, oy + 39)                       # p̃ → dot ④
arrow(480, r1, 480, oy + 39); arrow(500, oy + 39, 500, r1)   # acc ↔ dot ④
path(f'M{520},{r1 + 11} H{580} V{oy + 40}')        # acc → ÷ ℓ
path(f'M{450},{r2 + 11} H{606} V{oy + 40}')        # ℓ → ÷ ℓ

def blk(x, y, n):
    w(f'<rect x="{x}" y="{y}" width="44" height="40" rx="5" style="fill: var(--fig-1); stroke: var(--fig-1)"/>')
    T('tb', x + 22, y + 18, n + '_{j}', 'middle', 'fill: var(--fig-on-1)')
    T('s', x + 22, y + 32, 'B_{k}×d', 'middle', 'fill: var(--fig-on-1); opacity: .8')
sy0 = yS + 11
blk(236, sy0, 'K'); blk(433, sy0, 'V')
arrow(258, sy0, 258, oy + 39)                      # K_j → dot S
arrow(455, sy0, 455, oy + 39)                      # V_j → dot ④
w(f'<rect x="490" y="{sy0}" width="108" height="40" rx="5" fill="none" stroke="currentColor" stroke-opacity=".5" stroke-dasharray="4 3"/>')
T('s', 544, sy0 + 17, 'K_{j+1}、V_{j+1} 正在搬', 'middle'); T('s', 544, sy0 + 31, 'num_stages = 2', 'middle')
T('s', 296, sy0 + 17, '两个 slot 轮流用：', 'start'); T('s', 296, sy0 + 31, '算第 j 轮时 j + 1 在搬', 'start')

def col(x, y, n, ch, cwid, kinds, name, sub):
    for k in range(n):
        cell(x, y + k * ch, cwid, ch - 1.5, kinds[k], rx=1)
    T('s', x + cwid / 2, y + n * ch + 13, name, 'middle', 'opacity: 1; font-weight: 600')
    T('s', x + cwid / 2, y + n * ch + 26, sub, 'middle')
my = yM + 12
kv = ['read'] + ['later'] * (NJ - 1) + ['skip'] * (TK - NJ)
col(211, my, TQ, 14, 26, ['read' if i == I else 'mute' for i in range(TQ)], 'Q', '')
col(245, my, TK, 7, 26, kv, 'K', '')
col(442, my, TK, 7, 26, kv, 'V', '')
col(604, my, TQ, 14, 26, ['write' if i == I else 'mute' for i in range(TQ)], 'O', '')
arrow(224, my - 2, 224, r1 + 23)                   # Q_i → q（直接进 register）
arrow(258, my - 2, 258, sy0 + 41)                  # K → slot
arrow(455, my - 2, 455, sy0 + 41)                  # V → slot
T('s', 282, my + 12, 'tl.load：', 'start')
T('s', 282, my + 26, 'Q_{i} 只读一次，', 'start')
T('s', 282, my + 40, 'K_{j}、V_{j} 每轮一对', 'start')
T('s', 476, my + 12, 'S、P 不进显存；', 'start')
T('s', 476, my + 26, '只多存每行一个', 'start')
T('s', 476, my + 40, 'logsumexp 给反向', 'start')
path(f'M{617},{oy + 40} V{my - 2}', var='--fig-hi')    # ÷ ℓ → O_i
T('s', 610, sy0 + 17, 'tl.store', 'end', 'fill: var(--fig-hi); opacity: 1') if False else None
T('s', 611, r2 + 32, 'tl.store 只写一次', 'end', 'fill: var(--fig-hi); opacity: 1')

# 底下的步骤
yT = yM + 112 + 26
T('ttl', 10, yT, 'online softmax 在第 j 轮做的事（m、ℓ 每行一个数，acc 每行一起乘）')
rows = [('① m_{new} = max(m, rowmax(S_{ij}))', '这一块进来以后的行最大值'),
        ('② P̃_{ij} = exp(S_{ij} − m_{new})', '按新最大值取指数，还没除以分母'),
        ('③ ℓ ← exp(m − m_{new})·ℓ + rowsum(P̃_{ij})', '旧分母对齐到新最大值，再加这一块'),
        ('④ acc ← exp(m − m_{new})·acc + P̃_{ij}V_{j}', '旧输出乘同一个 exp(m − m_{new}) 对齐，再加这一块'),
        ('⑤ m ← m_{new}', '进入下一个 j'),
        ('扫完：O_{i} = acc / ℓ', '只在最后除一次，tl.store 写回')]
for k, (a, b) in enumerate(rows):
    T('lab', 24, yT + 22 + k * 18, a)
    T('s', 330, yT + 22 + k * 18, b)
T('lab2', 10, yT + 22 + 6 * 18 + 14, 'Triton 里看不到 warp：写的是整个 block 的 tl.dot，块内怎么把行分给 num_warps 个 warp 由编译器决定。')
H = yT + 22 + 6 * 18 + 26

LABEL = ("FlashAttention 的 tiling，按 Triton program 看。上：pid_m = 2 的 program 读 Q 的第 2 个 block，沿 j 依次读 K、V 的 block，"
         "S 只在片上算，最后写回 O 的第 2 个 block；causal 时对角线右上的 block 跳过。下：grid 每个 Q block 一个 program；program 里每轮 "
         "dot 算 S，online softmax 更新 m、ℓ，再 dot 乘 V 累加进 acc；K、V 经 shared memory 的两个 slot 轮流搬，q、acc、m、ℓ 一直在 register，扫完除以 ℓ 写回 O。")
defs = ''.join(m for _, m in MK.values())
svg = (f'<figure id="{FID}" class="fg-fig">\n<svg class="fg" viewBox="0 0 640 {H}" width="100%" role="img" aria-label="{LABEL}">\n' + CSS +
       f'\n  <defs>{defs}<pattern id="{FID}-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-1)" stroke-opacity=".55" stroke-width="1.3"/></pattern></defs>\n'
       + '\n'.join('  ' + x for x in B) + '\n</svg>')

if __name__ == '__main__':
    p = sys.argv[1]
    s = open(p).read()
    new, n = re.subn(r'<figure id="fig-1-2"[^>]*>\n<svg.*?</svg>', lambda m: svg, s, count=1, flags=re.S)
    assert n == 1
    open(p, 'w').write(new)
    print('H =', H)
