"""图 1-12（FlashAttention-2 反向的 tiling，按 Triton program 看）。画法、配色和格子沿用图 1-8（fig_1_2.py），
但每个 program 负责 S 的一列 block：K_j、V_j 只读一次，沿 i 读 Q、dO、L、D；dK_j、dV_j 在 tile 里累加，扫完写一次；
dQ_i 每一轮都加回 HBM（atomic add）。causal，B_q = 2 B_k，和图 1-8 同一个 S。
用法：python3 tools/flashattention-1-to-4/fig_1_12.py content/blog/flashattention-1-to-4/index.md
会把文中 <figure id="fig-1-12" ...> 里的 <svg> 换掉，figcaption 不动。
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_1_2 as g

g.B.clear(); g.MK.clear()
T, cell, block, arrow, w = g.T, g.cell, g.block, g.arrow, g.w

TQ, TK, J = 4, 8, 5            # 画 j = 5 的 program
I0 = J // 2                    # B_q = 2 B_k：这一列从 i = 2 开始要算
CW, RH, DW, VW = 22, 44, 30, 7 # 一个 key block 宽、一个 query block 高、d 方向宽、向量宽
SX, SY = 176, 148
SW, SH = TK * CW, TQ * RH
VY = 58; KY = VY + DW + 8      # 上面：Vᵀ、Kᵀ
BY1 = SY + SH + 24; BY2 = BY1 + DW + 8   # 下面：dKᵀ、dVᵀ
QX = SX - 16 - DW; DX = QX - 12 - VW; LX = DX - 8 - VW; OX = LX - 14 - DW   # 左边：dO、L、D、Q
RX = SX + SW + 52              # 右边：dQ
HTOT = BY2 + DW + 70
yc = lambda i: SY + i * RH + RH / 2 - 1
rowkind = lambda i: 'skip' if i < I0 else ('read' if i == I0 else 'later')

w(f'<rect class="band" x="4" y="4" width="632" height="{HTOT - 8}" rx="10"/>')
T('s', 624, 26, 'causal', 'end', 'fill: var(--fig-hi)')
T('t', 16, 26, 'dQ_{i} = Σ_{j} dS_{ij} K_{j}，  dK_{j} = Σ_{i} dS_{ij}^{ᵀ} Q_{i}，  dV_{j} = Σ_{i} P_{ij}^{ᵀ} dO_{i}')

# 上面：Vᵀ、Kᵀ（d × N_k），这个 program 只读第 j 个 block
for y, n in ((VY, 'V'), (KY, 'K')):
    for j in range(TK):
        cell(SX + j * CW, y, CW - 2, DW, 'read' if j == J else 'mute')
    T('tb', SX + SW + 8, y + 13, n + '^{ᵀ}')
    T('s', SX + SW + 8, y + 26, 'd × N_{k}')
T('s', SX + J * CW + (CW - 2) / 2, VY - 7, f'j = {J} 的 program', 'middle', 'opacity: 1; font-weight: 600')

# 左边：dO、L、D、Q（都按 query 切），沿 i 读
for x, bw, n in ((OX, DW, 'dO'), (LX, VW, 'L'), (DX, VW, 'D'), (QX, DW, 'Q')):
    for i in range(TQ):
        cell(x, SY + i * RH, bw, RH - 2, rowkind(i))
    T('tb', x + bw / 2, SY - 20, n, 'middle')
T('s', OX + DW / 2, SY - 7, 'N_{q} × d', 'middle')
T('s', QX + DW / 2, SY - 7, 'N_{q} × d', 'middle')
arrow(32, SY + I0 * RH, 32, SY + SH - 2)                          # 内层循环：i = 2 … 3
T('s', 0, 0, f'i = {I0} … {TQ - 1}', 'middle', '', f' transform="translate(24 {SY + (I0 * RH + SH) / 2}) rotate(-90)"')
for i in range(I0, TQ):
    arrow(QX + DW + 3, yc(i), SX - 3, yc(i))

# S：和图 1-8 同一个 causal 的 S，当前 program 那一列加框，其余列淡一点
w(f'<rect x="{SX - 3}" y="{SY - 3}" width="{SW + 4}" height="{SH + 4}" rx="4" fill="none" stroke="currentColor" stroke-opacity=".4" stroke-dasharray="4 3"/>')
for j in range(TK):
    if j != J: w('<g opacity=".45">')
    for i in range(TQ):
        block(SX + j * CW, SY + i * RH, CW - 2, RH - 2, SX, SY)
    if j != J: w('</g>')
w(f'<line x1="{SX}" y1="{SY}" x2="{SX + SW}" y2="{SY + SH}" stroke="currentColor" stroke-opacity=".55" stroke-dasharray="4 3"/>')
w(f'<rect x="{SX + J * CW - 2}" y="{SY - 2}" width="{CW + 2}" height="{SH + 2}" rx="3" fill="none" stroke="currentColor" stroke-width="1.6"/>')

# 右边：dQ（N_q × d），每一轮加回 HBM
for i in range(TQ):
    cell(RX, SY + i * RH, DW, RH - 2, 'write' if i >= I0 else 'mute')
T('tb', RX + DW / 2, SY - 20, 'dQ', 'middle')
T('s', RX + DW / 2, SY - 7, 'N_{q} × d', 'middle')
for i in range(I0, TQ):
    arrow(SX + SW + 6, yc(i), RX - 4, yc(i), var='--fig-hi')
tx = RX + DW + 10
T('s', tx, yc(I0) - 6, 'atomic add：', 'start', 'opacity: 1; font-weight: 600; fill: var(--fig-hi)')
T('s', tx, yc(I0) + 9, '每一轮把 dS_{ij} K_{j}')
T('s', tx, yc(I0) + 23, '加到 HBM 里的 dQ_{i} 上，')
T('s', tx, yc(I0) + 37, '别的 program 也在加')

# 下面：dKᵀ、dVᵀ（d × N_k），在 tile 里累加，扫完写一次
for y, n in ((BY1, 'dK'), (BY2, 'dV')):
    for j in range(TK):
        cell(SX + j * CW, y, CW - 2, DW, 'write' if j == J else 'mute')
    T('tb', SX + SW + 8, y + 13, n + '^{ᵀ}')
    T('s', SX + SW + 8, y + 26, 'd × N_{k}')
arrow(SX + J * CW + (CW - 2) / 2, SY + SH + 4, SX + J * CW + (CW - 2) / 2, BY1 - 3)
T('s', tx, BY1 + 13, f'dK_{{{J}}}、dV_{{{J}}}：', 'start', 'opacity: 1; font-weight: 600')
T('s', tx, BY1 + 28, '在 tile 里对 i 累加，')
T('s', tx, BY1 + 42, '扫完这一列写一次')

# 图例：两行
LY = BY2 + DW + 24
items = [[('read', f'这一轮读的（i = {I0}）'), ('later', '之后几轮读的'), ('write', '写回或加回 HBM 的'), ('mute', '别的 program 的')],
         [('full', '整块都算'), ('part', '部分 mask（−∞）'), ('skip', '整块跳过')]]
for r, row in enumerate(items):
    x = 16
    for kind, txt in row:
        y = LY + r * 20
        if kind == 'full': block(x, y, 12, 12, x + 100, y)
        elif kind == 'part': block(x, y, 12, 12, x, y)
        elif kind == 'skip': block(x, y, 12, 12, x - 100, y)
        else: cell(x, y, 12, 12, kind)
        T('s', x + 18, y + 10, txt)
        x += 18 + sum(10.5 if ord(ch) > 0x2E80 else 5.6 for ch in txt) + 22

LABEL = (f"FlashAttention-2 反向的 tiling，按 Triton program 看，causal：j = {J} 的 program 只读一次 K、V 的第 {J} 个 block，"
         f"沿 i 依次读 Q、dO、L、D 的 block；dK_{J}、dV_{J} 在 tile 里累加，扫完写回；每一轮把 dS_ij K_j 用 atomic add 加到 HBM 里的 dQ_i 上。"
         f"这一列在对角线右上方的 i = 0、1 两个 block 整块跳过，i = {I0} 部分 mask，i = 3 整块都算。")
defs = ''.join(m for _, m in g.MK.values())
DEFS = (f'\n  <defs>{defs}<pattern id="{g.FID}-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        f'<line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-1)" stroke-opacity=".55" stroke-width="1.3"/></pattern></defs>\n')
svg = (f'<svg class="fg" viewBox="0 0 640 {HTOT}" width="100%" role="img" aria-label="{LABEL}">\n' + g.CSS + DEFS
       + '\n'.join('  ' + x for x in g.B) + '\n</svg>').replace(f'{g.FID}-', 'fig-1-12-')

if __name__ == '__main__':
    p = sys.argv[1]
    s = open(p).read()
    s, n = re.subn(r'(<figure id="fig-1-12"[^>]*>\n)<svg.*?</svg>', lambda m: m.group(1) + svg, s, count=1, flags=re.S)
    assert n == 1, 'fig-1-12'
    open(p, 'w').write(s)
    print('H =', HTOT)
