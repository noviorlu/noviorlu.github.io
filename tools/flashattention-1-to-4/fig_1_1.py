"""图 1-1（两种看矩阵乘法的方式）。配色和图 1-2 一致：深蓝 = 第 1 轮读的，斜线 = 之后几轮读的，
浅蓝 = 片上算的 S、P，红 = 写回的输出。
用法：python3 tools/flashattention-1-to-4/fig_1_1.py content/blog/flashattention-1-to-4/index.md
"""
import re, sys

FID = "fig-1-1"
B = []
def w(s): B.append(s)

OFF = {'n': 0, 'sub': 3.5, 'sup': -5}
FS = {'sub': '72%', 'sup': '72%'}
def parse(t):
    out, buf, i = [], '', 0
    while i < len(t):
        if t[i] in '_^' and i + 1 < len(t) and t[i + 1] == '{':
            if buf: out.append((buf, 'n')); buf = ''
            j = t.index('}', i)
            out.append((t[i + 2:j], 'sub' if t[i] == '_' else 'sup')); i = j + 1
        else:
            buf += t[i]; i += 1
    if buf: out.append((buf, 'n'))
    return out
def T(cls, x, y, t, anchor='start'):
    cur, h = 0, ''
    for k, (s, l) in enumerate(parse(t)):
        if k == 0 and l == 'n':
            h += s; continue
        dy = OFF[l] - cur; cur = OFF[l]
        a = f' dy="{dy:g}"' if dy else ''
        f = f' font-size="{FS[l]}"' if l in FS else ''
        h += f'<tspan{a}{f}>{s}</tspan>'
    w(f'<text class="{cls}" x="{x:g}" y="{y:g}" text-anchor="{anchor}">{h}</text>')

STYLE = {
    'read': 'fill: color-mix(in srgb, var(--fig-1) 32%, transparent); stroke: var(--fig-1)',
    'later': f'fill: url(#{FID}-hatch); stroke: var(--fig-1)',
    'write': 'fill: color-mix(in srgb, var(--fig-hi) 30%, transparent); stroke: var(--fig-hi)',
    'chip': 'fill: color-mix(in srgb, var(--fig-3) 55%, transparent); stroke: var(--fig-2)',
    'chip_later': f'fill: url(#{FID}-hatch2); stroke: var(--fig-2)',
    'write_later': f'fill: url(#{FID}-hatch3); stroke: var(--fig-hi)',
}
C = 24
def cell(x, y, kind, label):
    w(f'<rect x="{x}" y="{y}" width="{C}" height="{C}" rx="1.5" style="{STYLE[kind]}" stroke-width="1"/>')
    T('os2', x + C / 2, y + 16, label, 'middle')
def arc(d):
    w(f'<path d="{d}" fill="none" stroke="currentColor" stroke-opacity=".55" stroke-width="1.4" marker-end="url(#{FID}-ar)"/>')
kj = lambda j: 'read' if j == 0 else 'later'      # 第 1 个 key 在第 1 轮读，其余之后几轮

# ---------- (a) S = q kᵀ ----------
T('os2-b', 8, 20, '(a) 向量乘向量 S = q kᵀ：S 的列是 q 的倍数')
for r in range(3):
    cell(16, 96 + r * C, 'read', 'q_{%d}' % (r + 1))
for j in range(4):
    cell(80 + j * C, 120, kj(j), 'k_{%d}' % (j + 1))
    arc(f'M{92 + j * C},120 C{92 + j * C},{80 - 11 * j} {19 + 6 * j},{80 - 11 * j} {19 + 6 * j},95')
    for r in range(3):
        cell(212 + j * C, 96 + r * C, 'chip' if j == 0 else 'chip_later', 's_{%d%d}' % (r + 1, j + 1))
    T('os2', 224 + j * C, 88, 'k_{%d}q' % (j + 1), 'middle')
T('os2-b', 194, 137, '=', 'middle')
T('os2-b', 28, 184, 'q', 'middle'); T('os2-s', 28, 198, '3 × 1', 'middle')
T('os2-b', 128, 160, 'kᵀ', 'middle'); T('os2-s', 128, 174, '1 × 4', 'middle')
T('os2-b', 260, 184, 'S', 'middle'); T('os2-s', 260, 198, '3 × 4', 'middle')
T('os2-b', 16, 240, 'S 的第 j 列 = k_{j} · q')

# ---------- (b) o = P v ----------
T('os2-b', 330, 20, '(b) 矩阵乘向量 o = P v：o 是 P 各列的组合')
px, py, vx, vy, ox = 340, 96, 500, 84, 572
for j in range(4):
    for r in range(3):
        cell(px + j * C, py + r * C, 'chip' if r == 0 else 'chip_later', 'P_{%d%d}' % (r + 1, j + 1))
    cell(vx, vy + j * C, 'read', 'v_{%d}' % (j + 1))
    yc, cx = vy + j * C + 12, px + j * C + 12
    xj, ty = 486 - 6 * j, 56 + 8 * j          # 先往左到 xj，再往上到 ty，再往左到第 j 列上方，互不交叉
    arc(f'M{vx},{yc} H{xj + 4} Q{xj},{yc} {xj},{yc - 4} V{ty + 4} Q{xj},{ty} {xj - 4},{ty} H{cx + 4} Q{cx},{ty} {cx},{ty + 4} V{py - 1}')
for r in range(3):
    cell(ox, py + r * C, 'write' if r == 0 else 'write_later', 'o_{%d}' % (r + 1))
T('os2-b', 452, 137, '×', 'middle'); T('os2-b', 548, 137, '=', 'middle')
T('os2-b', 388, 190, 'P', 'middle'); T('os2-s', 388, 204, '3 × 4', 'middle')
T('os2-b', 512, 198, 'v', 'middle'); T('os2-s', 512, 212, '4 × 1', 'middle')
T('os2-b', 584, 190, 'o', 'middle'); T('os2-s', 584, 204, '3 × 1', 'middle')
T('os2-b', 330, 240, 'o = v_{1} · P_{:,1} + v_{2} · P_{:,2} + v_{3} · P_{:,3} + v_{4} · P_{:,4}')

LABEL = ("(a) 列向量 q 乘行向量 k 转置等于 3 乘 4 的矩阵 S，箭头从 k 的每个元素 k_j 指向 q，S 的第 j 列等于 k_j 乘 q。"
         "(b) 3 乘 4 的矩阵 P 乘 4 乘 1 的列向量 v 等于 3 乘 1 的列向量 o，箭头从 v 的每个元素 v_j 指向 P 的第 j 列，o 等于 v_j 乘 P 第 j 列之和。"
         "格子换成 block 就是图 1-2：(a) 是 S_ij = Q_i K_j 转置，(b) 是 O_i 等于 P_ij V_j 之和。颜色和图 1-2 一样。")
svg = (f'<figure id="{FID}">\n<svg viewBox="0 0 680 252" width="100%" role="img" aria-label="{LABEL}">\n'
       '  <style>\n    .os2 { font-size: 11px; fill: currentColor; }\n    .os2-b { font-size: 12px; font-weight: 600; fill: currentColor; }\n'
       '    .os2-s { font-size: 10.5px; fill: currentColor; opacity: .72; }\n  </style>\n'
       f'<defs><marker id="{FID}-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor" fill-opacity=".6"/></marker>'
       f'<pattern id="{FID}-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-1)" stroke-opacity=".55" stroke-width="1.3"/></pattern>'
       f'<pattern id="{FID}-hatch2" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-2)" stroke-opacity=".6" stroke-width="1.3"/></pattern>'
       f'<pattern id="{FID}-hatch3" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" style="stroke: var(--fig-hi)" stroke-opacity=".6" stroke-width="1.3"/></pattern></defs>\n'
       + '\n'.join(B) + '\n</svg>')

if __name__ == '__main__':
    p = sys.argv[1]
    s = open(p).read()
    s, n = re.subn(r'<figure id="fig-1-1">\n<svg.*?</svg>', lambda m: svg, s, count=1, flags=re.S)
    assert n == 1
    s = re.sub(r'(<figcaption><strong>图 1-1</strong>)(.*?)(</figcaption>)', lambda m: m.group(1) +

        ' 两种看矩阵乘法的方式；格子换成 block 就是<a href="#fig-1-2">图 1-2</a>的两次乘法，配色也相同。' + m.group(3), s, count=1, flags=re.S)
    open(p, 'w').write(s)
