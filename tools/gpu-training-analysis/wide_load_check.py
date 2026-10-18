"""每个 thread 一条 ld.global 最多搬多少字节：分别写 4、16、32 B 的 PTX，用 ptxas 编给 sm_90a / sm_100a / sm_120a，看 SASS 里的 LDG（Appendix C）。

需要 CUDA 13 的 ptxas 和 cuobjdump（Triton 自带的 ptxas 只认 PTX 8.7）。
Run: PATH=/path/to/cuda13/bin:$PATH python3 wide_load_check.py
"""
import os, re, subprocess, tempfile

PTX = """.version 9.0
.target {arch}
.address_size 64
.visible .entry k(.param .u64 p){{
 .reg .b64 %rd<3>; .reg .b32 %r<9>;
 ld.param.u64 %rd1, [p];
 cvta.to.global.u64 %rd2, %rd1;
 {load}
 {sums}
 st.global.b32 [%rd2], %r1;
 ret;
}}
"""

for n in (1, 4, 8):                                   # 每个 thread 读 n 个 32-bit
    regs = ",".join(f"%r{i}" for i in range(1, n + 1))
    load = f"ld.global.b32 %r1, [%rd2];" if n == 1 else f"ld.global.v{n}.b32 {{{regs}}}, [%rd2];"
    sums = " ".join(f"add.u32 %r1, %r1, %r{i};" for i in range(2, n + 1))   # 每个 register 都用到，ptxas 才不会缩窄
    for arch in ("sm_90a", "sm_100a", "sm_120a"):
        d = tempfile.mkdtemp()
        src, out = os.path.join(d, "k.ptx"), os.path.join(d, "k.cubin")
        open(src, "w").write(PTX.format(arch=arch, load=load, sums=sums))
        r = subprocess.run(["ptxas", f"-arch={arch}", src, "-o", out], capture_output=True, text=True)
        if r.returncode:
            print(f"{4 * n:2d} B {arch}: ptxas 报错", next(l for l in r.stderr.splitlines() if "error" in l).split("error")[-1].strip(" :"))
            continue
        sass = subprocess.run(["cuobjdump", "-sass", out], capture_output=True, text=True).stdout
        print(f"{4 * n:2d} B {arch}:", " ".join(re.findall(r"LDG[^ ;]*", sass)))
