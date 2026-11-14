"""把压缩 npz 转成可内存映射的 .npy，避免每次训练把 600MB+ 全部载入内存。

长跑卡死的一个可疑诱因是内存压力（系统可用内存一度只剩 3.1/15.6GB）。
mmap 后操作系统按需分页，RSS 能大幅下降。
"""
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "data/research"
for L in (512, 1024):
    src = OUT / f"tok_{L}.npz"
    if not src.exists():
        continue
    z = np.load(src)
    for k in ("ids", "att", "label", "fold", "prompt_idx"):
        dst = OUT / f"tok_{L}_{k}.npy"
        if dst.exists():
            continue
        arr = z[k]
        np.save(dst, arr)
        print(f"  {dst.name}  {arr.shape}  {arr.dtype}")
    print(f"tok_{L}: 已转换\n")
