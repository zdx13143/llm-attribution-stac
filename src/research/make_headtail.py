"""生成「前 512 + 后 512」的 1024 token 版本，用于检验结尾是否带来额外信息。

对不足 1024 token 的文本：整篇都保留（head+tail 会重叠中间，无损失）。
对超长文本：保留开头 511 + 结尾 510，中间丢弃。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research"
MODEL = "answerdotai/ModernBERT-base"
L = 1024
HALF = 511          # 各留 511/510，加上两个特殊 token 正好 1024


def main() -> None:
    df = pd.read_parquet(OUT / "corpus_long.parquet")
    tok = AutoTokenizer.from_pretrained(MODEL)
    cls_id, sep_id, pad_id = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id

    n = len(df)
    ids = np.zeros((n, L), dtype=np.int32)
    att = np.zeros((n, L), dtype=np.int8)
    texts = df.text.tolist()

    t0 = time.time()
    n_trunc = 0
    for i, t in enumerate(texts):
        raw = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        if len(raw) > L - 2:
            n_trunc += 1
            body = raw[:HALF] + raw[-HALF:]
        else:
            body = raw
        seq = [cls_id] + body + [sep_id]
        m = len(seq)
        ids[i, :m] = seq
        att[i, :m] = 1
        if i % 20000 == 0:
            print(f"  {i}/{n}  已截断 {n_trunc}  {time.time()-t0:.0f}s", flush=True)

    np.save(OUT / f"ht_{L}_ids.npy", ids)
    np.save(OUT / f"ht_{L}_att.npy", att)
    print(f"完成：{n} 条，其中 {n_trunc} 条需要前后拼接 ({n_trunc/n*100:.1f}%)  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
