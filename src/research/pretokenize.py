"""一次性把语料分词成定长数组，去掉训练循环里的分词开销。

实测：训练 1.67s/step，瓶颈是 __getitem__ 里的实时分词（单线程），
GPU 只用了 2.3/8.2GB。预分词后可以把 bs 提到 64，吞吐提升数倍。

输出 data/research/tok_<maxlen>.npz：input_ids / attn / label / fold / prompt_idx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import LABEL2IDX  # noqa: E402

CORPUS = ROOT / "data/research/corpus_long.parquet"
OUT = ROOT / "data/research"
MODEL = "answerdotai/ModernBERT-base"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    df = pd.read_parquet(CORPUS)
    tok = AutoTokenizer.from_pretrained(MODEL)
    n, L = len(df), args.max_len
    ids = np.zeros((n, L), dtype=np.int32)
    att = np.zeros((n, L), dtype=np.int8)

    t0 = time.time()
    texts = df.text.tolist()
    B = 2048
    for b0 in range(0, n, B):
        enc = tok(texts[b0:b0 + B], truncation=True, max_length=L,
                  padding="max_length", return_attention_mask=True)
        m = len(enc["input_ids"])
        ids[b0:b0 + m] = np.asarray(enc["input_ids"], dtype=np.int32)
        att[b0:b0 + m] = np.asarray(enc["attention_mask"], dtype=np.int8)
        if (b0 // B) % 10 == 0:
            print(f"  {b0 + m}/{n}  {time.time()-t0:.0f}s", flush=True)

    pids = df.prompt_id.unique()
    pmap = {p: i for i, p in enumerate(pids)}
    prompt_idx = df.prompt_id.map(pmap).to_numpy(dtype=np.int32)
    label = df.label.map(LABEL2IDX).to_numpy(dtype=np.int16)
    fold = df.fold.to_numpy(dtype=np.int8)

    outp = OUT / f"tok_{L}.npz"
    np.savez_compressed(outp, ids=ids, att=att, label=label, fold=fold, prompt_idx=prompt_idx)
    print(f"已写 {outp}  shape={ids.shape}  {time.time()-t0:.0f}s")
    print(f"平均有效 token 长度: {att.sum(1).mean():.0f} / {L}")


if __name__ == "__main__":
    main()
