"""对已保存的权重单独打分（训练卡死时的补救手段）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research"
MODEL = "answerdotai/ModernBERT-base"
K = 18


class TokDS(Dataset):
    def __init__(self, ids, att, idx):
        self.ids, self.att, self.idx = ids, att, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(np.asarray(self.ids[j]).astype(np.int64)),
                torch.from_numpy(np.asarray(self.att[j]).astype(np.int64)))


def k2(pidx, idx, k=2, seed=0):
    rng = np.random.default_rng(seed)
    o = np.argsort(pidx[idx], kind="stable")
    i_s, p_s = idx[o], pidx[idx][o]
    keep, st = [], 0
    for i in range(1, len(p_s) + 1):
        if i == len(p_s) or p_s[i] != p_s[st]:
            g = i_s[st:i]
            keep.append(g if len(g) <= k else rng.choice(g, k, replace=False))
            st = i
    return np.concatenate(keep)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = lambda k: np.load(OUT / f"tok_{args.max_len}_{k}.npy", mmap_mode="r")
    ids, att = g("ids"), g("att")
    label = np.asarray(g("label")); fold = np.asarray(g("fold")); pidx = np.asarray(g("prompt_idx"))
    va = np.flatnonzero(fold == args.fold)
    va2 = k2(pidx, va, 2, seed=args.fold)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(Path(args.ckpt) / "model.pt", map_location="cpu"))
    model = model.to(dev).eval()

    res = {}
    for name, idx in (("K=all", va), ("K=2", va2)):
        dl = DataLoader(TokDS(ids, att, idx), batch_size=128, num_workers=0)
        preds = []
        for x, m in dl:
            lg = model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits.float()
            preds.append(lg.argmax(-1).cpu().numpy())
        pred = np.concatenate(preds)
        ba = balanced_accuracy_score(label[idx], pred)
        res[name] = float(ba)
        print(f"  {name}: {ba:.4f}  (n={len(idx)})", flush=True)
    print(f"\n权重来自 {args.ckpt}")


if __name__ == "__main__":
    main()
