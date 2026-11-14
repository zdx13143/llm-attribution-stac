"""独立评估 ArcFace 模型（训练脚本内评估在本机会卡死，故拆开）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_arcface import DS, subsample_per_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research"
K = 18


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--prefix", type=str, default="tok")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = lambda k, p="tok": np.load(OUT / f"{p}_{args.max_len}_{k}.npy", mmap_mode="r")
    ids, att = g("ids", args.prefix), g("att", args.prefix)
    label = np.asarray(g("label")); fold = np.asarray(g("fold")); pidx = np.asarray(g("prompt_idx"))

    va = np.flatnonzero(fold == args.fold)
    va2 = subsample_per_prompt(pidx, va, 2, seed=args.fold)

    meta = json.loads((Path(args.ckpt) / "meta.json").read_text(encoding="utf-8"))
    from train_arcface import Net
    from transformers import AutoModelForSequenceClassification
    src = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-base", num_labels=K, ignore_mismatched_sizes=True)
    net = Net(src.base_model, dim=768, s=meta.get("s", 30.0), m=meta.get("m", 0.3))
    net.load_state_dict(torch.load(Path(args.ckpt) / "model.pt", map_location="cpu"))
    net = net.to(dev).eval()
    print(f"权重 {args.ckpt} (s={meta.get('s')}, m={meta.get('m')})", flush=True)

    for name, idx in (("K=all", va), ("K=2", va2)):
        out = []
        dl = DataLoader(DS(ids, att, label, idx), batch_size=args.bs, num_workers=0)
        with torch.no_grad():
            for x, m, _ in dl:
                out.append(net(x.to(dev), m.to(dev)).argmax(-1).cpu().numpy())
        pred = np.concatenate(out)
        print(f"  {name}: {balanced_accuracy_score(label[idx], pred):.4f}  (n={len(idx)})", flush=True)


if __name__ == "__main__":
    main()
