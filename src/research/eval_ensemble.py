"""把普通 CE 模型和 ArcFace 模型的预测做集成，看错误是否互补。

两个模型同骨干同输入，只有损失函数不同 —— 若错误的样本不重叠，
概率平均后应高于任一单模型。零训练成本。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_arcface import Net, subsample_per_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research"
MODEL = "answerdotai/ModernBERT-base"
K = 18


class DS(Dataset):
    def __init__(self, ids, att, idx):
        self.ids, self.att, self.idx = ids, att, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(np.asarray(self.ids[j]).astype(np.int64)),
                torch.from_numpy(np.asarray(self.att[j]).astype(np.int64)))


@torch.no_grad()
def probs_ce(model, ids, att, idx, dev, bs=64):
    model.eval()
    out = []
    for x, m in DataLoader(DS(ids, att, idx), batch_size=bs, num_workers=0):
        lg = model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits.float()
        out.append(torch.softmax(lg, -1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def probs_arc(net, ids, att, idx, dev, bs=64):
    net.eval()
    out = []
    for x, m in DataLoader(DS(ids, att, idx), batch_size=bs, num_workers=0):
        lg = net(x.to(dev), m.to(dev)).float()
        out.append(torch.softmax(lg, -1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = lambda k: np.load(OUT / f"tok_{args.max_len}_{k}.npy", mmap_mode="r")
    ids, att = g("ids"), g("att")
    label = np.asarray(g("label")); fold = np.asarray(g("fold")); pidx = np.asarray(g("prompt_idx"))
    va2 = subsample_per_prompt(pidx, np.flatnonzero(fold == args.fold), 2, seed=args.fold)
    y = label[va2]
    print(f"K=2 子集 {len(va2)} 条", flush=True)

    ce = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True)
    ce.load_state_dict(torch.load(OUT / "ckpt/f0_L1024_step/model.pt", map_location="cpu"))
    ce = ce.to(dev)
    P_ce = probs_ce(ce, ids, att, va2, dev)
    del ce; torch.cuda.empty_cache()
    print(f"  CE     : {balanced_accuracy_score(y, P_ce.argmax(1)):.4f}", flush=True)

    src = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True)
    net = Net(src.base_model, dim=768, s=30.0, m=0.3)
    net.load_state_dict(torch.load(OUT / "ckpt/f0_arc3600_final/model.pt", map_location="cpu"))
    net = net.to(dev)
    P_arc = probs_arc(net, ids, att, va2, dev)
    ba_arc = balanced_accuracy_score(y, P_arc.argmax(1))
    print(f"  ArcFace: {ba_arc:.4f}", flush=True)

    print("\n=== 概率平均 ===")
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        P = w * P_ce + (1 - w) * P_arc
        print(f"  CE {w:.1f} + ArcFace {1-w:.1f} : {balanced_accuracy_score(y, P.argmax(1)):.4f}", flush=True)

    # 错误重叠度
    e1, e2 = P_ce.argmax(1) != y, P_arc.argmax(1) != y
    both = (e1 & e2).sum()
    print(f"\n错误重叠：CE 错 {e1.sum()} | ArcFace 错 {e2.sum()} | 同时错 {both} "
          f"| 仅 CE 错 {(e1 & ~e2).sum()} | 仅 ArcFace 错 {(e2 & ~e1).sum()}")


if __name__ == "__main__":
    main()
