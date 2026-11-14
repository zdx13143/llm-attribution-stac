"""用已训练的最优权重做混淆分析：18 个模型里，哪两个最难分开。

这是论文里最有意思的一张图——它反映的是「模型指纹」的可分性结构，
而不是单纯的调参结果。零训练成本，只做一次前向。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import LABELS  # noqa: E402

OUT = ROOT / "data/research"
MODEL = "answerdotai/ModernBERT-base"
K = 18


class TokDS(Dataset):
    def __init__(self, ids, att, label, idx):
        self.ids, self.att, self.label, self.idx = ids, att, label, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(self.ids[j].astype(np.int64)),
                torch.from_numpy(self.att[j].astype(np.int64)),
                int(self.label[j]))


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    for x, m, _ in loader:
        logits = model(input_ids=x.to(device), attention_mask=m.to(device)).logits.float()
        probs.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(probs)


def main() -> None:
    ckpt = Path("data/research/ckpt/f0_3ep_best")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(OUT / "tok_512.npz")
    ids, att, label, fold, pidx = z["ids"], z["att"], z["label"], z["fold"], z["prompt_idx"]
    va_idx = np.flatnonzero(fold == 0)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location="cpu"))
    model = model.to(device)

    dl = DataLoader(TokDS(ids, att, label, va_idx), batch_size=128, num_workers=0)
    P = predict_probs(model, dl, device)
    y = label[va_idx]
    pred = P.argmax(1)

    ba = balanced_accuracy_score(y, pred)
    print(f"验证集 {len(va_idx)} 条 | 均衡准确率 {ba:.4f}\n")

    # 每个 prompt 取 2 条（主协议）下的准确率
    rng = np.random.default_rng(0)
    order = np.argsort(pidx[va_idx], kind="stable")
    keep, start = [], 0
    ps = pidx[va_idx][order]
    for i in range(1, len(ps) + 1):
        if i == len(ps) or ps[i] != ps[start]:
            g = order[start:i]
            keep.append(g if len(g) <= 2 else rng.choice(g, 2, replace=False))
            start = i
    k2 = np.concatenate(keep)
    print(f"K=2 子集 {len(k2)} 条 | 均衡准确率 {balanced_accuracy_score(y[k2], pred[k2]):.4f}\n")

    # 每类召回
    cm = confusion_matrix(y, pred, labels=range(K))
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    order_r = np.argsort(recall)
    print("=== 最难识别的 6 个模型（召回率最低）===")
    for i in order_r[:6]:
        print(f"  {LABELS[i]:<24} 召回 {recall[i]:.3f}  (n={cm[i].sum()})")

    print("\n=== 最常被混淆的配对（真实→误判，前 8）===")
    pairs = []
    for i in range(K):
        for j in range(K):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], LABELS[i], LABELS[j]))
    pairs.sort(reverse=True)
    for n, a, b in pairs[:8]:
        print(f"  {a:<24} → {b:<24} {n} 次")

    np.save(OUT / "cm.npy", cm)
    (OUT / "analysis.json").write_text(json.dumps(dict(
        ba_all=float(ba),
        ba_k2=float(balanced_accuracy_score(y[k2], pred[k2])),
        recall={LABELS[i]: float(recall[i]) for i in range(K)},
        top_confusions=[dict(n=int(n), true=a, pred=b) for n, a, b in pairs[:15]],
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT/'cm.npy'} 和 analysis.json")


if __name__ == "__main__":
    main()
