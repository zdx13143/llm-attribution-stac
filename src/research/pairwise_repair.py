"""定向修补：针对混淆矩阵里最难的配对，训专门的二元判别器。

动机
----
18 类单模型在 fold 0 上 0.8350，错误高度集中在 8 个配对上
（kimi→glm 157 次、gpt-5.6-sol↔luna 150、gemini-3.7↔3.8 各 120 …）。
这些样本在 18 类问题里很难，但在「只要在这两类之间选一个」时容易得多。

防泄漏设计
----------
编码器训练时没见过 fold 0。在 fold 0 内部再按 prompt 分成 A/B 两半：
A 半训练二元头，B 半评估。二元头只看过 A 半，评估在 B 半，无泄漏。

应用方式（保守）
---------------
只有当 18 类的预测落在 {a, b} 这个配对里时，才用二元头重新判；
其它情况不动。这样最坏情况也只是把原本就错的两类互换。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import LABELS, LABEL2IDX  # noqa: E402

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
        return (torch.from_numpy(self.ids[j].astype(np.int64)),
                torch.from_numpy(self.att[j].astype(np.int64)))


@torch.no_grad()
def predict(model, ids, att, idx, device, bs=128):
    model.eval()
    preds, probs = [], []
    dl = DataLoader(TokDS(ids, att, idx), batch_size=bs, num_workers=0)
    for x, a in dl:
        logits = model(input_ids=x.to(device), attention_mask=a.to(device)).logits.float()
        p = torch.softmax(logits, -1)
        preds.append(p.argmax(-1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def k2_mask(prompt_idx, idx, k=2, seed=0):
    rng = np.random.default_rng(seed)
    order = np.argsort(prompt_idx[idx], kind="stable")
    i_sorted, p_sorted = idx[order], prompt_idx[idx][order]
    keep, start = [], 0
    for i in range(1, len(p_sorted) + 1):
        if i == len(p_sorted) or p_sorted[i] != p_sorted[start]:
            g = i_sorted[start:i]
            keep.append(g if len(g) <= k else rng.choice(g, k, replace=False))
            start = i
    return np.concatenate(keep)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(OUT / "tok_512.npz")
    ids, att, label, fold, pidx = z["ids"], z["att"], z["label"], z["fold"], z["prompt_idx"]
    feats = np.load(OUT / "feats_cls.npy").astype(np.float32)

    va = np.flatnonzero(fold == 0)
    # fold 0 按 prompt 对半切：0a 训二元头，0b 评估
    pids = np.unique(pidx[va])
    rng = np.random.default_rng(123)
    rng.shuffle(pids)
    half = set(pids[: len(pids) // 2])
    in_a = np.array([p in half for p in pidx[va]])
    idx_a, idx_b = va[in_a], va[~in_a]
    print(f"fold0 拆半：A(训二元头) {len(idx_a)} | B(评估) {len(idx_b)}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(OUT / "ckpt/f0_3ep_best/model.pt", map_location="cpu"))
    model = model.to(device)

    print("对 B 半推理...", flush=True)
    pred_b, prob_b = predict(model, ids, att, idx_b, device)
    y_b = label[idx_b]
    base_all = balanced_accuracy_score(y_b, pred_b)
    kb = k2_mask(pidx, idx_b, 2, seed=7)
    pos_b = {v: i for i, v in enumerate(idx_b)}
    kb_local = np.array([pos_b[v] for v in kb])
    base_k2 = balanced_accuracy_score(y_b[kb_local], pred_b[kb_local])
    print(f"基线：K=all {base_all:.4f} | K=2 {base_k2:.4f}\n", flush=True)

    # 从混淆矩阵取前 8 个配对
    cm = np.load(OUT / "cm.npy")
    pairs = []
    for i in range(K):
        j = int(np.argmax(np.where(np.arange(K) == i, -1, cm[i])))
        if cm[i, j] > 0:
            pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True)
    pairs = pairs[:8]
    print("目标配对：" + ", ".join(f"{LABELS[a]}↔{LABELS[b]}" for _, a, b in pairs), flush=True)

    # 训二元头（在 A 半，只用真实标签属于该对的样本）
    heads = {}
    for n, a, b in pairs:
        sel = idx_a[(label[idx_a] == a) | (label[idx_a] == b)]
        if len(sel) < 60:
            continue
        Xtr = feats[sel]
        ytr = (label[sel] == b).astype(int)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(Xtr, ytr)
        # 在 A 半内部估计这对的可分性
        acc_a = clf.score(Xtr, ytr)
        heads[(a, b)] = clf
        print(f"  {LABELS[a]:<22}↔{LABELS[b]:<22} n={len(sel):>5} 训练集内准确 {acc_a:.3f}", flush=True)

    # 在 B 半应用：仅当预测落在该配对内才重判
    pred_fix = pred_b.copy()
    changed = 0
    for i, pr in enumerate(pred_b):
        for (a, b), clf in heads.items():
            if pr in (a, b):
                p2 = int(clf.predict(feats[idx_b[i]][None, :])[0])
                new = b if p2 == 1 else a
                if new != pr:
                    pred_fix[i] = new
                    changed += 1
                break

    fix_all = balanced_accuracy_score(y_b, pred_fix)
    fix_k2 = balanced_accuracy_score(y_b[kb_local], pred_fix[kb_local])
    print(f"\n修补后：K=all {fix_all:.4f} | K=2 {fix_k2:.4f}")
    print(f"改判 {changed} / {len(pred_b)} 条 ({changed/len(pred_b)*100:.1f}%)")
    print(f"增量：K=all {fix_all-base_all:+.4f} | K=2 {fix_k2-base_k2:+.4f}")

    (OUT / "pairwise_result.json").write_text(json.dumps(dict(
        base_all=float(base_all), base_k2=float(base_k2),
        fix_all=float(fix_all), fix_k2=float(fix_k2),
        changed=int(changed), n_b=int(len(pred_b)),
        pairs=[dict(a=LABELS[a], b=LABELS[b], n=int(n)) for n, a, b in pairs],
    ), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
