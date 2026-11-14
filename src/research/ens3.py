"""三模型集成：1024 普通 CE + 1024 ArcFace + 512 普通 CE。

三个模型的输入长度、训练轨迹都不同，错误模式应当更分散。
概率按权重加权平均，先在 K=2 子集上网格搜索权重。
概率缓存到 npz，方便以后加成员时不用重复推理。
"""

from __future__ import annotations

import itertools
import json
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


def load_ce(ckpt, dev, L):
    g = lambda k: np.load(OUT / f"tok_{L}_{k}.npy", mmap_mode="r")
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True)
    m.load_state_dict(torch.load(Path(ckpt) / "model.pt", map_location="cpu"))
    return m.to(dev), g("ids"), g("att")


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab = np.load(OUT / "tok_1024_label.npy")
    fold = np.load(OUT / "tok_1024_fold.npy")
    pidx = np.load(OUT / "tok_1024_prompt_idx.npy")
    va2 = subsample_per_prompt(pidx, np.flatnonzero(fold == 0), 2, seed=0)
    y = lab[va2]
    print(f"K=2 子集 {len(va2)} 条", flush=True)

    cache = OUT / "ens_probs.npz"
    P = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        P = {k: z[k] for k in z.files}
        print(f"载入缓存 {list(P)}", flush=True)

    if "ce1024" not in P:
        m, ids, att = load_ce(OUT / "ckpt/f0_L1024_step", dev, 1024)
        P["ce1024"] = probs_ce(m, ids, att, va2, dev)
        del m; torch.cuda.empty_cache()
        print(f"  ce1024 : {balanced_accuracy_score(y, P['ce1024'].argmax(1)):.4f}", flush=True)
        np.savez(cache, **P)

    if "arc1024" not in P:
        ids = np.load(OUT / "tok_1024_ids.npy", mmap_mode="r")
        att = np.load(OUT / "tok_1024_att.npy", mmap_mode="r")
        src = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=K, ignore_mismatched_sizes=True)
        net = Net(src.base_model, dim=768, s=30.0, m=0.3)
        net.load_state_dict(torch.load(OUT / "ckpt/f0_arc3600_final/model.pt", map_location="cpu"))
        net = net.to(dev)
        P["arc1024"] = probs_arc(net, ids, att, va2, dev)
        del net; torch.cuda.empty_cache()
        print(f"  arc1024: {balanced_accuracy_score(y, P['arc1024'].argmax(1)):.4f}", flush=True)
        np.savez(cache, **P)

    if "ce512" not in P:
        m, ids, att = load_ce(OUT / "ckpt/f0_3ep_best", dev, 512)
        P["ce512"] = probs_ce(m, ids, att, va2, dev)
        del m; torch.cuda.empty_cache()
        print(f"  ce512  : {balanced_accuracy_score(y, P['ce512'].argmax(1)):.4f}", flush=True)
        np.savez(cache, **P, va2=va2)

    print("\n=== 两两组合 ===")
    keys = list(P)
    for a, b in itertools.combinations(keys, 2):
        best = max(((balanced_accuracy_score(y, (w*P[a]+(1-w)*P[b]).argmax(1)), w)
                    for w in (0.3, 0.4, 0.5, 0.6, 0.7)), key=lambda t: t[0])
        print(f"  {a} + {b}: {best[0]:.4f} @ w={best[1]:.1f}", flush=True)

    print("\n=== 三者加权（步长 0.1）===")
    best3 = (-1, None)
    for w1 in np.arange(0.1, 0.9, 0.1):
        for w2 in np.arange(0.1, 0.9 - w1 + 1e-9, 0.1):
            w3 = 1 - w1 - w2
            if w3 < 0.05:
                continue
            p = w1*P["ce1024"] + w2*P["arc1024"] + w3*P["ce512"]
            s = balanced_accuracy_score(y, p.argmax(1))
            if s > best3[0]:
                best3 = (s, (round(w1, 2), round(w2, 2), round(w3, 2)))
    print(f"  最优: {best3[0]:.4f} @ ce1024={best3[1][0]} arc1024={best3[1][1]} ce512={best3[1][2]}")
    (OUT / "ens3_result.json").write_text(json.dumps(
        dict(best=float(best3[0]), weights=best3[1],
             singles={k: float(balanced_accuracy_score(y, v.argmax(1))) for k, v in P.items()
                      if v.shape == (len(va2), K)}), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
