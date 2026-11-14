"""检验「结尾是否带来额外信息」。

用现有 1024 模型（只在「开头」上训练过）对比两种输入：
  A. 开头 1024（训练分布内）
  B. 前 511 + 后 510（分布外，但总长相同）

对 70% 未截断的样本两种输入完全相同，差异只来自 30% 的长文本。
若 B 在长文本上仍不输 A，说明结尾确实含信息 → 值得重新训练。
"""

from __future__ import annotations

import json
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
def run(model, ids, att, idx, dev, bs=96):
    model.eval()
    pred = []
    dl = DataLoader(DS(ids, att, idx), batch_size=bs, num_workers=0)
    for x, m in dl:
        lg = model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits.float()
        pred.append(lg.argmax(-1).cpu().numpy())
    return np.concatenate(pred)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab = np.load(OUT / "tok_1024_label.npy")
    fold = np.load(OUT / "tok_1024_fold.npy")
    pidx = np.load(OUT / "tok_1024_prompt_idx.npy")
    att_head = np.load(OUT / "tok_1024_att.npy", mmap_mode="r")
    ids_head = np.load(OUT / "tok_1024_ids.npy", mmap_mode="r")
    ids_ht = np.load(OUT / "ht_1024_ids.npy", mmap_mode="r")
    att_ht = np.load(OUT / "ht_1024_att.npy", mmap_mode="r")

    va = np.flatnonzero(fold == 0)
    # 被截断的样本（开头版有效长度顶到 1024）
    trunc = va[np.asarray(att_head[va]).sum(1) >= 1023]

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(OUT / "ckpt/f0_L1024_step/model.pt", map_location="cpu"))
    model = model.to(dev)

    print(f"验证集 {len(va)} 条，其中被截断 {len(trunc)} 条 ({len(trunc)/len(va)*100:.1f}%)", flush=True)

    res = {}
    for name, i_, a_ in (("A_开头", ids_head, att_head), ("B_前+后", ids_ht, att_ht)):
        p = run(model, i_, a_, va, dev)
        ba_all = balanced_accuracy_score(lab[va], p)
        pos = {v: k for k, v in enumerate(va)}
        sel = np.array([pos[v] for v in trunc])
        ba_tr = balanced_accuracy_score(lab[trunc], p[sel])
        res[name] = dict(all=float(ba_all), trunc=float(ba_tr))
        print(f"  {name}: 全体验证 {ba_all:.4f} | 仅长文本 {ba_tr:.4f}", flush=True)

    d_all = res["B_前+后"]["all"] - res["A_开头"]["all"]
    d_tr = res["B_前+后"]["trunc"] - res["A_开头"]["trunc"]
    print(f"\n差值：全体 {d_all:+.4f} | 长文本 {d_tr:+.4f}")
    (OUT / "headtail_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
