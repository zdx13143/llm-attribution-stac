"""用已训练的最优编码器抽取 [CLS] 向量，供后续二元判别器使用。

输出 data/research/feats_f0.npy (121k × 768, fp16)
只需要一次前向，约 40 分钟。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForSequenceClassification

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
        return (torch.from_numpy(self.ids[j].astype(np.int64)),
                torch.from_numpy(self.att[j].astype(np.int64)))


@torch.no_grad()
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(OUT / "tok_512.npz")
    ids, att = z["ids"], z["att"]
    n = len(ids)

    # 用分类头模型的 backbone，取 pooler / CLS
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True)
    sd = torch.load(OUT / "ckpt/f0_3ep_best/model.pt", map_location="cpu")
    m.load_state_dict(sd)
    backbone = m.base_model
    backbone = backbone.to(device).eval()

    all_idx = np.arange(n)
    dl = DataLoader(TokDS(ids, att, all_idx), batch_size=128, num_workers=0)
    feats = np.zeros((n, 768), dtype=np.float16)
    t0 = time.time()
    p = 0
    for x, a in dl:
        out = backbone(input_ids=x.to(device), attention_mask=a.to(device))
        h = out.last_hidden_state[:, 0]          # [CLS]
        h = torch.nn.functional.normalize(h.float(), dim=-1)
        b = h.shape[0]
        feats[p:p + b] = h.cpu().numpy().astype(np.float16)
        p += b
        if (p // 128) % 100 == 0:
            print(f"  {p}/{n}  {time.time()-t0:.0f}s", flush=True)

    np.save(OUT / "feats_cls.npy", feats)
    print(f"已写 {OUT/'feats_cls.npy'}  shape={feats.shape}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
