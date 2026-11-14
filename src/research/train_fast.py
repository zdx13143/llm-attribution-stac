"""从预分词缓存训练编码器（快版）。

相比 train_encoder.py：样本直接从 npz 切片，训练循环里没有分词，
bs 提到 64，实测吞吐提升数倍。

用法：
    python src/research/train_fast.py --fold 0 --epochs 1 --bs 64
    python src/research/train_fast.py --fold 0 --epochs 3 --bs 64 --tag full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research"
MODEL = "answerdotai/ModernBERT-base"
K = 18


class TokDS(Dataset):
    """直接切 numpy 切片，零分词开销。"""

    def __init__(self, ids, att, label, idx):
        self.ids, self.att, self.label, self.idx = ids, att, label, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(self.ids[j].astype(np.int64)),
                torch.from_numpy(self.att[j].astype(np.int64)),
                int(self.label[j]))


def subsample_per_prompt(prompt_idx, idx, k, seed=0):
    """在给定行集合里，每个 prompt 最多保留 k 条。"""
    rng = np.random.default_rng(seed)
    order = np.argsort(prompt_idx[idx], kind="stable")
    idx_sorted = idx[order]
    p_sorted = prompt_idx[idx_sorted]
    keep = []
    start = 0
    for i in range(1, len(p_sorted) + 1):
        if i == len(p_sorted) or p_sorted[i] != p_sorted[start]:
            grp = idx_sorted[start:i]
            keep.append(grp if len(grp) <= k else rng.choice(grp, k, replace=False))
            start = i
    return np.concatenate(keep)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for x, m, _ in loader:
        logits = model(input_ids=x.to(device), attention_mask=m.to(device)).logits.float()
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit-train", type=int, default=0, help="0=全部")
    ap.add_argument("--tag", type=str, default="fast")
    ap.add_argument("--ckpt", action="store_true", help="开梯度检查点（省显存、费算力）")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(OUT / f"tok_{args.max_len}.npz")
    ids, att, label, fold, pidx = z["ids"], z["att"], z["label"], z["fold"], z["prompt_idx"]

    tr_idx = np.flatnonzero(fold != args.fold)
    va_idx = np.flatnonzero(fold == args.fold)
    if args.limit_train and len(tr_idx) > args.limit_train:
        tr_idx = np.random.default_rng(1).choice(tr_idx, args.limit_train, replace=False)
    va2 = subsample_per_prompt(pidx, va_idx, 2, seed=args.fold)
    print(f"device={device} | train {len(tr_idx)} | val {len(va_idx)} → K=2 {len(va2)}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True).to(device)
    if args.ckpt:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    cnt = np.bincount(label[tr_idx], minlength=K).astype(np.float64)
    cw = torch.tensor(cnt.sum() / (K * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    dl_tr = DataLoader(TokDS(ids, att, label, tr_idx), batch_size=args.bs,
                       shuffle=True, num_workers=0, drop_last=True)
    dl_va = DataLoader(TokDS(ids, att, label, va_idx), batch_size=128, num_workers=0)
    dl_va2 = DataLoader(TokDS(ids, att, label, va2), batch_size=128, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = max(1, int(len(dl_tr) * args.epochs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.05, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    t0, done = time.time(), 0
    model.train()
    for ep in range(int(np.ceil(args.epochs))):
        for x, m, y in dl_tr:
            if done >= steps:
                break
            x, m, y = x.to(device), m.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = loss_fn(model(input_ids=x, attention_mask=m).logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            done += 1
            if done % args.log_every == 0 or done == 1:
                el = time.time() - t0
                print(f"  ep{ep} step {done}/{steps} loss {loss.item():.4f} "
                      f"{el/done:.2f}s/step eta {(steps-done)*el/done/60:.0f}min "
                      f"{torch.cuda.memory_allocated()/2**30:.1f}GB", flush=True)

    p2 = predict(model, dl_va2, device)
    pa = predict(model, dl_va, device)
    ba2 = balanced_accuracy_score(label[va2], p2)
    baa = balanced_accuracy_score(label[va_idx], pa)
    mins = (time.time() - t0) / 60
    print(f"\n=== fold {args.fold} | {args.epochs} ep | {mins:.1f} 分钟 ===")
    print(f"  K=2 {ba2:.4f} | K=all {baa:.4f}")

    res = dict(fold=args.fold, epochs=args.epochs, bs=args.bs, lr=args.lr,
               max_len=args.max_len, limit_train=args.limit_train,
               ba_k2=float(ba2), ba_kall=float(baa), minutes=mins, tag=args.tag)
    (OUT / f"res_f{args.fold}_{args.tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
