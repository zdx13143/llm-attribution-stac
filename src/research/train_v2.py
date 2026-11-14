"""编码器训练 v2：每轮评估 + 保存最优，支持续训。

相比 train_fast.py 的改进：
- 每个 epoch 结束都评估一次，记录曲线（判断是否过拟合/欠拟合）
- 保存 val 最优的权重，不必赌总轮数
- 显存/速度按实测最优配置：bs32 + 梯度检查点（1.74s/step, 2.3GB）
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
CKPT_DIR = OUT / "ckpt"
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


def subsample_per_prompt(prompt_idx, idx, k, seed=0):
    rng = np.random.default_rng(seed)
    order = np.argsort(prompt_idx[idx], kind="stable")
    idx_sorted = idx[order]
    p_sorted = prompt_idx[idx_sorted]
    keep, start = [], 0
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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--tag", type=str, default="v2")
    ap.add_argument("--resume", type=str, default="", help="从 ckpt 目录续训")
    ap.add_argument("--save-every", type=int, default=500, help="每 N 步保存一次权重")
    ap.add_argument("--prefix", type=str, default="tok", help="数据前缀 tok / ht")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mm = OUT / f"{args.prefix}_{args.max_len}_ids.npy"
    if mm.exists():   # 优先用 mmap 的 .npy，避免把 600MB+ 全载入内存
        g = lambda k, p=None: np.load(OUT / f"{p or args.prefix}_{args.max_len}_{k}.npy",
                                      mmap_mode="r")
        # label/fold/prompt_idx 与输入形式无关，统一从 tok_ 取
        ids, att = g("ids"), g("att")
        label = np.asarray(g("label", "tok"))
        fold = np.asarray(g("fold", "tok"))
        pidx = np.asarray(g("prompt_idx", "tok"))
        print(f"已用 mmap 数据（输入前缀 {args.prefix}）", flush=True)
    else:
        z = np.load(OUT / f"tok_{args.max_len}.npz")
        ids, att, label, fold, pidx = z["ids"], z["att"], z["label"], z["fold"], z["prompt_idx"]

    tr_idx = np.flatnonzero(fold != args.fold)
    va_idx = np.flatnonzero(fold == args.fold)
    va2 = subsample_per_prompt(pidx, va_idx, 2, seed=args.fold)
    print(f"device={device} | train {len(tr_idx)} | val {len(va_idx)} → K=2 {len(va2)}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True)
    if args.resume:
        sd = torch.load(Path(args.resume) / "model.pt", map_location="cpu")
        model.load_state_dict(sd)
        print(f"已续训：{args.resume}", flush=True)
    model = model.to(device)
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
    steps_per_ep = len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps_per_ep * args.epochs,
        pct_start=0.05, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    curve, best = [], -1.0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        ep_loss, n = 0.0, 0
        for x, m, y in dl_tr:
            x, m, y = x.to(device), m.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(input_ids=x, attention_mask=m).logits.float()
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            ep_loss += loss.item(); n += 1
            if n % 400 == 0:
                el = time.time() - t0
                print(f"  ep{ep} {n}/{steps_per_ep} loss {ep_loss/n:.4f} "
                      f"{el/(ep*steps_per_ep+n):.2f}s/step", flush=True)
            # 每 save_every 步落盘一次，长跑被卡死也不至于全丢
            if args.save_every and (ep * steps_per_ep + n) % args.save_every == 0:
                d = CKPT_DIR / f"f{args.fold}_{args.tag}_step"
                d.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), d / "model.pt")
                (d / "meta.json").write_text(json.dumps(
                    dict(fold=args.fold, tag=args.tag, epoch=ep + 1, step=n,
                         total_step=ep * steps_per_ep + n), indent=2), encoding="utf-8")
                print(f"  [ckpt] 已保存 step {n}", flush=True)
        p2 = predict(model, dl_va2, device)
        pa = predict(model, dl_va, device)
        ba2 = balanced_accuracy_score(label[va2], p2)
        baa = balanced_accuracy_score(label[va_idx], pa)
        curve.append(dict(epoch=ep + 1, train_loss=ep_loss / max(n, 1),
                          ba_k2=float(ba2), ba_kall=float(baa),
                          minutes=(time.time() - t0) / 60))
        print(f"=== ep{ep+1}: train_loss {ep_loss/max(n,1):.4f} | K=2 {ba2:.4f} | "
              f"K=all {baa:.4f} | {(time.time()-t0)/60:.0f}min ===", flush=True)
        if ba2 > best:
            best = ba2
            d = CKPT_DIR / f"f{args.fold}_{args.tag}_best"
            d.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), d / "model.pt")
            (d / "meta.json").write_text(json.dumps(
                dict(fold=args.fold, tag=args.tag, epoch=ep + 1, ba_k2=float(ba2)), indent=2),
                encoding="utf-8")
            print(f"  ↑ 保存最优 → {d}", flush=True)

    (OUT / f"curve_f{args.fold}_{args.tag}.json").write_text(
        json.dumps(dict(curve=curve, best_k2=float(best)), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n最优 K=2 = {best:.4f}")


if __name__ == "__main__":
    main()
