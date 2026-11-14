"""训练「模型指纹编码器」：判断一段文字由 18 个 LLM 中的哪一个写出。

设计要点
--------
1. **按 prompt 分组切分**：同题绝不跨训练/验证，逼模型学风格而不是话题。
2. **评估用 K=2 协议**：每个验证 prompt 最多留 2 条，接近真实部署
   （比赛经验：本地用每题 18 条评估，与榜单相关性只有 0.13；用 2 条是 0.91）。
3. **类别不平衡**：最常见的 claude-opus-4-8 有 2.3 万条，最少的 gemini-3.8-flash
   只有 2.3 千条，差 10 倍。用 class-weighted 交叉熵。
4. **ModernBERT-base**：149M，8GB 显存足够在 512 token / batch 32 上微调。

用法：
    python src/research/train_encoder.py --fold 0 --max-prompts 4000 --epochs 1   # 冒烟
    python src/research/train_encoder.py --fold 0 --max-prompts 0 --epochs 3      # 全量
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import LABEL2IDX, LABELS  # noqa: E402

CORPUS = ROOT / "data/research/corpus_long.parquet"
OUT = ROOT / "data/research"
MODEL = "answerdotai/ModernBERT-base"
K = len(LABELS)


class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding=False, return_attention_mask=True)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def collate(batch):
    keys = [k for k in batch[0] if k != "labels"]
    out = {k: torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True,
                                              padding_value=0) for k in keys}
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


def subsample_per_prompt(df, k, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in df.groupby("prompt_id", sort=False):
        if len(g) <= k:
            keep.append(g.index.to_numpy())
        else:
            keep.append(rng.choice(g.index.to_numpy(), size=k, replace=False))
    return df.loc[np.concatenate(keep)]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        y = batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.float()
        preds.append(logits.argmax(-1).cpu().numpy())
        if y is not None:
            ys.append(y.numpy())
    return np.concatenate(preds), (np.concatenate(ys) if ys else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-prompts", type=int, default=4000, help="0=全部")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--tag", type=str, default="mb")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    df = pd.read_parquet(CORPUS)
    if args.max_prompts and df.prompt_id.nunique() > args.max_prompts:
        pids = np.array(sorted(df.prompt_id.unique()))
        sub = set(np.random.default_rng(7).choice(pids, size=args.max_prompts, replace=False))
        df = df[df.prompt_id.isin(sub)].reset_index(drop=True)

    tr = df[df.fold != args.fold]
    va = df[df.fold == args.fold]
    va2 = subsample_per_prompt(va, 2, seed=args.fold)
    print(f"train {len(tr)} 条 / {tr.prompt_id.nunique()} prompt | "
          f"val {len(va)} → K=2 {len(va2)} 条", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    y_tr = tr.label.map(LABEL2IDX).to_numpy()
    ds_tr = TextDS(tr.text.tolist(), y_tr, tok, args.max_len)
    ds_va = TextDS(va.text.tolist(), va.label.map(LABEL2IDX).to_numpy(), tok, args.max_len)
    ds_va2 = TextDS(va2.text.tolist(), va2.label.map(LABEL2IDX).to_numpy(), tok, args.max_len)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, collate_fn=collate, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)
    dl_va2 = DataLoader(ds_va2, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)

    # 类别权重：逆频率，缓解 10:1 不平衡
    cnt = np.bincount(y_tr, minlength=K).astype(np.float64)
    w = (cnt.sum() / (K * np.maximum(cnt, 1)))
    class_w = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = max(1, int(len(dl_tr) * args.epochs))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.05,
        anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    t0 = time.time()
    done = 0
    model.train()
    for ep in range(int(np.ceil(args.epochs))):
        for i, batch in enumerate(dl_tr):
            if done >= steps:
                break
            y = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(**batch).logits.float()
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            done += 1
            if done % 50 == 0 or done == 1:
                print(f"  step {done}/{steps}  loss {loss.item():.4f}  "
                      f"{(time.time()-t0)/done:.2f}s/step  "
                      f"{torch.cuda.memory_allocated()/2**30:.1f}GB", flush=True)

    p_all, y_all = evaluate(model, dl_va, device)
    p_2, y_2 = evaluate(model, dl_va2, device)
    ba_all = balanced_accuracy_score(y_all, p_all)
    ba_2 = balanced_accuracy_score(y_2, p_2)
    print(f"\n=== fold {args.fold} ===")
    print(f"  K=all {ba_all:.4f} | K=2 {ba_2:.4f} | 用时 {(time.time()-t0)/60:.1f} 分钟")

    res = {"fold": args.fold, "max_prompts": args.max_prompts, "epochs": args.epochs,
           "max_len": args.max_len, "bs": args.bs, "lr": args.lr,
           "ba_k2": float(ba_2), "ba_kall": float(ba_all),
           "minutes": (time.time() - t0) / 60, "tag": args.tag}
    (OUT / f"enc_fold{args.fold}_{args.tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
