"""用角度间隔损失（ArcFace）重训分类头，把挤在一起的类强行拉开。

为什么这么做
------------
混淆矩阵显示：前 10 个混淆对占了 36.7% 的错误，且集中在 4 个小圈子
（gpt-5.6 三兄弟、gemini flash 两代、claude 两代、kimi/glm/claude 跨厂商）。
问题不是模型不够大，而是这些类在嵌入空间里挨得太近。
普通交叉熵只要求"分对"，容得下压线对；ArcFace 在余弦空间加大角度间隔，
强制同类聚拢、异类推远。

与失败的「二元判别器」的区别
--------------------------
那是外挂一个只在两类上训练的弱模型去覆盖主模型；这里是改进主模型自己的表征。

提速
----
从已训练的 1024 模型热启动，只换头，训 0.6 轮即可（约 2.5h，而非 4.5h）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research"
CKPT_DIR = OUT / "ckpt"
MODEL = "answerdotai/ModernBERT-base"
K = 18


class DS(Dataset):
    def __init__(self, ids, att, label, idx):
        self.ids, self.att, self.label, self.idx = ids, att, label, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(np.asarray(self.ids[j]).astype(np.int64)),
                torch.from_numpy(np.asarray(self.att[j]).astype(np.int64)),
                int(self.label[j]))


class ArcFace(nn.Module):
    """余弦空间 + 角度间隔。s 控制尺度，m 控制间隔强度。"""

    def __init__(self, dim, n_cls, s=30.0, m=0.3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, h, labels=None):
        h = F.normalize(h, dim=-1)
        w = F.normalize(self.W, dim=-1)
        cos = h @ w.t()
        if labels is None:
            return cos * self.s
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(labels, self.W.shape[0]).bool()
        return torch.where(onehot, target, cos) * self.s


class Net(nn.Module):
    def __init__(self, backbone, dim=768, s=30.0, m=0.3):
        super().__init__()
        self.backbone = backbone
        self.head = ArcFace(dim, K, s, m)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0]
        return self.head(h, labels)


def subsample_per_prompt(pidx, idx, k, seed=0):
    rng = np.random.default_rng(seed)
    o = np.argsort(pidx[idx], kind="stable")
    i_s, p_s = idx[o], pidx[idx][o]
    keep, st = [], 0
    for i in range(1, len(p_s) + 1):
        if i == len(p_s) or p_s[i] != p_s[st]:
            g = i_s[st:i]
            keep.append(g if len(g) <= k else rng.choice(g, k, replace=False))
            st = i
    return np.concatenate(keep)


@torch.no_grad()
def predict(net, ids, att, idx, dev, bs=96):
    net.eval()
    out = []
    dl = DataLoader(DS(ids, att, np.zeros(len(ids), dtype=np.int16), idx),
                    batch_size=bs, num_workers=0)
    for x, m, _ in dl:
        out.append(net(x.to(dev), m.to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3600, help="总步数（0.6 轮≈3600）")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1.5e-4)
    ap.add_argument("--lr-back", type=float, default=1.5e-5)
    ap.add_argument("--s", type=float, default=30.0)
    ap.add_argument("--m", type=float, default=0.30)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--prefix", type=str, default="tok")
    ap.add_argument("--init", type=str, default="data/research/ckpt/f0_L1024_step")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--tag", type=str, default="arc")
    ap.add_argument("--no-eval", action="store_true", help="训练后不评估（本机评估易卡死）")
    ap.add_argument("--no-sampler", action="store_true", help="不用类别平衡采样（与基线一致）")
    ap.add_argument("--freeze-steps", type=int, default=0,
                    help="前 N 步冻结骨干，只训新头（避免坏梯度污染预训练权重）")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = lambda k, p="tok": np.load(OUT / f"{p}_{args.max_len}_{k}.npy", mmap_mode="r")
    ids, att = g("ids", args.prefix), g("att", args.prefix)
    label = np.asarray(g("label")); fold = np.asarray(g("fold")); pidx = np.asarray(g("prompt_idx"))

    tr = np.flatnonzero(fold != args.fold)
    va = np.flatnonzero(fold == args.fold)
    va2 = subsample_per_prompt(pidx, va, 2, seed=args.fold)
    print(f"device={dev} | train {len(tr)} | val {len(va)} → K=2 {len(va2)} | "
          f"steps {args.steps} | s={args.s} m={args.m}", flush=True)

    # 骨干起点：--init 为空则用预训练权重（与基线完全对齐，只变损失函数）
    src = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=K, ignore_mismatched_sizes=True)
    if args.init:
        sd = torch.load(Path(args.init) / "model.pt", map_location="cpu")
        src.load_state_dict(sd)
        print(f"已热启动骨干：{args.init}", flush=True)
    else:
        print("从预训练权重开始（与基线一致）", flush=True)
    net = Net(src.base_model, dim=768, s=args.s, m=args.m).to(dev)
    if hasattr(net.backbone, "gradient_checkpointing_enable"):
        net.backbone.gradient_checkpointing_enable()
        print("已开启梯度检查点", flush=True)
    if hasattr(net.backbone, "config"):
        net.backbone.config.use_cache = False

    cnt = np.bincount(label[tr], minlength=K).astype(np.float64)
    cw = torch.tensor(cnt.sum() / (K * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.smooth)

    ds_tr = DS(ids, att, label, tr)
    if args.no_sampler:
        # 与基线一致：普通打乱，不做类别平衡采样
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                           num_workers=0, drop_last=True)
        print("不使用类别平衡采样（与基线一致）", flush=True)
    else:
        w_samp = (cnt.sum() / (K * np.maximum(cnt, 1)))[label[tr]]
        sampler = WeightedRandomSampler(torch.tensor(w_samp, dtype=torch.double),
                                        len(tr), replacement=True)
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, sampler=sampler,
                           num_workers=0, drop_last=True)
    dl_va = DataLoader(DS(ids, att, label, va), batch_size=96, num_workers=0)
    dl_va2 = DataLoader(DS(ids, att, label, va2), batch_size=96, num_workers=0)

    opt = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": args.lr_back},
        {"params": net.head.parameters(), "lr": args.lr_head},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_back, args.lr_head], total_steps=args.steps,
        pct_start=0.05, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    t0, done, run_loss = time.time(), 0, 0.0
    net.train()
    if args.freeze_steps > 0:
        for p in net.backbone.parameters():
            p.requires_grad = False
        print(f"前 {args.freeze_steps} 步冻结骨干，只训新头", flush=True)
    while done < args.steps:
        for x, m, y in dl_tr:
            if done >= args.steps:
                break
            if done == args.freeze_steps:
                for p in net.backbone.parameters():
                    p.requires_grad = True
                print(f"step {done}: 解冻骨干，开始全量微调", flush=True)
            x, m, y = x.to(dev), m.to(dev), y.to(dev)
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                logits = net(x, m, y)
                loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            done += 1; run_loss += loss.item()
            if done % 100 == 0 or done == 1:
                el = time.time() - t0
                print(f"  {done}/{args.steps} loss {run_loss/min(done,100):.4f} "
                      f"{el/done:.2f}s/step eta {(args.steps-done)*el/done/60:.0f}min "
                      f"{torch.cuda.memory_allocated()/2**30:.1f}GB", flush=True)
                run_loss = 0.0 if done % 200 == 0 else run_loss
            if args.save_every and done % args.save_every == 0:
                d = CKPT_DIR / f"f{args.fold}_{args.tag}_step"
                d.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), d / "model.pt")
                (d / "meta.json").write_text(json.dumps(
                    dict(fold=args.fold, tag=args.tag, step=done, s=args.s, m=args.m,
                         init=args.init), indent=2), encoding="utf-8")

    # 训练结束先落盘：本机在「训练脚本内部做评估」时会卡死，
    # 所以默认存完就退出，评估交给独立进程 eval_arc.py
    final = CKPT_DIR / f"f{args.fold}_{args.tag}_final"
    final.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), final / "model.pt")
    (final / "meta.json").write_text(json.dumps(
        dict(fold=args.fold, tag=args.tag, steps=args.steps, s=args.s, m=args.m,
             smooth=args.smooth, init=args.init, minutes=(time.time() - t0) / 60),
        indent=2), encoding="utf-8")
    print(f"已保存最终权重 → {final}", flush=True)

    if args.no_eval:
        print("跳过评估（--no-eval），请用 eval_arc.py 单独打分")
        return

    p2 = predict(net, ids, att, va2, dev)
    pa = predict(net, ids, att, va, dev)
    ba2 = balanced_accuracy_score(label[va2], p2)
    baa = balanced_accuracy_score(label[va], pa)
    mins = (time.time() - t0) / 60
    print(f"\n=== ArcFace fold{args.fold} | {args.steps} 步 | {mins:.0f} 分钟 ===")
    print(f"  K=2 {ba2:.4f} | K=all {baa:.4f}   （对照：普通 CE 的 1024 = 0.8421 / 0.8296）")
    (OUT / f"arcface_f{args.fold}_{args.tag}.json").write_text(json.dumps(dict(
        fold=args.fold, steps=args.steps, s=args.s, m=args.m, smooth=args.smooth,
        ba_k2=float(ba2), ba_kall=float(baa), minutes=mins), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
