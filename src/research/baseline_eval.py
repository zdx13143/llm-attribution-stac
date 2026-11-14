"""研究基线 + 双协议评估：量化"同题泄漏"。

协议
----
切分严格按 prompt 分组（同题只出现在一个 fold），所以不存在「同一题跨训练/测试」。
在此之上再控制**评估时每个 prompt 保留几条**：

  K=2   ：每个测试 prompt 最多留 2 条   —— 主协议，接近真实部署
  K=all ：每个测试 prompt 全部保留       —— 对照，暴露泄漏

两者之差就是"同题泄漏"带来的虚高。这是论文的第一张表。

用法：
    python src/research/baseline_eval.py --max-prompts 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import LABEL2IDX, LABELS  # noqa: E402

CORPUS = ROOT / "data/research/corpus_long.parquet"
OUT = ROOT / "data/research"


def build_xy(texts_tr, texts_te):
    """char_wb 3-5 + word 1-2，L2 归一化后按权重拼接。"""
    parts_tr, parts_te = [], []
    for analyzer, ng, w in (("char_wb", (3, 5), 1.0), ("word", (1, 2), 0.7)):
        kw = dict(analyzer=analyzer, ngram_range=ng, lowercase=True, sublinear_tf=True,
                  min_df=3, max_features=400_000, dtype=np.float32)
        if analyzer == "word":
            kw["token_pattern"] = r"(?u)\b\w+\b"
        v = TfidfVectorizer(**kw).fit(texts_tr)
        parts_tr.append(normalize(v.transform(texts_tr)) * w)
        parts_te.append(normalize(v.transform(texts_te)) * w)
    return sparse.hstack(parts_tr).tocsr(), sparse.hstack(parts_te).tocsr()


def subsample_per_prompt(df: pd.DataFrame, k: int, seed: int = 0) -> pd.DataFrame:
    """每个 prompt 最多保留 k 条（按 label 均匀抽样，保留多样模型）。"""
    if k <= 0:
        return df
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in df.groupby("prompt_id", sort=False):
        if len(g) <= k:
            keep.append(g.index.to_numpy())
        else:
            keep.append(rng.choice(g.index.to_numpy(), size=k, replace=False))
    return df.loc[np.concatenate(keep)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-prompts", type=int, default=8000, help="0=全部")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--min-models", type=int, default=0,
                    help="只保留有 >=N 个模型作答的 prompt（用于量化同题泄漏）")
    args = ap.parse_args()

    df = pd.read_parquet(CORPUS)
    if args.min_models > 0:
        n = df.groupby("prompt_id")["label"].nunique()
        keep = set(n[n >= args.min_models].index)
        df = df[df.prompt_id.isin(keep)].reset_index(drop=True)
        print(f"仅保留 >= {args.min_models} 个模型的 prompt: {df.prompt_id.nunique()} 个 / {len(df)} 条")
    if args.max_prompts and df.prompt_id.nunique() > args.max_prompts:
        pids = np.array(sorted(df.prompt_id.unique()))
        rng = np.random.default_rng(7)
        sub = set(rng.choice(pids, size=args.max_prompts, replace=False))
        df = df[df.prompt_id.isin(sub)].reset_index(drop=True)
        print(f"子采样到 {df.prompt_id.nunique()} 个 prompt / {len(df)} 条")

    y = df.label.map(LABEL2IDX).to_numpy()
    print(f"数据 {len(df)} 条 | {df.prompt_id.nunique()} prompt | {df.label.nunique()} 类\n")

    res = {"k2": [], "kall": []}
    t0 = time.time()
    for f in range(args.folds):
        tr = df[df.fold != f]
        te = df[df.fold == f]
        if len(te) == 0:
            continue
        Xtr, Xte = build_xy(tr.text.tolist(), te.text.tolist())
        clf = LinearSVC(C=args.C, class_weight="balanced", dual="auto", max_iter=3000)
        y_tr = tr.label.map(LABEL2IDX).to_numpy()
        clf.fit(Xtr, y_tr)

        te = te.copy()
        te["pred"] = clf.predict(Xte)
        te["y"] = te.label.map(LABEL2IDX)
        ba_all = balanced_accuracy_score(te.y, te.pred)
        te2 = subsample_per_prompt(te, 2, seed=f)
        ba_2 = balanced_accuracy_score(te2.y, te2.pred)
        res["kall"].append(ba_all)
        res["k2"].append(ba_2)
        print(f"  fold {f}: K=all {ba_all:.4f} | K=2 {ba_2:.4f} "
              f"({len(te)} → {len(te2)} 条, {time.time()-t0:.0f}s)", flush=True)

    out = {
        "max_prompts": args.max_prompts, "C": args.C, "folds": args.folds,
        "ba_k2_mean": float(np.mean(res["k2"])), "ba_k2_std": float(np.std(res["k2"])),
        "ba_kall_mean": float(np.mean(res["kall"])), "ba_kall_std": float(np.std(res["kall"])),
        "per_fold": res,
    }
    out["leakage_gap"] = out["ba_kall_mean"] - out["ba_k2_mean"]
    print(f"\n=== 结果 ===")
    print(f"  K=2（主协议） : {out['ba_k2_mean']:.4f} ± {out['ba_k2_std']:.4f}")
    print(f"  K=all（对照） : {out['ba_kall_mean']:.4f} ± {out['ba_kall_std']:.4f}")
    print(f"  同题泄漏虚高  : {out['leakage_gap']:+.4f}")
    (OUT / "baseline_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
