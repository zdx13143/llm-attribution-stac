"""把宽表语料整理成研究用的长表数据集 + prompt 不相交切分。

为什么不用比赛那套评估
----------------------
比赛里本地 CV 用「每个 prompt 全部 18 条」做 out-of-fold，与榜单相关性只有 0.13；
改成「每个 prompt 只留 2 条」，相关性到 0.91。原因是隐藏测试集每个 prompt 的段落很少，
同题多条会泄漏话题信息。

本项目按研究标准做：**切分按 prompt 分组，评估时每个 prompt 最多留 K 条**，
K=2 为主协议，K=全部作为对照（用来量化"同题泄漏"有多大）。

用法：
    python src/research/build_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import LABELS  # noqa: E402

SRC = ROOT / "data/external/response_corpus/external_all_wide.parquet"
OUT = ROOT / "data/research"

MIN_WORDS, MAX_WORDS = 40, 2000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    w = pd.read_parquet(SRC)
    lab = [c for c in w.columns if c in LABELS]
    assert len(lab) == 18, f"期望 18 个模型列，实际 {len(lab)}"

    # 长表化
    rows = []
    for m in lab:
        s = w[["prompt", m]].dropna()
        if len(s) == 0:
            continue
        rows.append(pd.DataFrame({
            "prompt": s["prompt"].to_numpy(),
            "text": s[m].astype(str).to_numpy(),
            "model": m,
        }))
    df = pd.concat(rows, ignore_index=True)

    # 词数过滤
    df["n_words"] = df["text"].str.split().str.len()
    n0 = len(df)
    df = df[(df.n_words >= MIN_WORDS) & (df.n_words <= MAX_WORDS)].copy()

    # 去重：同一 (prompt, model) 只留一条
    df["text_key"] = df["text"].str.strip()
    df = df.drop_duplicates(subset=["prompt", "model", "text_key"])

    # prompt 用内容哈希作 id（避免依赖原表行号）
    df["prompt_id"] = df["prompt"].map(
        lambda p: hashlib.blake2s(str(p).strip().encode("utf-8"), digest_size=8).hexdigest()
    )
    df["label"] = df["model"]
    df = df[["prompt_id", "prompt", "text", "label", "n_words"]].reset_index(drop=True)

    # 统计
    g = df.groupby("prompt_id")["label"].nunique()
    k_dist = g.value_counts().sort_index()
    print(f"原始 {n0} 条 → 过滤去重后 {len(df)} 条")
    print(f"prompt {df.prompt_id.nunique()} 个 | 模型 {df.label.nunique()} 个")
    print(f"≥2 模型作答的 prompt: {(g >= 2).sum()} | 全 18 模型: {(g == 18).sum()}")
    print(f"词数: 中位 {df.n_words.median():.0f}, p10 {df.n_words.quantile(.1):.0f}, "
          f"p90 {df.n_words.quantile(.9):.0f}")
    print("\n每模型条数：")
    print(df.label.value_counts().to_string())

    # prompt 不相交 5 折（GroupKFold 等价，但确定性）
    pids = np.array(sorted(df.prompt_id.unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(pids)
    fold_of = {p: i % 5 for i, p in enumerate(pids)}
    df["fold"] = df.prompt_id.map(fold_of).astype(np.int8)

    df.to_parquet(OUT / "corpus_long.parquet", index=False)

    meta = {
        "rows": int(len(df)),
        "prompts": int(df.prompt_id.nunique()),
        "labels": int(df.label.nunique()),
        "median_words": float(df.n_words.median()),
        "prompts_ge2": int((g >= 2).sum()),
        "prompts_18": int((g == 18).sum()),
        "folds": 5,
        "min_words": MIN_WORDS,
        "max_words": MAX_WORDS,
    }
    (OUT / "corpus_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写 {OUT/'corpus_long.parquet'}（{len(df)} 行）")


if __name__ == "__main__":
    main()
