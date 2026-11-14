# LLM Attribution (STAC)

给定一段文本，判断它由 18 个 LLM 中的哪一个生成。

本项目是一份**独立的本地研究记录**：不使用外部模型的 API，全部训练在单张 8GB 笔记本显卡上完成。
重点是**诚实的评估协议**和**可复现的实验记录**（包括失败的实验）。

---

## 结果概览

评估协议见下文。数字均为 **balanced accuracy**（18 类等权，避免被多数类拉高）。

| 方法 | K=2（主协议） | K=all（对照） |
|---|---|---|
| TF-IDF + LinearSVC（子集 3,000 题） | 0.6015 | 0.6090 |
| TF-IDF + LinearSVC（密集子集 ≥7 模型） | 0.8082 | 0.8235 |
| ModernBERT-base, 512 tok, 1 epoch | 0.8134 | 0.7973 |
| ModernBERT-base, 512 tok, 3 epochs | 0.8350 | 0.8317 |
| **ModernBERT-base, 1024 tok, 1 epoch** | **0.8421** | 0.8296 |
| 三模型加权集成 | **0.8580** | — |

随机基线为 1/18 ≈ **0.056**。

---

## 评估协议

这份数据是「同一道题 × 18 个模型」的结构，因此**同一 prompt 的多条回答会共享话题信息**。
如果按行随机切分，模型只需要认出话题就能猜出作者 —— 这是本任务最大的陷阱。

因此本项目采用两级控制：

1. **切分按 prompt 分组**：同一个 prompt 的所有样本只会出现在训练集或验证集之一。
2. **评估时限制每题条数**：主协议 `K=2` 为每个验证 prompt 最多保留 2 条，
   对照组 `K=all` 保留全部，两者之差即「同题泄漏」的大小。

在本数据上，`K=all` 相对 `K=2` 只高约 **+0.015** —— 远小于常见报告中的差距。
原因见 [docs/FINDINGS.md](docs/FINDINGS.md)：泄漏主要来自「分组指派」类方法，
而不是单样本分类器本身。

---

## 数据

原始宽表：**70,980 行 × 18 个模型列**，共 **164,954** 条带标签段落。

清洗后（词数 40–2000、去重）：

| 项 | 值 |
|---|---|
| 段落数 | 120,946 |
| 去重 prompt 数 | 43,554 |
| 模型数 | 18 |
| 词数中位数 | 479（p90 = 1,243） |
| 每模型条数 | 2,356 ~ 23,568（约 10:1 不平衡） |

长文本是这份数据的优势：比赛版本只有 40–250 词，而这里中位数是 479 词，
因此**上下文长度成为一个有效变量**（见 FINDINGS）。

数据本身不随仓库分发，`src/research/build_corpus.py` 从原始宽表生成
`data/research/corpus_long.parquet`。

---

## 代码结构

```
src/common.py                18 个标签与机构映射
src/research/
  build_corpus.py            宽表 → 长表 + prompt 分组 5 折
  pretokenize.py             一次性分词，训练时零分词开销（用 mmap 的 .npy）
  baseline_eval.py           TF-IDF + LinearSVC 基线，含 K=2 / K=all 双协议
  train_encoder.py           初版训练（实时分词，较慢）
  train_fast.py              从分词缓存训练
  train_v2.py                每轮评估 + 每 N 步存档
  train_arcface.py           角度间隔损失实验（负结果）
  eval_ckpt.py               对已存权重单独打分
  eval_arc.py                ArcFace 模型的独立评估
  eval_ensemble.py           两模型集成 + 权重扫描
  ens3.py                    三模型集成 + 概率缓存
  analyze_confusion.py       混淆矩阵与各类召回
  make_headtail.py           前 511 + 后 510 的输入构造
  test_headtail.py           免训练探针：结尾是否带来额外信息
  pairwise_repair.py         二元判别器修补（负结果）
  extract_feats.py           抽取 [CLS] 特征
  convert_npy.py             npz → mmap 友好的 .npy
```

`results/metrics/` 存放每次实验的指标 JSON，`results/logs/` 存放训练日志原文。

---

## 复现

```bash
pip install -r requirements.txt

# 1) 生成数据集（需要原始宽表 parquet）
python src/research/build_corpus.py

# 2) 分词缓存
python src/research/pretokenize.py --max-len 1024
python src/research/convert_npy.py

# 3) 训练单个编码器（约 4.5 小时 / 8GB 显卡）
python src/research/train_v2.py --fold 0 --epochs 1 --bs 16 --max-len 1024 --tag my_run

# 4) 单独评估（务必与训练分开进程，见下）
python src/research/eval_ckpt.py --ckpt data/research/ckpt/f0_my_run_final --max-len 1024

# 5) 集成
python src/research/ens3.py
```

### 两个工程注意点

* **训练与评估必须分成两个进程。** 在单张 8GB 卡上，如果在训练脚本末尾直接做验证，
  会稳定地卡死（三次复现）。现在的脚本一律 `--no-eval`，评估交给独立进程。
* **必须每 N 步存档。** `train_v2.py` 的 `--save-every` 曾在一次卡死中救回 6,000 步的成果；
  早期版本没有存档，一次卡死直接损失 5.5 小时。

---

## 结论摘要

完整版见 [docs/FINDINGS.md](docs/FINDINGS.md)，这里只列四条：

1. **数据量 > 模型架构。** 同一套代码，9.7k 条训练数据得 0.568，97k 条得 0.813。
2. **上下文长度是真实变量。** 512 → 1024 token，+0.029（约 4 个标准误）。
3. **给强模型叠加弱专家必然亏。** 二元修补头 −0.012；比赛中的「家族门控」零效果；
   外部团队加入弱成员 −0.011。三次独立复现。
4. **集成要看信息差异，不是方法差异。** 两个普通 CE 模型（仅输入长度不同）集成 0.8536，
   优于 CE + 角度间隔损失模型的 0.8488。

---

## 局限

* 编码器结果均来自**单一 fold**（fold 0），未做 5 折平均，因此没有误差棒。
  K=2 子集为 10,926 条，标准误约 0.006，据此判断差异是否显著。
* 集成权重在验证集上网格搜索，存在轻微过拟合风险；未使用嵌套验证。
* 仅使用 ModernBERT-base（149M）。更大骨干受 8GB 显存限制，未系统探索。
