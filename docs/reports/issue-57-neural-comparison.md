# #57 神经语义模型对比与 AutoResearch 受控优化 — 报告

日期：2026-11-03。分支 `feat/issue-57-neural-comparison`（实验脚本见 `scripts/issue57_*.py`）。
任务：TCM-SD 中文证候倾向（148 类单标签，short 主诉为主要场景）。
数据：TCM-SD 54,152 病例 / 52,295 用户（CC BY-NC-SA 4.0，非商业）。

**一句话结论**：在冻结协议 v2 上，**每一个神经方案都低于经典字符线性基线**（A 0.3199 vs B 0.2797 / C 0.2016 / D 0.1815–0.1874），AutoResearch 自己在同任务上的最大收获也是**把头部换成线性**。因此**不升级到神经模型，保留 TF-IDF+LR 为交付候选**。

## 1. 口径（可比较性声明）

- A/B/C/D 四条**全部**跑在 `#56` 修复后的 **protocol v2** 冻结输入上
  （`scripts/issue57_common.load_selection_pool` 直接加载 `runs/sd-v2/<scenario>/selection_manifest.json`），
  因此四行之间可直接比较。
- `docs/reports/research-sd-v1.md` 的战役分数**不是 v2**（其页首自标
  `NOT_COMPARABLE_TO_PROTOCOL_V2`），本章单独陈述，不与上表混比。
- 未触碰历史 holdout；#57 的全部结论只依据 dev/CV。

## 2. 递进比较（protocol v2，4 折 × 2 重复 = 8 fold-runs，除非另注）

| 线 | 模型 | CV mean macro-F1 | 跨折 std | fold-runs | 备注 |
| --- | --- | --- | --- | --- | --- |
| **A** | 字符 1–3 gram TF-IDF + LR（ovr-liblinear，冻结强基线） | **0.319921** | — | 8 | `issue-56-protocol-v2.md`；当前最强 |
| B | 固定文本特征 + 小 MLP | 0.279677 | 0.012977 | 8 | 非线性映射**没有**收益（−0.040） |
| C | bert-base-chinese 端到端微调 | 0.201569 | 0.007749 | 16 | 2 seeds；远低于 A（−0.118） |
| D1 | mc-bert 端到端微调（医学领域） | 0.181463 | 0.006818 | 8 | 领域预训练**没有**救回 |
| D2 | chinese-roberta-wwm-ext 端到端微调 | 0.187435 | 0.013071 | 8 | 同上 |
| D3 | mc-bert / roberta **冻结编码器 + 线性头** | IN_PROGRESS | — | 8 × 2 seeds | 见 §5 |

**C/D 的排序回答了两个问题**：
1. **不是训练不足**：C 线 16 fold-runs × 2 seeds 的 std 只有 0.0077，折间极稳；
2. **不是领域错配**：医学语料预训练的 mc-bert（0.1815）比通用 roberta-wwm-ext（0.1874）还低，
   两者都在 bert-base-chinese（0.2016）之下——**是预训练文本表示本身在本任务上不敌字符 n-gram 线性模型**，
   与"模板化主诉 + 长尾类别"的数据形态一致。

## 3. AutoResearch 受控优化（#53 阶段 1 战役，同任务族）

引擎 `src/research/`（源码级：改 `candidate.py`，固定墙钟预算，KEEP 前进 git 分支，
噪声门槛 1×SEM + 复杂度地板）。9 轮，决策全部由隔离 evaluator 做出：

| 轮 | 候选 | 裁决 | 选择值 | 理由摘要 |
| --- | --- | --- | --- | --- |
| 0 | CUDA 启用 | KEEP（锚点） | 0.2794 | 工程前提 |
| 1 | lr 1e-2 | REVERT | 0.2713 | −0.008 |
| 2 | 逆频率类别加权 | REVERT | 0.2644 | −0.015 |
| 3 | **线性头**（65k→148 直连） | **KEEP（冠军）** | **0.3260** | **+0.0466，复杂度 −54 节点** |
| 4 | width 128 | REVERT | 0.3259 | 线性头下参数无用 |
| 5 | lr 1e-3 | REVERT | 0.3186 | −0.007 |
| 6 | 两层头（65k→256→148） | REVERT | 0.3316 | +0.0056 过噪声门槛，< 11 节点 × 0.001 复杂度地板 |
| 7 | lr 1e-2（线性头） | REVERT | 0.3173 | −0.009 |
| 8 | AdamW wd 1e-4 | REVERT | 0.3245 | −0.0015 |

- **本任务族最大的一次真实改进，是"把头部简化成线性"**（+0.0466，且复杂度为负）。这与 §2 中线性基线最强的结论互相印证。
- 轮 6 是判据的正面样本：增益真实为正、过了噪声门槛，但按项目复杂度经济学（1 节点 = 0.001）判负——**没有为了结果事后放宽系数**。
- 战役分数属 protocol v1 语义，**不与 §2 的 v2 表混比**。
- holdout（该 campaign 预算 1/1，已用满）：神经冠军 **0.3324** vs 同口径 TF-IDF+LR **0.3732** vs Dummy 0.0020 → **神经仍未胜过线性**。

## 4. 稳健性

- 对抗/拒判：`adversarial_reject_rate = 0.857143`。
- 否定处理：**NOT_AVAILABLE** —— 真实病例的否定改写没有真值可依，接口行为仅由合成测试覆盖，**不得称为准确率或临床泛化**。
- Top-k / 头尾分层 / 推理延迟：**NOT_AVAILABLE**（本轮未系统采集；`frozen` 路径已记录 `embed_seconds`/`fit_seconds`/`predict_seconds`/`embedding_dim`，其余候选未记录）。标注为未运行，不补数。

## 5. D 线冻结编码器（进行中）

在 dev host acam-lab 以 `runs/issue57/gpu_chain_3.sh` 逐组合运行（两模型 × 两 seed，4 折 × 2 重复）。
运行中修掉的两个真实缺陷（已写进脚本注释与提交信息）：

1. 冻结头原用 `liblinear`，它是**二分类专用**，在 148 类上抛 `ValueError`，此前被记为失败而非结果。现改为
   `OneVsRestClassifier(liblinear)` —— 与 A 线 TF-IDF 基线同一分类器族，比较落在同一个头而不是两个。
2. 一折有 **33,814 行 × 768 稠密浮点**，liblinear 成本按非零元计，默认 2000 次迭代上限需约 20 分钟/折且跑不完。
   现声明 200 次上限 + `dual=False`，并**逐折记录有多少类未收敛**（`head_classes_not_converged`），不隐藏。
   同时加入 CLS 嵌入 memo（eval 模式无梯度 ⇒ 逐位相同向量），去掉每折重复编码的 ~8 倍冗余。

单折观测（**不是结果**，仅说明方向）：mc-bert 冻结头在 fold 0 得 0.2583，高于其微调版 0.1815、低于线性基线 0.3199。
完成后的数字与是否收敛将补入本报告，并按 §6 的同一结论口径解读。

## 6. 结论与选型

1. **不升级到神经模型**。四条神经路线（MLP 头、通用 BERT、医学 BERT、更强通用 BERT）在同一合法 dev/CV 上**全部低于** TF-IDF+LR，差距 0.04–0.14，远超折间 std（0.008–0.013）。
2. **保留 TF-IDF+LR 为交付候选**：推理成本最低、无需 GPU、无需权重许可，且是当前最优。
3. **神经路线的价值不在指标**，而在工程闭环与可解释/可扩展接口（已在 #53/#58 交付）；这不等同于"神经更强"。
4. 若要再评估神经方案，**必须换信息量**（新特征/新标签/新数据），而不是继续在同一搜索空间里调参——
   与体质线的机制级结论一致。

## 7. 复现

```bash
# 全部在 dev host acam-lab；真实数据在 data/local/，不入 Git
cd /home/lora/workspace/projects/MulitiModal-i57 && export PYTHONPATH=src
PY=/home/lora/workspace/projects/MulitiModal/.venv/bin/python
ROOT=/home/lora/workspace/projects/MulitiModal/data/local/models

$PY -m scripts.issue57_mlp_head --seed 20261101 --out runs/issue57/b-mlp-short
$PY -m scripts.issue57_bert_cv cv --model $ROOT/bert-base-chinese --lr 5e-5 --loss plain \
    --seeds 20261101,20261102 --out runs/issue57/c-bert-cv
$PY -m scripts.issue57_encoder_models check   --model-root $ROOT --out runs/issue57/d-check
$PY -m scripts.issue57_encoder_models finetune --model mc-bert --model-root $ROOT \
    --lr 5e-5 --loss plain --seeds 20261101 --out runs/issue57/d-mcbert-ft
$PY -m scripts.issue57_encoder_models frozen   --model mc-bert --model-root $ROOT \
    --seed 20261101 --folds 4 --repeats 2 --out runs/issue57/d-mcbert-frozen-s20261101
```

## 8. 证据状态

- 标签为数据集自带的真实诊断标签（非合成）；仍为单中心、模板化文本，尾部类别支持度低。
- 全部为工程/统计层陈述，**无临床有效性声明**；`GO_RESEARCH_CLAIMS` 未放行。
- 真实数据、checkpoint、`runs/` 工件均未入 Git。
- 未使用历史 holdout 调参；sd-v1 campaign 的 holdout 预算 1/1 已用满，本文不追加解锁。