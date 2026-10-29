# MulitiModal

按摩前（Pre-only）多模态生理状态评估与预期疲劳缓解预测（BPSI/EFRI），以及基于公开数据的中医体质九维模型。第一阶段交付**双模型工程闭环**（Dataset → Label → Train → Eval → Save → Infer）与**受限 AutoResearch/RSI V1**。不是多模态 LLM，不做按摩硬件控制，不产出临床疗效声明。

## 当前状态

仅仓库脚手架（#51），无任何模型 / 数据功能代码。

- 分支：`feature/clean-slate`（空起点；旧历史仅见 GitHub commit `9fc5087d`，已删除，不作依赖）
- 主 Epic：#22（双模型 + AutoResearch/RSI V1）
- P0 优先：#48 双模型 Neural Core、#49 训练/评估基础设施；#50 中医模型先行真实训练
- 一手需求源：`docs/source/pre-massage-fatigue-prediction-framework.pdf`（研究设计草稿，含 `[value]` 占位符；SHA-256 `29727dace…d781`）

## 硬边界（全文见 [AGENTS.md](AGENTS.md)）

1. 推理只用按摩前（Pre）信息；按摩中 EEG / 按摩后测量仅用于训练期标签构造（#23/#31/#42）。
2. 数据划分一律 subject-level，同一受试者不跨折（#27）。
3. Scaler、特征筛选、指标权重只在训练折内拟合（#29/#42）。
4. 真实数据、检查点、工件不入 Git；synthetic fixture 必须显式标记（#26/#35）。
5. 标签分层 RULE / PUBLISHED / LLM_SILVER 不可互换，LLM_SILVER 永不作真值（#38）。
6. 三层放行独立判定：GO_ENGINEERING / GO_REAL_DATA_TRAINING / GO_RESEARCH_CLAIMS（#47）。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
ruff check .
```

要求 Python ≥ 3.11（本地 3.14 已验证）。

## 仓库结构

```
docs/source/            原始需求 PDF 与一手资料
tests/                  pytest（fast 默认；slow / synthetic 用 marker）
```

后续模块目录由对应 issue 落位：#48（`src/core` 双模型契约与网络骨架）、#25/#42/#43（数据契约与规则引擎）、#49（Trainer/Evaluator/CLI）。不要凭空恢复旧目录名（representation/、massage_decision/、legacy_research/）。
