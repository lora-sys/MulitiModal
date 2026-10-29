# MulitiModal

按摩前（Pre-only）多模态生理状态评估与预期疲劳缓解预测（BPSI/EFRI），以及基于公开数据的中医体质九维模型。第一阶段交付**双模型工程闭环**（Dataset → Label → Train → Eval → Save → Infer）与**受限 AutoResearch/RSI V1**。不是多模态 LLM，不做按摩硬件控制，不产出临床疗效声明。

## 当前状态

脚手架（#51）+ 共享神经核心（#48）+ TCM 体质预测模型 V1（#39）+ 训练运行时（#49）+ TCM 可信评估与推理 API（#40），全部纯合成数据工程验证（SMOKE_ONLY）。

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
src/core/               #48 神经核心：specs / batch / encoders / losses / models / registry
src/runtime/            #49/#40 训练运行时：batches / trainer / evaluation / checkpoint / protocols / tcm_eval / tcm_infer / cli
tests/core/             核心契约测试（synthetic；含 During/Post 拒绝、缺模态、NaN、mask、保存重载）
tests/runtime/          运行时测试（synthetic；fit/断点续训一致性/泄漏断言/受试者级指标/CLI smoke）
tests/                  pytest（fast 默认；slow / synthetic 用 marker）
```

`src/core` 要点：输入契约由 `InputSpecSet` 驱动（TCM `tcm-input-v0`、生理 `physio-input-pre-only-v0`）；缺模态 = features 缺键、行不可用 = availability mask，零填充嵌入、永不插补；非契约模态（如生理模型的 During/Post 特征）validate 阶段直接拒绝。TCM V1（#39）分类头优先：九维独立 sigmoid 多标签（逐类 observed mask），互斥来源才用 softmax-CE；0–100 回归头独立且默认关闭，仅真实问卷评分可用时启用。`registry` 保存/加载带 SHA-256 校验与 input_spec 严格重建。

`src/runtime` 要点（#49/#40）：`SyntheticBatchProvider`（模拟 subject manifest + #27 分区断言）、`Trainer.fit`（固定种子、CPU 默认设备、AMP 可选、梯度裁剪、早停、断点续训一致性）、`Evaluator`（versioned masked 指标，类别指标仅在标签存在时计算）、训练 checkpoint（weights + optimizer/epoch/RNG 状态 + SMOKE_ONLY 标记）、`train_candidate`/`trusted_score` 为 #45 预留的稳定协议。TCM 可信评估（#40）：冻结 `EvalManifest`（train/dev/holdout 不可交叉）、`TCMTrustedEvaluator` 主指标 `category_probs/subject_mae`、受试者级 MAE/RMSE + 逐维 n/missing/positive_rate、SMOKE_ONLY 报告并列明尚不可计算的真实指标；`TCMPredictor` 提供单条/批量 Pre-only 推理（工件版本检查、未知字段报错、无可用模态时 ABSTAIN 拒判）。CLI（`PYTHONPATH=src python -m runtime.cli`）：`list-models` / `validate-architecture` / `smoke --task tcm|physio` / `checkpoint-reload`，全部只读合成 fixture（SMOKE_ONLY）。

后续模块目录由对应 issue 落位：#45（AutoResearch 隔离 harness）、#25/#42/#43（数据契约与规则引擎）。不要凭空恢复旧目录名（representation/、massage_decision/、legacy_research/）。
