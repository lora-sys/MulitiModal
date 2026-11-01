# #53 阶段 1 源码级 AutoResearch 战役报告（sd-v1）

日期：2026-11-02。分支 `feature/tcm-sd-syndrome-v1`。任务：TCM-SD 中文证候倾向
（148 类单标签，短文本场景为主口径）。引擎：`src/research/`（任务参数化后），
worktree `/home/lora/workspace/ar-sd`，分支 `autoresearch/sd-v1`。

## 1. 数据与任务契约（冻结）

- TCM-SD：54,152 病例 / 52,295 用户 / 148 证型（CC BY-NC-SA 4.0，非商业）。
  审计详见 `docs/data/tcm-sd-syndrome-audit.md`。
- 受试者级重划分（官方 split 有 50 个用户跨文件泄漏，弃用）：seed 20261101，
  确定性可重放。
- **评估集单向去重（按场景输入文本基准）**：short 场景剔 dev 2,065 / test 1,650
  例（主诉与先前 partition 逐字节相同）；clinical 基准剔 595/495。train 全量保留。
  仲裁记录：先以临床全文为基准只收回泄漏的一小部分（short −0.0136），主线复核
  后改为按场景基准（short 实测 LR 0.3835→0.3449，−0.0386）。
- 泄漏记录剔除：user 484293（证候名逐字出现在 detection），加载层 `EXCLUDED_LEAK_RECORDS`。
- 特征：清单冻结的无状态 HashingVectorizer（char_wb 1-3 gram，2^16 桶，L2）——
  无拟合统计量，折内拟合规则空泛满足，评估器可从清单参数确定性重建（不破坏
  state_dict-only 隔离契约）。input spec `tcm-sd-text-v2:…:hashed_text:65536`。
- dev 真实基线（去重后）：TF-IDF+LR（ovr-liblinear）0.3449 > BERT 微调 0.1658
  （short）；clinical 0.5754 > 0.2249。详见模型卡。

## 2. 战役记录（9 轮，3 折 × 2 重复，90 s/折，seed 0）

proposer 全部为真实 agent 提案（driver `scripts/run_sd_v1.py`，anchor 断言补丁），
裁决由引擎独立做出（噪声门槛 1×SEM + 复杂度地板 0.001/节点）：

| 轮 | 候选 | 裁决 | macro-F1（观察类） | 理由摘要 |
| --- | --- | --- | --- | --- |
| 0 | CUDA 启用 | **KEEP**（基线锚点） | 0.2794 | 65k 维输入 CPU 算不动，设备启用是工程前提 |
| 1 | lr 1e-2 | REVERT | 0.2713 | −0.008 未过门槛 |
| 2 | 逆频率类别加权 | REVERT | 0.2644 | −0.015 未过门槛 |
| 3 | **线性头**（65k→148 直连） | **KEEP**（冠军） | **0.3260** | +0.0466，复杂度 −54 节点 |
| 4 | width 128 | REVERT | 0.3259 | 线性头下参数无用，得分不变 |
| 5 | lr 1e-3 | REVERT | 0.3186 | −0.007 未过门槛 |
| 6 | 两层头（65k→256→148） | REVERT | 0.3316 | +0.0056 过噪声门槛但 < 11 节点 × 0.001 复杂度地板 |
| 7 | lr 1e-2（线性头） | REVERT | 0.3173 | −0.009 未过门槛 |
| 8 | AdamW wd 1e-4 | REVERT | 0.3245 | −0.0015 未过门槛 |

PLATEAU=5（连续 REVERT：轮 4–8）按声明早停。

**裁决质量**：轮 6 是引擎判据的正面样本——实测增益真实为正、过了噪声门槛，
但按项目复杂度经济学（1 节点 = 0.001 增益）判负。与 v10 的 tcm-0004 裁决同口径；
主线未事后放宽系数（为结果拟合指标是本项目反对的病理）。

## 3. 最终交付（holdout 解锁 1/1，预算用满）

台账事件 `00dd8687b86f4518…`（`runs/holdout-unlock-ledger-sd.jsonl`，预算
MAX_UNLOCKS_SD=1）。冠军 tcm-0003（commit `1dbdbe5`）在选择池 46,373 行重训
（seed 0，90.67 s），holdout 4,049 例在隔离 evaluator 子进程一次评分：

| 模型 | holdout macro-F1（观察类） |
| --- | --- |
| TF-IDF+LR（冻结配置，池训练，解锁后同口径评分） | **0.3732** |
| 神经冠军（线性头，AutoResearch 优化） | 0.3324 |
| 多数类 Dummy | 0.0020 |

工件（acam-lab `runs/sd-campaign/champion/`，不入 Git）：weights.pt SHA
`1a180268f0132360…`、weights-architecture.json SHA `31776a27f770e9ac…`、
finalize.json、pool-manifest.json。候选代码 SHA `7e586d6effafbc6a…`。
LR 基线对照：`runs/sd-campaign/lr-holdout-evaluation.json`（池训练 46,373 行、
清单 holdout 4,049 行逐行一致、冻结 seed 20261101）。

**诚实解读**：AutoResearch 把神经模型从冒烟 0.1379 提升到 0.3324（+0.14），
并且正确识别了架构方向（该特征空间线性可分，瓶颈编码器是纯损耗）。但两次独立
交付（体质线 0.3702 vs BMI 0.3761；本线 0.3324 vs LR 0.3732）都是**神经未胜过
简单基线**。在模板化病历文本 + 尾部类别支持度低的条件下，经典字符 n-gram 线性
模型仍是正确第一选择；神经交付物的价值在工程闭环与可解释接口，不在指标优势。

## 4. 证据状态与边界

- 本任务三层证据：GO_ENGINEERING 达成（真实数据全链路 + 隔离链 + 台账）；
  GO_REAL_DATA_TRAINING 达成（PUBLISHED 标签、QC 通过、受试者级划分）；
  GO_RESEARCH_CLAIMS **未达成**（无独立参考数据；跨来源泛化 NOT_COMPUTABLE——
  TCMEval-SDT 对齐覆盖 18.9% 且多标签语义不匹配，TCM-Lung 无名称解码无许可，
  详见数据审计 §外部候选）。
- 独立普通用户/专家验证：NOT_AVAILABLE。
- 数据许可 CC BY-NC-SA 4.0：仅科研非商业；商业部署需单独授权确认。
- 本划分 holdout 解锁预算 1/1 已用满；再评估需新数据里程碑或用户明确授权。
- 本报告所有数字均可从 `runs/`（acam-lab）工件与台账独立重算；不构成诊断
  准确率或临床有效性声明。

## 5. 复现

```bash
# 输入构建（真实数据在 dev host data/local/tcm-sd/）
PYTHONPATH=src python -m scripts.build_sd_campaign_inputs --scenario short
# 战役（每轮一个真实补丁，引擎裁决）
PYTHONPATH=src python scripts/run_sd_v1.py
# 冠军 refit + holdout 一次评分（扣台账）
PYTHONPATH=src python -m runtime.cli research-src-finalize --worktree <wt> \
  --branch autoresearch/sd-v1 --data-manifest runs/sd-campaign/manifest.json \
  --dataset runs/sd-campaign/records.json --out runs/sd-campaign \
  --task syndrome --editable src/research/candidate_sd.py --budget-s 90 --seed 0
# 测试
.venv/bin/python -m pytest -q   # acam-lab 393 passed @ 237cdab 前后基线
```
