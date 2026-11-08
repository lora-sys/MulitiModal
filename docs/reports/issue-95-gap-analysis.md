# Issue #95 Gap Analysis 与测试基线（2026-11-08）

**证据级别**：`LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`。
本报告只评估工程实现与测试，不声明真实疲劳预测、按摩疗效或设备控制安全。

## 1. 环境与工作树

| 项 | 本地（macOS） | 远程开发机（acam-lab / lora） |
|---|---|---|
| 仓库 | `/Users/lora/repos/MulitiModal` | `/home/lora/workspace/projects/MulitiModal` |
| 分支 / SHA | `feature/clean-slate` / `379ee12` | 同左 |
| 提交内容 | `src/physio/`、`tests/physio/`、`scripts/physio_*`、`docs/design/physio-realtime-v1.md`、`docs/reports/issue-95-*.md`、`AGENTS.md`、`README.md` | 已同步同一工作树；保留远端私有 `data/`、`artifacts/`、`tcm-50*` |
| Python | 3.13（`.venv`），无 torch | 3.12.3，torch 2.15.0+cu132，numpy 2.5.4，sklearn 1.9.1 |
| 私有数据 | 仅 Mendeley / TCM-SD / TCM 评测集；**无 94 字段会话导出，无 5 场共用账号会话** | 同上 |

## 2. 基线测试

### 本地

```text
.venv/bin/python -m pytest -q
→ 578 passed, 17 skipped

.venv/bin/python -m ruff check src/physio tests/physio scripts/physio_realtime_demo.py
→ All checks passed!
```

### 远程（首次同步后）

```text
.venv/bin/python -m pytest -q
→ 1 failed, 671 passed, 2 skipped

失败：tests/physio/test_acceptance.py::test_acceptance_all_checks_pass
原因：有 torch 时 acceptance 的 fake model 走了 ModelBatch 路径，fake input spec 不满足
core.ModelBatch.validate；本地无 torch 时不会触发。
处理：已改为不暴露 input_spec 的 duck-typed fake model，本地与远程均应通过。
```

```text
.venv/bin/python -m ruff check src tests scripts
→ 8 errors（均为既有文件，不在本轮新增范围内）
```

既有 ruff 错误位置：`src/research/constitution_value_review.py`（2×E501）、
`src/syndrome/inference.py`（1×E501）、`src/syndrome/value_review.py`（3×E501）、
`tests/scripts/test_tcm_input_scenarios.py`（B905）、
`tests/syndrome/test_inference_robustness.py`（I001）。本轮不修改冻结研究文件。

## 3. 能力清单（整改后状态）

> 本节状态为本轮整改完成后的实际状态；第 2 节保留整改前的基线数字。
> 合成 fixture 通过不等于真实数据验证。

| Issue | 能力 | 状态 | 源码 / 测试 |
|---|---|---|---|
| #96 | 三来源统一事件/Session、字段 QC、单位/零值、断线乱序 | `IMPLEMENTED_AND_TESTED` | `src/physio/{schema,metrics,adapters,stream}.py`；`tests/physio/test_{contract,field_metrics,adapters,stream}.py` |
| #96 | 94 字段离线导出映射、memberId/快照/完成态/EEG 时间冲突 | `IMPLEMENTED_AND_TESTED`（合成 fixture；真实映射 `BLOCKED_BY_DATA_OR_DEVICE`） | `src/physio/legacy_export.py`；`tests/physio/test_legacy_export.py` |
| #97 | 因果滑窗、离线/在线等价、折内 scaler、拒判门禁 | `IMPLEMENTED_AND_TESTED` | `src/physio/{windows,preprocessing,inference,model}.py`；对应测试 |
| #97 | 跨会话缓存污染、模型/Scaler 工件绑定、损坏工件 | `IMPLEMENTED_AND_TESTED` | `src/physio/artifacts.py` 以特征/目标版本 + scaler sha256 + 模型 state sha256 绑定工件，缺失/篡改/未验证一律 `NOT_AVAILABLE`；`validation_status` 仍是外部审核结论，绑定只保证模型与 scaler 不可互换 |
| #97 | 训练入口：synthetic smoke、前向反向、有限梯度、保存/重载、manifest、resume | `IMPLEMENTED_AND_TESTED`（远端 torch 主机） | `src/physio/training.py`；`tests/physio/test_training.py` |
| #98 | Post−Pre delta、masked loss、y_pri 不可用 | `IMPLEMENTED_AND_TESTED` | `src/physio/response.py`；`tests/physio/test_response.py` |
| #98 | 同记录冒充 Pre/Post、补值不参与标签、受试者级划分 | `IMPLEMENTED_AND_TESTED` | `src/physio/dataset.py`；`tests/physio/test_dataset.py` |
| #99 | 推荐状态机、静态候选、SAFE_STOP、无控制指令 | `IMPLEMENTED_AND_TESTED` | `src/physio/{programs,recommendation}.py`；对应测试 |
| #100 | TCMExamV1、审核知识 fail-closed | `IMPLEMENTED_AND_TESTED` | `src/physio/tcm_exam.py`；`tests/physio/test_tcm_exam.py` |
| #95 | 统一验收、模拟 E2E | `IMPLEMENTED_AND_TESTED` | `src/physio/acceptance.py`；`scripts/physio_realtime_demo.py`；`tests/physio/test_acceptance.py` |
| #95 | 逐步回放、benchmark、模型测评接口、统一测试入口、验收报告 | `IMPLEMENTED_AND_TESTED` | `scripts/physio_benchmark.py`、`src/physio/evaluation.py`、`scripts/physio_verify.py`、`docs/reports/issue-95-acceptance.md` |

## 4. 真实数据与标签缺口（硬阻塞）

1. 仓库与两台机器均**没有 94 字段会话导出**，也没有 Issue 所述 5 场共用账号会话；
   因此无法完成真实字段名映射、真实 QC 回归或真实受试者级划分。
2. 没有独立疲劳标签、配对 Pre/During/Post 真值或设备字段字典；
   `GO_REAL_DATA_TRAINING(physio)` 保持 `NOT_GRANTED`。
3. `hrv` 在不同健康记录类型中的语义未确认；本轮实现为显式拒绝歧义字段，
   不做“一律等同 HRV / 微循环”的猜测。
4. 上述缺口只能标 `BLOCKED_BY_DATA_OR_DEVICE`，不得用合成数据冒充真实数据，
   不得为了让训练跑通而补造标签。

## 5. 本轮整改顺序

1. 修复远程 acceptance 失败，恢复双环境一致。
2. 新增离线导出注册表与导入器（合成 94 字段 golden fixture），覆盖 Issue 列出的
   memberId、会话 ID、快照/完成态、重复/乱序、时区、跨日、EEG 时间、duration 冲突。
3. 新增数据集层：observed/valid/imputed mask 分离、训练折内 imputer、受试者级划分、
   y_delta 准入（禁止补值与同记录冒充）与可复现 QC 摘要。
4. 新增训练/测评/benchmark：无有效标签拒绝训练；真实样本不足返回 `INSUFFICIENT_EVIDENCE`；
   benchmark 只输出实测延迟/内存，不设未经验证的医学阈值。
5. 扩展 E2E 为按事件时间逐步回放，新增统一验证入口与验收报告，再跑本地 + 远程全量测试。
6. 通过 council 评审（Claude / Codex / Gemini）逐项核实并迭代到目标分数。

## 6. 明确不做

- 不提交真实会话、个人标识、设备流水或含原始数值的日志。
- 不修改 `src/core/{specs,batch,models}.py` 的冻结语义，不修改两条 AutoResearch 候选文件。
- 不训练真实疲劳模型、不生成真实疲劳分数、不连接或控制真实按摩椅。
- 不把合成 fixture 的通过写成真实生理效果验证。
