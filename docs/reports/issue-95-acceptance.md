# Issue #95 验收报告：实时生理工程主线（2026-11-08）

**证据级别**：`LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`。
本报告只证明工程契约与测试可运行，**不证明**真实疲劳预测、按摩疗效或设备控制安全。
真实 94 字段导出、5 场共用账号会话、独立配对标签与厂商控制协议均未提供，
相关能力标 `BLOCKED_BY_DATA_OR_DEVICE` 或 `INSUFFICIENT_EVIDENCE`。

## 1. 运行环境

| 项 | 本地（macOS） | 远程开发机（acam-lab / lora） |
|---|---|---|
| 路径 | `/Users/lora/repos/MulitiModal` | `/home/lora/workspace/projects/MulitiModal` |
| Git SHA | `379ee12` + 未提交工作树 | 同左（rsync 镜像，未 `--delete`） |
| Python | 3.13.12（`.venv`） | 3.12.3（`.venv`） |
| PyTorch | 未安装 | 2.15.0+cu132 |
| 平台 | macOS-27.0.1-arm64，10 CPU | Linux 6.17 x86_64，24 CPU |
| 私有数据 | 缺失真实导出与会话 | 保留远端 `data/`、`artifacts/`、`tcm-50*`，同样无 94 字段导出 |

## 2. 实际运行的命令与结果

### 2.1 本地

```text
.venv/bin/python -m pytest -q
→ 654 passed, 18 skipped  (skip 均为“无 torch”或“私有数据仅远端齐备”)

.venv/bin/python -m ruff check src/physio tests/physio \
  scripts/physio_realtime_demo.py scripts/physio_benchmark.py scripts/physio_verify.py
→ All checks passed!

.venv/bin/python -m ruff check src tests scripts --output-format=concise
→ 8 errors，全部位于本轮未改动的既有文件：
   src/research/constitution_value_review.py（2×E501）
   src/syndrome/inference.py（1×E501）
   src/syndrome/value_review.py（3×E501）
   tests/scripts/test_tcm_input_scenarios.py（B905）
   tests/syndrome/test_inference_robustness.py（I001）

PYTHONPATH=src .venv/bin/python -m scripts.physio_realtime_demo
→ acceptance PASS 13/13；inference NOT_AVAILABLE；control_command None

.venv/bin/python scripts/physio_verify.py
→ acceptance PASS (13/13)；pytest tests/physio PASS；scoped ruff PASS；OVERALL PASS
```

### 2.2 远程开发机

```text
.venv/bin/python -m pytest -q
→ 756 passed, 2 skipped, 1 warning  (torch 训练测试实际执行并通过；
   skip 仅为 Zenodo 引用文件未缓存)

.venv/bin/python -m ruff check src/physio tests/physio \
  scripts/physio_realtime_demo.py scripts/physio_benchmark.py scripts/physio_verify.py
→ All checks passed!

PYTHONPATH=src .venv/bin/python -m scripts.physio_realtime_demo
→ acceptance PASS 13/13；inference NOT_AVAILABLE；control_command None；
  timeline 候选随窗口增长：[] → [M01, M02] → [M01, M02] → [M01, M02]

.venv/bin/python scripts/physio_verify.py
→ acceptance PASS (13/13)；pytest tests/physio PASS；scoped ruff PASS；OVERALL PASS
```

本地与远程测试数差异来自环境能力：本地无 torch，`tests/physio/test_training.py`
与 `test_realtime_model.py` 按仓库惯例整体 skip；远程实际训练并验证了
masked loss、有限梯度、checkpoint 保存/重载和共享账号显式 opt-in。

## 3. Benchmark 实测（240 事件 / 5 次重复，合成数据）

| 指标 | 本地（macOS arm64） | 远程（Linux x86_64） |
|---|---|---|
| 导入吞吐 | 11,366 events/s | 13,248 events/s |
| 窗口构建 p50 / p95 | 0.732 / 0.824 ms | 0.600 / 0.719 ms |
| 推理 p50 / p95 | 0.759 / 0.820 ms | 0.665 / 0.713 ms |
| 峰值 traced 内存 | 483,693 B | 473,713 B |
| 推理状态 | `NOT_AVAILABLE` | `NOT_AVAILABLE` |
| 过期输入 | `ABSTAIN` | `ABSTAIN` |
| 会话事件 / gap / 重复 | 240 / 0 / 0 | 240 / 0 / 0 |

**注意**：这是合成事件流上的工程耗时，不是真实设备端到端延迟，也不构成任何
医学或产品验收阈值。benchmark 代码本身不定义阈值。

## 4. 13 项统一验收检查

| # | 检查 | Issue | 结果 |
|---|---|---|---|
| 1 | 设备/服务/离线协议等价 | #96 | PASS |
| 2 | 乱序回放窗口等价 | #97 | PASS |
| 3 | 未知字段/单位、SpO2 0、rr 0、非法 HR/BP、非数值拒绝 | #96 | PASS |
| 4 | 未绑定/未验证/篡改工件不评分；过期输入拒判 | #97 | PASS |
| 5 | Post−Pre 为有符号观测；`y_pri` 恒 null | #98 | PASS |
| 6 | 仅静态候选；过期遥测拒判；用户停止为 SAFE_STOP；无控制指令 | #99 | PASS |
| 7 | 中医标签只映射已审核知识；缺审核/许可即排除 | #100 | PASS |
| 8 | 推理不 import 离线适配器；demo 无 CSV 路径；TCM 不 import Mendeley | #95 | PASS |
| 9 | 快照/完成态确定性裁决；同一记录不得跨 Pre/Post | #96 | PASS |
| 10 | 折内插补保持 mask；`y_delta` 需独立有效端点；`y_pri` null | #98 | PASS |
| 11 | 无标签、无可信身份均拒绝训练 | #97 | PASS |
| 12 | 小样本固定 `INSUFFICIENT_EVIDENCE` | #95 | PASS |
| 13 | 工件绑定：缺失或篡改 scaler / 模型 state、错误特征/目标版本均拒判 | #97 | PASS |

## 5. 能力状态

| 能力 | 状态 | 证据 |
|---|---|---|
| 三来源统一事件/Session、字段 QC、断线/乱序 | `IMPLEMENTED_AND_TESTED` | `src/physio/{schema,metrics,adapters,stream}.py` + `tests/physio/` |
| 94 字段导出注册表、逐行 QC、确定性导入 | `IMPLEMENTED_AND_TESTED`（合成 fixture） | `src/physio/legacy_export.py`，17 tests |
| mask 分离、训练折插补、受试者划分、标签准入 | `IMPLEMENTED_AND_TESTED` | `src/physio/dataset.py`，13 tests |
| torch-optional smoke 训练、manifest、resume、state 摘要 | `IMPLEMENTED_AND_TESTED`（远端 torch） | `src/physio/training.py`，8 tests 远端通过 |
| 模型/Scaler 工件绑定与篡改拒判 | `IMPLEMENTED_AND_TESTED` | `src/physio/artifacts.py`，10 tests + acceptance 第 13 项 |
| 回归/分类测评与证据门禁 | `IMPLEMENTED_AND_TESTED` | `src/physio/evaluation.py`，8 tests |
| benchmark（吞吐/延迟/内存/异常计数） | `IMPLEMENTED_AND_TESTED` | `src/physio/benchmark.py`，7 tests |
| 按事件时间逐步回放 E2E | `IMPLEMENTED_AND_TESTED` | `scripts/physio_realtime_demo.py`, `test_integration_chain.py` |
| 跨模块确定性/单调性属性检查 | `IMPLEMENTED_AND_TESTED` | `tests/physio/test_properties.py` |
| 真实字段名映射与真实 QC 回归 | `BLOCKED_BY_DATA_OR_DEVICE` | 无真实 94 字段导出 |
| 真实疲劳标签与训练有效性 | `INSUFFICIENT_EVIDENCE` | 无独立配对标签；`GO_REAL_DATA_TRAINING(physio)` 仍 `NOT_GRANTED` |
| 真实设备控制 | `NOT_CONNECTED` | 无厂商协议与安全验证 |

## 6. 本轮新增/修改文件

新增：

- `src/physio/legacy_export.py`、`src/physio/dataset.py`、`src/physio/training.py`、
  `src/physio/artifacts.py`、`src/physio/evaluation.py`、`src/physio/benchmark.py`
- `scripts/physio_benchmark.py`、`scripts/physio_verify.py`
- `tests/physio/test_legacy_export.py`、`test_dataset.py`、`test_training.py`、
  `test_artifacts.py`、`test_evaluation.py`、`test_benchmark.py`、
  `test_integration_chain.py`、`test_properties.py`
- `docs/reports/issue-95-gap-analysis.md`、本报告

修改：

- `src/physio/acceptance.py`：检查数 8 → 13，修复 torch 路径下的 fake model 与
  `dataset_label_gate` 断言
- `scripts/physio_realtime_demo.py`：新增 `timeline` 逐步回放，保留原有字段
- `tests/physio/test_acceptance.py`：期望 13 项检查
- `README.md`、`AGENTS.md`、`docs/design/physio-realtime-v1.md`：补充离线导入、
  数据纪律、训练/测评/benchmark/verify 入口与限制

未修改：`src/core/{specs,batch,models}.py`、两条 AutoResearch 候选文件、
全部冻结评价器与历史实验语义。

## 7. 未解决项与后续条件

1. **真实数据映射**：需要一份经许可的 94 字段导出及字段字典，才能把
   `BLOCKED_BY_DATA_OR_DEVICE` 转为真实映射验证。
2. **真实标签**：需要独立于厂商 `fatigue_index` 的配对标签；在此之前
   `y_pri` 保持不可用、推理保持 `NOT_AVAILABLE`。
3. **独立受试者**：5 场共用账号会话只能做管线回归，不能当作 5 名独立受试者。
4. **设备控制**：需要厂商控制 API 与安全验证，当前固定 `NOT_CONNECTED`。
5. **既有 lint 债务**：8 个非本轮 ruff 错误保留原状，避免修改冻结研究文件。

## 8. 证据边界声明

- 所有数值来自合成事件与本地/远程开发机，标识为 `LOCAL_EXPERIMENTAL`；
  隔离级别为 `AGENT_ISOLATION_UNVERIFIED`。
- 未提交、未推送、未创建 PR；工作树改动保留给后续审查。
- 未写入任何真实 Session、会员标识、设备流水、凭据或含原始数值的日志。

## 9. 独立评审与整改

两轮独立只读评审（Claude、Gemini）在本轮结束后给出 8.6/10 与 8.7/10，
均已逐项核实。第三家 provider（Codex/Grok/FX/Kiro/Pi）因额度或鉴权不可用，
未虚构第三方评审。

已修复的评审发现：

| 级别 | 发现 | 整改 |
|---|---|---|
| P1 | Gap Analysis 称“已实现 session 指纹与版本绑定”，但推理层无 checkpoint/sha 校验 | 已实现 `ModelArtifact`：绑定特征/目标版本、scaler sha256 与模型 state sha256；缺失或篡改一律 `NOT_AVAILABLE` |
| P1 | `test_interval_too_long_rejects_all_metrics` 端点无 measurement，断言恒真 | 补合法 measurement，并断言 `ADMISSION_INTERVAL_TOO_LONG` |
| P1 | EEG 阻断分支为死代码 | 删除该死分支，注册表注明 EEG 仅审计、不产生 measurement，并加测试断言 |
| P1 | README benchmark 命令缺 `PYTHONPATH=src` | 命令补全路径 |
| P2 | 训练循环手写前向，与 `model.forward` 重复 | 改为 `model.forward(ModelBatch, feature_mask=...)` |
| P2 | 协议字段解析失败复用时间戳/时长冲突状态码 | 新增 `ISSUE_INVALID_VALUE` |
| P2 | `_normalise_rows` 过滤非 Mapping 行后行号偏移 | 保留原始绝对行号 |
| P2 | 独立 Pre/Post 各自 sequence 从 0 起会被误判同记录 | `source_record` 相异时优先按记录判定，并加测试 |
| P2 | `y_pri` 同记录场景 reason 粒度粗 | 改为 `ADMISSION_SAME_RECORD` |
| P2 | `test_properties.py` 无用 builder 残留 | 删除 |


## 10. 第二轮评审结果与 P2 收尾

| 评审方 | 分数 | P0/P1 | 结论 |
|---|---|---|---|
| Claude `claude-opus-5-5` | **9.1 / 10** | 无 | 无需立即复评 |
| Gemini `gemini-3.8-flash-high` | **9.3 / 10** | 无 | 无需复评，建议合入 |
| Codex `gpt-6.1-sol` | 不可用 | — | 账号额度耗尽（2026-11-11 后恢复） |

第二轮评审提出的 P2 已收尾：

- interval 分支的 `y_pri` reason 改为 `ADMISSION_INTERVAL_TOO_LONG`
- 单侧带 `source_record`、另一侧缺失的混合 provenance 直接拒绝为 `ADMISSION_UNVERIFIED_RECORD`，不再回退 sequence
- 重复 completed 记录在 `received_at`/`sequence` 相同时用行内容 sha256 作为二级确定性排序，乱序导入摘要一致
- 训练拒绝信息附带行号、session 与 subject
- 跨日判定严格 `>` 的边界语义写入代码注释
- `SMOKE_ONLY` 提权拒绝负例移入无 torch 依赖的 `test_artifacts.py`
- demo 与 benchmark 脚本自行注入 `src`，裸跑不再依赖 `PYTHONPATH`

保留的 P2（不触及证据边界）：`INSUFFICIENT_EVIDENCE` 下仍计算描述性 mae/rmse（`status` 为权威门禁）；16 项攻击面尚未抽成独立负例套件。两者均不影响本次评分与交付边界。
