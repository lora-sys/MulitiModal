# ML 生命周期现状审计（#112）

**状态**：`LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`
**性质**：工程基线审计，不是结果报告。未宣称任何预测性能、疲劳改善、因果效应或临床有效性。
**基线**：PR #101（`95788fb`）与 PR #103（`e76e2b3`）均已合并；Epic #104 定义唯一生命周期。
**审计时点**：`src/lifecycle/` 工程层落地后、`docs/design/ml-lifecycle.md` 单一索引建立后。

## 1. 方法

按 #112 要求逐路径审计：已合并代码、`AGENTS.md`、PDF 原方案、研发/产品/TCM 三条未完成线，
输出"现有 / 缺失 / 重复 / 冻结 / blocked"矩阵。审计以**实测**为准：所有"通过"数字都来自
本机实际执行的命令，不以测试数量本身当作质量证据。

复现命令：

```bash
python -m pytest tests/lifecycle/test_contracts_fast.py -m fast -q   # 纯 stdlib 契约门禁
python -m pytest tests/lifecycle -q           # 合成生命周期套件
python -m ruff check src tests scripts        # 全仓 lint
python scripts/lifecycle_verify.py            # 统一验收（含 15 项检查 / 10 项负例）
python scripts/lifecycle_verify.py --torch-free-smoke   # 无 torch 时纯契约可导入
```

## 2. 七阶段状态矩阵

| 阶段 | Issue | 现有（合并前） | 本次新增 | 缺失 / 冻结 / blocked |
|---|---|---|---|---|
| 1 任务与标签协议 | #104 | `src/core/specs.py`（Pre-only BPSI/EFRI 契约）、`src/physio/schema.py`、`evidence/registry.py` | `src/lifecycle/contracts.py`：12 类不可变 manifest/spec + 三门禁 + evidence 级 | **冻结**：`src/research/constitution.py` / `prepare.py` 评估器、两条冠军线；`y_pri` 恒不可用；真实标签 BLOCKED |
| 2 数据源与管道 | #105 | `src/physio/{adapters,legacy_export,real_export,real_data_audit}.py`（94 字段映射与逐列审计） | `src/lifecycle/pipeline.py`：增量幂等六阶段、checkpoint/cursor、缓存键、隔离区、ready-gate | **blocked**：厂商字段字典、`脑波` 选择规则、`is_complete` 语义、`response_hash` 构造、`15.41` 补值来源 |
| 3 特征与标签 | #107 | `src/physio/dataset.py`（observed/valid/imputed mask、FoldImputer）、`src/physio/metrics.py` | `src/lifecycle/features.py`：版本化注册表、BPSI/EFRI 声明规则、target/head 兼容检查 | **冻结**：`PhysioBPSIEFRIModel` 输入契约（BODY_DIM=12 / PHYSIO_DIM=7）；不得改冻结模型去迁就不相容目标 |
| 4 训练与评估 | #108 | `src/physio/{model,training,evaluation,artifacts}.py`、`src/core/models.py`、`src/runtime/*`、AutoResearch Skill | `src/lifecycle/training.py`：prepare→train→validate→compare→package、RunManifest、合成 smoke | **冻结**：真实训练 `GO_REAL_DATA_TRAINING=NOT_GRANTED`；历史 holdout 预算耗尽；无 AutoResearch champion 提升 |
| 5 工件与部署 | #109 | `src/physio/artifacts.py`（ModelArtifact sha256 绑定、篡改检测） | `src/lifecycle/release.py`：ArtifactBundle、原子 candidate→champion、rollback、实测 benchmark | **blocked**：无 VALIDATED 真实工件；`validation_status=VALIDATED` 需 APPROVED + 非合成证据 |
| 6 网关与实时服务 | #110 | `src/physio/{inference,recommendation,tcm_exam}.py`、`src/syndrome/service.py` | `src/lifecycle/gateway.py`：薄版本化 envelope、三模式隔离、幂等/超时/节流、fail-closed | **冻结**：`control_command` 恒 null；`NOT_CONNECTED`；C 模式不得从生理数值推断证候 |
| 7 监控与迭代 | #111 | `src/research/dashboard.py`、`scripts/physio_benchmark.py` | `src/lifecycle/observe.py`：append-only 哈希链、匿名聚合、数据触发仅刷新、KEEP/REVERT/CRASH 记录 | **blocked**：自动训练 / 自动提升 / 自动发布一律禁止 |

## 3. 重复与收敛

| 重复面 | 处理 |
|---|---|
| 数据字典 / 版本化 spec 散落在 `core/specs.py`、`physio/model.py`、`physio/dataset.py` | `contracts.py` 提供**唯一词汇**；其余模块保持原样不改写，避免动冻结研究线 |
| 工件哈希绑定同时存在于 `physio/artifacts.py` 与 `lifecycle/release.py` | 前者是推理时 fail-closed 绑定，后者是发布期原子提升/回滚；职责不同，不合并 |
| `src/physio/acceptance.py`（16/16）与 `src/lifecycle/acceptance.py`（15/15） | 分别是生理工程验收与生命周期验收；`lifecycle_verify.py` 不重复跑前者，`physio_verify.py` 仍独立 |
| 旧 `src/data/*` 与 `src/lifecycle/pipeline.py` | 前者是既有研究线数据读取；后者是生命周期增量编排，不复制其实现 |

## 4. 审计发现的真实缺陷与修复（禁止为了让 CI 绿而扩大假通过）

| ID | 严重度 | 问题 | 修复 |
|---|---|---|---|
| A1 | **P1** | `src/core/__init__.py` 顶层 eager 导入 `.models`（后者 `import torch`），干净 CI 无 torch 时连**纯 stdlib** 的 `core.specs` 都无法 import。实测：屏蔽 torch 后 `import core.specs` 抛 `ModuleNotFoundError` | 改为 PEP 562 `__getattr__` 惰性解析 torch 相关符号；`specs` 仍顶层导出。实测：无 torch 时 `core.specs` 可用、`core.ModelBatch` 明确拒绝；有 torch 时 `from core import ModelBatch, PhysioBPSIEFRIModel` 与 `dir(core)` 语义不变 |
| A2 | P2 | 全局 `ruff check src tests scripts` 有 8 处遗留错误（6×E501、1×B905、1×I001） | 逐处机械修复：长字符串改为相邻字面量、`zip(..., strict=True)`、import 排序。研究线 281 passed / 7 skipped 确认字符串值未变 |
| A3 | P2 | `pyproject.toml` 声明了 `fast`/`slow` 标记，但**全仓 0 个测试**使用 `fast`，`pytest -m fast` 会 NO_TESTS_COLLECTED | 新增 `tests/lifecycle/test_contracts_fast.py`（19 项纯 stdlib 契约测试，标 `fast`）；启用 `--strict-markers`；新增 `real_data` 标记说明 |
| A4 | P2 | 新增 `tests/lifecycle/test_training.py` 与既有 `tests/physio/test_training.py` basename 撞名，pytest 收集期 `import file mismatch` | 改名 `test_train_ready.py`；未加 `tests/lifecycle/__init__.py`，避免遮蔽 `src/lifecycle` 包 |
| A5 | P2 | 无 `.github/workflows`，无法证明干净环境可复现 | 新增 `ci.yml`：3 层 job（fast 契约 × Py3.11/3.12/3.13、合成 E2E × Py3.11/3.13、无 torch 诚实 skip），无私有数据、无联网厂商服务、无 GPU |
| A9 | **P1** | Step 5 Preview 评审复现：同一目录先跑默认配置、再把 `min_subjects` 改 3，第二次因 `processed` 游标仅按 `dataset_sha256` 继承而跳过全部批次，**把 7 行的好 snapshot 覆盖成 0 行**；`sources.py` 的 SIMULATED+CONNECTED 守卫条件写反，带空 conformance 的假真设备可构造成功 | **已修**：`deps_match` 必须同时匹配 dataset/schema/code/config/rules；SIMULATED+CONNECTED 无条件拒判；`_batch_key` 加入 raw sha256，同名不同内容按"修正"隔离登记并强制 re-ingest；缓存 QC 损坏时清空游标全量重算；6 个新回归测试逐个经突变验证 |
| A8 | **P1** | 无 torch 时 `pytest -m fast` 会**收集全仓**，`tests/runtime/*` / `tests/core/*` 顶层 import torch-backed 模块 → 5 个 collection error，CI fast job 直接崩（实测） | fast 门禁改为明确路径 `pytest tests/lifecycle/test_contracts_fast.py -m fast -q`；fast 门禁本就只应跑 stdlib 契约文件，不是缩窄通过数。实测无 torch：fast 19 passed、lifecycle 146 passed（当时 121）、lifecycle+physio 全绿；本机有 torch 全仓 979 passed / 9 skipped |
| A6 | P3 | 管道 `snapshot.created_at` 取墙钟时间，导致同一输入两次运行 snapshot hash 不同；`PipelineState` 的 `run_id`/时间戳同理 | `created_at` 改为数据内最早 `event_time`（确定性 as-of）；缓存命中时继承 prior 的 run 身份与时间戳，state 文件逐字节一致。由 `test_repeated_run_is_byte_identical_and_has_no_duplicates` 钉住 |
| A7 | P3 | 训练 `stage="validate"` 与管道数据校验阶段同名，语义冲突 | 训练侧改 `validate_model`，`STAGES` 增 `prepare` |

## 5. 测试基线分层（不膨胀通过数）

| 层 | 标记 | 命令 | 内容 |
|---|---|---|---|
| 快测 | `fast` | `pytest -m fast` | 纯 stdlib 不可变契约：门禁独立、假设备拒判、受试者泄漏、VALIDATED 证据约束、cache key 依赖、QC 无静默跳过 |
| 合成 E2E | `synthetic` | `pytest tests/lifecycle` | 增量幂等、乱序/迟到收敛、失败恢复、drift 阻断、折内拟合、release 原子提升/回滚、gateway 负例、observe 哈希链 |
| 安全负例 | — | `lifecycle_verify.py` | 15 项统一检查含 10 项负例：注入失败、ready-gate 拒判、坏行带理由、训练不自动提升、工件不自提升 |
| 慢测 | `slow` | `pytest -m slow` | 长耗时集成/Bootstrap；CI 按需启用，未纳入常规通过数 |
| 真实数据 | `real_data` | 默认跳过 | 需 `data/local/` 私有数据；**不计入常规 CI 通过数** |

**明确不做的假通过**：不把 skip 计入通过；不因测试变绿而提权 `GO_REAL_DATA_TRAINING` /
`GO_RESEARCH_CLAIMS`；不为了让 lint 干净而放宽规则；不把 synthetic golden 写成真实结论。

## 6. 三门禁与逐阶段 GO/NO-GO

| 门禁 | 判定 | 依据 |
|---|---|---|
| `GO_ENGINEERING` | **GRANTED**（本次工程 PR） | 合成 CI + 本地安全负例通过；`lifecycle_verify.py` OVERALL PASS |
| `GO_REAL_DATA_TRAINING` | **NOT_GRANTED** | 无独立可信配对标签、无生理受试者身份字典、历史 holdout 预算耗尽 |
| `GO_RESEARCH_CLAIMS` | **NOT_GRANTED** | 无独立验证、无真实模型、无外部数据 |

`gate_bundle(engineering_go=True)` 在本层强制后两者为 `NOT_GRANTED`，且 `DatasetManifest.trainable()`
以 `approval=APPROVED` + `identity_trust=TRUSTED` + `frozen=True` + 非 SMOKE 证据为硬门禁。
测试通过**不会**改变这三者。

## 7. 干净环境 / 版本 / 平台差异

- `requirements-dev.txt` 声明 `pytest>=8`、`ruff>=0.6`、`numpy>=1.26`、`scikit-learn>=1.5`、
  `joblib>=1.3`、`xlrd>=2.0`、`openpyxl>=3.1`、`transformers>=4.40`；**torch 不在其中**，
  仓库约定神经栈只在专用训练机安装，本地缺 torch 时对应用例 skip。
- CI 固定 `actions/checkout@v4` + `actions/setup-python@v5`，矩阵 Py3.11/3.12/3.13，
  并打印 `pip freeze` 与 `python --version` 供复现。
- 无 torch 环境：`lifecycle.contracts` / `lifecycle.pipeline` / `core.specs` 实测可导入；
  神经测试以 `pytest.importorskip("torch")` 显式 skip，不伪装通过。
- **已知未固定**：`requirements-dev.txt` 用 `>=` 下界而非精确 pin，CI 会随上游小版本漂移。
  这是当前依赖管理的真实缺口，已在 `docs/reports/ml-lifecycle-acceptance.md` 列为遗留项，
  未通过放宽 lint 或跳过测试掩盖。

## 8. 旧 Issue 迁移核对（#104 表）

| 旧 Issue 组 | 接替者 | 状态 |
|---|---|---|
| #22 旧总控 | #104 Epic | superseded |
| #23–#31 physio 契约/数据/标签/模型 | #105/#106/#107/#108 | superseded |
| #34–#36 QA / 验收 / 研究阻塞 | #112 + 三门禁 | superseded |
| #41–#43 BPSI/EFRI 规则与双分支 | #107（声明规则，非临床真值）+ #108（合成 smoke） | superseded |
| #46 端到端运行验收 | #112 生命周期验收 | superseded |

保留独立 open：#52 / #54 / #90 / #91（TCM 产品与数据阻塞事项），本次不触碰。
#102 已 CLOSED、#103 已 MERGED（`e76e2b3`，2026-11-10 `11:36 UTC`），合并单独审查完成。

**迁移注释的时态更正（持续同步）**：17 个旧 Issue 的关闭评论均写于 2026-11-10，
其中多处写「PR #103 仍待审查 / remains open」。该描述在本次合并后已过时；
历史评论保留作审计轨迹，不改写，但**当前事实以本行与 PR #103 状态为准**：
`#103 = MERGED`，`#102 = CLOSED`。实测逐条 `stateReason = NOT_PLANNED`，
关闭原因均为 `superseded/not_planned` 且明确"不代表所有需求已实现"，
与 #104 的迁移口径一致，无一条被误标为"已完成"。

**迁移未覆盖风险**：旧 #41–#43 的"真实 BPSI/EFRI 拟合"需求被拆到 #113/#114/#115 并标记
BLOCKED，属**迁移而非实现**；本报告不声称每项历史要求已完成。

## 9. 未解除边界

- 真实 94 字段完整导出与厂商字段字典：`BLOCKED_BY_DATA_OR_DEVICE`
- 独立配对疲劳标签 / 生理受试者隔离：`INSUFFICIENT_EVIDENCE` / `PHYSICAL_SUBJECT_ISOLATION_UNVERIFIED`
- 真实设备控制：`NOT_CONNECTED`
- 结论级别：`LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`

## 10. 评审团第一轮汇总（进行中）

用户要求三模型评审团（Gemini 3.8 Flash / Step 5 Preview / Claude Opus 5.5）对
质量与实现程度两轴打分，均须 ≥9.0；按汇总结果继续优化后复评。

| 评审者 | 质量 | 实现 | 总评 | 状态 |
|---|---|---|---|---|
| Step 5 Preview (`pi`) | 8.7 | 8.6 | **8.6** | 13 项 finding 已全部处置，见 `e334cb0`；待复评 |
| Gemini 3.8 Flash (`antigravity`) | 待返回 | 待返回 | 待返回 | 运行中 |
| Claude Opus 5.5 (`claudeAgent`) | 待返回 | 待返回 | 待返回 | 运行中 |

**Step 5 Preview 的阻塞项（已修，`e334cb0`）**：S1 假真设备可构造（P1）、
S2 依赖变化把 7 行 snapshot 覆盖成 0 行（P1）、S3 同名不同字节被当重复静默丢弃（P2）、
S9 缓存 QC 损坏崩溃（P2）。其余：S4 README fast 命令、S5 伪造的"110 项"、
S6 过时计数、S7 CI 无 torch 注释、S8 冻结研究线无 CI 保护、S13 内容级泄漏检查、
S10/S11/S12 时间戳与语义问题。每项修复都用"撤回修复看测试是否失败"验证过。

**首轮教训**：三个 P1/P2 缺陷是我自己在批量 `sed` 替换与增量语义优化时引入的，
说明"为了修一个发现而改动共享路径"必须同时跑全量回归 — 本次靠评审团的
独立复现捕获，未流入合并。

**未达标说明**：首轮 8.6 < 9.0，故本分支**不记为验收通过**；三份评分齐全后
按汇总再优化并复评，达标前不合并 PR。
