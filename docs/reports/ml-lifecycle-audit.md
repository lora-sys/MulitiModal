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
| 1 任务与标签协议 | #104 | `src/core/specs.py`（Pre-only BPSI/EFRI 契约）、`src/physio/schema.py`、`evidence/registry.py` | `src/lifecycle/contracts.py`：14 类不可变 manifest/spec + 三门禁 + evidence 级 | **冻结**：`src/research/constitution.py` / `prepare.py` 评估器、两条冠军线；`y_pri` 恒不可用；真实标签 BLOCKED |
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
| A8 | **P1** | 无 torch 时 `pytest -m fast` 会**收集全仓**，`tests/runtime/*` / `tests/core/*` 顶层 import torch-backed 模块 → 5 个 collection error，CI fast job 直接崩（实测） | fast 门禁改为明确路径 `pytest tests/lifecycle/test_contracts_fast.py -m fast -q`；fast 门禁本就只应跑 stdlib 契约文件，不是缩窄通过数。实测无 torch：fast 19 passed、lifecycle 149 passed（当时 121；补回归后为 159）、lifecycle+physio 全绿；本机有 torch 全仓 982 passed / 9 skipped（补测后为 992） |
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
  已提供伴随式 `requirements-lock.txt`（本机实测复现快照），但跨 Py3.11/3.12/3.13 矩阵
  不强制启用；残留风险与理由见 `ml-lifecycle-acceptance.md` §7，未通过放宽 lint 或跳过测试掩盖。

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
| Gemini 3.8 Flash (`antigravity`) | 9.6 | 9.8 | **9.6** | PASS；G1–G5 五个 P3 已全部处置，见 `02c7e25`；待复评 |
| Claude Opus 5.5 (`claudeAgent`) | 待返回 | 待返回 | 待返回 | 运行中 |

**Step 5 Preview 的阻塞项（已修，`e334cb0`）**：S1 假真设备可构造（P1）、
S2 依赖变化把 7 行 snapshot 覆盖成 0 行（P1）、S3 同名不同字节被当重复静默丢弃（P2）、
S9 缓存 QC 损坏崩溃（P2）。其余：S4 README fast 命令、S5 伪造的"110 项"、
S6 过时计数、S7 CI 无 torch 注释、S8 冻结研究线无 CI 保护、S13 内容级泄漏检查、
S10/S11/S12 时间戳与语义问题。每项修复都用"撤回修复看测试是否失败"验证过。

**首轮教训**：三个 P1/P2 缺陷是我自己在批量 `sed` 替换与增量语义优化时引入的，
说明"为了修一个发现而改动共享路径"必须同时跑全量回归 — 本次靠评审团的
独立复现捕获，未流入合并。

**Gemini 3.8 Flash 的 G1–G5（已修，`02c7e25`）**：G1 `bpsi()` 对 Mapping 用位置权重
会按 dict 插入序静默绑定 → 改要求 Mapping 权重；G2 冷启动 `run_id` 用墙钟 →
改为 dataset/schema/code/config/rules 的 digest；G3 队列溢出策略未命名 → 显式
`TAIL_DROP` 契约；G4 客户端时钟超前会产生负 lag → 钳为 0 并报 `CLOCK_SKEW_DETECTED`；
G5 依赖下界 → 加伴随式 `requirements-lock.txt`（本机复现快照，**不强制 CI**，
因矩阵跨 Py3.11/3.12/3.13）。五项均配回归测试，G1/G2/G4 经突变验证。

**未达标说明**：Step 5 Preview 首轮 8.6 < 9.0，虽已全部修复但**尚未复评**，
故本分支仍**不记为验收通过**；三份评分齐全、修复后复评两轴均 ≥9.0 前不合并 PR。
两份评审都针对 `55dd039`（修复前），复评必须针对最新 HEAD。

### 10.1 迟到评审（针对 `d5cc1b3`）的处置

| 项 | 评审主张 | 复现结论 | 处置 |
|---|---|---|---|
| C1 | 碰撞回滚未移除旧批次已接受行，`n_accepted + n_rejected + n_duplicate` 不能守恒 | **主诉为误报**：旧批次被接受的行的 idempotency key 在 re-ingest 时被覆盖，实测 `record_ids ['1','3','4','5']`，被删除的 `r2` 不会残留。但它暴露了**真实 P1**（见下） | 保留并强化回滚计数；新增「修正批次删除旧行」回归测试 |
| C2 | 冲突重复被静默 last-writer-wins | 成立（P3） | quarantine 记录补 `resolution: first_write_wins` / `kept_record_id` / `kept_batch_id` |

**C1 暴露的真实 P1（已修，`36068dc`）**：任意行级重复（批内或跨批）都会让
`QCReport` 守恒式失败并抛 `ContractError`，管道直接崩溃——含重复幂等键的真实
导出根本无法导入。修复：守恒式改为
`n_accepted + n_rejected + n_duplicate == n_input`；同批次字节相同的一次运行内重放
改记为 drift 而不是 duplicate；冲突重复记录保留策略。新增 4 项回归测试
（批内重复、跨批重复、修正批次删除旧行、同批次重放），契约与重放两项经突变验证
（各自撤回修复即 2 failed）。

### 10.2 评审团第二轮（HEAD `c087d30`）

| 评审者 | 质量 | 实现 | 总评 | 结论 |
|---|---|---|---|---|
| Claude Opus 5.5 | 8.8 | 9.3 | **8.8** | 否决：质量轴 < 9.0 |
| Step 5 Preview | 9.1 | 9.2 | **9.1** | 通过，建议同 PR 内补齐两个 P1 |
| Gemini 3.8 Flash | — | — | — | 复评未能执行：模型容量不足（见 10.3） |

**P1-1（两家独立发现，已修）**：`pipeline_state.json` / `snapshot.json` 被截断时
管道抛裸 `JSONDecodeError`；`resume=True` 是默认值，一次中断即可让之后每次运行
都崩，必须人工删 `runs/`。这与 `pipeline.py` docstring 自己声明的「中断可恢复」
矛盾，同目录下 `qc_report.json` 却已有兜底。修法：两个读取函数把不可读的 prior
当作「无 prior」走全量重算；5 个状态文件改为 `temp + fsync + os.replace` 原子写；
并补上「游标有效但快照不可复用 → 否则会跳过全部批次写出空 snapshot」的
`processed.clear()`。

**P1-1'（Claude 独立发现，已修）**：同名批次修正时的回滚会连带删除**仍被声明的
其他批次**拥有的行，且留下幻影 `n_duplicate`；更严重的是 snapshot 依赖批次声明
顺序。实测复现：

```
[A,B,A2]  input=2 acc=1 dup=1 rows=['src-2']      ← src-1 被静默删除
[B,A,A2]  input=2 acc=2 dup=0 rows=['src-1','src-2']
```

修法：`_ingest` 重写为「候选池 + 确定性胜出」——每个幂等键保留全部候选行，胜出者
为 `(batch_id, record_id)` 最小者，修正只摘除该批次贡献的候选；重复数在解析阶段
统一计算。修复后两种声明顺序输出逐字节相同。这也把重复仲裁从「先写先得」改成
与顺序无关的确定性规则。

**P1-2（Step 5 Preview，已修）**：设计索引第 139 行仍写
`n_accepted + n_rejected == n_input`，代码已是含 `n_duplicate` 的版本。已同步。

**其余 P2 处置**：

| 评审者 | 项 | 处置 |
|---|---|---|
| Claude | 回滚过滤 `ws.rejections` 是死代码（`_validate` 在全部 ingest 之后才跑） | 随 `_ingest` 重写删除 |
| Claude / Step 5 | `private_data_not_in_git` 验收项近乎恒真 | 改为真跑 `git ls-files` 并断言无 `data/raw/`、`data/local/`、`runs/`、`artifacts/`、`*.pt`/`*.pth` |
| Step 5 | 新增的 `resolution` / `kept_record_id` / `kept_batch_id` 零测试覆盖（突变验证不挂） | 在跨批重复用例里断言这三个字段 |
| Step 5 | 缓存命中未校验 `qc.sha256 == snapshot.qc_sha256`，可产出自相矛盾的 `DatasetManifest` | 缓存命中时校验，不符即全量重算并记 drift |
| Step 5 | 文档称「12 类不可变 frozen dataclass」，实际 14 类 | 三处同步为 14 |
| Step 5 | 碰撞回滚触发面比文档暗示的窄 | 写入 `_ingest` docstring（跨运行修正必改 `dataset_sha256`，只会全量重算） |

**关于远程 `1000 passed / 4 skipped` 的可核验性（Step 5 P2-5）**：该数字此前只在
PR 描述里出现，未给命令与差异来源。现已核对并在此记录对账：

| 项 | 本地（macOS / Py3.13，无 torch） | 远程 `acam-lab`（Linux / Py3.12，torch 2.15 + transformers） |
|---|---|---|
| 收集用例 | 1000 | 1008 |
| 通过 / 跳过 | 992 / 9 | 1004 / 4 |

差 8 个收集项来自 `tests/syndrome/test_bert_model.py`（本地缺 `transformers`，
整模块跳过故不收集）；本地 6 个 `tests/scripts/test_tcm_minimal_demo.py` 因缺
`data/local/` 与 `artifacts/` 跳过、远程实际执行；本地 `tests/physio/test_real_data_audit.py`
27 passed，远程 25 passed + 2 skipped（该机缺私有导出）。逐项相加与两边总数一致。

### 10.3 第二轮 Gemini 复评的状态

Gemini 3.8 Flash 的复评任务在 `c087d30` 上启动后返回
`No capacity available for model gemini-3.8-flash-high on the server`，未产出分数。
该行按缺失处理，不虚构评分：首轮的 9.6 针对 `55dd039`，不能代表修复后的 HEAD。

处置：修复落盘后在新 HEAD 上重试；若持续无容量，则以 Claude 与 Step 5 Preview
两份**针对最终 HEAD** 的评分为准，并在 PR 中如实标注 Gemini 缺位。

### 10.4 第二轮之后的修复（当前 HEAD）

两家评审提出的 P1 与 P2 已全部落盘：

- `_ingest` 重写为「候选池 + 确定性胜出」，消除碰撞回滚删除其他批次行与声明顺序依赖；
- 状态/快照读取容错 + 全部状态文件原子写；
- 缓存命中校验 `qc.sha256 == snapshot.qc_sha256`；
- 存档游标在无可用快照时清空，避免跳过全部批次写出空 snapshot；
- 验收项 `private_data_not_in_git` 改为真跑 `git ls-files`（实测 222 个跟踪文件，无私有路径）；
- 设计索引的守恒式与「12 类」计数同步为实际契约（14 类）。

新增 4 项回归测试，全部经「撤回修复即失败」的突变验证。
本地实测：全仓 `992 passed, 9 skipped`、`tests/lifecycle` `159 passed`、fast `19 passed`、
`ruff` 零告警、`lifecycle_verify.py` `15/15 PASS`。
远程 `acam-lab` 实测：全仓 `1004 passed, 4 skipped`、`tests/lifecycle` `159 passed`、
`lifecycle_verify.py` `15/15 PASS`。

### 10.5 第三轮评审（HEAD `dbaa579`）与 15 项 finding 的处置

第三轮由 Codex 在三个被评审 commit 上共提出 14 项 finding（4 项 P1、10 项 P2），
Step 5 Preview 给出 9.2 / 9.5「通过、建议合并」（另提 1 项非阻塞 P2），
Claude Opus 5.5 给出 8.9 / 9.5 并否决，唯一否决理由是「确定性胜出排序无测试保护」，
另指出验收报告一处陈旧计数。Gemini 3.8 Flash 因模型容量不足未返回评分，如实标注缺位。

**全部 14 项 Codex finding + Claude 2 项 P1 + 两家 P2 已落盘：**
（列中的编号为本轮复现脚本 `/tmp/repro_findings.py` 的 F1–F15，含已由 `90b2cb7`
修复并补回归锁定的 F1）

| 编号 | 级别 | 问题 | 处置 |
|---|---|---|---|
| F1 | P1 | 崩溃恢复无可用快照时游标跳过全部批次（旧 finding） | 已由 `90b2cb7` 修复，本轮补回归锁定 |
| F2 | P1 | 请求自带 `validated` / `safety_authorized` 即可让实时模式发分 | 模式 B 固定 `NOT_AVAILABLE`，不信任请求侧声明 |
| F3 | P2 | 回滚不校验 previous 工件文件 | 回滚前重建并 `verify()`，失败即拒且保持 champion |
| F4 | P1 | 缓存命中丢失 QC blockers，`train_ready` 由假翻真 | `_readiness` 消费 `qc.blocked`；失败路径先记 blocker 再算 QC |
| F5 | P2 | cutoff / 排序 / gap 用字符串比较 ISO 时间 | `_parse_event_time` 归一为 UTC-aware instant |
| F6 | P2 | SessionWindow 接受别的 session 事件 | 拒绝并登记理由（不静默丢弃） |
| F7 | P2 | 声明后改写 raw bytes 仍被摄入 | `_read_rows` 先校验 raw sha256 与声明一致 |
| F8 | P1 | SAFE_STOP 被幂等缓存与限流挡住 | 停止信号在缓存/限流之前求值 |
| F9 | P2 | `legal_range` 未约束观测值与折内中位数 | 越界观测记 observed/无效/0.0 占位，且不进中位数 |
| F10 | P2 | 缓存 snapshot 内容被篡改仍被复用 | `Snapshot.rows_sha256`，解析时校验，不符即全量重算 |
| F11 | P2 | 两个 store 实例共用日志会接出断链 | append 在共享锁内重读 durable 链尾并校验 |
| F12 | P2 | smoke 清单的 commit/environment 用默认值 | `run_synthetic_smoke` 透传 `TrainConfig` 溯源 |
| F13 | P2 | dataset 缓存键不含 batch-id 绑定 | 键改为 `sorted((batch_id, sha256))` |
| F14 | P2 | 重复提升把 `previous` 覆盖成自身 | 已是 champion 即拒，不改状态 |
| F15 | P2 | 结构化 `fields` 的 `timepoint=post` 绕过 Pre-only 守卫 | 提取时保留 phase 元数据 |
| C-P1 | P1 | 确定性胜出排序无回归保护 | 新增「声明顺序无关、snapshot 逐字节相同」用例 |
| C-P2 | P2 | 验收报告仍写「12 类」；私有路径分类器零覆盖；恒真断言 | 改为 14 类；补正负对照单测；断言改为字节级比较 |

新增 `tests/lifecycle/test_review_round3.py`（33 项）+ `test_train_ready.py` 1 项回归，
每项均为负例。**15 项机制修复逐项做突变验证**：撤回任一修复后对应回归测试即失败
（M1–M15 全部命中，包含 C-P1 的确定性排序）。F4/F15 首轮突变未命中系验证脚本
自身问题（`.pyc` 时间-大小碰撞、只替换两处相同代码中的一处），修正脚本后 15/15 命中。

**远程基线的可核验性（Claude P2-5 / Step 5 P2-5）**：此前 PR 只给数字未给来源，
本轮补齐 host、user、实测时间、精确命令与差异逐项对账：

| 项 | 本地（macOS / CPython 3.13.12，无 torch/transformers） | 远程 `acam-lab`（Linux / CPython 3.12.3，torch 2.15 + transformers） |
|---|---|---|
| 精确命令 | `PYTHONPATH=src:. .venv/bin/python -m pytest -q` | 同左 |
| 收集用例（`--collect-only -q`） | 1034 | 1042 |
| 通过 / 跳过 | 1038 / 9 | 1050 / 4 |
| `pytest tests/lifecycle -q` | 205 passed | 205 passed |
| `ruff check src tests scripts` | 零告警 | 零告警 |
| `lifecycle_verify.py` | 15/15 PASS | 15/15 PASS |

收集用例差 8 项，全部来自 `tests/syndrome/test_bert_model.py`：本地无
`transformers`，整模块在收集期跳过（0 项收集、1 条 module-level skip 结果），
远程 8 项实跑全过。结果数因此差 7（1035 → 1042）。逐文件对账：
`test_tcm_minimal_demo.py` 本地 3 passed / 6 skipped、远程 9 passed（缺私有工件）；
`test_real_data_audit.py` 本地 27 passed、远程 25 passed / 2 skipped（该机缺私有导出）；
`test_ccmq60.py` 两端均 11 passed / 2 skipped；其余文件两端同为 985 项。
按文件相加与两端总数一致，差异均有明确环境原因，没有把 skip 写成 PASS。

证据级别不变：`LOCAL_EXPERIMENTAL` / `AGENT_ISOLATION_UNVERIFIED` / `SMOKE_ONLY`；
三门禁仍为 `GO_ENGINEERING=GRANTED`、`GO_REAL_DATA_TRAINING=NOT_GRANTED`、
`GO_RESEARCH_CLAIMS=NOT_GRANTED`，CI 绿灯不自动提权。

### 10.6 终轮复评（HEAD `8a5c5f6`）与 2 项新 P2

对 `8a5c5f6` 发起终轮复评。Claude Opus 5.5 与 Step 5 Preview 的路由本次均返回
`fetch failed`（未返回评分，如实标注缺位，不虚构分数）；`opencode` 路由因额度耗尽
返回 `Insufficient balance`。Codex（gpt-6.1-sol）完成复评，给出 8.5 / 8.7 并否决，
列出 2 项本轮新引入的 P2：

| 编号 | 级别 | 问题 | 处置 |
|---|---|---|---|
| R4-1 | P2 | 同一 `batch_id` 的两个不同版本在单次运行内声明时，`dataset_sha256` 用排序后的 `(batch_id, sha256)` 集合，丢失修正顺序；`_ingest` 却按「最后一个版本胜出」，导致 `[old,new]` 与 `[new,old]` 缓存键相同但结果不同（实测 warm 复用得 90、cold 得 70） | 改为按 `batch_id` 取最后一个声明的 `sha256`（effective bindings）再排序求摘要，与 `_ingest` 的胜出语义一致 |
| R4-2 | P2 | 网关幂等缓存只以 `request_id` 为键；先提交模式 A/S1，再用同一 ID 提交模式 B/S2，会拿到模式 A 的 `OK` 与 BPSI/EFRI 数值，绕过模式 B 的不可用保护 | 缓存身份绑定 mode/session/trace/版本/payload；冲突复用返回 `REJECTED` 且 `prediction=None`，合法重放仍命中 |

两项均由复现脚本先确认，再各补 1 项负例回归；3 项突变（撤回 effective bindings、
去掉身份校验、不写入身份）全部命中。新增 2 项回归后：本地 `1028 passed, 9 skipped`、
`tests/lifecycle` `195 passed`；远程 `acam-lab` `1040 passed, 4 skipped`、
`tests/lifecycle` `195 passed`；两端 `ruff` 零告警、`lifecycle_verify.py` `15/15 PASS`。

10.5 的远程基线表按最终 HEAD 的数字更新（1028 / 1040、195 passed），收集用例差与
逐文件对账不变；10.4 保留为第二轮当时的历史快照，不回写。

累计 17 项机制修复（15 + 2）全部经撤回验证，共 36 项新增回归。证据级别与三门禁不变。

### 10.7 第五轮复评（HEAD `87eaaff`）与身份摘要的 P2

第五轮对 `87eaaff` 的确认复评：Codex（gpt-6.1-sol）给出 8.6 / 8.7 并否决，
确认 R4-1 / R4-2 两项 P2 均已真正解决，但指出本轮修复引入一项新 P2：
`_store_cache` 先写 `_cache` 再对完整 payload 做严格 JSON 摘要；`Envelope` 与字典入口
都接受 `NaN`、`bytes` 等值，此时 `digest` 抛 `ValueError` / `TypeError`，
SAFE_STOP 也随之中断，并留下「cache 有 1 项、identities 0 项」的部分状态。

处置：
- 新增 `_stable()` 递归归一化（Mapping / Sequence / set / 非有限 float / bytes / 任意对象
  均映射为确定性 JSON-safe 值），`_request_identity` 变为全函数，不再抛异常；
- `_store_cache` 先算身份再写 cache 与 identities，摘要失败时不产生部分状态；
- SAFE_STOP 在缓存查找之前求值，且返回路径不再依赖可失败的摘要。

验证：NaN / inf / bytes / set 四类 payload × 评分请求 / SAFE_STOP 共 8 个用例全部
fail-closed（评分请求 `NOT_AVAILABLE` 且 `prediction=None`，SAFE_STOP 返回 `SAFE_STOP`），
cache 与 identities 长度一致；另补 1 项原子性测试（monkeypatch 令身份抛错后两者均为空）
与 1 项「归一化后仍能识别模式/会话冲突」测试。2 项突变（撤回 `_stable`、把 cache 写回
身份之前）全部命中；连同第三轮 15 项与第四轮 3 项，累计 20 项突变检查全部命中。
累计 18 项 finding（15 + 2 + 1）全部落盘、46 项新增回归。

本地 `1038 passed, 9 skipped`、`tests/lifecycle` `205 passed`；
远程 `acam-lab` `1050 passed, 4 skipped`、`tests/lifecycle` `205 passed`；
两端 `ruff` 零告警、`lifecycle_verify.py` `15/15 PASS`。10.5 的基线表按最终 HEAD 更新，
10.4 保留为第二轮历史快照。证据级别与三门禁不变。
