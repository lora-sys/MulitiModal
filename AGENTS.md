# MulitiModal：工程规范与项目事实手册

本文档供所有参与本仓库研发与实验的 Agent 查阅与执行。改动前须通读相关任务文件；若具体任务最新冻结契约与本文档产生冲突，以任务最新冻结契约或交付模块为准，严禁私自改写研究目标与评估口径。

---

## 1. 核心研发与工程规范

1. **奥卡姆剃刀与极简原则**：
   - 实现保持简单、职责清晰；优先复用现有依赖与已验证的轻量级方案。
   - 已废弃的过时路径应彻底清理，不保留无意义的过渡兼容层与废弃注释。
2. **AutoResearch 双轨单文件约束**：
   - 证候任务线（TCM-SD）：唯一可修改文件为 `src/research/train.py`。
   - 体质任务线（Constitution）：唯一可修改文件为 `src/research/constitution_train.py`。
   - 外部冻结评价器（`prepare.py` / `constitution.py`）、输入数据、标签字典、交叉验证划分、时间预算与随机种子一律冻结，严禁在实验轮次中改动。
3. **闭环决策与记录契约**：
   - 每一轮必须基于上一轮证据提出单一可验证假设；
   - 严格依据外部评价器输出裁决：**KEEP**（指标严格提升且实现紧凑）、**REVERT**（未提升则强制回滚至 Champion）、**CRASH**（异常报错或超时，限修 2 次）；
   - 所有决策原子化追加写入各任务的 `runs/<tag>/results.tsv`。
4. **代码提交规范**：
   - 遵循 Conventional Commits 规范，一个 commit 对应一个独立可验证的目标；
   - Schema 改动必带契约测试，Adapter 改动必带 shape/mask/负例测试，Bug 修复必带回归测试。
   - 运行环境使用本机 Python 与可用 CPU/GPU，不强求引入 Docker、多 Agent 集群或分布式编排。

---

## 2. 数据与科学证据边界红线

1. **受试者级物理隔离与防泄露**：
   - 同一受试者的所有时序或多模态记录必须严格归属于同一折（Partition），严禁跨折拆分；
   - 特征标准化（均值、标准差、中位数）、缺失值插补、特征选择与权重拟合仅限在当前训练折内部拟合，严禁接触测试或验证样本；
   - 历史 holdout 测试集解锁预算（体质线 2/2 次、证候线 1/1 次）已彻底耗尽，模型探索与候选筛选绝对禁止触碰历史 holdout。
2. **真实性与不猜测原则**：
   - 体质线第 9 维平和质在源数据中完全未观测（0/274），模型输出必须为 `(n, 8)`，推理端统一返回 `ABSTAIN`，严禁补零、插值或伪造；
   - 标签体系（RULE、PUBLISHED、LLM_SILVER）必须保留原始血统来源且不可互换；LLM_SILVER 不作为参考真值；规则缺项一律 `ABSTAIN`。
3. **证据分级声明**：
   - 开发机本地实验结果一律标识为 `LOCAL_EXPERIMENTAL`，隔离级别记为 `AGENT_ISOLATION_UNVERIFIED`；
   - `GO_ENGINEERING`、`GO_REAL_DATA_TRAINING`、`GO_RESEARCH_CLAIMS` 三级门禁独立判定，任何本地 dev/CV 指标均不得外推为临床有效性定论；
   - 原始数据集只读，导入前后核对 sha256；真实数据、二进制模型工件、日志、`runs/` 及 `artifacts/` 严禁提交至 Git 仓库。
   - 合成数据必须显式标注 `SMOKE_ONLY`。

---

## 3. 三大研究主线事实与交付状态

### 1. 证候分类研究线 (Syndrome Track)
- **数据规格**：TCM-SD 数据集，54,152 例中文症状主诉，148 类证候分类，数据许可为 CC BY-NC-SA 4.0（非商用）。
- **基线与交付**：protocol-v2 下字符 1~4 gram TF-IDF + LogisticRegression 的 148 类 `macro-F1` 约为 0.3199，优于同等条件下的复杂神经方案，作为交付候选。
- **交付工件与接口**：
  - 持久化工件：`artifacts/sd-v2-ar1-champion/`（模型权重、标签定义、阈值绑定与 sha256 清单）；
  - 交付验收入口：[src/syndrome/delivery.py](src/syndrome/delivery.py)；
  - 只读价值复核：[src/syndrome/value_review.py](src/syndrome/value_review.py)（Top-k、置信度阈值、长尾逐类 F1 与置信区间审计）；
  - 服务产品契约：[src/syndrome/service.py](src/syndrome/service.py) 提供双模式（症状分析与实时非诊断知识）安全产品契约，支持 fail-closed 拒判保护与工件哈希校验。

### 2. 体质多维回归研究线 (Constitution Track)
- **数据规格**：Mendeley `Share data.xls`，274 名真实受试者，8 个已观测维度连续评分（依据 ZYYXH/T157-2009 标准，0~100 分）。
- **已交付 Champion**：第一轮 18 轮探索已顺利冻结并交付（Commit `16c26582`，Huber 线性与 kNN 局部中位数的逐维自适应软集成，选用 9 维基础特征），折外 `masked MAE` 达到 **14.1437**（复核实测 14.1444，较均值基线降低 0.3530，95% CI `[-0.6087, -0.0870]`）。
- **9 维基础特征**：性别 (gender)、年龄 (age)、文化程度 (education)、职业类别 (occupation)、慢性病史 (chronic dz)、手术史 (surgery hx)、身高 (BH)、体重 (BW)、BMI；均为基础客观指标，无需量表问卷。
- **交付工件与接口**：
  - 持久化工件：`artifacts/constitution-v1-champion/`（模型权重、标签定义、带 sha256 清单的 `registry_entry.json`）；
  - 交付验收入口：[src/research/constitution_delivery.py](src/research/constitution_delivery.py)（`deliver` 训练并固化工件，`verify` 快速只读审计）；
  - 只读价值复核：[src/research/constitution_value_review.py](src/research/constitution_value_review.py)（训练折均值基线对比、配对差值、逐维显著性与 Bootstrap CI 审计）。

### 3. 生理状态与实时知识研究线 (Physiology Track)
- **产品主线（#95–#100）**：按摩过程中接收带时间戳的生理事件，结合静态基线与截至当前时刻的滑窗估计**当前疲劳相关状态**，并在安全约束内提供可解释的程序候选；中医设备结果只做有出处的独立辅助映射。
- **统一契约**：设备直连、厂商服务端与离线研究导入共用 `PhysioEventV1` + `PhysioSessionV1`（[src/physio/schema.py](src/physio/schema.py)）；CSV 仅是离线研究入口，不得成为实时推理依赖。实时推理入口见 [src/physio/inference.py](src/physio/inference.py)，设计与模块边界见 [docs/design/physio-realtime-v1.md](docs/design/physio-realtime-v1.md)。
- **离线导入与数据纪律**：`legacy_export`（[src/physio/legacy_export.py](src/physio/legacy_export.py)）冻结 94 字段映射并对 identity、快照/完成态、Pre/Post 冒充、重复/乱序、时区/跨日、EEG/duration 冲突和补值逐行记录 QC；`dataset`（[src/physio/dataset.py](src/physio/dataset.py)）分离 observed/valid/imputed mask，插补仅在训练折内拟合，受试者级划分拒绝共用账号泛化声明，`y_pri` 恒不可用。
- **训练、测评与 benchmark**：`training` 为 torch-optional masked-loss smoke 训练，无有效标签或无可信身份时拒绝启动；`evaluation` 在样本或受试者不足时固定返回 `INSUFFICIENT_EVIDENCE`；`benchmark` 只报告实测延迟、吞吐与峰值内存，不设未经验证的医学或产品阈值。
- **门禁与安全边界**：在真实配对标签、独立验证和厂家控制 API 到位前，疲劳预测固定 `NOT_AVAILABLE`，推荐只给经审查的静态候选，设备控制固定 `NOT_CONNECTED`；不得用合成模型分数冒充实测，也不得按未经验证的“疲劳下降”排序或宣称某程序最优。PR #101（`95788fb`）与 PR #103（`e76e2b3`，2026-11-10 11:36 UTC）均已合并，Issue #102 已关闭；历史注释中称“#103 仍 open / 待审”已属陈旧事实。真实 94 字段厂商字典未提供，5 场数据为 1 `member_id_hash` 映射 5 个 `subject_key`，物理受试者隔离未经验证。
- **防越界护栏**：实时知识模式绝不从生理数值推断中医证候或体质，严格切断生理数值与疾病/证候的伪造关联；当前未配对真值数据状态为 `BLOCKED`。历史 Pre-only `PhysioBPSIEFRIModel` 研究继续归档，不定义产品主线，也不修改其冻结实验语义。

### 4. ML 生命周期工程层（#104–#112）
- **唯一设计索引**：[docs/design/ml-lifecycle.md](docs/design/ml-lifecycle.md)（Single Source of Truth：7 阶段能力矩阵、Go/No-Go 表、可复现命令；#104–#112），现状审计见 [docs/reports/ml-lifecycle-audit.md](docs/reports/ml-lifecycle-audit.md)，端到端验收见 [docs/reports/ml-lifecycle-acceptance.md](docs/reports/ml-lifecycle-acceptance.md)。
- **薄编排层（9 大模块）**：[src/lifecycle/](src/lifecycle/) 包含 9 个模块（`contracts`, `pipeline`, `sources`, `features`, `training`, `release`, `gateway`, `observe`, `acceptance`），提供不可变契约、增量幂等数据管道、源能力管理、特征/目标注册、训练前工程准备、工件原子提升与回滚、薄推理网关、监控哈希链及统一验收。严格作为顶层薄编排，**绝不改写** `src/physio/`、`src/core/`、`src/runtime/` 的既有研究线或冻结评估器 `prepare.py` / `constitution.py`。
- **三门禁独立判定**：由 `contracts.gate_bundle` 强制隔离，低阶通过绝不自动提权高阶：
  - `GO_ENGINEERING`：**GRANTED**（合成 CI 绿灯、本地 15 项验收与 10 项安全负例全部拦截通过、全仓 ruff 零告警）；
  - `GO_REAL_DATA_TRAINING`：**NOT_GRANTED**（真实数据未提供物理隔离受试者与客观生理真值，绿色 CI 绝不自动提权）；
  - `GO_RESEARCH_CLAIMS`：**NOT_GRANTED**（历史 holdout 预算已耗尽，未获外部多中心临床独立验证）。
- **实测基线与验收数据**：
  - 全仓 `pytest -q` 实测 `1026 passed, 9 skipped`；
  - 全仓 `ruff check` 零告警；
  - 纯 stdlib 契约快测 `pytest tests/lifecycle/test_contracts_fast.py -m fast -q` 实测 `19 passed`；
  - 合成生命周期套件 `pytest tests/lifecycle -q` 实测 `193 passed`；
  - 统一验收脚本 `python scripts/lifecycle_verify.py` 执行 15/15 项验收（含 10 项安全负例全部拦截，实测 `OVERALL: PASS`）。
- **CI 与环境适配**：[.github/workflows/ci.yml](.github/workflows/ci.yml) 设 4 个 jobs（fast contracts 覆盖 Py3.11/3.12/3.13，synthetic E2E 覆盖 Py3.11/3.13，frozen-lines 冻结研究线回归、no-torch 诚实 skip 验证）。因 `pyproject.toml` 未声明 `[build-system]`，CI 显式声明 `PYTHONPATH: "src:."`；`core/__init__.py` 采用 PEP 562 惰性解析 Torch 相关符号，确保无 Torch 环境下导入纯标准库 `core.specs` 与 `lifecycle.contracts` 畅通无阻；`python scripts/lifecycle_verify.py --torch-free-smoke` 验证无 Torch 纯契约可导入性。无私有数据、无厂商网络服务、无 GPU。
- **数据纪律与持续迭代**：raw 只读、派生可重建；缓存键 = 源 hash + 代码版本 + 配置 + 规则版本；被拒行必须带理由进隔离区，禁止 silent skip；新数据只自动刷新导入/QC/特征，绝不自动触发训练、模型提升或部署发布。
- **历史工单迁移结论**：历史工单 #22–#43、#46、#36 均已统一关闭（stateReason 为 `NOT_PLANNED` / `superseded`），代表向 #104–#112 体系迁移与收敛，**绝非声称其历史功能已逐项实现**。中医独立工单 #52/#54/#90/#91 继续保持 open。
- **已知未决遗留项（诚实记录）**：
  1. `requirements-dev.txt` 采用 `>=` 下界；已提供伴随式 `requirements-lock.txt` 作本机实测复现快照，但 CI 因跨 Py3.11/3.12/3.13 矩阵不强制启用，跨版本漂移仍可见于 job 的 `pip freeze` 输出；
  2. 真实按摩椅设备恒定为 `NOT_CONNECTED`，厂商控制协议未接入；
  3. 94 字段厂商字典与原始数据格式受阻，标为 `BLOCKED_BY_DATA_OR_DEVICE`；
  4. 真实物理受试者身份隔离未获验证（共用测试账号导致 1 member_hash 映射 5 个 subject_key）。

### 5. 辅助验收与演示工具
- **双轨最小演示入口**：[scripts/tcm_minimal_demo.py](scripts/tcm_minimal_demo.py)（加载真实工件与冻结特征，体验端到端推理、异常拒判与生理防越界护栏）；
- **实际输入场景鲁棒性套件**：[scripts/tcm_input_scenarios.py](scripts/tcm_input_scenarios.py)（覆盖日常口语表达、特征全缺/部分缺、极端合理值与矛盾症状 4 类场景；仅做鲁棒性检查，显式标为 `SMOKE_ONLY`，严禁产出准确率）；
- **实时生理模拟入口**：[scripts/physio_realtime_demo.py](scripts/physio_realtime_demo.py)（按事件时间逐步回放设备流、断线恢复、窗口推理门禁、静态候选与中医侧车；`SMOKE_ONLY`，未对真实设备验证）；
- **生理 benchmark 与统一验证入口**：[scripts/physio_benchmark.py](scripts/physio_benchmark.py) 输出实测导入/窗口/推理延迟与峰值内存（不设未经验证阈值），[scripts/physio_verify.py](scripts/physio_verify.py) 一次运行 acceptance + pytest + scoped ruff。
