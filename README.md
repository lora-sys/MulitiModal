# MulitiModal

面向按摩椅与日常身心健康场景的中医多模态参考研究系统。本项目探索**证候分类**、**体质多维评分**与**按摩前生理状态分析**三条互相平行的科学研究路线。

> [!IMPORTANT]
> **合规与伦理声明**：本项目的全部输出与算法模型均为科研探索成果，仅作为健康参考，**不构成医学临床诊断依据，亦不宣称任何临床疗效**。项目数据许可、工程事实与安全硬规则参见 [AGENTS.md](AGENTS.md)。

---

## 核心机制：AutoResearch 自主科研框架

本项目内嵌了受 [Karpathy AutoResearch](https://github.com/karpathy/autoresearch) 与 [Tencent Hyra](https://hunyuan.tencent.com/research/hyra2) 启发的极简科研实验闭环（驱动规范详见 [AutoResearch Skill](skills/autoresearch/SKILL.md)）。其核心设计原则是：**将科学边界的定义权留给研究者，将受约束的实验探索权交付给 Coding Agent，使每轮研究都能沉淀可复用的确定性证据**。

```
                    ┌────────────────────────┐
                    │  研究者预设固化简报    │
                    │ (数据、指标、基线门槛) │
                    └───────────┬────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │                 初始基线复现                    │
        └───────────────────────┬─────────────────────────┘
                                │
                                ▼
  ┌─────────────────> 提出单一可检验假设
  │                             │
  │                             ▼
  │                   仅修改指定候选源码文件
  │                             │
  │                             ▼
  │                   提交 Git Commit 隔离记录
  │                             │
  │                             ▼
  │                 进程外外部固化评价器独立打分
  │                             │
  │                             ▼
  │                   KEEP / REVERT / CRASH 决策
  │                             │
  │                             ▼
  └─── 下一轮假设 <───  记录证据至 results.tsv
```

- **单文件突变**：每条研究线仅开放唯一的候选文件供 Agent 编辑，评价器代码与运行环境严格冻结。
- **外部独立评测**：候选代码运行在隔离子进程中，仅能通过输入文件获取训练数据与待测样本，指标由外部评测器统一计算，杜绝自评与数据泄露。
- **Git 驱动的胜负机制**：指标严格提升且架构精简合理才保留（KEEP），退化或未改善则强制回滚至当前 Champion（REVERT），异常超时则限修后回滚（CRASH）。
- **无状态只读监控**：通过纯原生轻量看板可视化呈现迭代历史，页面仅作数据消费，不反向控制训练逻辑。

---

## 两大自主研究任务线

项目现已支持两条完全平行的 AutoResearch 研究主线，各项任务的输入数据、标签体系与评价标准完全物理隔离：

### 1. 证候研究线 (TCM-SD Syndrome Track)
- **任务目标**：根据患者的中文短主诉文本，预测 148 类中医证候类别。
- **数据与许可**：TCM-SD 数据集（54,152 例病历），采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可，严格限于非商业科研使用。
- **评价协议与基线**：protocol-v2，受试者分组，折内单向文本去重，4 折 × 2 次重复验证。基线为字符 1~4 gram TF-IDF + LogisticRegression（148 类 `macro-F1` 约为 0.3199）。
- **实测预测价值与能力边界**（详见 [2026-11-06 复核报告](docs/reports/2026-11-06-inference-value-review.md)）：
  - 验证集 Top-1 / Top-3 / Top-5 命中率分别达 **61.3% / 84.3% / 90.2%**，`macro-F1` 为 0.3202（95% bootstrap CI `[0.2702, 0.3251]`）；
  - 存在显著长尾稀疏性（148 类中 55 类在 dev 集中 F1 为 0）；
  - 内置置信度阈值机制，决策覆盖率约 90.1%，拒判率约 9.9%，对抗异常输入（乱码、无关问答）100% 触发拒判防护。
- **可编辑候选**：`src/research/train.py`
- **交付工件与入口**：`artifacts/sd-v2-ar1-champion/`（由 [src/syndrome/delivery.py](src/syndrome/delivery.py) 训练并验收）。
- **详细程序契约**：[skills/autoresearch/program.md](skills/autoresearch/program.md)
- **任务背景说明**：[docs/autoresearch-tcm-sd.md](docs/autoresearch-tcm-sd.md)

### 2. 体质研究线 (Constitution Regression Track)
- **任务目标**：基于人口学与基础人体测量特征，预测受试者的 8 维中医偏颇体质连续得分（0~100 分，依据 ZYYXH/T157-2009 行业标准）。
- **数据源与特征**：Mendeley 数据集（274 名真实受试者）。模型选用 **9 维基础特征**：性别 (gender)、年龄 (age)、文化程度 (education)、职业类别 (occupation)、慢性病史 (chronic dz)、手术史 (surgery hx)、身高 (BH)、体重 (BW)、BMI；均为非侵入客观指标，无需填写问卷量表。
- **平和质安全拒判**：第 9 维平和质在源数据中完全未观测（0/274），模型输出形状锁定为 `(n, 8)`，在推理端恒定返回 `ABSTAIN`，严禁补零、猜测或伪造。
- **评价指标**：`constitution/masked_mae`（8 维平均绝对误差，越低越好）。
- **首轮已交付成果与实测价值**（详见 [2026-11-06 复核报告](docs/reports/2026-11-06-inference-value-review.md)）：
  - 产出并冻结冠军模型 `16c26582`（Huber 鲁棒线性与 kNN 局部中位数的逐维自适应软集成）；
  - 折外 masked MAE 为 **14.1444**，相较训练折均值基线（14.4974）降低 0.3530（95% bootstrap CI `[-0.6087, -0.0870]`），8 折全部正向改善；
  - 局限性：相对改善幅度较小（约 2.4%），逐维仅 4/8 维（阴虚、痰湿、气郁、特禀）置信区间严格排除 0，属于开发集内部折外探索性证据，不可外推至临床。
- **可编辑候选**：`src/research/constitution_train.py`
- **交付工件与入口**：`artifacts/constitution-v1-champion/`（由 [src/research/constitution_delivery.py](src/research/constitution_delivery.py) 交付验收）。
- **详细程序契约**：[skills/autoresearch/program-constitution.md](skills/autoresearch/program-constitution.md)

### 3. 实时生理疲劳与安全推荐（产品工程主线，`NOT_VALIDATED`）
- **产品目标**：按摩椅运行期间接收带时间戳的生理事件，结合静态基线与历史窗口估计**当前疲劳相关状态**，并在安全约束内给出可解释的程序候选；中医健康信息作为独立辅助能力。
- **当前能力边界**：只有版本化协议、适配器、滑窗特征、门禁、推荐状态机与 SMOKE_ONLY 模拟演示；**没有真实配对标签、没有独立验证、没有真实设备协议**。推理固定返回 `NOT_AVAILABLE`，设备控制固定 `NOT_CONNECTED`，不得展示虚构疲劳分数或自动下发强度指令。
- **统一契约**：设备直连 / 厂商云 / 离线研究导入收敛到 `PhysioEventV1` + `PhysioSessionV1`；CSV 仅是离线研究入口，不是实时推理依赖。SpO2 单位、零值语义、缺失/乱序/过期/断线均有显式处理。
- **离线导入与数据纪律**：`legacy_export` 冻结 94 字段导出映射并按行记录 QC 问题（未知字段、快照/完成态冲突、同一记录冒充 Pre/Post、重复/乱序、时区/跨日、EEG 与 duration 冲突、补值标记）；`dataset` 分离 observed/valid/imputed 三种 mask，插补只在训练折内拟合，受试者级划分拒绝共用账号泛化，`y_pri` 恒不可用。
- **训练与测评**：`training` 为 torch-optional 的 masked-loss smoke 训练（无有效标签、无可信身份时拒绝启动）；`artifacts` 用特征/目标版本、scaler sha256 与模型 state sha256 绑定工件，缺失或篡改一律 `NOT_AVAILABLE`；`evaluation` 在样本或受试者不足时固定返回 `INSUFFICIENT_EVIDENCE`。本地无 torch 时训练测试 skip，远端训练机实际执行。
- **工程入口**：[src/physio/](src/physio/)（`schema`、`metrics`、`adapters`、`legacy_export`、`windows`、`preprocessing`、`artifacts`、`inference`、`dataset`、`training`、`evaluation`、`response`、`recommendation`、`tcm_exam`、`benchmark`）；设计契约见 [docs/design/physio-realtime-v1.md](docs/design/physio-realtime-v1.md)。
- **模拟演示与验证**：`PYTHONPATH=src python -m scripts.physio_realtime_demo --pretty`（按事件时间逐步回放，`SMOKE_ONLY`）；`PYTHONPATH=src python scripts/physio_benchmark.py --pretty` 输出实测延迟/内存；`python scripts/physio_verify.py` 一次运行 acceptance + pytest + scoped ruff。
- **硬阻塞与报告**：PR #101（`95788fb`）与 PR #103（`e76e2b3`，2026-11-10 11:36 UTC）均已合并，Issue #102 已关闭（历史注释中“#103 仍 open / 待审”已属陈旧事实）；真实 94 字段导出、5 场共用账号会话（1 `member_id_hash` 映射 5 个 `subject_key`，物理受试者隔离未经验证）、独立疲劳标签和厂商控制协议均未提供，相关项标 `BLOCKED_BY_DATA_OR_DEVICE`，真实设备恒定 `NOT_CONNECTED`；能力清单见 [docs/reports/issue-95-gap-analysis.md](docs/reports/issue-95-gap-analysis.md)，验收证据见 [docs/reports/issue-95-acceptance.md](docs/reports/issue-95-acceptance.md)。
- **生命周期工程编排**：[src/lifecycle/](src/lifecycle/) 为其上层的薄工程编排层（契约、管道、源能力、特征、发布、网关、监控与验收），不改写生理研究线与冻结评估器；详见唯一设计索引 [docs/design/ml-lifecycle.md](docs/design/ml-lifecycle.md)。
- **历史研究线**：`PhysioBPSIEFRIModel` 的 Pre-only EFRI/BPSI 研究继续归档，不定义新的产品主线，也不修改其冻结实验语义。

---

## 快速上手与操作指南

### 1. 运行测试套件
```bash
pytest -q
```
> [!NOTE]
> 仓库采用分层依赖设计，若本地未安装 `torch` / `transformers`，相关神经栈测试会自动安全跳过（全仓实测基线 `1038 passed, 9 skipped`；无 torch 时神经栈安全 skip），经典机器学习交付线完全不受影响。

### 2. 运行双轨最小演示（推荐体验入口）
加载真实冻结工件与受试者特征，体验全链路端到端预测与安全拒判行为：
```bash
# 双轨综合演示（体质预测 + 默认主诉证候分析）
python -m scripts.tcm_minimal_demo

# 自定义主诉文本输入
python -m scripts.tcm_minimal_demo --symptom "反复胸闷隐痛3年余，加重伴憋喘1周"

# 仅生理测量输入（触发防越界安全护栏，严格拒判，不伪造证候/体质关联）
python -m scripts.tcm_minimal_demo --physiology-only
```

### 3. 运行双模式服务演示
体验症状分析（SYMPTOM_ANALYSIS）与实时非诊断性健康知识（REALTIME_KNOWLEDGE）双模式接口：
```bash
python -m scripts.syndrome_service_demo
```

### 4. 运行预测价值复核与实际输入场景测试
对模型进行完全只读的价值审计（不训练、不读 holdout）与极端/日常输入鲁棒性测试：
```bash
# 体质模型与基线对比、Bootstrap 置信区间审计
python -m research.constitution_value_review

# 证候模型 Top-k、长尾逐类 F1 与置信度阈值审计
python -m syndrome.value_review

# 日常表达、缺值、极端合理值与矛盾症状输入鲁棒性评测（SMOKE_ONLY）
python -m scripts.tcm_input_scenarios
```

### 5. 冠军模型验证与交付验收
一键完成冠军模型的拟合、工件打包与哈希完整性校验：
```bash
# 体质线冠军交付验收
python -m research.constitution_delivery deliver   # 拟合并生成持久化工件
python -m research.constitution_delivery verify    # 只读模式校验哈希与边界

# 证候线冠军交付验收
python -m syndrome.delivery \
    --repo . --raw-dir data/local/tcm-sd \
    --manifest data/local/tcm-sd/protocol-v2/short/selection_manifest.json \
    --dataset data/local/tcm-sd/protocol-v2/short/selection_records.json \
    --results runs/ar1/results.tsv \
    --artifact-dir artifacts/sd-v2-ar1-champion \
    --report-dir runs/delivery/ar1
```

### 6. 启动只读监控仪表盘
可视化查看任意 AutoResearch 研究线迭代过程与折外表现：
```bash
# 监控体质线实验
./solve.sh dashboard --run runs/constitution-v1

# 监控证候线实验
./solve.sh dashboard --run runs/<tag>
```

---

## ML 生命周期工程入口

```bash
python scripts/lifecycle_verify.py                      # 统一验收：fast 契约 + 合成生命周期套件 + 全仓 lint + 15/15 验收（含 10 项安全负例，实测 PASS）
pytest tests/lifecycle/test_contracts_fast.py -m fast -q # 纯 stdlib 契约快测（CI 第一道门禁，实测 19 passed）
pytest tests/lifecycle -q                               # 合成生命周期套件（实测 205 passed）
python scripts/lifecycle_verify.py --torch-free-smoke   # 证明无 torch 时纯契约可导入
```

- **薄编排层（9 大模块）**：[src/lifecycle/](src/lifecycle/) 由 9 个模块构成（`contracts`、`pipeline`、`sources`、`features`、`training`、`release`、`gateway`、`observe`、`acceptance`），`src/physio/`、`src/core/`、`src/runtime/` 的研究线与冻结评估器 `prepare.py` / `constitution.py` **不被改写**。
- **三门禁独立判定**：`GO_ENGINEERING` = GRANTED（合成 CI 与本地 10 项负例全通）；`GO_REAL_DATA_TRAINING` = NOT_GRANTED（无物理隔离受试者与客观真值，绿色 CI 绝不提权）；`GO_RESEARCH_CLAIMS` = NOT_GRANTED（无外部多中心临床验证）。
- **CI 门禁路径**：见 [.github/workflows/ci.yml](.github/workflows/ci.yml)（4 个 jobs：fast 契约覆盖 Py3.11/3.12/3.13，synthetic E2E 覆盖 Py3.11/3.13，frozen-lines 冻结研究线回归、no-torch 诚实 skip；因 pyproject 无 `[build-system]`，CI 声明 `PYTHONPATH: "src:."`；无私有数据、无厂商网络服务、无 GPU）。
- **完整规格与审计**：详见唯一索引 [docs/design/ml-lifecycle.md](docs/design/ml-lifecycle.md)，现状审计见 [docs/reports/ml-lifecycle-audit.md](docs/reports/ml-lifecycle-audit.md)，端到端验收见 [docs/reports/ml-lifecycle-acceptance.md](docs/reports/ml-lifecycle-acceptance.md)。

---

## 科学证据边界与硬性红线

1. **证据分级明确**：开发机本地跑出的所有指标均标注为 `LOCAL_EXPERIMENTAL`，隔离级别标识为 `AGENT_ISOLATION_UNVERIFIED`。未经独立可信评价服务（见 [docs/trusted-evaluator.md](docs/trusted-evaluator.md)）验签与异源数据复现的结果，不得作为正式学术定论。
2. **测试集耗尽封存**：证候线与体质线的历史 holdout 测试集解锁预算已全部用尽，任何探索循环与模型筛选严禁触碰历史 holdout。
3. **数据泄露零容忍**：所有特征标准化统计量（均值、方差、中位数）与模型参数必须严格在各交叉验证折内部拟合，严禁跨折泄露。
4. **真实与合成隔离**：合成数据一律显式打上 `SMOKE_ONLY` 标记，不可与真实数据混淆。

---

## 项目文档导航

- [AGENTS.md](AGENTS.md)：面向 Coding Agent 的全局工程规范、硬性红线与项目事实手册。
- [docs/reports/2026-11-06-inference-value-review.md](docs/reports/2026-11-06-inference-value-review.md)：**2026-11-06 最新复核报告**，详尽记录推理正确性、预测价值与实际输入场景鲁棒性评测结果。
- [docs/models/tcm-sd-syndrome-v1-card.md](docs/models/tcm-sd-syndrome-v1-card.md)：证候分类模型卡（Model Card）。
- [skills/autoresearch/SKILL.md](skills/autoresearch/SKILL.md)：AutoResearch 技能的标准定义与交互契约。
- [docs/autoresearch-brief.md](docs/autoresearch-brief.md)：新实验立项必须填写的科研简报模板。
- [docs/autoresearch-design.md](docs/autoresearch-design.md)：AutoResearch 架构设计理念与边界阐述。
- [docs/trusted-evaluator.md](docs/trusted-evaluator.md)：独立可信评价环境部署契约。
- **[docs/design/ml-lifecycle.md](docs/design/ml-lifecycle.md)：ML 生命周期唯一设计索引（Single Source of Truth）** —— 统领 7 大阶段能力矩阵、三独立门禁与可复现工程命令（#104–#112）；现状审计与真实缺陷清单见 [docs/reports/ml-lifecycle-audit.md](docs/reports/ml-lifecycle-audit.md)，端到端验收与门禁结论见 [docs/reports/ml-lifecycle-acceptance.md](docs/reports/ml-lifecycle-acceptance.md)。
- [docs/design/physio-realtime-v1.md](docs/design/physio-realtime-v1.md)：实时生理疲劳与安全推荐的统一事件/窗口/推荐契约（工程实现，`NOT_VALIDATED`）。
- [docs/design/physio-next-phase-plan.md](docs/design/physio-next-phase-plan.md)：按摩前生理预测模型下一阶段最小实施方案（`NOT_VALIDATED`，是设计不是结果）。
- [docs/design/autoresearch-rsi-v1.md](docs/design/autoresearch-rsi-v1.md)：AutoResearch 自主循环的历史设计，仅作沿革参考。
- [docs/reports/](docs/reports/)：历史科研实验报告归档。
