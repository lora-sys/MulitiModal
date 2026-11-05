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
- **可编辑候选**：`src/research/train.py`
- **详细程序契约**：[skills/autoresearch/program.md](skills/autoresearch/program.md)
- **任务背景说明**：[docs/autoresearch-tcm-sd.md](docs/autoresearch-tcm-sd.md)

### 2. 体质研究线 (Constitution Regression Track)
- **任务目标**：基于人口学特征与基础人体测量特征，预测受试者的 8 维中医偏颇体质连续得分（0~100 分，依据 ZYYXH/T157-2009 行业标准）。
- **数据源与特征**：Mendeley 数据集（274 名真实受试者，9 维基础特征）。
- **平和质安全拒判**：第 9 维平和质在源数据中完全未观测（0/274），模型输出形状锁定为 `(n, 8)`，在推理端恒定返回 `ABSTAIN`，严禁补零、猜测或伪造。
- **评价指标**：`constitution/masked_mae`（8 维平均绝对误差，越低越好）。
- **首轮已交付成果**：完成 18 轮自动迭代，产出并冻结冠军模型 `16c26582`（Huber 鲁棒线性与 kNN 局部中位数的逐维自适应软集成，折外 masked MAE 达到 **14.1437**）。
- **可编辑候选**：`src/research/constitution_train.py`
- **详细程序契约**：[skills/autoresearch/program-constitution.md](skills/autoresearch/program-constitution.md)

---

## 快速上手与操作指南

### 1. 启动只读监控仪表盘
查看任意研究线迭代过程与折外表现：
```bash
# 监控体质线实验
./solve.sh dashboard --run runs/constitution-v1

# 监控证候线实验
./solve.sh dashboard --run runs/<tag>
```

### 2. 体质冠军模型验证与交付验收
一键完成第一轮体质冠军模型的全量拟合、工件打包与自动化鲁棒性验收：
```bash
export PYTHONPATH=src
python -m research.constitution_delivery deliver   # 拟合并生成持久化工件与校验清单
python -m research.constitution_delivery verify    # 纯校验模式：验证哈希完整性与推理边界
```
生成的持久化工件位于 `artifacts/constitution-v1-champion/`，包含 `model.joblib`、`labels.json` 以及带 sha256 清单和完整 provenance 的 `registry_entry.json`。

### 3. 运行双模式服务演示
```bash
python -m scripts.syndrome_service_demo
```

---

## 科学证据边界与硬性红线

1. **证据分级明确**：开发机本地跑出的所有指标均标注为 `LOCAL_EXPERIMENTAL`，隔离级别标识为 `AGENT_ISOLATION_UNVERIFIED`。未经独立可信评价服务（见 [docs/trusted-evaluator.md](docs/trusted-evaluator.md)）验签与异源数据复现的结果，不得作为正式学术定论。
2. **测试集耗尽封存**：证候线与体质线的历史 holdout 测试集解锁预算已全部用尽，任何探索循环与模型筛选严禁触碰历史 holdout。
3. **数据泄露零容忍**：所有特征标准化统计量（均值、方差、中位数）与模型参数必须严格在各交叉验证折内部拟合，严禁跨折泄露。
4. **真实与合成隔离**：合成数据一律显式打上 `SMOKE_ONLY` 标记，不可与真实数据混淆。

---

## 项目文档导航

- [AGENTS.md](AGENTS.md)：面向 Coding Agent 的全局工程规范、硬性红线与项目事实手册。
- [skills/autoresearch/SKILL.md](skills/autoresearch/SKILL.md)：AutoResearch 技能的标准定义与交互契约。
- [docs/autoresearch-brief.md](docs/autoresearch-brief.md)：新实验立项必须填写的科研简报模板。
- [docs/autoresearch-design.md](docs/autoresearch-design.md)：AutoResearch 架构设计理念与边界阐述。
- [docs/trusted-evaluator.md](docs/trusted-evaluator.md)：独立可信评价环境部署契约。
- [docs/reports/](docs/reports/)：历史科研实验报告归档。
