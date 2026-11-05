---
name: autoresearch
description: Run, resume, or monitor consecutive model research experiments using local CPU/GPU, a frozen evaluation protocol, Git KEEP/REVERT/CRASH mechanics, and a read-only dashboard. Supports dual independent research tracks (TCM-SD 148-class syndrome classification and Mendeley 8-dimension constitution score regression). Use when asked to train and improve models, run AutoResearch loops, inspect experiment history, or launch the monitoring dashboard.
---

# AutoResearch 技能规范

本 Skill 由当前 Coding Agent 单主干执行，不启动额外的 Agent 编排服务，亦不引入常驻外部队列。

## 1. 核心定位与原则

AutoResearch 旨在实现**以证据为中心、假设驱动的自主模型科研循环**：
- **科学边界**：研究目标、冻结数据集、评价指标与晋级门槛由研究者在研究简报中预先固化；Agent 负责在受约束范围内提出科学假设并自主推进多轮实验。
- **轻量独立**：依托本机 Python 与可用 CPU/GPU 算力运行，不依赖 Docker、复杂 Sandbox 或分布式中间件。

## 2. 双研究线独立映射

本项目包含两条互相平行的 AutoResearch 任务线，数据、代码与评价协议严格物理隔离：

| 维度 | 证候研究线 (Syndrome Track) | 体质研究线 (Constitution Track) |
| :--- | :--- | :--- |
| **任务目标** | 148 类中文短主诉证候多分类 | 8 维体质量表评分连续回归（0~100 分） |
| **数据源** | TCM-SD（54,152 例病历，CC BY-NC-SA 4.0） | Mendeley `Share data.xls`（274 名受试者） |
| **主评价指标** | `macro-F1`（越大越好，方向 `maximize`） | `constitution/masked_mae`（越小越好，方向 `minimize`） |
| **唯一可编辑候选** | `src/research/train.py` | `src/research/constitution_train.py` |
| **冻结评价入口** | `python -m research.prepare` | `python -m research.constitution` |
| **详细程序契约** | [program.md](program.md) | [program-constitution.md](program-constitution.md) |
| **交付成果规范** | `src/syndrome/delivery.py` | `src/research/constitution_delivery.py` |

> [!IMPORTANT]
> 两条研究线的数据集、标签映射、受试者划分、指标计算与候选源码严禁混用或交叉导入。

## 3. 铁律规范与防泄露红线

1. **单文件修改约束**：每轮实验仅允许修改对应任务指定的单个候选文件。任何改动评价器、特征提取器接口、划分清单、随机种子、评估预算或参考标签的行为均属违规。
2. **严格禁止数据泄露**：预处理统计量、特征归一化、插补及权重拟合必须严格限制在当前训练折内；严禁使用任何已耗尽预算的历史 holdout 测试集。
3. **真实性与不猜测原则**：
   - 体质研究线中，第 9 维平和质在源数据中完全未观测（0/274），任何候选输出必须为 `(n, 8)`，第 9 维在推理端统一返回 `ABSTAIN`，绝不补零、伪造或猜测。
4. **独立分支隔离**：所有实验迭代必须在独立的 `autoresearch/<tag>` 分支或独立 worktree 上进行，严禁将未经验收的中间试验提交至主分支。

## 4. 实验循环标准流程

每次科研执行严格按以下步进推进，直至用尽设定轮次/时间预算或达到终止条件：

```
1. 初始基线复现 ──> 2. 提出单一可检验假设 ──> 3. 编辑单候选文件并 Git Commit
                             │
                             ▼
5. 结果决策与记录 <── 4. 运行冻结评价协议（进程外独立打分）
   ├── KEEP: 指标严格改善，刷新 Champion
   ├── REVERT: 未改善或复杂度不划算，回滚至 Champion
   └── CRASH: 运行时报错或超时，限修 2 次，否则回退
                             │
                             ▼
                  6. 基于证据推进下一轮迭代
```

1. **首轮基线验证**：
   - 证候线：复现字符 1~4 gram TF-IDF + LogisticRegression 基线（macro-F1 约 0.3199）。
   - 体质线：复现 8 维训练折中位数回归基线（masked MAE 约 15.65）。
2. **假设提出**：分析上一轮成败与折外误差分布，提出单一明确的归纳偏置或特征假设（拒绝盲目暴力调参）。
3. **代码变更与提交**：仅编辑指定候选文件并创建 Conventional Commit。
4. **统一协议评估**：通过冻结命令调用外部评价器（子进程隔离执行，外部计算指标），不采纳候选代码自报的分数。
5. **决策判定**：
   - **KEEP**：主指标严格优于当前 Champion，且代码复杂度提升具备合理性。
   - **REVERT**：指标未改善或提升不抵复杂度，强制 `git reset` 回滚至 Champion。
   - **CRASH**：超时或执行异常，查看日志后最多修复 2 次，仍失败则归零记录并回滚。
6. **落盘决策**：调用对应模块的 `record` 子命令，将 Git Commit、分值、耗时、假设描述及决策原子写入 `runs/<tag>/results.tsv`。

## 5. 监控与交付

- **只读监控**：通过 `./solve.sh dashboard --run runs/<tag>` 启动原生轻量级可视化看板，实时查阅轮次趋势与逐折表现。网页为无状态只读层，绝不反向操控训练进程。
- **阶段交付**：实验结束时统一交付：
  1. 最佳 Champion Commit 标识及权重；
  2. 完整的 `results.tsv` 实验沿革与消融分析；
  3. 经过双向哈希校验与单元测试的交付工件（如 `artifacts/constitution-v1-champion/`）；
  4. 监控看板导出状态与详尽模型卡。

## 6. 证据状态与合规声明

- 本机执行的所有实验结果一律标注为 `LOCAL_EXPERIMENTAL`，其评价隔离状态为 `AGENT_ISOLATION_UNVERIFIED`。
- 本项目严禁将开发机本地 CV/dev 分数外推为外部临床有效性，亦不构成任何临床诊断依据。
