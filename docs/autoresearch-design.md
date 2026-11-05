# AutoResearch 架构与设计规范

## 1. 核心设计理念

一次严谨的模型探索需要将**“候选假设的有效性”**与**“实验测量的可靠性”**在物理和逻辑层面解耦：
- **研究者职责**：在实验前固化研究问题、数据集划分、特征范围、评价指标与晋级门槛；
- **Coding Agent 职责**：在受限的边界内逐轮提出科学假设，实施局部代码改动，推动迭代；
- **固化外部评价器**：使用完全冻结的代码在独立进程外客观计算验证集指标，阻断任何形式的作弊与自评虚高；
- **确定性证据沉淀**：每轮迭代保留代码提交、假设阐述、指标变化、失利归因与逐折明细，使每一次尝试（无论 KEEP 还是 REVERT）都成为后续探索可查证的经验证据。

```
研究简报固化 ──> 基线复现 ──> 假设与单文件修改 ──> 隔离子进程运行 ──> 外部固化评价
                                ↑                                      │
                                │                                      ▼
                             下一项假设 ◄── 记录证据至 results.tsv ◄── KEEP / REVERT
```

---

## 2. 职责划分与安全隔离边界

| 参与角色 | 职责范围 | 操作权限 |
| :--- | :--- | :--- |
| **研究者 (Researcher)** | 制定并提供 [研究简报](autoresearch-brief.md)，固化数据源、评价指标、随机种子、时间预算与验收标准。 | 完全控制评价契约与基线标准。 |
| **Agent (Coding Engine)** | 在冻结框架下自主提出假设，修改唯一的指定候选文件，记录实验决策并推进迭代。 | 仅限修改对应的候选源码文件，严禁修改评价器或数据集。 |
| **外部评价器 (Evaluator)** | 在隔离子进程中装载数据并运行候选代码，独立计算评估指标，返回真实得分与资源耗时。 | 拥有数据标签与折划分的只读权限，候选进程不可见该评价逻辑。 |
| **监控看板 (Dashboard)** | 读取 `results.tsv`、`status.json` 与 `round-*.json`，以无依赖原生 SVG 渲染折线与逐折分布。 | 纯只读数据消费层，无写入与反向调度权。 |

> [!NOTE]
> 当前开发机候选进程由 Python 子进程在临时目录中启动，属于轻量进程隔离，尚未实现严格的 OS/容器级网络与文件硬沙箱（状态标识为 `AGENT_ISOLATION_UNVERIFIED`）。因此，本地实验结果严格标记为 `LOCAL_EXPERIMENTAL`，独立可信生产环境的部署标准参见 [独立评价环境契约](trusted-evaluator.md)。

---

## 3. 双任务线最小模块布局

系统将科研框架（循环驱动、历史记录、只读看板）与具体科研任务严格解耦：

| 模块类别 | 证候研究线 (TCM-SD) | 体质多维回归线 (Constitution) | 核心职责 |
| :--- | :--- | :--- | :--- |
| **通用驱动 Skill** | [SKILL.md](../skills/autoresearch/SKILL.md) | [SKILL.md](../skills/autoresearch/SKILL.md) | 统一规范实验执行生命周期与决策动作 |
| **任务程序契约** | [program.md](../skills/autoresearch/program.md) | [program-constitution.md](../skills/autoresearch/program-constitution.md) | 声明任务专属的数据清单、指标方向与约束 |
| **唯一可编辑候选** | [train.py](../src/research/train.py) | [constitution_train.py](../src/research/constitution_train.py) | 存放模型网络、特征工程与超参数的单文件源码 |
| **固化外部评价器** | [prepare.py](../src/research/prepare.py) | [constitution.py](../src/research/constitution.py) | 执行多折交叉验证并在外部计算指标 |
| **交付与验收入口** | [delivery.py](../src/syndrome/delivery.py) | [constitution_delivery.py](../src/research/constitution_delivery.py) | Champion 工件全量拟合、打包、哈希校验与导出 |
| **只读仪表盘** | [dashboard.py](../src/research/dashboard.py) | [dashboard.py](../src/research/dashboard.py) | 原生轻量化 SVG 实验趋势与状态监控看板 |

---

## 4. 证据分级与晋级策略

1. **严格晋级 (Strict Promotion)**：默认模式要求主指标必须实现可复现的严格正向提升（证候线 macro-F1 更大，体质线 masked MAE 更小），且改动带来的复杂度上升必须物有所值。
2. **防回归与失败处理**：
   - 负向改动或无明显增益：果断执行 REVERT 回退至上一个 Champion；
   - 异常奔溃或超时：标记为 CRASH，仅允许就地排查 2 次语法/环境错误，未恢复则立即回退；CRASH 轮次不产生有效评分。
3. **经验积累原则**：
   - 任何单次得分只能视为观察（Observation）；
   - 唯有在独立重复验证或不同随机种子下表现稳定的改进，才可沉淀为可靠科研经验；
   - 严禁为了刷分而在事后挑选或调整评价协议。
