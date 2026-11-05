# AutoResearch 任务契约：TCM-SD 证候分类

本契约规范 Coding Agent 在 TCM-SD 数据集上的自主模型探索闭环，参考 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 的极简科研工程理念。当前任务专注于 148 类中医中文短主诉证候分类，以 148 类 `macro-F1` 为核心评价指标（数值越高越好）。

## 1. 任务前置与环境设置

1. **阅读规范**：执行前仔细通读 [AGENTS.md](../../AGENTS.md)、[SKILL.md](SKILL.md) 以及评价器实现 [prepare.py](../../src/research/prepare.py)。
2. **分支隔离**：新建独立的 `autoresearch/<tag>` 实验分支或独立 git worktree，所有探索性 commit 严禁直接推送到主分支。
3. **核查数据集与冻结哈希**：
   - 清单路径：`runs/sd-v2/short/selection_manifest.json`
   - 记录路径：`runs/sd-v2/short/selection_records.json`
   - 固化 protocol-v2 评估协议：受试者严格分组、折内单向文本去重、148 类 macro-F1、4 折 × 2 次重复（共 8 折次验证）、种子 `20261104`。
   - **禁止触碰 holdout**：历史 holdout 测试集解锁预算已耗尽，严禁用于本轮候选筛选或调参。
4. **资源约束**：使用本机 Python 环境与可用 CPU/GPU。单折训练时间上限为 300 秒（`--budget 300`），整轮 8 次训练加上评估开销总墙钟时间约为 40 分钟。超时将触发超时熔断，提前收敛则提前结束。
5. **记录目录**：在 `.gitignore` 保护的 `runs/<tag>/` 目录下保存 `results.tsv`、逐轮配置 `round-*.json`、训练日志以及 `status.json`。

## 2. 核心研发守则

- **唯一可编辑文件**：**仅允许修改 `src/research/train.py`**。特征工程、模型架构、超参数、损失函数与优化策略由 Agent 全权自主决定。
- **严禁篡改评价环境**：严禁修改 `src/research/prepare.py`、外部评价逻辑、数据加载器、标签空间、验证折划分或种子；不得擅自引入外部未声明的三方重量级依赖。
- **防止统计量泄露**：所有特征提取器（如 TF-IDF 词表）、标量变换及分类器拟合必须在各训练折内部完成，严禁接触验证折或折外信息。
- **摒弃复杂盲目崇拜**：不过度假设深层神经网络一定超越传统轻量模型。在指标相当时优先选择紧凑可维护的模型；晋级（KEEP）必须以可复现的验证集分数严格提升为硬性前提。
- **证据分级透明**：本机训练生成的结果标注为 `LOCAL_EXPERIMENTAL`，不作为独立认证的科研定论或临床结论。

## 3. 标准评估运行命令

每轮实验递增指定输出路径，通过外部独立评价器打分：

```bash
export PYTHONPATH=src
python -m research.prepare \
  --manifest runs/sd-v2/short/selection_manifest.json \
  --dataset runs/sd-v2/short/selection_records.json \
  --candidate src/research/train.py \
  --out runs/<tag>/round-000.json \
  --folds 4 --repeats 2 --seed 20261104 --budget 300
```

> [!NOTE]
> 最终指标由 `prepare.py` 在候选子进程外部计算并产出，候选代码自报指标不予采纳。首轮必须运行原始未经修改的 `train.py` 以沉淀基准分数（基线参考值约为 0.3199）。

## 4. 实验迭代闭环流程

持续自治循环执行，直到用尽规划预算或收到用户停止指令：

1. **提出假设**：分析当前 Champion、历史胜负记录与误差分布，形成单一清晰且可证伪的改进假设。
2. **编码与提交**：仅编辑 `src/research/train.py`，完成修改后执行 `git commit`。
3. **运行统一评价**：执行上述评测命令，捕获各折详细评分与日志。
4. **判定决策**：
   - **KEEP**：macro-F1 严格高于 Champion，且引入的复杂度具有可接受的产出比。
   - **REVERT**：指标未改善或发生负向退化，执行 `git reset --hard <champion_sha>` 回滚代码。
   - **CRASH**：发生执行异常或运行超时，查看日志并允许就地修补最多 2 次；若仍未恢复则果断回滚，该轮指标视作失效。
5. **记录决策**：
   ```bash
   python -m research.prepare record \
     --results runs/<tag>/results.tsv \
     --result runs/<tag>/round-xxx.json \
     --commit <sha> \
     --decision KEEP|REVERT|CRASH \
     --description "<假设与改动摘要>"
   ```
6. **启动仪表盘**：
   ```bash
   ./solve.sh dashboard --run runs/<tag>
   ```

## 5. 最终交付与交接 (Handoff)

当达到实验终止条件或用户要求交付时：
- 保留最佳 Champion Commit，并在交付总结中清晰列明演进历程与核心消融分析。
- 模型持久化与工件化、推理部署验收、逐类准确率报告均通过标准交付模块承接（参见 [src/syndrome/delivery.py](../../src/syndrome/delivery.py)），不侵入单轮实验循环。
- 未经独立异源真实数据验证前，坚守科学严谨性，不宣称跨机构或外部泛化有效性。
