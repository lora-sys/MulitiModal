# AutoResearch: TCM-SD

让 Coding Agent 参考 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 自主改进中医证候分类模型。当前只研究 TCM-SD 的 148 类中文短主诉分类，主指标为 macro-F1，越高越好。其他中医体质和 PDF 生理预测任务另行研究。

## Setup

1. 阅读 `AGENTS.md`、`skills/autoresearch/SKILL.md`、`src/research/prepare.py` 和 `src/research/train.py`。新建 `autoresearch/<tag>` 分支或独立 worktree。
2. 核查本机真实 TCM-SD 开发选择集及冻结哈希：`runs/sd-v2/short/selection_manifest.json`、`selection_records.json`。固定 protocol v2 的受试者分组、折内单向文本去重、148 类 macro-F1、4 折 × 2 次重复和 seed。禁止使用已耗尽解锁预算的历史 holdout。
3. 训练使用开发机 Python 和实际可用的 CPU/GPU，不使用 Docker。**每折最多 5 分钟墙钟时间**，即 `--budget 300`。一轮包含 8 次训练，最多约 40 分钟加评估开销。当前实现是超时上限，模型提前完成就提前结束，并非强制训练满 5 分钟。
4. 在 Git 忽略目录 `runs/<tag>/` 保存 `results.tsv`、`round-*.json`、日志和 `status.json`。第一轮运行原样 `train.py` 建立基线；历史约 0.3199 仅供对照，不能代替本轮真实评分。

## Rules

- **只允许修改 `src/research/train.py`**。模型、特征、超参数、损失函数和训练策略由 Agent 自主决定。
- **不得修改 `prepare.py`、评价器、数据、标签、划分、seed 或训练预算**。不额外安装依赖；训练折之外的数据不能用于拟合。
- 不假设神经网络比传统模型更好。性能接近时优先简单模型，但已有记录规则要求 KEEP 的分数必须严格提高。
- 固定实验协议内比较候选。合成或缩减折数的调试结果不参与冠军排名。真实本机结果仅标为 `LOCAL_EXPERIMENTAL`。

## Run

每轮使用新输出路径，固定如下正式评价命令：

```bash
export PYTHONPATH=src
python -m research.prepare \
  --manifest runs/sd-v2/short/selection_manifest.json \
  --dataset runs/sd-v2/short/selection_records.json \
  --candidate src/research/train.py \
  --out runs/<tag>/round-000.json \
  --folds 4 --repeats 2 --seed 20261104 --budget 300
```

将 `<tag>` 替换为研究目录；后续使用 `round-001.json` 等。评分由 `prepare.py` 计算，不采纳候选自己报告的指标。

## Experiment loop

**持续执行，直到用户停止。没有总时长或总轮数限制。**

1. 查看当前 champion、Git HEAD、历史结果和上轮失败原因，提出一个可验证的假设。
2. 只修改 `train.py`，提交 Git commit，运行完整固定评价，保存日志和逐折结果。
3. 用 `python -m research.prepare record` 将本轮 commit、macro-F1、假设和 KEEP、REVERT 或 CRASH 写入 `results.tsv`。
4. **KEEP**：得分严格超过 champion，且复杂度值得接受。**REVERT**：未改善或不值得保留，重置到 champion。**CRASH**：超时、报错或结果无效，检查日志，最多修复两次，否则回退。CRASH 不产生有效分数。
5. 根据本轮结果选择下一项实验，不提前排定全部实验，不逐轮向用户请示。

现有只读网页用于查看结果：

```bash
./solve.sh dashboard --run runs/<tag>
```

## Handoff

用户停止研究或要求交付时，保留最佳 commit、完整实验记录与必要的运行说明。模型保存与重载、推理验收、逐类分析和外部泛化验证在独立的交付任务中完成，不把它们加入每轮 KEEP/REVERT，也不重复使用历史 holdout。没有独立外部数据时，不宣称跨来源泛化已经验证。
