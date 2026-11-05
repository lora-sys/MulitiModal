# TCM-SD 证候分类研究适配说明

本文档详细说明 [AutoResearch Skill](../skills/autoresearch/SKILL.md) 在中医证候多分类任务（TCM-SD）上的具体实现契约与运行标准。

---

## 1. 任务背景与核心规格

- **预测目标**：输入患者的单条中文症状主诉短文本，预测其所属的中医证候类别（共 148 类别）。本任务仅作为健康参考，明确不提供临床诊断。
- **数据集特征**：基于 TCM-SD 真实数据集，共计 54,152 例临床短文本，遵循 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 知识共享许可协议（非商业研究）。
- **评价协议**：protocol-v2
  - **受试者级物理隔离**：同一受试者的全部记录不可跨折拆分；
  - **文本去重保障**：在训练折内部对验证折执行单向精确文本去重，消除简单文本记忆带来的虚高表现；
  - **评价指标**：148 类宏平均 `macro-F1`（值在 [0, 1] 区间，数值越高越好）；
  - **折次与种子**：标准评估执行 4 折 × 2 次重复（共 8 折次验证），随机种子固化为 `20261104`；
  - **测试集状态**：历史 holdout 测试集解锁预算已彻底耗尽，严禁用于本轮候选的训练或选择。

---

## 2. 模块与入口职责

- **唯一可编辑候选源码**：`src/research/train.py`。
- **固化外部评价入口**：`src/research/prepare.py`。
- **原生轻量只读看板**：`src/research/dashboard.py`。
- **交付与部署契约**：`src/syndrome/delivery.py`。

---

## 3. 标准操作指南

### 3.1 真实数据正式评估命令
每轮探索更新指定输出的 JSON 路径：

```bash
export PYTHONPATH=src
python -m research.prepare \
  --manifest runs/sd-v2/short/selection_manifest.json \
  --dataset runs/sd-v2/short/selection_records.json \
  --candidate src/research/train.py \
  --out runs/<tag>/round-000.json \
  --folds 4 --repeats 2 --seed 20261104 --budget 300
```

### 3.2 快速合成冒烟评估 (Smoke Mode)
用于在无真实数据或仅验证工程通路时快速执行：

```bash
export PYTHONPATH=src
python -m research.prepare --synthetic \
  --candidate src/research/train.py \
  --out runs/<tag>/smoke-000.json \
  --budget 60 --folds 2 --repeats 1
```

### 3.3 决策记录命令
```bash
python -m research.prepare record \
  --results runs/<tag>/results.tsv \
  --result runs/<tag>/round-xxx.json \
  --commit <sha> \
  --decision KEEP|REVERT|CRASH \
  --description "<假设与实现描述>"
```

### 3.4 启动实时只读仪表盘
```bash
./solve.sh dashboard --run runs/<tag>
```

---

## 4. 证据边界与合规声明

- 本地评估由 Python 子进程在受限临时目录下运行，当前属于 `AGENT_ISOLATION_UNVERIFIED` 状态；
- 本机生成的全部实验结果统一归属于 `LOCAL_EXPERIMENTAL` 级别，不可直接外推为具有外部通用性或临床证据力的学术结论。
