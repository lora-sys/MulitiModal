"""中医体质 AutoResearch 第二阶段研究程序（最小、独立、冻结）。

在证候线 program.md 之后启动的第二条研究线。复用同一 Skill、Git
KEEP/REVERT/CRASH 循环、results.tsv 契约与 `research.dashboard` 只读监控；
不新建训练平台、不使用 Docker、不启动多 Agent、不安装新依赖。

## 研究目标与冻结条件

- 数据：Mendeley `Share data.xls`，274 名受试者，8 维 PUBLISHED 体质评分
  （ZYYXH/T157-2009，0–100）。平和质是第 9 维且**完全未观测**：ABSTAIN，
  不伪造、不补零、不猜测，任何候选输出必须是 (n, 8)。
- 主指标：`constitution/masked_mae`（8 维平均绝对误差），**越低越好**，
  direction=minimize。RULE 派生标签与 PUBLISHED 评分不可互换。
- 划分：selection-only 分组 4 折 × 2 重复，seed 20261105；同一受试者不跨
  分区；holdout 为空（历史体质 holdout 2/2 解锁预算已耗尽，不得重复使用）。
- 冻结哈希：`CONSTITUTION_DATASET_SHA256` = 原始 xls sha256；manifest 与
  records 的 sha256 冻结在 `src/research/constitution_data.py`。
- 预算：每折最多 300 秒（5 分钟墙钟），不额外安装依赖。
- 证据状态：`LOCAL_EXPERIMENTAL`；评估器隔离 `AGENT_ISOLATION_UNVERIFIED`。
  体质线 `GO_RESEARCH_CLAIMS` 仍为 `NOT_GRANTED`（缺独立参照数据），
  任何 dev/CV 分数都不得推导科研结论或临床有效性。

## 先验证，再研究

1. **真实数据验证**：`python -m research.constitution_data verify` 核对哈希、
   274 行、8 维、平和质未观测、缺失特征 mask、holdout 为空、标签来源。
2. **回归基线**：`round-000.json` 记录基线候选（8 维训练折中位数），
   以 `record --decision KEEP` 建立冠军。
3. **自主实验**：之后每轮提出一个可检验假设，只改
   `src/research/constitution_train.py`，提交 commit、完整评估、记录决策。

## 固定评价命令

```bash
export PYTHONPATH=src
python -m research.constitution \
  --manifest data/local/mendeley-cxtph3tjsg-v1/protocol-v1/selection_manifest.json \
  --dataset data/local/mendeley-cxtph3tjsg-v1/protocol-v1/selection_records.json \
  --xls "data/local/mendeley-cxtph3tjsg-v1/Share data.xls" \
  --candidate src/research/constitution_train.py \
  --out runs/constitution-v1/round-000.json \
  --folds 4 --repeats 2 --seed 20261105 --budget 300
```

评分由 `research.constitution` 在候选进程外计算，不采纳候选自报指标。

## 实验循环

1. 查看当前 champion、Git HEAD、`results.tsv` 与上轮结果，提出一个可检验假设。
2. 只修改 `src/research/constitution_train.py`，提交 commit，运行完整固定评价。
3. 用 `python -m research.constitution record` 将 commit、masked_mae、假设和
   KEEP / REVERT / CRASH 写入 `results.tsv`。
4. **KEEP**：masked_mae 严格低于 champion 且实现值得保留。
   **REVERT**：未改善，重置到 champion。**CRASH**：超时或结果无效，
   检查日志、最多修复两次，否则回退；CRASH 不产生有效分数。
5. 根据本轮结果选择下一假设，不提前排定全部实验，不逐轮请示。

只读监控（复用现有 Dashboard）：

```bash
./solve.sh dashboard --run runs/constitution-v1
```

## 禁止事项

- 不修改冻结评价协议、数据、标签、划分、seed、预算、主指标或方向；
- 不修改证候线冠军模型与 `src/research/train.py`；
- 不把缺失特征补零、不把未观测平和质当 0 分或任何猜测值；
- 不读取任何历史 holdout、不宣称外部泛化已验证；
- 不引入 Docker、多 Agent、新训练平台或新依赖。
