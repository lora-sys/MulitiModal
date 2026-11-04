# 合成回归：迁移示例

这个任务只验证 AutoResearch 的迁移与运行流程，不使用真实数据，也不证明科研效果。

| 项 | 本例的冻结定义 |
| --- | --- |
| 目标 | 根据单个数字预测另一个数字 |
| 数据 | 内置合成训练与选择样本；不提供最终测试集 |
| 指标 | 选择集平均绝对误差，越低越好 |
| 基线 | `train.py` 的均值预测 |
| 可修改文件 | `examples/linear_research/train.py` |
| 固定部分 | 本目录 `prepare.py` 的数据与评价、容器运行器、预算和镜像 |
| 预算 | 每轮 20 秒，种子 1；实验轮数由研究者指定 |
| 证据 | `SMOKE_ONLY`；本地结果未独立认证 |

在 Linux 上启用用户 systemd、激活 Python 环境并准备本地 Docker 镜像后，从可信检出运行；将 `<worktree>` 和 `<run>` 换成新建的专用 worktree 与忽略目录的绝对路径：

```bash
PYTHONPATH=src python examples/linear_research/prepare.py \
  --candidate <worktree>/examples/linear_research/train.py --out <run>/round-000.json
PYTHONPATH=src python examples/linear_research/prepare.py record \
  --repo <worktree> --commit <baseline-commit> --result <run>/round-000.json \
  --results <run>/results.tsv --decision KEEP --description 'mean baseline'
./solve.sh dashboard --run <run>
```

随后可提出“线性拟合比均值预测更合适”的假设，将候选文件的 `MODEL` 从 `mean` 改为 `linear`，提交该文件，再重复评估和记录。指标改善则保留提交，否则回退专用 worktree。实际科研必须由研究者提供对应的真实数据协议和独立可信评价环境。
