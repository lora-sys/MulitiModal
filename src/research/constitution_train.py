"""第二阶段可编辑候选：八维体质评分回归基线。

冻结评价器只给本进程两个 JSON 文件（train.json / predict.json），运行在隔离
目录、``python -I`` 之下，看不到仓库、冻结清单或验证标签。基点模型只使用训练
折的 8 个观测维度，输出 (n, 8) 的分数；平和质完全未观测，绝不预测。

基线策略：每维训练折中位数（对 MAE 稳健），不使用任何特征。后续轮次由
AutoResearch 循环在本文件内自主提出假设并修改。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--predict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    training = json.loads(Path(args.train).read_text(encoding="utf-8"))
    inputs = json.loads(Path(args.predict).read_text(encoding="utf-8"))
    rows = training["rows"]
    n_dims = len(training["score_dims"])
    if n_dims != 8 or any(len(row["scores"]) != n_dims for row in rows):
        raise ValueError("baseline expects exactly 8 observed constitution scores")
    medians = [statistics.median(row["scores"][index] for row in rows)
               for index in range(n_dims)]
    predictions = [list(medians) for _ in inputs["rows"]]
    Path(args.output).write_text(json.dumps(predictions), encoding="utf-8")


if __name__ == "__main__":
    main()
