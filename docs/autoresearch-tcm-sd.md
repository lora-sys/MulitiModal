# 本项目的证候研究适配

这份说明是通用 AutoResearch Skill 的一个任务实例；任务事实不写进 Skill。

- 目标：以短症状主诉预测证候类别；当前任务不提供临床诊断。
- 历史选择口径：TCM-SD protocol-v2，受试者分组、折内对验证集做单向精确文本去重，148 类 macro-F1，越高越好。现有 holdout 预算已用尽。
- 候选文件：`src/research/train.py`。固定任务协议位于 `src/research/prepare.py`；候选执行由 `src/research/sandbox.py` 负责，监控由 `src/research/dashboard.py` 提供。
- 当前状态：仅支持内置合成 fixture 的工程 smoke。候选容器隔离已做定向检查，评价器独立于提案 Agent 的隔离尚未验证；真实数据入口拒绝运行，不能用 smoke 得分作研究结论。
- 运行前先建立专用、干净的 Git worktree 和新的忽略目录 `runs/<tag>`。每轮使用同一预算、seed、folds、repeats 和镜像。完整流程见 [通用 Skill](../skills/autoresearch/SKILL.md) 和 [程序契约](../skills/autoresearch/program.md)。

从可信代码检出目录执行一次合成评估（路径按实际 worktree 修改）：

```bash
PYTHONPATH=src python -m research.prepare --synthetic \
  --candidate /absolute/path/worktree/src/research/train.py \
  --out /absolute/path/runs/tag/round-000.json --budget 60 --folds 2 --repeats 1
```

用 `python -m research.prepare record --help` 查看记录命令；记录要求候选文件与 Git 提交哈希一致，且不允许混合协议。`./solve.sh dashboard --run /absolute/path/runs/tag` 启动本机只读监控。本地结果文件没有独立认证，不能视为可信科研评分。
