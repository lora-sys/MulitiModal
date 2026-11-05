# 本项目的证候研究适配

这份说明是通用 AutoResearch Skill 的一个任务实例；任务事实不写进 Skill。

- 目标：以短症状主诉预测证候类别；当前任务不提供临床诊断。
- 历史选择口径：TCM-SD protocol-v2，受试者分组、折内对验证集做单向精确文本去重，148 类 macro-F1，越高越好。现有 holdout 预算已用尽。
- 候选文件：`src/research/train.py`。固定任务协议、候选执行和结果记录位于 `src/research/prepare.py`；只读监控位于 `src/research/dashboard.py`。
- 当前状态：候选由本机 Python 子进程在临时目录运行，未通过 OS/容器级隔离，也未阻断网络。`AGENT_ISOLATION_UNVERIFIED` 仍成立；本地分数不能作为可信科研评分或临床结论。
- 运行前先建立专用、干净的 Git worktree 和新的忽略目录 `runs/<tag>`。使用固定预算、seed、folds 和 repeats，不需要 Docker 镜像。完整流程见 [通用 Skill](../skills/autoresearch/SKILL.md) 和 [程序契约](../skills/autoresearch/program.md)。

从可信代码检出目录执行一次合成评估（路径按实际 worktree 修改）：

```bash
PYTHONPATH=src python -m research.prepare --synthetic \
  --candidate /absolute/path/worktree/src/research/train.py \
  --out /absolute/path/runs/tag/round-000.json --budget 60 --folds 2 --repeats 1
```

用 `python -m research.prepare record --help` 查看记录命令；记录要求候选文件与 Git 提交哈希一致，且不允许混合协议。`./solve.sh dashboard --run /absolute/path/runs/tag` 启动本机只读监控。本地结果文件没有独立认证，不能视为可信科研评分。
