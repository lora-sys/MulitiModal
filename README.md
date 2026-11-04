# MulitiModal

按摩椅场景下的中医健康参考研究。当前研究入口是 TCM-SD 症状主诉到证候倾向；项目不提供诊断或临床疗效结论。

## AutoResearch

按照 [AutoResearch Skill](skills/autoresearch/SKILL.md) 操作。研究 Agent 只修改 [train.py](src/research/train.py)，每轮基于上一轮结果提出一个假设；固定协议负责评分（当前仅支持合成数据 smoke），保留或回退改动。`./solve.sh dashboard` 提供只读监控页面。

先启用项目的 Python 环境；候选训练在本机已有的 `multimodal-research-worker:2.7.0-cpu` Docker 镜像中执行。

- [AGENTS.md](AGENTS.md)：项目规范、数据边界和研究背景。
- [program.md](skills/autoresearch/program.md)：每轮提案格式。
- [solve.sh](solve.sh)：研究命令入口，例如 `./solve.sh dashboard --run runs/R`。
- [研究运行代码](src/research/)：冻结数据处理、评估、隔离和运行记录。
- [研究报告](docs/reports/)：历史实验、方法和证据。

真实研究的 protocol-v2 选择池和指标保持冻结；新入口目前只运行内置合成数据。Holdout 预算已用尽；选择轮次不得使用 holdout。隔离的端到端状态为 `AGENT_ISOLATION_UNVERIFIED`，不应据此宣称系统已验证。TCM-SD 采用 CC BY-NC-SA 4.0，仅限许可范围内使用。

历史报告中的旧研究脚本与运行方式可在重写前的 Git 提交 `afe7580` 查阅；新入口不直接运行这些旧命令。
