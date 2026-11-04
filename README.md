# MulitiModal

面向按摩椅场景的中医健康参考研究。项目探索证候、体质与按摩前生理状态三条独立路线；结果不构成诊断或临床疗效声明。项目事实、数据许可与硬规则见 [AGENTS.md](AGENTS.md)。

## AutoResearch：让每轮研究留下可复用的东西

[AutoResearch Skill](skills/autoresearch/SKILL.md) 驱动研究者定义的实验循环：先测基线，再根据上一轮的证据提出假设，修改一个指定文件，固定评价，记录并保留或回退。每轮留下提交、协议标识、结果和判断；有价值的观察需要带证据与适用范围，经过独立复现后才能当作可靠经验。

研究者在开始前需要定义研究问题、数据和许可、训练/选择/锁定测试划分、基线、主指标与方向、固定评价器、允许修改的文件、运行环境、资源与总预算、停止和交付条件。使用 [研究简报模板](docs/autoresearch-brief.md) 固定这些内容。缺失评价或隔离边界时，不运行真实数据研究。

```text
研究简报 → 基线 → 假设与单文件修改 → 隔离运行 → 固定评价
                  ↑                         ↓
               下一项假设 ← 记录证据 ← KEEP / REVERT
```

Skill 描述通用方法；任务各自提供数据与评价代码。候选执行、结果历史和只读监控是共享的最小工具，不承担任务的科学定义。运行期数据、结果和检查点留在本地，不提交 Git。

[设计说明](docs/autoresearch-design.md) 解释评价边界、证据和当前未实现的能力。

## 从合成示例开始

[线性回归示例](examples/linear_research/README.md) 是与本项目任务无关的最小迁移演示：固定合成数据和平均绝对误差，先运行均值基线，再尝试线性拟合。它使用一个可修改训练文件、容器运行器和同一监控页。示例为 `SMOKE_ONLY`，只验证流程。

启用项目 Python 环境并准备本地 `multimodal-research-worker:2.7.0-cpu` Docker 镜像后，按示例说明创建专用 Git worktree 与 run 目录。`./solve.sh dashboard --run <run>` 在本机启动只读监控；本地结果记录未独立认证。

## 本项目的研究适配

[证候任务说明](docs/autoresearch-tcm-sd.md) 记录当前数据、指标、运行命令和证据边界。新入口目前只允许运行内置合成数据；候选容器的定向检查不等于可信评价器已独立于提案 Agent，真实数据入口保持关闭（`AGENT_ISOLATION_UNVERIFIED`）。体质和证候 holdout 的解锁预算已用尽，不得用于候选选择。

[历史研究报告](docs/reports/) 保留原有结论与局限。报告引用的旧脚本和运行方式可在重写前的 Git 提交 `afe7580` 查阅；新入口不直接运行旧命令。

## 参考

- [Karpathy AutoResearch](https://github.com/karpathy/autoresearch)：单文件修改、固定预算、基线和逐轮保留/回退。
- [Hyra 2.0](https://hunyuan.tencent.com/research/hyra2)：以可复用工具、证据记录和经验积累支持后续研究。
- [Hyra Results](https://github.com/Tencent-Hunyuan/Hyra-results)：公开研究产物、运行脚本和可复查证据。本仓库借鉴可复用产物的思路，不声称复现其内部系统。
