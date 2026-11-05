---
name: autoresearch
description: Run, resume or monitor consecutive model research experiments with local CPU/GPU, a frozen evaluation protocol, Git KEEP/REVERT and a read-only dashboard. Use when asked to train and improve models, run AutoResearch, compare candidates, inspect experiments or show the monitoring page.
---

# AutoResearch

本 Skill 由当前 Coding Agent 执行，不启动另一个 Agent 服务。

1. 先读 `AGENTS.md`、当前研究线对应的 program 文件和当前研究任务资料。研究目标与冻结条件由研究者确定，Agent 负责逐轮提出假设和实施实验。
   - 证候线（TCM-SD 148 类）：`skills/autoresearch/program.md`。
   - 体质线（274 人八维评分）：`skills/autoresearch/program-constitution.md`。
2. 两条研究线互相独立，数据、标签、划分、指标、候选文件与冻结评价协议不得混用。证候线循环只允许修改 `src/research/train.py`；体质线循环只允许修改 `src/research/constitution_train.py`。两条线都禁止在轮次中修改各自的评价器、真实数据、标签、划分、指标和历史 holdout。
3. 使用本机 Python、CPU/GPU 运行，不要求 Docker、sandbox、独立服务或常驻工作队列。用 Git 管理候选，用 `runs/<tag>/results.tsv` 记录决策。
4. 第一次运行必须先复现 TF-IDF + LR 基线。之后每轮读上轮证据，提出单个可检验假设，修改候选、提交、用同一协议评估、KEEP 或 REVERT。失败不得晋级，结果不得事后挑选。
5. 使用 `./solve.sh dashboard --run runs/<tag>` 打开现有只读监控。网页只读取状态与实验记录，不控制训练。
6. 开始真实数据实验前先确认本机入口已经完成适配，且通过用户分组、冻结哈希、无 holdout 和评估一致性测试。若入口仍仅允许合成数据，不可假装已开始真实训练。
7. 当前本机实验标记 `LOCAL_EXPERIMENTAL`。不宣称具有独立防篡改、外部泛化、诊断或临床有效性。
8. 按研究者的总预算连续执行，不在每轮等待批准；结束时交付最佳 commit、训练权重、完整报告和监控截图。
