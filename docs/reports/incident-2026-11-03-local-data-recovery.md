# 事故与恢复记录：开发机 data / runs / artifacts 被删除（2026-11-03）

## 1. 事实

在开发机 `acam-lab:~/workspace/projects/MulitiModal` 上执行了一条多源删除式同步：

```
rsync -a --delete src/ scripts/ <dest>/      # 错误命令
```

rsync 的 `--delete` 作用于**目标根目录**，而多源形式把 `src/`、`scripts/` 的**内容**摊到 `<dest>` 根下，于是克隆根目录被清空：`.git`、`.venv`、`docs/`、`runs/`、`artifacts/`、`data/` 全部丢失。

本机克隆（`/Users/lora/repos/MulitiModal`）完好，未受影响；这正是恢复的起点。

## 2. 幸存与已恢复（含证据）

| 项 | 现状 | 来源 / 证据 |
| --- | --- | --- |
| 源码与历史 | `.git` 由本机 `git bundle --all` 重新 clone | base `fde897c`（#73） |
| `data/` | 1.7G，布局与本机逐文件一致（22 文件） | 本机 `data/local` 856M 回传 |
| `mendeley-cxtph3tjsg-v1/Share data.xls` | 284K | 重下（官方 files API，DOI `10.17632/cxtph3tjsg.1`）；**SHA-256 `6206f432b683f8a7b9a75dae23e598a4f01a41a4f8af84b099924b9d85d08a96`，与 `docs/data/dataset-audit-report.md` 记录一致** |
| `models/bert-base-chinese` | 633M | 重下公开权重（HF 模型 ID `bert-base-chinese`，经 hf-mirror）。仓库未记录该目录的 SHA-256，故以加载与 `tests/syndrome/test_bert_model.py` 通过为验证 |
| `models/{chinese-roberta-wwm-ext,mc-bert}` | 各 393M | 本机回传 |
| `runs/` | 553M | `projects/MulitiModal-{i56,i57,issue-56}/runs` 三处幸存目录合并：`sd-v2`（含 `selection_manifest.json`/`selection_records.json`/`cv-report.json`）、`issue57`、`data-manifests` |
| `artifacts/` | 4.6M | 本机 `artifacts/tcm-50` v1–v7，**含 `weights.pt` / `training_state.pt`** |
| `.venv` | 重建 | 补装 `matplotlib`、`xlrd`——`requirements-dev.txt` 未列出，但旧 venv 实装有；版本对齐文档（`transformers 5.22.1` / `accelerate 1.15.0`） |
| AutoResearch dashboard | 恰好 1 进程，`127.0.0.1:8765` HTTP 200 | runs root `~/workspace/asr-runs` 未在项目内，事故中幸存 |

跨机冗余已建立：`runs/`、`bert-base-chinese`、Mendeley 工作簿同时存在于本机与开发机。

## 3. 已确认丢失（两台机器全盘搜索无）与处置

1. **`runs/holdout-unlock-ledger-sd.jsonl`（及同名 `holdout-unlock-ledger.jsonl`）——不重建。**
   理由：冻结决定已明确「v10 的第一次解锁发生在台账机制存在之前，由本报告与治理文档记录，**不追补台账条目**」（`docs/reports/research-src-v9.md:207`）。台账是「读权限被授予」的行为记录，不是可由报告反推的哈希；写入事后编造的条目比缺失更糟。已记录的事实：sd 线首次事件 `00dd8687b86f4518…`、预算 `MAX_UNLOCKS_SD=1`、冠军 `tcm-0003`（commit `1dbdbe5`）——见 `docs/reports/research-sd-v1.md:54-62`。
2. **`runs/real-tcm-v5..v8`、`runs/research-v9/v10/v11*`、`runs/sd-campaign`、`runs/smoke`、`runs/manifests`、`runs/tcm-sd-syndrome-v1`**：历史运行目录（其中被文档引用的名字见 `docs/design/autoresearch-rsi-v1.md` §11、`docs/reports/research-src-v{9,11}.md`）。它们可由文档化命令重跑再生。
3. **`artifacts/tcm-50-v8`——`#50` config 时代真实训练（v8）的交付产物目录，未自动再生。**
   该线由 `runtime/harness.py` 产出（`docs/design/autoresearch-rsi-v1.md:284`「`runtime/harness.py`（v1–v8）」），重跑需要 GPU 工时而该线含 holdout 读取（v7 产物内有 `chart-regression-holdout.png`），属治理决策，须经明确授权后执行。其**数字与溯源仍在 git 内**：`docs/reports/tcm-50-real-training.md`、`README.md`「#50 交付结论」。

## 4. 恢复后的验证（开发机）

- `pytest tests` → **597 passed, 9 skipped, 0 failed**
- `PYTHONPATH=src python scripts/review_eval.py` → **52/52**
- `ruff check .` → clean
- 真实数据路径已恢复可执行：`tests/runtime/test_harness_real.py`（#50 端到端）、`tests/syndrome/test_bert_model.py`、`tests/syndrome/test_syndrome_real_data.py` 均实际运行（此前因缺数据/权重 skip）。
- 恢复过程中暴露并修复了两条**只在缺数据时才 skip、因而从未在绿色基线里跑过**的陈旧测试（#73，已合并）：BERT fixture 未拟合 `max_len` 就调用 `predict_proba`/`save`（#53 守卫正确，改为按训练配置默认值声明，不放宽守卫）；真实数据分区测试仍断言已改名的 `official_file_counts`（现为 `development_file_counts`）。

## 5. 纪律修订

- **禁止多源 + `--delete`**。目录传输只允许单源 `rsync -a <src>/ <dst>/`。
- 任何 `--delete` / `rm -rf` 必须先 `--dry-run` 或先列出待删项；对克隆根目录一律排除 `.git/`、`.venv/`。
- 真实数据、权重与运行产物照旧不入 Git，但恢复后应立即建立**跨机第二副本**（本次已做）。

## 6. 未改变的工程判断（原「去重复」计划取消）

原计划删除 `src/runtime/harness.py`、`src/runtime/agent.py`、`runtime.cli research*` 及 8 条只驱动它们的审计检查（e39/e40/e41/e57/e60/e64/e71/e74）。**该计划与冻结文档冲突，已取消**：

- `docs/design/autoresearch-rsi-v1.md:284` 把 `runtime/harness.py` 列为 v1–v8 的生产路径，并要求「**保留作为回归与冒烟路径**」。
- `docs/reports/issue-61-architecture-audit.md:29` 的收敛规则是「config-only loop 与 `runtime.cli research*` 标记 deprecated，**保留可运行以维持回归**」。
- `README.md` 已就入口收敛作出标注：「#50 交付结论（v8 —— config 时代结论，已被源码级 v9/v10 取代）」，源码级入口章节指向 `src/research/`。

即：旧路径是**被标注取代但保留的回归线**，不是可删的死代码；删除它会同时移除 v8 交付线的再生能力与 8 条冻结审计检查。