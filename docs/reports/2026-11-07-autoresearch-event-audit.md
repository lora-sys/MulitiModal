# AutoResearch 实验审计修复说明（任务 01）

日期：2026-11-07 · 证据等级：`LOCAL_EXPERIMENTAL` · 隔离级别：`AGENT_ISOLATION_UNVERIFIED`

## 1. 缺陷与影响

`src/research/prepare.py` 的 `record_main()` 原先对所有 `CRASH` 强制写入 `SMOKE_ONLY`：

```python
evidence=EVIDENCE_LOCAL if args.decision != "CRASH" and evidence == EVIDENCE_LOCAL
else EVIDENCE_SMOKE
```

两个后果：

1. 真实开发集（`LOCAL_EXPERIMENTAL`）实验崩溃后，证据等级被降级为 `SMOKE_ONLY`；
2. `append_history()` 要求同一 run 的所有行 `evidence` 一致，因此在已有
   `LOCAL_EXPERIMENTAL` 行的 run 里记录 CRASH 会直接抛
   `ValueError: result metadata differs from this run's earlier results`——
   崩溃根本进不了审计记录。

这不是分数错误，而是审计链断裂：外部只能看到最终 `REVERT`，看不到失败尝试。

## 2. 修复内容

| 位置 | 修改 |
| --- | --- |
| `prepare.record_main()` | 删除 CRASH 的 SMOKE_ONLY 特例。证据等级按 `--evidence` > `--result` 文件中的 evidence > 该 run 已有 results.tsv 行 > 默认 `LOCAL_EXPERIMENTAL` 依次确定；有分数的轮次仍以 result 文件为准 |
| `prepare.resolved_evidence()` / `history_evidence()` | 崩溃常常没有 result 文件，改由同 run 的既有行继承证据等级 |
| `prepare.append_history()` | 完全相同的记录命令重放为 no-op（不产生重复行）；同一 commit 若此前只有未记分的 CRASH 行，允许追加修复重试，不覆盖原崩溃；已有带分数决策的 commit 仍拒收第二个带分数决策 |
| `prepare.append_event()` | 新增 append-only `events.jsonl`，事件带 `event_id` / `candidate_id` / `attempt` / `repair_of` / `decision` / `evidence` / `protocol_sha256` / `result_path` / `log_path`；同一 `event_id` 重复写入为 no-op，同 id 不同内容报错 |
| `prepare.event_id_for()` | 由（results 路径、candidate_id、attempt、commit、decision、description、value、evidence、repair_of）生成稳定 sha256，保证重复命令幂等 |
| `prepare.add_event_arguments()` / `record_event()` | 共享 CLI 与事件写入，证候线与体质线 `record_main` 都产出事件 |
| `constitution.record_main()` | 本就没有 SMOKE_ONLY 缺陷；现在同样写入事件，CRASH 保持 `LOCAL_EXPERIMENTAL` |

`results.tsv` 仍是冻结的决策历史，不改写、不补写历史事件；事件日志是新增的旁路审计层。

两点如实说明：

- 修改 `src/research/prepare.py` 会改变 `fixed_code_hashes()` 返回的评估器哈希。
  历史 `round-*.json` 里记录的仍是修改前的哈希，因此这些旧结果**不能用新版
  `record` 重新提交**，也不会被重新验证；本轮没有重新提交任何历史结果。冠军交付
  （`syndrome.delivery`）不依赖该哈希，未受影响。
- `record` 仍要求给出候选 commit（`committed_source` 绑定），所以事件始终与具体
  代码版本关联；若候选在提交前就崩溃，需要先用当时的提交哈希或明确标注的占位
  标识记录，不能凭空编造 commit。

## 3. 事件记录方式

```bash
# 第一次崩溃（无 result 文件，evidence 从同一 run 的既有行继承）
python -m research.prepare record \
  --repo . --results runs/<tag>/results.tsv --commit <sha> \
  --decision CRASH --description "round-017 crash: timeout" \
  --candidate-id round-017 --attempt 1 --log runs/<tag>/round-017.log

# 同一候选的修复重试：显式链接到崩溃事件，attempt 递增
python -m research.prepare record \
  --repo . --results runs/<tag>/results.tsv --commit <sha2> \
  --result runs/<tag>/round-017b.json --decision REVERT \
  --description "round-017 repaired once: ..." \
  --candidate-id round-017 --attempt 2 --repair-of <crash_event_id>

# 重放同一条命令：results.tsv 与 events.jsonl 均不产生重复
```

- `--candidate-id` 是崩溃与修复重试共享的轮次标识；`--repair-of` 指向被修复的
  崩溃 `event_id`；`--attempt` 记录第几次尝试。三者共同构成可追溯的修复链。
- `--event-log` 可覆盖默认路径（results.tsv 同目录的 `events.jsonl`）。
- `--event-id` 可显式指定幂等键；默认由事件身份哈希生成。
- 事件日志追加写、带文件锁、原子替换；日志末行不完整时拒绝追加而非静默修复。
- KEEP / REVERT / CRASH 都能同时写入结果行和事件；CRASH 的 `value` 为空、
  `protocol_sha256` 为空（未产生可验证分数），但 `evidence` 与候选信息完整。

## 4. 历史 `round-017` 的真实情况（不补造事件）

生产机 `runs/ar1/` 的实际证据：

- `round-017.log` 存在超时 traceback；`round-017b.json` 与 `round-017b.log` 存在；
- `results.tsv` 第 18 行（索引 17）只有一条 `REVERT`：
  `round-017 (CRASH repaired once) real L1 l1_ratio=1.0 max_iter=200: 0.314997 vs 0.325958 champion; original penalty=l1/max_iter=2000 timed out (see round-017.log)`，
  `evidence=LOCAL_EXPERIMENTAL`，`value=0.31499734725112366`；
- `runs/ar1/results.tsv` 共 32 行决策，全部为 `KEEP`/`REVERT`，**0 条 CRASH 行**；
  `runs/constitution-v1/results.tsv` 共 18 行，同样 0 条 CRASH 行；
- 两个 run 目录下都不存在 `events.jsonl`。

结论：`round-017` 的崩溃尝试在当时**没有被记录为独立事件**，只在后续 REVERT 的
描述里被提及。这是历史数据的真实缺口。本次修复**不创建、不回填、不推测**该历史
事件；只保证今后崩溃与修复重试都能被完整记录。任何声称「历史 50 轮包含 CRASH
事件」的说法都不成立。

## 5. 未改动项与回归测试

生产机 sha256（修复后核对，与本轮之前的冻结值一致）：

| 文件 | sha256 |
| --- | --- |
| `runs/ar1/results.tsv` | `f956c8c3f65d7327ec8f190f6767dfe5849c6bb3d463f41d842f2791898e03ef` |
| `runs/constitution-v1/results.tsv` | `4f34a8fabf1a74ff8eaf455b216c5d40e2825f618a8f480a0dac5c460f826db9` |
| `src/research/train.py` | `bd1bed98ab0407141e0059e361ee1872bd068b4680b4061f3f50a00bea844dcb` |
| `src/research/constitution_train.py` | `53240a285314e1b38b05c9a78711c5e875092c7237572750bfd5e7fe553a71ac` |
| `artifacts/sd-v2-ar1-champion/model.joblib` | `944b325d273d79f7b11b3d4a5e5e07340f99aa973a86e15ead53ca798ae5e74f` |
| `artifacts/sd-v2-ar1-champion/labels.json` | `65d797c4c43a6977453aab98e36c2dab4fb0596df79eb05f6def7aaf2009285d` |
| `artifacts/sd-v2-ar1-champion/registry_entry.json` | `a73830d8e577195a66bf7d0e0b3751aac816533bd804fce36e2904332cbf246f` |

旧 50 轮记录、分数、冠军工件与两个冻结候选源文件均未改动；没有重训练、没有调阈值、
没有写 `results.tsv`、没有解锁历史 holdout。

回归测试（`tests/research/test_prepare.py`）：

- `test_local_crash_keeps_its_evidence_level_and_records_a_linked_retry`：
  LOCAL 行之后的 CRASH 仍为 `LOCAL_EXPERIMENTAL`，重复写入不产生重复，
  修复重试通过 `repair_of` 显式链接且不覆盖崩溃事件；
- `test_keep_revert_crash_all_record_and_retry_after_crash_is_allowed`：
  三种决策都可记录，精确重复是 no-op，崩溃后同 commit 可追加修复决策；
- `test_event_log_is_append_only_and_idempotent`：事件冲突、损坏日志、三种决策事件；
- `test_constitution_record_crash_writes_local_experimental_event`：体质线事件链。

所有实验结论仍限于 `LOCAL_EXPERIMENTAL`；本修复不改变任何模型的有效性判断。
