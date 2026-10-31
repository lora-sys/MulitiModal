# #50 交付基准审查 Eval（v1，2026-10-31 冻结）

来源：两轮独立零上下文对抗审查（Antigravity / Claude-Opus-4.6-Thinking 与
Pi），经主流程逐条核验后确认为真实问题的条目。此清单是后续所有审查轮次的
**固定基准**：每一轮修复后必须全部 PASS 才允许收口；新一轮审查发现的新问题
追加为新条目（E26+），不得重写既有条目。

可执行检查统一由 `scripts/review_eval.py` 执行（本机或开发机）：
`PYTHONPATH=src python scripts/review_eval.py [--summary artifacts/tcm-50/summary.json]`
退出码非 0 = 存在回归。单元级检查同时以 pytest 用例固化。

| ID | 来源 | 问题（一句话） | 修复后行为 | 验证 |
| --- | --- | --- | --- | --- |
| E01 | Pi R-01 | QC 敏感性对照用了重新 shuffle 的划分（~13% 受试者换分区，且 273 在 holdout） | `exclude_subjects_from_manifest` 保持原划分仅剔除 QC 行，scaler 在缩减 train 折内重拟合 | review_eval `e01_qc_split_preserved` + pytest |
| E02 | Pi R-02 | macro-F1 分母随模型漂移（不预测稀有维免罚） | macro-F1 固定对全部可观测维度平均，未定义 F1 计 0，`n_undefined_f1` 一并报告 | review_eval `e02_macro_f1_fixed_denominator` |
| E03 | Pi R-03 | 基线没有 subject MAE，与选择指标脱节 | 三个基线 + 神经模型均报告 subject_mae（同一聚合口径） | review_eval `e03_baselines_report_subject_mae` |
| E04 | Pi R-04 | 可用性指示位只给神经模型，基线拿到"平均人"行 | 基线设计矩阵附带每模态可用性位（bmi_only 附 body 位） | review_eval `e04_availability_bits_in_baselines` |
| E05 | Pi R-05 | 全文无模态可用率，full 可能实际是 body-only | summary.json/report.md 按分区×模态报告 n_available/n_rows | review_eval `--summary` 检查 `availability` 字段 |
| E06 | Pi R-06 + Anti R-19 | `_fixed_training` 在模型构建后才播种，固定配置任务不可独立复现 | 构建前 `torch.manual_seed(0)`；两次训练权重逐位相同 | review_eval `e06_fixed_training_deterministic` |
| E07 | Pi R-08 | 图表把 None 画成 0（回归 MAE 的 None=完美分） | None → NaN（缺失柱不渲染），图题注明"缺失≠0" | review_eval `e07_charts_treat_none_as_missing` |
| E08 | Pi R-09 + Anti R-05 | `compute_subject_level_metrics` 空观测返回 0.0 且多行受试者重复计数 | 空观测返回 None；按受试者池化聚合，与 TrustedEvaluator 同数 | review_eval `e08_subject_metrics_fail_closed` + pytest |
| E09 | Pi R-10 | scaler (0,1) 回退让退化列按原始量纲进网络 | 退化列在 manifest 中标记，transform 恒置 0 | review_eval `e09_degenerate_columns_zeroed` |
| E10 | Pi R-11 | ≥40 切点随题数漂移、TENDENCY 并入负类未声明 | report.md 明示有效切点与 TENDENCY 语义 | review_eval `--summary`/`--report` 文本检查 |
| E11 | Pi R-12 | 回归交付无平凡参照、聚合口径未声明 | 逐维 train-mean 预测器参照 + `aggregation` 字段 | review_eval `e11_regression_has_reference` |
| E12 | Pi R-13 | QC 排除硬编码 273，score_mask/fingerprint_anomalies 无人消费 | QC 行从 `provenance["fingerprint_anomalies"]` 派生；`derive_classification_targets` 校验 score_mask 一致性 | review_eval `e12_qc_rows_derived_from_provenance` |
| E13 | Pi R-14 | support 图用 dev、F1 图用 holdout，互相矛盾 | 两图固定 holdout | review_eval `e13_support_chart_partition` |
| E14 | Pi R-15 | manifest 不去重，重复受试者静默泄漏 | `build_real_manifest` 拒绝重复 subject_index | review_eval `e14_duplicate_subjects_rejected` |
| E15 | Pi R-16 | 报告格式化假定非 None，诱导未来把 None 补成 0 | 全部格式化走 `None → N/A` 分支，空轮次不崩溃 | review_eval `e15_report_tolerates_none` |
| E16 | Pi R-17 | summary.json 缺复算参数（阈值/聚合/权重哈希/种子） | summary 含 `metric_params` + `champion_provenance` | review_eval `--summary` 检查 |
| E17 | Anti R-01 | 训练无 epoch 级 shuffle，batch 恒按受试者序 | train provider 确定性 epoch shuffle（manifest 哈希+轮次播种），dev/holdout 有序 | review_eval `e17_train_epoch_shuffle` |
| E18 | Anti R-06/R-15 | 0.0 primary 可成永久冠军（与 E08 同源） | E08 修复后评估器不可能产出 0.0；`_primary_value` 仍拒绝 None/NaN | 由 E08 覆盖 |
| E19 | Anti R-14 | `trusted_score` 对真实 checkpoint 硬编码合成评估 + smoke_only=True | 拒绝 smoke_only=False 的 checkpoint | review_eval `e19_trusted_score_refuses_real` |
| E20 | Anti R-07/R-25 | `torch.load(weights_only=False)` pickle 执行风险；checkpoint_summary 不验权重哈希 | 先验权重 SHA-256，state 用 `weights_only=True` 反序列化 | pytest 全套（checkpoint 往返） |
| E21 | Anti R-08 | `PYTHON*` 前缀放行 PYTHONSTARTUP（启动代码注入） | 白名单精确匹配 8 个变量 | review_eval `e21_env_whitelist_exact` |
| E22 | Anti R-12 | fractions 造成空 train 分区时 IndexError | 各分区非空校验，报错信息明确 | review_eval `e22_empty_partition_rejected` |
| E23 | Anti R-16 | `apply_config_patch` 静默丢弃白名单外键 | docstring 声明选择性复制语义（调用方负责补 data/out_dir） | 文档审查（已落地） |
| E24 | Pi R-07 | 消融与冠军配置/轮数不一致 + winner's curse 未声明 | 消融沿用冠军 model/trainer 配置仅换特征集；报告明示 winner's curse 与单 seed | review_eval `e24_ablation_uses_champion_config` |
| E25 | Pi R-11/R-17 补充 | holdout 多次解锁、单 seed、n<50 未声明 | report.md 限制小节逐项披露（v2 起） | review_eval `--report` 文本检查 |

**已核验非问题**（保留记录，防止重复误报）：
- `trainer.batch_size` 补丁经 `BATCH_SIZE_KEY` 专门路由，实际生效（Anti 审查疑点）。
- `_nan_aware`/`_fit_scaler` 用总体标准差（ddof=0）与 sklearn StandardScaler 一致（Pi R-02 的"inconsistent with sklearn"说法有误）。
- 特征列 0–62 与计分列 63–70 不相交，无特征/标签泄漏；holdout 仅报告期解锁。
- 每候选独立子进程 + 冻结种子，无 RNG 交叉污染。

## 第二轮审查追加条目（Pi round-2，R-01–R-12 → E26–E36）

| ID | 来源 | 问题（一句话） | 修复后行为 | 验证 |
| --- | --- | --- | --- | --- |
| E26 | Pi R-05 | `core/batch.py` availability 文档极性写反（写"unusable"，实现 True=可用） | 文档改为 "True = fully observed and usable"；与 encoder/real_data 写入语义一致 | review_eval `e26_availability_doc_polarity` |
| E27 | Pi R-06 | checkpoint 存 last-epoch 权重却标注 best_eval_loss | **修订（有意契约变更）**：v3 A/B 实测 best-eval-loss 恢复在不平衡阈值化目标上选中欠拟合基率轮（dev F1 0.03 vs last-epoch 0.27），故默认 last-epoch 并以 `weight_epoch` 字段显式声明；`restore_best=True` 为 opt-in 机制且确实恢复 best 权重 | review_eval `e27_trainer_restore_semantics` + pytest |
| E28 | Pi R-01 | 报告路径不校验 checkpoint↔manifest 绑定（worker 路径有、报告路径没有） | `assert_champion_matches_manifest`：smoke_only / manifest_sha256 / dataset_sha256 三重 fail-closed；champion_provenance 落盘绑定哈希 | review_eval `e28_report_binds_champion_to_manifest` + `--summary` 检查 |
| E29 | Pi R-02/R-11 | 派生 manifest（回归/deployable）依赖默认 fractions 巧合一致 | `derived_manifest` 传入冻结 fractions + exclusions 并断言分区成员一致；`build_real_manifest` 改为全量洗牌后剔除，剔除不再改变其余受试者分区 | review_eval `e29_derived_manifests_mirror_frozen_split` + pytest |
| E30 | Pi R-09 | 所有 real manifest 共用 `tcm-input-real-v1` 版本串，无区分度 | 版本串编入 modality_set 与各模态维度（`tcm-input-real-v1:<set>:<name>=<dim>,…`） | review_eval `e30_input_spec_version_discriminates` + pytest |
| E31 | Pi R-04 | 报告硬编码 0.29/0.345/0.000/0.275/"6 个候选"/"<50 人" | 全部由 summary 插值（正类率取 Dummy holdout 逐维平均）；summary 增加 records_total / subjects_in_manifest / report_version | review_eval `e31_report_interpolates_no_hardcoded_metrics` + `--summary` 检查 |
| E32 | Pi R-10 | baselines 自定义 CANONICAL_DIMS 与 labels.standards 重复定义 | 直接 `from labels.standards import CANONICAL_ORDER as CANONICAL_DIMS`（同一对象） | review_eval `e32_canonical_dims_single_source` |
| E33 | Pi R-03 | QC 敏感性对照分母不同（41 人 dev vs 40 人缩减 dev） | 冠军模型也在缩减 dev 上评估（`dev_subject_mae_champion_on_reduced`），同分母对照 | review_eval `e33_qc_sensitivity_same_denominator` |
| E34 | Pi R-08 | scaler 逐列有限值拟合的语义与行数未入档 | manifest 记录 `scaler_policy` + `scaler_fit_rows` | review_eval `e34_scaler_semantics_recorded` |
| E35 | Pi R-12 | rounds 图 x 轴用 1..n 位置序，REVERT 轮压缩坐标 | x 用 run_id 解析的真实轮次号（tcm-0003→3），刻度标 run_id | review_eval `e35_chart_rounds_real_axis` |
| E36 | Pi R-07/R-12 | 受试者身份=行序属未验证假设；holdout 解锁未成枚举记录 | summary/report 声明 subject_identity=UNVERIFIED；报告枚举 holdout 解锁记录（v1/v2/本次）并要求下次换划分；QC 判定与 holdout 无关的声明补入 | review_eval `e36_subject_identity_declared` + `--summary` 检查 |
| E37 | GLM round-3 | QC 对照把冠军喂给 QC manifest 重拟合的 scaler（输入契约错配）；QC 派生 manifest 的 `scaler_fit_rows` 过期 | 冠军在缩减 dev 上用**自己的** manifest（训练时 scaler/spec）评估（`RealBatchProvider(subject_ids=…)` 限定分区子集，跨分区拒绝）；`exclude_subjects_from_manifest` 重拟合后同步更新 `scaler_fit_rows` | review_eval `e37_champion_evaluated_under_own_scaler` + pytest |

## 第四轮审查追加条目（Pi round-4，R-01–R-05 → E38–E42）

| 编号 | 来源 | 发现 | 冻结修复 | 验证 |
| --- | --- | --- | --- | --- |
| E38 | Pi R-01 | run manifest 无完整性保护：连续性检查只读 entry 0，条目编辑/删除/重排静默改变冠军与预算 | 每条 append 携带 `prev_entry_sha256`+`entry_sha256` 链，`load()` 验链；编辑/中间删除/重排 → RuntimeError。已知局限（尾部截断、整文件无链重写）不可检测——真隔离是 #45 AGENT_ISOLATION 验证，已在 docstring 声明 | review_eval `e38_manifest_chain_tamper_evidence` + pytest |
| E39 | Pi R-02 | `DeterministicStubProvider` 用内部计数器，resume 时 canned plan 从头重放已试候选 | plan 位置改由 harness 传入的 `round_index`（=已记录轮数）索引，resume 续接不重放 | review_eval `e39_resume_plan_alignment` + pytest |
| E40 | Pi R-03 | `apply_config_patch` 在静态检查 try 之外；train `_worker` JSON 解码错误未捕获；`provider.propose` 异常直接杀死循环——三者都绕过 fail-to-REVERT | apply_config_patch 移入 static-check try；train 调用点补 `json.JSONDecodeError`；`run()` 捕获 provider ValueError 并记录 `agent_error` REVERT 条目 | review_eval `e40_fail_to_revert_round_paths` + pytest |
| E41 | Pi R-04 | `_worker_env` 保留宿主 `PYTHONPATH`，污染路径可在"隔离"子进程内遮蔽第三方导入 | worker `PYTHONPATH` 一律设为仓库 `src/`，不继承宿主值 | review_eval `e41_worker_env_drops_inherited_pythonpath` + pytest |
| E42 | Pi R-05 | real 训练把 dev 接成 Trainer eval provider；若未来设置 `early_stop_patience`/`restore_best` 会让 dev 参与拟合（与选择双重使用） | `_train_real` fail-closed：这两个 trainer 键非 None 即拒绝（当前行为不变，latent 泄漏关闭） | review_eval `e42_real_training_forbids_dev_steering` + pytest |

Round-4 裁决：Antigravity opus-4-6-thinking 因配额耗尽未产出；Pi GLM-5.3-Flash 确认循环核心不变量成立（dev-only 选择、作用域校验、强制种子、holdout 不可达、无 patch 逃逸），R-01 为唯一高严重度缺陷，已随 E38–E42 冻结修复。

## 第五轮审查追加条目（Pi round-5，R-01–R-07）

| 编号 | 来源 | 发现 | 冻结修复 | 验证 |
| --- | --- | --- | --- | --- |
| E43 | Pi round-5 R-01 | "legacy" 形状（无 `entry_sha256`）的行在任意位置都被跳过验链——链后伪造/插入未链行可静默通过并被 `champion()` 采纳 | 只允许**开头** legacy 行；任何已链行之后出现未链行即 RuntimeError（"unchained entry"） | review_eval `e38`（forged line 用例）+ pytest |
| E44 | Pi round-5 R-02 | 无密钥哈希链挡不住能重算整条链的写手；E38 原表述夸大覆盖面 | 声明改诚实：仅防意外/朴素就地编辑；对确定写手的真控制是 #45 AGENT_ISOLATION OS/container 验证（manifest docstring 已同步） | review_eval `e38` 注释 + docstring |
| E45 | Pi round-5 R-03 | `run()` 只捕 `ValueError`——后端超时/连接错误等直接杀死循环且不留 manifest 记录 | `except Exception` 记录 `agent_error`（含异常类型名） | review_eval `e40`（TimeoutError 用例）+ pytest |
| E46 | Pi round-5 R-05 | score 成功后 `_artifact_sha256` 在守卫之外，weights.pt 丢失 → OSError 裸崩 | 哈希计算纳入守卫 → `score_failed` REVERT 条目 | review_eval `e40`（hasher 替换用例） |
| E47 | Pi round-5 R-06 | `run()` 摘要的 `decisions` 漏掉 `agent_error` 轮（与 `rounds_completed` 不一致） | agent_error 条目同时进入 `decisions` | review_eval `e40` + pytest |
| E48 | Pi round-5 R-07 | worker rc=0 且 stdout 为空 → `splitlines()[-1]` IndexError，不在任何捕获列表 | 空输出显式 RuntimeError（已被捕获）→ REVERT | pytest（`_worker` 路径经 e40/既有 crash 测试覆盖） |
| E49 | Pi round-5 R-04 | e39–e42 以源码 grep 钉住修复，语义回归可通过 | e39/e40/e41/e42 全部改写为行为断言（真调用 provider/harness/_worker_env/train_candidate） | review_eval 本身 |

Round-5 裁决：**NO**——#50 数字不被推翻（实际运行的冠军选择由进程内可信评分驱动，报告端有 `assert_champion_matches_manifest` 三重哈希绑定）；但 E38 的防篡改声明须按 E44 限定后方可作为 #50 provenance 引用。

## 第六轮审查追加条目（Pi round-6，manifest 健壮性 → E50–E52）

Round-6 delegate 在产出最终报告前再次被 turn-limit 截断；以下发现从其 activity 时间线的完整分析中收割并经本地核实。

| 编号 | 来源 | 发现 | 冻结修复 | 验证 |
| --- | --- | --- | --- | --- |
| E50 | round-6 分析 | `append()` 接受已带 `entry_sha256` 的条目并原样落盘（不接链）→ 下一次 `load()` 验链必败，store 永久毒化 | `append()` 显式拒绝预哈希条目（ValueError） | pytest `test_manifest_chain_round6_hardening` |
| E51 | round-6 分析 | 撕裂写（进程在 append 中途被杀，行尾缺 `\n`）后，下一次 append 把两个 JSON 对象拼成一行 → store 永久不可解析 | `append()` 落盘前检查文件尾字节，缺 `\n` 先补写再接链 | pytest 同上（torn-write 用例） |
| E52 | round-6 分析 | 只有 `entry_sha256` 没有 `prev_entry_sha256` 的行抛 KeyError，与"完整性失败 = RuntimeError"的契约不一致 | `_verify_chain` 改用 `entry.get(...)`，统一 RuntimeError（"prev_entry_sha256 mismatch"） | pytest 同上 |

Round-6 结论：三条均为 manifest 健壮性加固（fail-closed 语义下的一致性与可恢复性），不涉及训练数值路径，#50 数字与 provenance 不受影响；审查循环至此连续一轮无高严重度缺陷，round-4/5 的 impactful findings 均已关闭并冻结进 eval。
