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
