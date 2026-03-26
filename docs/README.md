# 项目文档

本目录包含多模态按摩椅舒适度预测系统的完整文档。

---

## 📁 文档结构

```
docs/
├── README.md                    # 本文档
├── guides/                      # 指南类文档
│   ├── BEST_MODEL_GUIDE.md      # 最佳模型选择指南
│   ├── COMPLETE_GUIDE.md        # 完整指南
│   └── MODEL_UPGRADE_PATH.md    # 模型升级路径
├── experiments/                 # 实验文档
│   ├── EXPERIMENT_LOG.md        # 实验日志
│   ├── EXPERIMENT_REPORT.md     # 实验报告
│   └── FINAL_RESULTS_SUMMARY.md # 最终结果汇总
├── hyperopt/                    # 超参数优化文档
│   ├── README.md                # 超参数优化使用说明
│   └── HYPEROPT_PLAN.md         # 超参数优化计划
├── validation/                  # 验证文档
│   └── REAL_DATA_VALIDATION_PLAN.md # 真实数据验证计划
└── internal/                    # 内部文档
    ├── README.md                # 实验模块说明
    ├── enhanced_dataset_info.md # 增强数据集信息
    ├── frameworktest_target.md  # 框架测试目标
    └── transform_learning_target.md # 迁移学习目标
```

---

## 📖 文档说明

### guides/ - 指南类文档

这些文档提供项目的使用指南和最佳实践：

- **BEST_MODEL_GUIDE.md**: 如何选择最适合的模型
- **COMPLETE_GUIDE.md**: 完整的项目指南
- **MODEL_UPGRADE_PATH.md**: 模型升级和改进路径

### experiments/ - 实验文档

记录所有实验的详细信息和结果：

- **EXPERIMENT_LOG.md**: 15个阶段的实验日志
- **EXPERIMENT_REPORT.md**: 详细的实验报告
- **FINAL_RESULTS_SUMMARY.md**: 最终实验结果汇总

### hyperopt/ - 超参数优化文档

超参数优化框架的完整文档：

- **README.md**: 超参数优化使用说明
- **HYPEROPT_PLAN.md**: 超参数优化计划和架构设计

### validation/ - 验证文档

真实数据验证相关文档：

- **REAL_DATA_VALIDATION_PLAN.md**: 真实数据验证计划

### internal/ - 内部文档

项目内部的开发文档：

- **README.md**: 实验模块说明
- **enhanced_dataset_info.md**: 增强数据集信息
- **frameworktest_target.md**: 框架测试目标
- **transform_learning_target.md**: 迁移学习目标

---

## 🎯 快速导航

### 新手入门
1. 阅读 `guides/COMPLETE_GUIDE.md` 了解项目全貌
2. 阅读 `README.md` 了解项目概述
3. 查看 `experiments/FINAL_RESULTS_SUMMARY.md` 了解最终结果

### 选择模型
1. 阅读 `guides/BEST_MODEL_GUIDE.md` 了解模型选择建议
2. 查看 `experiments/EXPERIMENT_REPORT.md` 了解模型性能对比

### 运行实验
1. 阅读 `experiments/EXPERIMENT_LOG.md` 了解实验流程
2. 参考 `internal/README.md` 了解实验模块

### 超参数优化
1. 阅读 `hyperopt/README.md` 了解如何使用
2. 查看 `hyperopt/HYPEROPT_PLAN.md` 了解设计细节

---

## 📊 项目成果

### 性能指标

**分类任务**：
- 最佳模型：baseline_b
- 准确率：98.88%
- F1分数：0.9892

**回归任务**：
- 最佳模型：baseline_c
- MAE：3.57
- R²：0.92
- Pearson：0.96

### 完成阶段

15个实验阶段全部完成，详见 `experiments/EXPERIMENT_LOG.md`

---

## 🔗 相关资源

- **项目仓库**: `https://github.com/lora-sys/MulitiModal`
- **代码位置**: `./`
- **实验结果**: `./experiment/results`

---

## 📝 文档更新

- **创建日期**: 2026-03-26
- **最后更新**: 2026-03-26
- **版本**: 1.0.0

---

## 💡 维护说明

### 添加新文档

1. 根据文档类型选择合适的子目录
2. 使用清晰的命名规范
3. 在本文档中添加新文档的说明
4. 更新相关的索引和引用

### 更新现有文档

1. 确保文档与代码同步
2. 更新日期和版本信息
3. 记录重要的变更
4. 保持文档的一致性

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- **项目Issue**: 在GitHub仓库提交Issue
- **代码审查**: 提交Pull Request
- **文档改进**: 直接编辑文档并提交

---

**文档版本**: 1.0.0
**最后更新**: 2026-03-26