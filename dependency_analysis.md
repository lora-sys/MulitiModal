# 依赖分析报告
生成时间: 2026-08-15

## 标准库 (已安装)
- argparse, csv, dataclasses, datetime, gc, json, logging
- math, os, pathlib, pickle, random, re, shutil, subprocess
- sys, time, typing, warnings
- __future__, collections, itertools, functools

## 核心科学计算
- numpy>=1.24
- scipy>=1.11
- pandas>=2.0
- matplotlib>=3.7
- seaborn>=0.12

## 深度学习框架
- torch>=2.0
- torchvision (可选,用于数据增强)
- torchaudio (可选,用于音频处理)

## 机器学习工具
- scikit-learn>=1.3
- optuna>=3.4

## 演示和可视化
- gradio>=4.0
- tqdm>=4.65

## 数据处理
- pyyaml>=6.0
- h5py (用于 WESAD 数据读取)

## 其他工具
- pillow (图像处理)

## 模块依赖图

### 当前工业代码 (目标)
```
demo/ (Gradio演示)
  ├── 核心: numpy, torch, gradio
  ├── 路径: pathlib, sys, warnings
  └── 历史: ft_transformer, models.encoders

src/ (尚未实现)
  └── 待定

representation/ (尚未实现)
  └── 待定

massage_decision/ (尚未实现)
  └── 待定
```

### 历史代码 (legacy_research/)
```
oplri 模型:
  ├── 核心: torch, numpy, sklearn
  ├── 实验: optuna, pandas, matplotlib, seaborn
  └── 配置: yaml, pathlib, logging

tcm_ft_transformer:
  ├── 核心: torch, numpy
  ├── 可视化: matplotlib, seaborn
  └── 工具: pandas, tqdm

experiments/early_multimodal:
  ├── 数据: pandas, numpy, sklearn
  ├── 实验: optuna, matplotlib, seaborn
  └── 工具: pathlib, pickle, tqdm
```

### 实验归档 (experiment/archive/) - 已废弃
```
deprecated_train: torch, numpy, sklearn, pandas
deprecated_models: torch, numpy, sklearn
deprecated_generate: torch, numpy, scipy, pandas, seaborn
deprecated_utils: torch, numpy, neurokit2, yaml
```

## 依赖冲突风险
- **Python 3.14**: 某些旧包可能不兼容,特别是 neurokit2
- **numpy 2.x**: seaborn<0.13 可能需要 numpy<2
- **torch + cuda**: 需要 GPU 支持确认

## 分离策略

### 工业代码 (industrial/)
- 最小化依赖: numpy, torch, gradio
- 待开发: representation, massage_decision

### 历史代码 (legacy/)
- 实验工具: optuna, pandas, matplotlib, seaborn
- 数据处理: scipy, sklearn

### 废弃代码 (archived/)
- 已废弃,不维护
