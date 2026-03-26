"""
超参数优化示例脚本

展示如何使用Optuna + Hyperband进行超参数优化
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime
from typing import Optional
import json

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from experiment.utils.logger import setup_logging, get_logger
from experiment.hyperopt.config import (
    HyperoptConfig,
    create_classification_config,
    create_regression_config,
)
from experiment.hyperopt.objective import create_objective_function
from experiment.dataset.unified_source import UnifiedNPZDataSource
from experiment.dataset.unified_dataset import UnifiedMultimodalDataset
from torch.utils.data import DataLoader, random_split

import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler


def prepare_data(config: HyperoptConfig) -> tuple:
    """准备数据

    Args:
        config: 超参数优化配置

    Returns:
        tuple: (train_loader, val_loader, device)
    """
    logger = get_logger(__name__)

    # 选择数据集
    # 优先使用配置中的dataset_path，否则根据task_type选择默认路径
    if hasattr(config, 'dataset_path') and config.dataset_path:
        dataset_path = config.dataset_path
    elif config.task_type == "classification":
        dataset_path = "experiment/model/unified_dataset_expanded.npz"
    else:
        dataset_path = "experiment/model/unified_dataset_regression.npz"

    # 解析为绝对路径
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(project_root, dataset_path)

    logger.info(f"加载数据集: {dataset_path}")

    # 加载数据
    source = UnifiedNPZDataSource(dataset_path)
    if not source.initialize():
        raise RuntimeError("数据源初始化失败")

    # 根据任务类型选择数据集
    if config.task_type == "classification":
        dataset = UnifiedMultimodalDataset(source)
    else:
        # 回归任务使用简单包装器，保持float目标值
        class RegressionDataset:
            def __init__(self, source):
                self.source = source
                self._data = source._data
                self._sample_list = source._sample_list

            def __len__(self):
                return len(self._sample_list)

            def __getitem__(self, idx):
                sample = self.source.load_sample(idx)
                return {
                    'dynamic': torch.tensor(sample.dynamic, dtype=torch.float32),
                    'static_basic': torch.tensor(sample.static_basic, dtype=torch.float32),
                    'static_scores': torch.tensor(sample.static_scores, dtype=torch.float32),
                    'constitution': torch.tensor(sample.constitution, dtype=torch.long),
                    'scores': torch.tensor(sample.scores, dtype=torch.float32),  # 保持float，不-1
                }

        dataset = RegressionDataset(source)

    # 划分数据集
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    logger.info(f"数据集划分 - 训练集: {train_size}, 验证集: {val_size}")

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 默认值，会被超参数覆盖
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    return train_loader, val_loader, device


def create_pruner(config: HyperoptConfig) -> Optional[optuna.pruners.BasePruner]:
    """创建剪枝器

    Args:
        config: 超参数优化配置

    Returns:
        Optional: 剪枝器实例
    """
    if not config.pruning.enabled:
        return None

    pruner_type = config.pruning.pruner_type.lower()

    if pruner_type == "median":
        pruner = MedianPruner(
            n_startup_trials=config.pruning.n_startup_trials,
            n_warmup_steps=config.pruning.n_warmup_steps,
            interval_steps=config.pruning.interval_steps
        )
    elif pruner_type == "successive_halving":
        pruner = SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=config.pruning.reduction_factor,
            min_early_stopping_rate=config.pruning.min_early_stopping_rate
        )
    else:
        pruner = MedianPruner()

    return pruner


def run_hyperopt(
    model_type: str = "baseline_c",
    task_type: str = "classification",
    n_trials: int = 50,
    timeout: Optional[int] = 3600,
    output_dir: str = "experiment/results/hyperopt"
) -> None:
    """运行超参数优化

    Args:
        model_type: 模型类型
        task_type: 任务类型
        n_trials: 试验次数
        timeout: 超时时间（秒）
        output_dir: 输出目录
    """
    # 初始化日志
    setup_logging(log_dir=f"{output_dir}/logs", level="INFO")
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("超参数优化开始")
    logger.info("=" * 60)
    logger.info(f"模型类型: {model_type}")
    logger.info(f"任务类型: {task_type}")
    logger.info(f"试验次数: {n_trials}")
    logger.info(f"超时时间: {timeout}秒")

    # 创建配置
    if task_type == "classification":
        config = create_classification_config(model_type)
    else:
        config = create_regression_config(model_type)

    config.n_trials = n_trials
    config.timeout = timeout
    config.output_dir = output_dir
    config.study_name = f"{model_type}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 准备数据
    logger.info("准备数据...")
    train_loader, val_loader, device = prepare_data(config)

    # 创建objective函数
    logger.info("创建objective函数...")
    objective = create_objective_function(config, train_loader, val_loader, device)

    # 创建Study
    logger.info("创建Optuna Study...")
    storage_path = os.path.join(output_dir, config.study_name, "db.sqlite3")
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)

    sampler = TPESampler(seed=config.seed)
    pruner = create_pruner(config)

    study = optuna.create_study(
        direction=config.get_direction(),
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{storage_path}",
        study_name=config.study_name,
        load_if_exists=False
    )

    # 运行优化
    logger.info("开始优化...")
    start_time = datetime.now()

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        show_progress_bar=True,
        catch=(Exception,),
        callbacks=None
    )

    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    # 输出结果
    logger.info("=" * 60)
    logger.info("优化完成")
    logger.info("=" * 60)
    logger.info(f"总耗时: {elapsed_time:.2f}秒")
    logger.info(f"完成试验数: {len(study.trials)}")
    logger.info(f"剪枝试验数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

    # 检查是否有成功的试验
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        logger.warning("没有成功的试验，无法输出最佳参数")
        return None
    logger.info(f"失败试验数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    best_trial = study.best_trial
    logger.info(f"最优指标: {best_trial.value:.4f}")
    logger.info(f"最优超参数:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")

    # 保存结果
    results_path = os.path.join(output_dir, config.study_name, "best_params.json")
    results = {
        "model_type": model_type,
        "task_type": task_type,
        "best_value": float(best_trial.value),
        "best_params": best_trial.params,
        "n_trials": len(study.trials),
        "elapsed_time": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"结果已保存到: {results_path}")

    # 可视化（可选）
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt

        # 优化历史
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(output_dir, config.study_name, "optimization_history.png"))

        # 超参数重要性
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(output_dir, config.study_name, "param_importances.png"))

        logger.info("可视化图表已生成")
    except Exception as e:
        logger.warning(f"生成可视化图表失败: {e}")

    logger.info("超参数优化完成！")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="超参数优化示例")
    parser.add_argument("--model", type=str, default="baseline_c", choices=["baseline_a", "baseline_b", "baseline_c"], help="模型类型")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "regression"], help="任务类型")
    parser.add_argument("--n_trials", type=int, default=50, help="试验次数")
    parser.add_argument("--timeout", type=int, default=3600, help="超时时间（秒）")
    parser.add_argument("--output_dir", type=str, default="experiment/results/hyperopt", help="输出目录")

    args = parser.parse_args()

    run_hyperopt(
        model_type=args.model,
        task_type=args.task,
        n_trials=args.n_trials,
        timeout=args.timeout,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()