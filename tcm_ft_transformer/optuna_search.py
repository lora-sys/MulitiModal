"""
Optuna 超参数搜索模块
使用贝叶斯优化搜索最佳超参数组合
"""

import os
import json
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
import torch

from config import DATA_CONFIG, OPTUNA_CONFIG, CV_CONFIG, TRAIN_CONFIG, MODEL_CONFIG, OUTPUT_FILES
from train import train_single_fold


def objective(trial, X, y, n_splits=5, num_epochs=20):
    """
    Optuna 目标函数
    
    Args:
        trial: Optuna trial 对象
        X: 特征矩阵
        y: 标签矩阵
        n_splits: 交叉验证折数
        num_epochs: 每个 trial 的训练轮数（减少以加快搜索）
        
    Returns:
        mean_val_loss: 平均验证损失
    """
    # =====================================================================
    # 定义搜索空间（连续分布，充分利用 TPE 贝叶斯优化）
    # =====================================================================
    model_params = {
         'n_features': DATA_CONFIG['n_features'],
        'n_classes': DATA_CONFIG['n_classes'],
        'd_token': 64,  # 固定
        'n_heads': 4,  # 固定
        'n_layers': trial.suggest_int('n_layers', 2, 4),  # 连续整数搜索
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05),  # 连续搜索，步长 0.05
    }

    train_config = {
        'batch_size': TRAIN_CONFIG['batch_size'],
        'num_epochs': num_epochs,  # 减少轮数以加快搜索
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),  # 对数空间连续搜索
        'weight_decay': TRAIN_CONFIG['weight_decay'],
        'warmup_ratio': TRAIN_CONFIG['warmup_ratio'],
        'grad_clip_max_norm': TRAIN_CONFIG['grad_clip_max_norm'],
        'patience': 3,  # 减少耐心值以加快搜索
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': './checkpoints/optuna'
    }
    
    # =====================================================================
    # 交叉验证
    # =====================================================================
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=CV_CONFIG['random_state'])
    val_losses = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练单个 fold
        _, val_loss = train_single_fold(
            X_train, y_train,
            X_val, y_val,
            model_params=model_params,
            train_config=train_config,
            fold_idx=fold_idx
        )
        
        val_losses.append(val_loss)
        
        # 报告中间结果（用于剪枝）
        intermediate_value = np.mean(val_losses)
        trial.report(intermediate_value, fold_idx)
        
        # 检查是否应该剪枝
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    mean_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)
    
    return mean_val_loss


def run_optuna_search(
    X,
    y,
    n_trials=20,
    n_splits=5,
    num_epochs=20,
    study_name='ft_transformer_optuna',
    storage=None
):
    """
    运行 Optuna 超参数搜索
    
    Args:
        X: 特征矩阵
        y: 标签矩阵
        n_trials: 试验次数
        n_splits: 交叉验证折数
        num_epochs: 每个 trial 的训练轮数
        study_name: 研究名称
        storage: 存储路径（用于持久化）
        
    Returns:
        study: Optuna 研究对象
        best_params: 最佳参数
    """
    print("=" * 60)
    print("Optuna 超参数搜索")
    print("=" * 60)
    print(f"试验次数: {n_trials}")
    print(f"交叉验证折数: {n_splits}")
    print(f"每个 trial 训练轮数: {num_epochs}")
    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)
    
    # 创建采样器和剪枝器
    sampler = TPESampler(seed=CV_CONFIG['random_state'])
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    # 创建研究（如果已存在则加载，否则创建新的）
    study = optuna.create_study(
        study_name=study_name,
        direction=OPTUNA_CONFIG['direction'],
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True
    )

    # 提示用户是否加载了历史试验（只计算已完成的试验）
    n_completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    if n_completed > 0:
        print(f"\n⚠️ 加载已有研究，包含 {n_completed} 个已完成的试验")

        # 如果已有已完成试验数量 >= 目标数量，直接跳过优化
        if n_completed >= n_trials:
            print(f"   已有 {n_completed} 个已完成试验，满足要求")
            print(f"   跳过优化，直接使用已有结果")
            print(f"   如需重新开始，请删除 {storage} 文件")
        else:
            remaining_trials = n_trials - n_completed
            print(f"   本次将再添加 {remaining_trials} 个新试验（目标: {n_trials}）")
            print(f"   如需重新开始，请删除 {storage} 文件")

    # 运行优化（只在需要时运行）
    if n_completed < n_trials:
        remaining_trials = n_trials - n_completed
        study.optimize(
            lambda trial: objective(trial, X, y, n_splits, num_epochs),
            n_trials=remaining_trials,
            n_jobs=1,
            show_progress_bar=True
        )
    else:
        print(f"\n✅ 跳过 Optuna 优化，使用已有的 {n_completed} 个已完成试验结果")

    # 重新计算已完成试验数量
    n_completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

    # 打印结果
    print("\n" + "=" * 60)
    print("搜索完成！")
    print("=" * 60)

    if n_completed > 0:
        print(f"最佳试验: {study.best_trial.number}")
        print(f"最佳验证损失: {study.best_value:.6f}")
        print(f"最佳参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
    else:
        print("没有已完成的试验")

    print("=" * 60)
    
    # 保存结果
    results = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }
    
    results_path = os.path.join(os.path.dirname(OUTPUT_FILES["best_model"]), 'optuna_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n已保存搜索结果: {results_path}")
    
    return study, study.best_params


def visualize_optuna_results(study, save_path=None):
    """
    可视化 Optuna 搜索结果
    
    Args:
        study: Optuna 研究对象
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 优化历史
    ax = axes[0, 0]
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in trials]
    ax.plot(range(len(values)), values, 'o-')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Optimization History')
    ax.grid(True)
    
    # 2. 参数重要性
    ax = axes[0, 1]
    importance = optuna.importance.get_param_importances(study)
    params = list(importance.keys())
    values = list(importance.values())
    ax.barh(params, values)
    ax.set_xlabel('Importance')
    ax.set_title('Parameter Importance')
    
    # 3. 学习率 vs 损失
    ax = axes[1, 0]
    lrs = [t.params['learning_rate'] for t in trials]
    losses = [t.value for t in trials]
    ax.scatter(lrs, losses, alpha=0.6)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Learning Rate vs Loss')
    ax.set_xscale('log')
    ax.grid(True)
    
    # 4. Dropout vs 损失
    ax = axes[1, 1]
    dropouts = [t.params['dropout'] for t in trials]
    ax.scatter(dropouts, losses, alpha=0.6)
    ax.set_xlabel('Dropout')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Dropout vs Loss')
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存可视化结果: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # 测试 Optuna 搜索
    print("测试 Optuna 搜索...")
    
    # 创建虚拟数据
    X = np.random.randn(1000, 4).astype(np.float32)
    y = np.random.dirichlet(np.ones(9), size=1000).astype(np.float32)
    
    # 运行搜索（减少试验次数以加快测试）
    study, best_params = run_optuna_search(
        X, y,
        n_trials=5,
        n_splits=3,
        num_epochs=3
    )
    
    # 可视化结果
    visualize_optuna_results(
        study,
        save_path='./checkpoints/optuna_results.png'
    )