"""
可视化工具
生成训练历史、交叉验证结果等图表
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from config import OUTPUT_FILES, CONSTITUTION_NAMES


def plot_training_history(history, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. 损失曲线
    ax = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (KL Divergence)', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. 学习率曲线
    ax = axes[1]
    ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存训练历史: {save_path}")
    
    plt.close()


def plot_cv_comparison(cv_results, save_path=None):
    """
    绘制交叉验证结果对比
    
    Args:
        cv_results: 交叉验证结果字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. 各折验证损失
    ax = axes[0]
    folds = range(1, len(cv_results['fold_val_losses']) + 1)
    ax.bar(folds, cv_results['fold_val_losses'], color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(y=cv_results['mean_val_loss'], color='red', linestyle='--', linewidth=2, label=f'Mean: {cv_results["mean_val_loss"]:.6f}')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Cross-Validation Results by Fold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. 训练损失 vs 验证损失（各折）
    ax = axes[1]
    train_losses = cv_results['fold_train_losses']
    val_losses = cv_results['fold_val_losses']
    
    x = np.arange(len(folds))
    width = 0.35
    
    ax.bar(x - width/2, train_losses, width, label='Train Loss', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, val_losses, width, label='Val Loss', color='coral', alpha=0.7)
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Train vs Val Loss by Fold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存交叉验证对比: {save_path}")
    
    plt.close()


def plot_prediction_distribution(y_true, y_pred, save_path=None):
    """
    绘制预测分布对比
    
    Args:
        y_true: 真实标签 (N, n_classes)
        y_pred: 预测标签 (N, n_classes)
        save_path: 保存路径
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(9):
        ax = axes[i]
        
        # 真实分布
        ax.hist(y_true[:, i], bins=30, alpha=0.5, label='True', color='blue', density=True)
        
        # 预测分布
        ax.hist(y_pred[:, i], bins=30, alpha=0.5, label='Pred', color='red', density=True)
        
        ax.set_xlabel('Probability', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{CONSTITUTION_NAMES[i]}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存预测分布: {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签 (N, n_classes)
        y_pred: 预测标签 (N, n_classes)
        save_path: 保存路径
    """
    from sklearn.metrics import confusion_matrix
    
    # 获取预测类别
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    # 检查是否所有预测都是同一类别
    unique_pred_labels = np.unique(y_pred_labels)
    if len(unique_pred_labels) == 1:
        print(f"⚠️  警告：所有预测都是同一类别 '{CONSTITUTION_NAMES[unique_pred_labels[0]]}'")
        print(f"   这可能表明模型过拟合、测试数据单一，或者模型存在问题。")
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 动态调整刻度和标签
    # 只显示实际出现的类别
    unique_labels_true = np.unique(y_true_labels)
    unique_labels_pred = np.unique(y_pred_labels)
    all_unique_labels = np.unique(np.concatenate([unique_labels_true, unique_labels_pred]))
    
    # 如果所有样本都是同一类别，使用该类别的标签
    if len(all_unique_labels) == 1:
        tick_labels = [CONSTITUTION_NAMES[all_unique_labels[0]]]
        ax.set(xticks=[0],
               yticks=[0],
               xticklabels=tick_labels,
               yticklabels=tick_labels,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
    else:
        # 正常情况：显示完整的混淆矩阵
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=CONSTITUTION_NAMES,
               yticklabels=CONSTITUTION_NAMES,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
    
    # 旋转 x 轴标签（只有多个类别时才旋转）
    if len(all_unique_labels) > 1:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存混淆矩阵: {save_path}")
    
    plt.close()


def save_cv_results(cv_results, save_path=None):
    """
    保存交叉验证结果为 JSON
    
    Args:
        cv_results: 交叉验证结果字典
        save_path: 保存路径
    """
    if save_path is None:
        save_path = OUTPUT_FILES["cv_results"]
    
    # 转换 numpy 类型为 Python 类型
    results_serializable = {
        'mean_val_loss': float(cv_results['mean_val_loss']),
        'std_val_loss': float(cv_results['std_val_loss']),
        'mean_train_loss': float(cv_results['mean_train_loss']),
        'std_train_loss': float(cv_results['std_train_loss']),
        'fold_val_losses': [float(x) for x in cv_results['fold_val_losses']],
        'fold_train_losses': [float(x) for x in cv_results['fold_train_losses']],
        'overfit_gaps': [float(x) for x in cv_results['overfit_gaps']],
        'mean_overfit_gap': float(cv_results['mean_overfit_gap']),
        'best_params': cv_results.get('best_params', {}),
    }
    
    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"已保存交叉验证结果: {save_path}")


def load_cv_results(load_path=None):
    """
    加载交叉验证结果
    
    Args:
        load_path: 加载路径
        
    Returns:
        cv_results: 交叉验证结果字典
    """
    if load_path is None:
        load_path = OUTPUT_FILES["cv_results"]
    
    with open(load_path, 'r') as f:
        cv_results = json.load(f)
    
    return cv_results


if __name__ == "__main__":
    # 测试可视化
    print("测试可视化...")
    
    # 创建虚拟训练历史
    history = {
        'train_loss': [0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25, 0.24, 0.23, 0.22],
        'val_loss': [0.55, 0.48, 0.45, 0.43, 0.42, 0.41, 0.40, 0.40, 0.41, 0.42],
        'learning_rate': [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 1e-5]
    }
    
    plot_training_history(history, save_path='./checkpoints/test_training_history.png')
    
    # 创建虚拟交叉验证结果
    cv_results = {
        'mean_val_loss': 0.35,
        'std_val_loss': 0.02,
        'mean_train_loss': 0.30,
        'std_train_loss': 0.01,
        'fold_val_losses': [0.34, 0.36, 0.35, 0.33, 0.37],
        'fold_train_losses': [0.29, 0.31, 0.30, 0.28, 0.32],
        'overfit_gaps': [0.05, 0.05, 0.05, 0.05, 0.05],
        'mean_overfit_gap': 0.05,
        'best_params': {'n_layers': 3, 'learning_rate': 0.001, 'dropout': 0.3}
    }
    
    plot_cv_comparison(cv_results, save_path='./checkpoints/test_cv_comparison.png')
    save_cv_results(cv_results, save_path='./checkpoints/test_cv_results.json')
    
    print("可视化测试完成！")