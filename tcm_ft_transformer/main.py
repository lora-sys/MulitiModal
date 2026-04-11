"""
主入口文件
实现完整的训练流程：Optuna 搜索 -> 交叉验证 -> 最终测试
"""

import os
import sys
import argparse
import numpy as np
import torch
from sklearn.model_selection import KFold

from config import (
    DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, OPTUNA_CONFIG, CV_CONFIG, OUTPUT_FILES
)
from preprocess import load_and_preprocess_data
from train import train_single_fold
from optuna_search import run_optuna_search, visualize_optuna_results
from visualize import (
    plot_training_history, plot_cv_comparison, save_cv_results,
    plot_prediction_distribution, plot_confusion_matrix
)


def set_random_seed(seed=42):
    """
    设置随机种子以保证可复现性
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_full_pipeline(
    data_path=None,
    n_trials=20,
    n_splits=5,
    num_epochs_search=20,
    num_epochs_final=50,
    device='cuda'
):
    """
    运行完整训练流程
    
    Args:
        data_path: 数据文件路径
        n_trials: Optuna 试验次数
        n_splits: 交叉验证折数
        num_epochs_search: 搜索阶段每个 trial 的训练轮数
        num_epochs_final: 最终训练的轮数
        device: 设备
    """
    print("=" * 80)
    print("FT-Transformer 中医体质分类 - 完整训练流程")
    print("=" * 80)
    
    # 设置随机种子
    set_random_seed(DATA_CONFIG['random_state'])
    
    # 设置设备
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    
    # =====================================================================
    # 阶段 0: 数据预处理
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 0: 数据预处理")
    print("=" * 80)
    
    X_pool, y_pool, X_test, y_test, scaler_params = load_and_preprocess_data(
        data_path=data_path,
        test_split=DATA_CONFIG['test_split'],
        random_state=DATA_CONFIG['random_state'],
        epsilon=DATA_CONFIG['epsilon']
    )
    
    # =====================================================================
    # 阶段 1: Optuna 超参数搜索
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 1: Optuna 超参数搜索")
    print("=" * 80)
    
    study, best_params = run_optuna_search(
        X=X_pool,
        y=y_pool,
        n_trials=n_trials,
        n_splits=n_splits,
        num_epochs=num_epochs_search,
        study_name='ft_transformer_optuna',
        storage='sqlite:///checkpoints/optuna.db'
    )
    
    # 可视化搜索结果
    visualize_optuna_results(
        study,
        save_path=os.path.join(os.path.dirname(OUTPUT_FILES["best_model"]), 'optuna_results.png')
    )
    
    # =====================================================================
    # 阶段 2: 最终模型训练与验证
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 2: 最终模型训练与验证 (5折交叉验证)")
    print("=" * 80)
    
    print(f"\n使用最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # 合并最佳参数
    final_model_params = {
        'd_token': MODEL_CONFIG['d_token'],
        'n_heads': MODEL_CONFIG['n_heads'],
        'n_classes': MODEL_CONFIG['n_classes'],
        **best_params
    }
    
    final_train_config = {
        'batch_size': TRAIN_CONFIG['batch_size'],
        'num_epochs': num_epochs_final,
        'learning_rate': best_params['learning_rate'],
        'weight_decay': TRAIN_CONFIG['weight_decay'],
        'warmup_ratio': TRAIN_CONFIG['warmup_ratio'],
        'grad_clip_max_norm': TRAIN_CONFIG['grad_clip_max_norm'],
        'patience': TRAIN_CONFIG['patience'],
        'device': device,
        'checkpoint_dir': os.path.dirname(OUTPUT_FILES["best_model"])
    }
    
    # 5折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=CV_CONFIG['random_state'])
    
    fold_val_losses = []
    fold_train_losses = []
    fold_histories = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_pool)):
        print(f"\n{'=' * 80}")
        print(f"Fold {fold_idx + 1}/{n_splits}")
        print(f"{'=' * 80}")
        
        X_train, X_val = X_pool[train_idx], X_pool[val_idx]
        y_train, y_val = y_pool[train_idx], y_pool[val_idx]
        
        # 训练单个 fold
        history, best_val_loss = train_single_fold(
            X_train, y_train,
            X_val, y_val,
            model_params=final_model_params,
            train_config=final_train_config,
            fold_idx=fold_idx
        )
        
        fold_val_losses.append(best_val_loss)
        fold_train_losses.append(min(history['train_loss']))
        fold_histories.append(history)
        
        print(f"\nFold {fold_idx + 1} 完成:")
        print(f"  最佳验证损失: {best_val_loss:.6f}")
        print(f"  最佳训练损失: {min(history['train_loss']):.6f}")
        print(f"  过拟合差距: {best_val_loss - min(history['train_loss']):.6f}")
    
    # 计算统计结果
    mean_val_loss = np.mean(fold_val_losses)
    std_val_loss = np.std(fold_val_losses)
    mean_train_loss = np.mean(fold_train_losses)
    std_train_loss = np.std(fold_train_losses)
    overfit_gaps = np.array(fold_val_losses) - np.array(fold_train_losses)
    mean_overfit_gap = np.mean(overfit_gaps)
    
    print(f"\n{'=' * 80}")
    print("交叉验证结果汇总")
    print(f"{'=' * 80}")
    print(f"平均验证损失: {mean_val_loss:.6f} ± {std_val_loss:.6f}")
    print(f"平均训练损失: {mean_train_loss:.6f} ± {std_train_loss:.6f}")
    print(f"平均过拟合差距: {mean_overfit_gap:.6f}")
    print(f"{'=' * 80}")
    
    # 保存交叉验证结果
    cv_results = {
        'mean_val_loss': mean_val_loss,
        'std_val_loss': std_val_loss,
        'mean_train_loss': mean_train_loss,
        'std_train_loss': std_train_loss,
        'fold_val_losses': fold_val_losses,
        'fold_train_losses': fold_train_losses,
        'overfit_gaps': overfit_gaps.tolist(),
        'mean_overfit_gap': mean_overfit_gap,
        'best_params': best_params,
    }
    
    save_cv_results(cv_results, save_path=OUTPUT_FILES["cv_results"])
    
    # 绘制交叉验证对比图
    plot_cv_comparison(cv_results, save_path=OUTPUT_FILES["cv_comparison"])
    
    # 绘制最佳 fold 的训练历史
    best_fold_idx = np.argmin(fold_val_losses)
    plot_training_history(
        fold_histories[best_fold_idx],
        save_path=OUTPUT_FILES["training_history"]
    )
    
    # =====================================================================
    # 阶段 3: 最终测试（可选）
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 3: 最终测试（使用锁定的测试集）")
    print("=" * 80)
    
    # 加载最佳模型
    from ft_transformer import get_model
    from preprocess import create_dataloaders
    
    model = get_model(
        n_features=8,
        n_classes=9,
        **final_model_params
    )
    
    checkpoint_path = os.path.join(
        os.path.dirname(OUTPUT_FILES["best_model"]),
        f'fold_{best_fold_idx}',
        OUTPUT_FILES["best_model"]
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"已加载最佳模型 (Fold {best_fold_idx + 1})")
        
        # 在测试集上评估
        test_loader = create_dataloaders(
            X_test, y_test,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        test_loss = 0.0
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                
                # 计算 KL 损失（注意：outputs 已经是概率分布，需要 log）
                log_probs = torch.log(torch.clamp(outputs, min=1e-8))
                loss = torch.nn.functional.kl_div(log_probs, batch_y, reduction='batchmean')
                
                test_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_trues.append(batch_y.cpu().numpy())
        
        test_loss /= len(test_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        
        print(f"\n测试集结果:")
        print(f"  测试损失: {test_loss:.6f}")
        
        # 绘制预测分布
        plot_prediction_distribution(
            all_trues, all_preds,
            save_path=os.path.join(os.path.dirname(OUTPUT_FILES["best_model"]), 'prediction_distribution.png')
        )
        
        # 绘制混淆矩阵
        plot_confusion_matrix(
            all_trues, all_preds,
            save_path=os.path.join(os.path.dirname(OUTPUT_FILES["best_model"]), 'confusion_matrix.png')
        )
        
        # 保存测试结果
        test_results = {
            'test_loss': test_loss,
            'best_fold': best_fold_idx + 1,
            'best_params': best_params,
        }
        
        test_results_path = os.path.join(os.path.dirname(OUTPUT_FILES["best_model"]), OUTPUT_FILES["test_results"])
        import json
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"已保存测试结果: {test_results_path}")
    else:
        print(f"警告: 未找到模型检查点: {checkpoint_path}")
        print("跳过最终测试")
    
    # =====================================================================
    # 完成
    # =====================================================================
    print("\n" + "=" * 80)
    print("训练流程完成！")
    print("=" * 80)
    print("\n交付物清单:")
    print(f"  1. 模型权重: {OUTPUT_FILES['best_model']}")
    print(f"  2. 标准化参数: {OUTPUT_FILES['scaler_params']}")
    print(f"  3. 训练历史: {OUTPUT_FILES['training_history']}")
    print(f"  4. 交叉验证对比: {OUTPUT_FILES['cv_comparison']}")
    print(f"  5. 交叉验证结果: {OUTPUT_FILES['cv_results']}")
    print(f"  6. Optuna 搜索结果: optuna_results.json")
    print(f"  7. Optuna 可视化: optuna_results.png")
    print("=" * 80)


def run_search_only(data_path=None, n_trials=20, n_splits=5, num_epochs=20):
    """
    仅运行 Optuna 搜索
    """
    print("=" * 80)
    print("Optuna 超参数搜索模式")
    print("=" * 80)
    
    # 设置随机种子
    set_random_seed(DATA_CONFIG['random_state'])
    
    # 数据预处理
    X_pool, y_pool, _, _, _ = load_and_preprocess_data(data_path=data_path)
    
    # 运行搜索
    study, best_params = run_optuna_search(
        X=X_pool,
        y=y_pool,
        n_trials=n_trials,
        n_splits=n_splits,
        num_epochs=num_epochs,
        study_name='ft_transformer_optuna',
        storage='sqlite:///checkpoints/optuna.db'
    )
    
    print(f"\n最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='FT-Transformer 中医体质分类训练')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'search'],
                        help='运行模式: full (完整流程) 或 search (仅搜索)')
    parser.add_argument('--data', type=str, default=None,
                        help='数据文件路径')
    parser.add_argument('--trials', type=int, default=20,
                        help='Optuna 试验次数')
    parser.add_argument('--epochs_search', type=int, default=20,
                        help='搜索阶段每个 trial 的训练轮数')
    parser.add_argument('--epochs_final', type=int, default=50,
                        help='最终训练的轮数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda 或 cpu')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_full_pipeline(
            data_path=args.data,
            n_trials=args.trials,
            n_splits=5,
            num_epochs_search=args.epochs_search,
            num_epochs_final=args.epochs_final,
            device=args.device
        )
    elif args.mode == 'search':
        run_search_only(
            data_path=args.data,
            n_trials=args.trials,
            n_splits=5,
            num_epochs=args.epochs_search
        )
    else:
        print(f"未知模式: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
