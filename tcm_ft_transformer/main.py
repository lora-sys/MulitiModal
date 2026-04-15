"""
主入口文件
实现完整的训练流程：Optuna 搜索 -> 交叉验证 -> 最终测试
"""

import json
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
        'n_layers': best_params.get('n_layers', 3),
        'dropout': best_params.get('dropout', 0.3),
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
    # 阶段 3: 最终测试（Ensemble 多模型集成预测）
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 3: 最终测试（Ensemble 集成预测）")
    print("=" * 80)

    # 加载所有 5 个 fold 的模型
    from ft_transformer import get_model
    from preprocess import create_dataloaders
    from train import KLDivLossWithLogSoftmax
    from config import CHECKPOINT_DIR

    all_models = []
    for fold_idx in range(n_splits):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'fold_{fold_idx}', OUTPUT_FILES["best_model"])
        
        if not os.path.exists(checkpoint_path):
            print(f"警告: 未找到 Fold {fold_idx + 1} 的模型检查点: {checkpoint_path}")
            continue

        model = get_model(
            n_features=4,
            n_classes=9,
            **final_model_params
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        all_models.append((fold_idx, model))
        
        print(f"已加载 Fold {fold_idx + 1} 模型")

    if len(all_models) == 0:
        print("错误: 未找到任何模型检查点，跳过最终测试")
    else:
        # 检查模型数量是否足够
        if len(all_models) < n_splits:
            print(f"\n⚠️ 警告: 仅加载 {len(all_models)}/{n_splits} 个 fold 模型, "
                  f"Ensemble 结果可能不准确")
        if len(all_models) < 3:
            print("错误: 可用模型少于 3 个，Ensemble 结果不可靠")
        else:
            print(f"\n使用 {len(all_models)} 个模型进行 Ensemble 预测")

        # 在测试集上进行集成预测
        test_loader = create_dataloaders(
            X_test, y_test,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0
        )

        # 复用训练时的损失函数（确保数值保护一致）
        criterion = KLDivLossWithLogSoftmax()

        ensemble_loss = 0.0
        all_preds = []
        all_trues = []
        individual_losses = [[] for _ in all_models]

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                # 收集所有模型的预测
                model_preds = []
                for i, (fold_idx, model) in enumerate(all_models):
                    pred = model(batch_X)
                    model_preds.append(pred)

                    # 计算单个模型的损失（复用 KLDivLossWithLogSoftmax）
                    single_loss = criterion(pred, batch_y)
                    individual_losses[i].append(single_loss.item())

                # Ensemble 预测：平均所有模型的输出
                ensemble_pred = torch.stack(model_preds, dim=0).mean(dim=0)

                # 计算 Ensemble 损失（复用 KLDivLossWithLogSoftmax）
                loss = criterion(ensemble_pred, batch_y)

                ensemble_loss += loss.item()
                all_preds.append(ensemble_pred.cpu().numpy())
                all_trues.append(batch_y.cpu().numpy())

        ensemble_loss /= len(test_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)

        print(f"\n{'=' * 80}")
        print("Ensemble 预测结果")
        print(f"{'=' * 80}")
        print(f"Ensemble 测试损失: {ensemble_loss:.6f}")
        
        # 打印每个模型的单独表现
        print(f"\n各模型单独表现:")
        for i, (fold_idx, _) in enumerate(all_models):
            avg_loss = np.mean(individual_losses[i])
            print(f"  Fold {fold_idx + 1}: {avg_loss:.6f}")
        
        # 计算 Ensemble 提升
        individual_mean = np.mean([np.mean(losses) for losses in individual_losses])
        improvement = (individual_mean - ensemble_loss) / individual_mean * 100
        print(f"\n单模型平均损失: {individual_mean:.6f}")
        print(f"Ensemble 损失: {ensemble_loss:.6f}")
        print(f"Ensemble 相对提升: {improvement:.2f}%")
        print(f"{'=' * 80}")
        
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
            'ensemble_test_loss': ensemble_loss,
            'individual_model_losses': [np.mean(losses) for losses in individual_losses],
            'individual_mean_loss': individual_mean,
            'ensemble_improvement_percent': improvement,
            'n_models_in_ensemble': len(all_models),
            'best_params': best_params,
        }
        
        test_results_path = os.path.join(os.path.dirname(OUTPUT_FILES["best_model"]), OUTPUT_FILES["test_results"])
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"已保存测试结果: {test_results_path}")

    # =====================================================================
    # 阶段 4: 全局最终训练（基于 CV 最优 epoch）
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 4: 全局最终训练（基于 CV 信息）")
    print("=" * 80)

    # 计算 5-fold CV 的平均最优 epoch
    fold_optimal_epochs = []
    for history in fold_histories:
        # 找到验证损失最小的 epoch
        best_epoch = np.argmin(history['val_loss']) + 1
        fold_optimal_epochs.append(best_epoch)

    avg_optimal_epoch = int(round(np.mean(fold_optimal_epochs)))
    
    print(f"各 Fold 最优 Epoch:")
    for i, epoch in enumerate(fold_optimal_epochs):
        print(f"  Fold {i + 1}: 第 {epoch} 轮")
    print(f"平均最优 epoch: {avg_optimal_epoch}")
    print(f"训练数据量: {len(X_pool)} 样本 (100% 训练池)")
    print(f"使用参数: 最佳参数（来自 Optuna 搜索）")
    print("=" * 80)

    # 创建最终训练用的 DataLoader（使用全部 pool 数据）
    final_train_loader = create_dataloaders(
        X_pool, y_pool,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )

    # 创建最终模型
    final_model = get_model(
        n_features=4,
        n_classes=9,
        **final_model_params
    )

    # 训练最终模型
    from train import Trainer
    from config import CHECKPOINT_DIR
    
    final_checkpoint_dir = CHECKPOINT_DIR
    final_trainer = Trainer(
        model=final_model,
        train_loader=final_train_loader,
        val_loader=None,  # 无验证集，使用全部数据
        device=device,
        learning_rate=best_params['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        warmup_ratio=TRAIN_CONFIG['warmup_ratio'],
        num_epochs=avg_optimal_epoch,  # 使用 CV 找到的最优 epoch 数
        patience=None,  # 无验证集，早停自然跳过，训练完整 epochs
        grad_clip_max_norm=TRAIN_CONFIG['grad_clip_max_norm'],
        checkpoint_dir=final_checkpoint_dir
    )

    # 训练最终模型
    final_history = final_trainer.train()
    
    # 保存最终模型（Trainer.train() 已自动保存，这里再保存一个明确命名的副本）
    final_model_path = os.path.join(final_checkpoint_dir, 'final_model.pth')
    final_trainer.save_checkpoint(final_model_path)
    
    print(f"\n✅ 全局最终模型已保存: {final_model_path}")
    print(f"\n最终模型训练历史:")
    print(f"  训练轮数: {len(final_history['train_loss'])}")
    print(f"  最终训练损失: {final_history['train_loss'][-1]:.6f}")

    # =====================================================================
    # 完成
    # =====================================================================
    print("\n" + "=" * 80)
    print("训练流程完成！")
    print("=" * 80)
    print("\n交付物清单:")
    print(f"  1. 最佳模型权重 (Fold {best_fold_idx + 1}): checkpoints/best_model.pth")
    print(f"  2. 全局最终模型: checkpoints/final_model.pth ⭐ (推荐用于部署)")
    print(f"  3. Ensemble 模型: checkpoints/fold_0~4/best_model.pth (5 个模型)")
    print(f"  4. 标准化参数: {OUTPUT_FILES['scaler_params']}")
    print(f"  5. 训练历史: {OUTPUT_FILES['training_history']}")
    print(f"  6. 交叉验证对比: {OUTPUT_FILES['cv_comparison']}")
    print(f"  7. 交叉验证结果: {OUTPUT_FILES['cv_results']}")
    print(f"  8. Optuna 搜索结果: optuna_results.json")
    print(f"  9. Optuna 可视化: optuna_results.png")
    print(f"  10. 测试结果: {OUTPUT_FILES['test_results']}")
    print("=" * 80)
    print("\n模型使用说明:")
    print("  - 部署/生产环境: 使用 final_model.pth（全局最终模型）")
    print("  - 研究/对比实验: 使用 fold_*/best_model.pth（Ensemble）")
    print("  - 评估模型性能: 参考阶段 3 的 Ensemble 测试结果")
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
