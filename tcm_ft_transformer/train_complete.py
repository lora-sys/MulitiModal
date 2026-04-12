"""
统一训练流程 - 确保所有条件一致，最终输出最佳模型

改进点：
1. 使用中位数而不是平均数计算最优 epoch（避免异常值影响）
2. 为全局最终训练添加验证集（从训练池中划分）
3. 训练多个候选模型（不同 epoch 数），在验证集上选择最佳
4. 最终在测试集上评估，输出一个明确的最佳模型
5. 所有训练使用相同的随机种子和数据划分
"""

import os
import json
import sys
import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from datetime import datetime

from config import (
    DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, OPTUNA_CONFIG, CV_CONFIG, OUTPUT_FILES, CHECKPOINT_DIR
)
from preprocess import load_and_preprocess_data, create_dataloaders
from train import train_single_fold, Trainer, KLDivLossWithLogSoftmax
from optuna_search import run_optuna_search, visualize_optuna_results
from ft_transformer import get_model
from visualize import (
    plot_training_history, plot_cv_comparison, save_cv_results,
    plot_prediction_distribution, plot_confusion_matrix
)


def set_random_seed(seed=42):
    """设置随机种子以保证可复现性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_model(model, data_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            
            total_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    return avg_loss, all_preds, all_trues


def run_complete_training(data_path=None, n_trials=20, n_splits=5, device='cuda'):
    """
    运行完整训练流程，确保所有条件一致
    
    流程：
    1. Optuna 超参数搜索
    2. 5折交叉验证（确定最优 epoch 范围）
    3. 训练多个候选模型（不同 epoch 数）
    4. 在验证集上选择最佳模型
    5. 在测试集上评估最佳模型
    6. 输出最佳模型
    """
    print("=" * 80)
    print("FT-Transformer 中医体质分类 - 统一训练流程")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置随机种子（确保所有阶段使用相同的种子）
    set_random_seed(DATA_CONFIG['random_state'])
    
    # 设置设备
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    
    # =====================================================================
    # 阶段 0: 数据预处理（一次性划分，确保一致性）
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 0: 数据预处理")
    print("=" * 80)
    
    # 加载数据
    X_pool, y_pool, X_test, y_test, scaler_params = load_and_preprocess_data(
        data_path=data_path,
        test_split=DATA_CONFIG['test_split'],
        random_state=DATA_CONFIG['random_state'],
        epsilon=DATA_CONFIG['epsilon']
    )
    
    print(f"训练池样本数: {len(X_pool)}")
    print(f"测试集样本数: {len(X_test)}")
    
    # 从训练池中划分验证集（用于最终模型选择）
    # 使用 80% 训练，20% 验证，使用分层采样保持类别比例
    stratify_target = y_pool.argmax(axis=1) if y_pool.ndim > 1 else y_pool
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_pool, y_pool,
        test_size=0.2,
        random_state=DATA_CONFIG['random_state'],
        shuffle=True,
        stratify=stratify_target
    )
    
    print(f"最终训练集样本数: {len(X_train_final)}")
    print(f"验证集样本数: {len(X_val_final)}")
    
    # =====================================================================
    # 阶段 1: Optuna 超参数搜索（使用训练池的 80% 用于训练，20% 用于验证）
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 1: Optuna 超参数搜索")
    print("=" * 80)
    
    # 检查是否已有 Optuna 搜索结果
    optuna_results_path = 'optuna_results.json'
    if os.path.exists(optuna_results_path):
        print(f"✅ 检测到已有 Optuna 搜索结果: {optuna_results_path}")
        print("   直接使用现有结果，跳过 Optuna 搜索阶段")
        
        with open(optuna_results_path, 'r') as f:
            optuna_results = json.load(f)
        
        best_params = optuna_results['best_params']
        print(f"\n从文件读取的最佳参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        study = None  # 不需要 study 对象
    else:
        print("未找到现有 Optuna 搜索结果，开始新的搜索...")
        
        study, best_params = run_optuna_search(
            X=X_train_final,
            y=y_train_final,
            n_trials=n_trials,
            n_splits=n_splits,
            num_epochs=20,
            study_name='ft_transformer_optuna',
            storage='sqlite:///checkpoints/optuna.db'
        )
        
        # 可视化搜索结果
        visualize_optuna_results(
            study,
            save_path=os.path.join(CHECKPOINT_DIR, 'optuna_results.png')
        )
        
        print(f"\n最佳参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    
    # =====================================================================
    # 阶段 2: 5折交叉验证（确定最优 epoch 范围）
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 2: 5折交叉验证（确定最优 epoch 范围）")
    print("=" * 80)

    # 合并最佳参数
    final_model_params = {
        'd_token': MODEL_CONFIG['d_token'],
        'n_heads': MODEL_CONFIG['n_heads'],
        'n_layers': best_params.get('n_layers', 3),
        'dropout': best_params.get('dropout', 0.3),
    }

    final_train_config = {
        'batch_size': TRAIN_CONFIG['batch_size'],
        'num_epochs': 50,  # 设置较大的值，让早停机制工作
        'learning_rate': best_params['learning_rate'],
        'weight_decay': TRAIN_CONFIG['weight_decay'],
        'warmup_ratio': TRAIN_CONFIG['warmup_ratio'],
        'grad_clip_max_norm': TRAIN_CONFIG['grad_clip_max_norm'],
        'patience': TRAIN_CONFIG['patience'],
        'device': device,
        'checkpoint_dir': CHECKPOINT_DIR
    }

    # 检查是否已有交叉验证结果
    cv_results_path = os.path.join(CHECKPOINT_DIR, OUTPUT_FILES["cv_results"])
    if os.path.exists(cv_results_path):
        print(f"✅ 检测到已有交叉验证结果: {cv_results_path}")
        print("   直接使用现有结果，跳过阶段2")

        with open(cv_results_path, 'r') as f:
            cv_results = json.load(f)

        fold_val_losses = cv_results['fold_val_losses']
        fold_train_losses = cv_results['fold_train_losses']
        fold_optimal_epochs = cv_results['fold_optimal_epochs']

        median_optimal_epoch = cv_results['median_optimal_epoch']
        mean_optimal_epoch = cv_results['mean_optimal_epoch']
        mean_val_loss = cv_results['mean_val_loss']
        std_val_loss = cv_results['std_val_loss']
        mean_train_loss = cv_results['mean_train_loss']
        std_train_loss = cv_results['std_train_loss']
        mean_overfit_gap = cv_results['mean_overfit_gap']

        fold_histories = []  # 暂时不需要历史

        print(f"\n从文件读取的交叉验证结果:")
        print(f"  各 Fold 最优 Epoch: {fold_optimal_epochs}")
        print(f"  中位数最优 epoch: {median_optimal_epoch}")
        print(f"  平均最优 epoch: {mean_optimal_epoch}")
        print(f"  平均验证损失: {mean_val_loss:.6f} ± {std_val_loss:.6f}")

        # 从 cv_results 中获取 overfit_gaps
        overfit_gaps = np.array(cv_results['overfit_gaps'])
    else:
        # 5折交叉验证
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=CV_CONFIG['random_state'])

        fold_val_losses = []
        fold_train_losses = []
        fold_histories = []
        fold_optimal_epochs = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_final)):
            print(f"\n{'=' * 80}")
            print(f"Fold {fold_idx + 1}/{n_splits}")
            print(f"{'=' * 80}")

            X_train, X_val = X_train_final[train_idx], X_train_final[val_idx]
            y_train, y_val = y_train_final[train_idx], y_train_final[val_idx]

            # 训练单个 fold
            history, best_val_loss = train_single_fold(
                X_train, y_train,
                X_val, y_val,
                model_params=final_model_params,
                train_config=final_train_config,
                fold_idx=fold_idx
            )

            # 找到最优 epoch
            best_epoch = np.argmin(history['val_loss']) + 1
            fold_optimal_epochs.append(best_epoch)

            fold_val_losses.append(best_val_loss)
            fold_train_losses.append(min(history['train_loss']))
            fold_histories.append(history)

            print(f"\nFold {fold_idx + 1} 完成:")
            print(f"  最优 epoch: {best_epoch}")
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

        # 使用中位数而不是平均数（避免异常值影响）
        median_optimal_epoch = int(np.median(fold_optimal_epochs))
        mean_optimal_epoch = int(round(np.mean(fold_optimal_epochs)))

        print(f"\n{'=' * 80}")
        print("交叉验证结果汇总")
        print(f"{'=' * 80}")
        print(f"各 Fold 最优 Epoch: {fold_optimal_epochs}")
        print(f"中位数最优 epoch: {median_optimal_epoch} ⭐ (推荐)")
        print(f"平均最优 epoch: {mean_optimal_epoch}")
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
            'fold_optimal_epochs': fold_optimal_epochs,
            'median_optimal_epoch': median_optimal_epoch,
            'mean_optimal_epoch': mean_optimal_epoch,
            'overfit_gaps': overfit_gaps.tolist(),
            'mean_overfit_gap': mean_overfit_gap,
            'best_params': best_params,
        }

        save_cv_results(cv_results, save_path=OUTPUT_FILES["cv_results"])

        # 绘制交叉验证对比图
        plot_cv_comparison(cv_results, save_path=OUTPUT_FILES["cv_comparison"])

        # 绘制最佳 fold 的训练历史（如果有）
        if len(fold_histories) > 0:
            best_fold_idx = np.argmin(fold_val_losses)
            plot_training_history(
                fold_histories[best_fold_idx],
                save_path=OUTPUT_FILES["training_history"]
            )
        else:
            print("  跳过训练历史绘制（无历史数据）")
        print(f"{'=' * 80}")
        
        X_train, X_val = X_train_final[train_idx], X_train_final[val_idx]
        y_train, y_val = y_train_final[train_idx], y_train_final[val_idx]
        
        # 训练单个 fold
        history, best_val_loss = train_single_fold(
            X_train, y_train,
            X_val, y_val,
            model_params=final_model_params,
            train_config=final_train_config,
            fold_idx=fold_idx
        )
        
        # 找到最优 epoch
        best_epoch = np.argmin(history['val_loss']) + 1
        fold_optimal_epochs.append(best_epoch)
        
        fold_val_losses.append(best_val_loss)
        fold_train_losses.append(min(history['train_loss']))
        fold_histories.append(history)
        
        print(f"\nFold {fold_idx + 1} 完成:")
        print(f"  最优 epoch: {best_epoch}")
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
    
    # 使用中位数而不是平均数（避免异常值影响）
    median_optimal_epoch = int(np.median(fold_optimal_epochs))
    mean_optimal_epoch = int(round(np.mean(fold_optimal_epochs)))
    
    print(f"\n{'=' * 80}")
    print("交叉验证结果汇总")
    print(f"{'=' * 80}")
    print(f"各 Fold 最优 Epoch: {fold_optimal_epochs}")
    print(f"中位数最优 epoch: {median_optimal_epoch} ⭐ (推荐)")
    print(f"平均最优 epoch: {mean_optimal_epoch}")
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
        'fold_optimal_epochs': fold_optimal_epochs,
        'median_optimal_epoch': median_optimal_epoch,
        'mean_optimal_epoch': mean_optimal_epoch,
        'overfit_gaps': overfit_gaps.tolist(),
        'mean_overfit_gap': mean_overfit_gap,
        'best_params': best_params,
    }
    
    save_cv_results(cv_results, save_path=OUTPUT_FILES["cv_results"])
    
    # 绘制交叉验证对比图
    plot_cv_comparison(cv_results, save_path=OUTPUT_FILES["cv_comparison"])

    # 绘制最佳 fold 的训练历史（如果有）
    if len(fold_histories) > 0:
        best_fold_idx = np.argmin(fold_val_losses)
        plot_training_history(
            fold_histories[best_fold_idx],
            save_path=OUTPUT_FILES["training_history"]
        )
    else:
        print("  跳过训练历史绘制（无历史数据）")
    
    # =====================================================================
    # 阶段 3: 训练多个候选模型（不同 epoch 数）
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 3: 训练多个候选模型（不同 epoch 数）")
    print("=" * 80)

    # 定义损失函数（用于阶段5）
    criterion = KLDivLossWithLogSoftmax()

    # 基于 CV 结果，选择几个候选 epoch 数
    candidate_epochs = [
        max(3, median_optimal_epoch - 3),  # 比中位数少 3
        median_optimal_epoch,               # 中位数
        median_optimal_epoch + 3,           # 比中位数多 3
        mean_optimal_epoch,                 # 平均数
    ]
    # 去重并排序
    candidate_epochs = sorted(list(set(candidate_epochs)))

    # 检查是否已有候选模型
    existing_candidate_models = []
    for epoch_num in candidate_epochs:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'candidate_model_epoch_{epoch_num}.pth')
        if os.path.exists(checkpoint_path):
            existing_candidate_models.append(epoch_num)

    if len(existing_candidate_models) == len(candidate_epochs):
        print(f"✅ 检测到已有所有候选模型，跳过阶段3")
        print(f"   现有模型: {existing_candidate_models}")

        # 加载现有候选模型
        candidate_models = []
        for epoch_num in candidate_epochs:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'candidate_model_epoch_{epoch_num}.pth')
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            candidate_models.append({
                'epoch_num': epoch_num,
                'val_loss': checkpoint['val_loss'],
                'checkpoint_path': checkpoint_path,
                'history': checkpoint['history'],
            })
            print(f"  已加载: Epoch {epoch_num} (Val Loss: {checkpoint['val_loss']:.6f})")
    else:
        print(f"\n候选 epoch 数: {candidate_epochs}")
        print(f"训练数据量: {len(X_train_final)} 样本")
        print(f"验证数据量: {len(X_val_final)} 样本")

        # 创建训练和验证 DataLoader
        train_loader = create_dataloaders(
            X_train_final, y_train_final,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0
        )

        val_loader = create_dataloaders(
            X_val_final, y_val_final,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0
        )

        candidate_models = []

        for epoch_num in candidate_epochs:
            print(f"\n{'=' * 80}")
            print(f"训练候选模型: {epoch_num} 轮")
            print(f"{'=' * 80}")

            # 创建新模型
            model = get_model(
                n_features=8,
                n_classes=9,
                **final_model_params
            )

            # 训练模型
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                learning_rate=best_params['learning_rate'],
                weight_decay=TRAIN_CONFIG['weight_decay'],
                warmup_ratio=TRAIN_CONFIG['warmup_ratio'],
                num_epochs=epoch_num,
                patience=None,  # 不使用早停，训练完整 epoch 数
                grad_clip_max_norm=TRAIN_CONFIG['grad_clip_max_norm'],
                checkpoint_dir=CHECKPOINT_DIR
            )

            history = trainer.train()

            # 在验证集上评估
            val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)

            print(f"\n候选模型 ({epoch_num} 轮) 完成:")
            print(f"  最终训练损失: {history['train_loss'][-1]:.6f}")
            print(f"  验证损失: {val_loss:.6f}")

            # 保存模型
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'candidate_model_epoch_{epoch_num}.pth')
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'history': history,
                'model_params': final_model_params,
                'train_config': final_train_config,
            }, checkpoint_path)

            candidate_models.append({
                'epoch_num': epoch_num,
                'val_loss': val_loss,
                'checkpoint_path': checkpoint_path,
                'history': history,
            })
    
    # =====================================================================
    # 阶段 4: 在验证集上选择最佳模型
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 4: 在验证集上选择最佳模型")
    print("=" * 80)
    
    print(f"\n候选模型验证损失对比:")
    for i, model_info in enumerate(candidate_models):
        print(f"  {i+1}. Epoch {model_info['epoch_num']}: {model_info['val_loss']:.6f}")
    
    # 选择验证损失最小的模型
    best_model_idx = np.argmin([m['val_loss'] for m in candidate_models])
    best_model_info = candidate_models[best_model_idx]
    
    print(f"\n✅ 最佳模型: Epoch {best_model_info['epoch_num']}")
    print(f"   验证损失: {best_model_info['val_loss']:.6f}")
    
    # 复制最佳模型为 best_model.pth
    import shutil
    best_model_path = os.path.join(CHECKPOINT_DIR, OUTPUT_FILES["best_model"])
    shutil.copy(best_model_info['checkpoint_path'], best_model_path)
    print(f"   已保存: {best_model_path}")
    
    # =====================================================================
    # 阶段 5: 在测试集上评估最佳模型
    # =====================================================================
    print("\n" + "=" * 80)
    print("阶段 5: 在测试集上评估最佳模型")
    print("=" * 80)
    
    # 加载最佳模型
    best_model = get_model(
        n_features=8,
        n_classes=9,
        **final_model_params
    )
    
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.to(device)
    
    # 创建测试 DataLoader
    test_loader = create_dataloaders(
        X_test, y_test,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # 在测试集上评估
    test_loss, test_preds, test_trues = evaluate_model(best_model, test_loader, criterion, device)
    
    print(f"\n{'=' * 80}")
    print("最佳模型测试结果")
    print(f"{'=' * 80}")
    print(f"模型: Epoch {best_model_info['epoch_num']}")
    print(f"验证损失: {best_model_info['val_loss']:.6f}")
    print(f"测试损失: {test_loss:.6f}")
    print(f"{'=' * 80}")
    
    # 绘制预测分布
    plot_prediction_distribution(
        test_trues, test_preds,
        save_path=os.path.join(CHECKPOINT_DIR, 'prediction_distribution.png')
    )
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        test_trues, test_preds,
        save_path=os.path.join(CHECKPOINT_DIR, 'confusion_matrix.png')
    )

    # 保存测试结果
    # 转换 NumPy 类型为 Python 原生类型
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    test_results = {
        'best_model_epoch': best_model_info['epoch_num'],
        'val_loss': best_model_info['val_loss'],
        'test_loss': test_loss,
        'candidate_epochs': candidate_epochs,
        'candidate_val_losses': [m['val_loss'] for m in candidate_models],
        'best_params': best_params,
        'cv_results': convert_to_native(cv_results),
    }

    test_results_path = os.path.join(CHECKPOINT_DIR, OUTPUT_FILES["test_results"])
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"已保存测试结果: {test_results_path}")

    # =====================================================================
    # 完成
    # =====================================================================
    print("\n" + "=" * 80)
    print("训练流程完成！")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🏆 最佳模型:")
    print(f"   路径: {best_model_path}")
    print(f"   Epoch: {best_model_info['epoch_num']}")
    print(f"   验证损失: {best_model_info['val_loss']:.6f}")
    print(f"   测试损失: {test_loss:.6f}")
    print("\n交付物清单:")
    print(f"  1. 最佳模型: {best_model_path} ⭐")
    print(f"  2. 候选模型: checkpoints/candidate_model_epoch_*.pth")
    print(f"  3. 标准化参数: {OUTPUT_FILES['scaler_params']}")
    print(f"  4. 训练历史: {OUTPUT_FILES['training_history']}")
    print(f"  5. 交叉验证对比: {OUTPUT_FILES['cv_comparison']}")
    print(f"  6. 交叉验证结果: {OUTPUT_FILES['cv_results']}")
    print(f"  7. Optuna 搜索结果: optuna_results.json")
    print(f"  8. Optuna 可视化: optuna_results.png")
    print(f"  9. 测试结果: {test_results_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_complete_training(
        data_path='data/vital_signs_dataset_final.csv',
        n_trials=20,
        n_splits=5,
        device='cuda'
    )