"""
Optuna 超参数调优脚本

使用 Optuna 自动搜索最佳超参数组合

调优参数：
- 学习率
- 批次大小
- 编码器学习率比例
- Dropout 比例
- 隐藏层维度
- 门控维度

用法:
    python optuna_tune.py --data_root /path/to/WESAD --n_trials 50
"""

import os
import sys
import argparse
import optuna
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'model'))
sys.path.insert(0, os.path.join(script_dir, 'dataset'))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Optuna 超参数调优 - 双门控融合模型')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='WESAD 数据集根目录')
    
    # 调优参数
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Optuna 试验次数')
    parser.add_argument('--timeout', type=int, default=None,
                        help='超时时间（秒）')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='并行任务数')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='每次试验的训练轮数（调优时减少以加快速度）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    
    # Optuna 参数
    parser.add_argument('--study_name', type=str, default='wesad_dual_gating',
                        help='研究名称')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db',
                        help='Optuna 存储路径')
    parser.add_argument('--direction', type=str, default='minimize',
                        choices=['minimize', 'maximize'],
                        help='优化方向（minimize=最小化损失，maximize=最大化R²）')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def objective(trial, data_root, num_epochs, device, args):
    """
    Optuna 目标函数
    
    Args:
        trial: Optuna Trial 对象
        data_root: 数据集路径
        num_epochs: 训练轮数
        device: 设备
        args: 命令行参数
        
    Returns:
        metric: 优化目标（验证损失或 R²）
    """
    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 1. 定义超参数搜索空间
    # 
    # 重要性排序（基于模型特点）：
    # ⭐⭐⭐⭐⭐: learning_rate, encoder_lr_ratio, gate_dim (核心门控网络)
    # ⭐⭐⭐⭐: batch_size, dropout
    # ⭐⭐⭐: hidden_dim, weight_decay
    # ⭐⭐: shared_dim (固定为 128)
    
    # === 最重要参数（必须仔细调优） ===
    
    # 学习率：对数均匀分布（影响收敛速度和最终性能）
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    
    # 编码器学习率比例：均匀分布（影响压力编码器的微调程度）
    # TCM_Encoder 参数已冻结，但压力编码器需要适当微调
    encoder_lr_ratio = trial.suggest_float('encoder_lr_ratio', 0.05, 0.2)
    
    # 门控网络维度：分类选择（核心创新参数）
    # 32: 轻量级门控，适合小数据集
    # 64: 平衡门控，推荐起点
    # 96: 强力门控，适合中等数据集
    # 128: 充分门控，适合大数据集
    # 注意：太大会过拟合，太小则门控能力弱
    gate_dim = trial.suggest_categorical('gate_dim', [32, 64, 96, 128])
    
    # === 重要参数（影响较大） ===
    
    # 批次大小：分类选择（影响训练稳定性和泛化能力）
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Dropout 比例：均匀分布（防止过拟合）
    # 对于双门控模型，适度的 dropout 很重要
    dropout = trial.suggest_float('dropout', 0.2, 0.4)
    
    # === 中等重要参数（有影响但不如前5个） ===
    
    # 隐藏层维度：分类选择（影响模型容量）
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256])
    
    # 权重衰减：对数均匀分布（L2 正则化）
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # === 固定参数（不调优） ===
    
    # 共享维度：固定为 128（与 TCM_Encoder 输出匹配）
    shared_dim = 128
    
    # 打印当前超参数
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: 超参数配置")
    print(f"{'='*60}")
    
    # 按重要性分组显示
    print(f"\n🔥 最重要参数（核心）:")
    print(f"  learning_rate:      {learning_rate:.6f}")
    print(f"  encoder_lr_ratio:   {encoder_lr_ratio:.3f}")
    print(f"  gate_dim:           {gate_dim}  ⬅️ 核心门控网络")
    
    print(f"\n🔥 重要参数:")
    print(f"  batch_size:         {batch_size}")
    print(f"  dropout:            {dropout:.3f}")
    
    print(f"\n🔥 中等重要参数:")
    print(f"  hidden_dim:         {hidden_dim}")
    print(f"  weight_decay:       {weight_decay:.6f}")
    
    print(f"\n🔥 固定参数:")
    print(f"  shared_dim:         {shared_dim}")
    
    # 2. 加载数据集
    from dataset_weasad import create_wesad_dataset
    
    try:
        train_dataset, val_dataset = create_wesad_dataset(data_root, train_ratio=0.8)
        
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        # 返回一个较大的损失值（如果是最小化）
        return float('inf') if args.direction == 'minimize' else float('-inf')
    
    # 3. 创建模型
    from model_dual_gating import create_dual_gating_model
    
    try:
        model = create_dual_gating_model(
            device=device,
            shared_dim=shared_dim,
            gate_dim=gate_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        model = model.to(device)
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return float('inf') if args.direction == 'minimize' else float('-inf')
    
    # 4. 训练配置
    criterion = nn.MSELoss()
    
    # 参数分组
    param_groups = model.get_param_groups(
        base_lr=learning_rate,
        encoder_lr_ratio=encoder_lr_ratio
    )
    
    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )
    
    # 5. 训练循环
    best_val_loss = float('inf')
    patience = 5  # 早停耐心值
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        train_samples = 0
        
        for batch in train_loader:
            dynamic = batch['dynamic'].to(device)
            static_basic = batch['static_basic'].to(device)
            targets = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(dynamic, static_basic)
            loss = criterion(outputs, targets.unsqueeze(1).repeat(1, 2))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            train_samples += targets.size(0)
        
        train_loss /= train_samples
        
        # 验证
        model.eval()
        val_loss = 0
        val_samples = 0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in val_loader:
                dynamic = batch['dynamic'].to(device)
                static_basic = batch['static_basic'].to(device)
                targets = batch['label'].to(device)
                
                outputs = model(dynamic, static_basic)
                loss = criterion(outputs, targets.unsqueeze(1).repeat(1, 2))
                
                val_loss += loss.item() * targets.size(0)
                val_samples += targets.size(0)
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
        
        val_loss /= val_samples
        
        # 更新学习率
        scheduler.step()
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # 打印进度
        print(f"  Epoch [{epoch+1:2d}/{num_epochs}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Best: {best_val_loss:.6f}")
        
        # 早停
        if no_improve_count >= patience:
            print(f"  ⏹️  早停触发（{patience} epochs 无改善）")
            break
    
    # 6. 计算最终指标
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse = mean_squared_error(all_targets, all_predictions[:, 0])  # 使用放松度预测
    r2 = r2_score(all_targets, all_predictions[:, 0])
    
    print(f"\n  最终指标:")
    print(f"    Val Loss (MSE): {mse:.6f}")
    print(f"    R²: {r2:.6f}")
    
    # 7. 返回优化目标
    if args.direction == 'minimize':
        # 最小化验证损失
        return mse
    else:
        # 最大化 R²
        return r2


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("🔬 Optuna 超参数调优 - 双门控融合模型")
    print("=" * 60)
    print(f"数据集: {args.data_root}")
    print(f"试验次数: {args.n_trials}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"设备: {args.device}")
    print(f"优化方向: {args.direction}")
    print(f"存储: {args.storage}")
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"\n⚠️  CUDA 不可用，切换到 CPU")
        args.device = 'cpu'
    
    # 检查数据集
    if not os.path.exists(args.data_root):
        print(f"\n❌ 错误: WESAD 数据集不存在: {args.data_root}")
        return
    
    # 创建或加载研究
    print(f"\n📊 创建 Optuna 研究...")
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=args.direction,
    )
    
    print(f"  研究名称: {args.study_name}")
    print(f"  存储路径: {args.storage}")
    print(f"  优化方向: {args.direction}")
    
    # 定义回调函数
    def print_callback(study, trial):
        print(f"\n{'='*60}")
        print(f"📈 Trial {trial.number} 完成")
        print(f"{'='*60}")
        print(f"  当前最佳 {args.direction}: {study.best_value:.6f}")
        print(f"  最佳参数: {study.best_params}")
    
    # 开始调优
    print(f"\n🚀 开始调优...")
    print("-" * 60)
    
    study.optimize(
        lambda trial: objective(trial, args.data_root, args.num_epochs, args.device, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        callbacks=[print_callback],
        show_progress_bar=True,
    )
    
    # 打印结果
    print(f"\n{'='*60}")
    print("📊 调优结果")
    print(f"{'='*60}")
    print(f"\n最佳试验: {study.best_trial.number}")
    print(f"最佳 {args.direction}: {study.best_value:.6f}")
    print(f"\n最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 保存最佳超参数
    import json
    save_dir = 'experiment/results/optuna'
    os.makedirs(save_dir, exist_ok=True)
    
    best_params_path = os.path.join(save_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump({
            'study_name': args.study_name,
            'best_value': float(study.best_value),
            'best_params': study.best_params,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    
    print(f"\n💾 最佳超参数已保存到: {best_params_path}")
    
    # 绘制调优历史
    try:
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(save_dir, 'optimization_history.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(os.path.join(save_dir, 'param_importances.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 调优可视化已保存到: {save_dir}")
    except Exception as e:
        print(f"⚠️  可视化保存失败: {e}")
    
    print(f"\n✅ 调优完成！")
    print(f"\n使用最佳超参数训练完整模型:")
    print(f"python main_wesad.py --data_root {args.data_root} \\")
    for key, value in study.best_params.items():
        print(f"  --{key} {value}")


if __name__ == "__main__":
    main()
