"""
使用最佳超参数训练完整模型

基于 Optuna 找到的最佳超参数进行完整训练

用法:
    python train_with_best_params.py --data_root /path/to/WESAD --best_params_path experiment/results/optuna/best_params.json
"""

import os
import sys
import argparse
import json
import torch

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'model'))
sys.path.insert(0, os.path.join(script_dir, 'dataset'))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用最佳超参数训练完整模型')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='WESAD 数据集根目录')
    
    # 最佳参数路径
    parser.add_argument('--best_params_path', type=str,
                        default='experiment/results/optuna/best_params.json',
                        help='最佳超参数 JSON 文件路径')
    
    # 训练参数（可选，会覆盖最佳参数）
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='训练轮数（默认使用最佳参数）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（默认使用最佳参数）')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率（默认使用最佳参数）')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str,
                        default='experiment/results/wesad_best',
                        help='保存目录')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def load_best_params(best_params_path):
    """加载最佳超参数"""
    with open(best_params_path, 'r') as f:
        data = json.load(f)

    print(f"✅ 加载最佳超参数:")
    print(f"  来源: {best_params_path}")
    if data.get('best_value') is not None:
        print(f"  最佳值: {data['best_value']:.6f}")
    if data.get('n_trials') is not None:
        print(f"  试验次数: {data['n_trials']}")
    if data.get('timestamp') is not None:
        print(f"  时间戳: {data['timestamp']}")
    print(f"\n  超参数:")
    for key, value in data.get('best_params', data).items():
        print(f"    {key}: {value}")

    return data.get('best_params', data)


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("🎯 使用最佳超参数训练完整模型")
    print("=" * 60)
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"\n⚠️  CUDA 不可用，切换到 CPU")
        args.device = 'cpu'
    
    # 检查数据集
    if not os.path.exists(args.data_root):
        print(f"\n❌ 错误: WESAD 数据集不存在: {args.data_root}")
        return
    
    # 加载最佳超参数
    if not os.path.exists(args.best_params_path):
        print(f"\n❌ 错误: 最佳参数文件不存在: {args.best_params_path}")
        print("请先运行 optuna_tune.py 进行超参数调优")
        return
    
    best_params = load_best_params(args.best_params_path)
    
    # 合并命令行参数（命令行参数优先）
    if args.num_epochs is not None:
        best_params['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        best_params['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        best_params['learning_rate'] = args.learning_rate
    
    # 设置默认训练轮数（如果最佳参数中没有）
    if 'num_epochs' not in best_params:
        best_params['num_epochs'] = 100  # 完整训练使用更多轮数

    if 'batch_size' not in best_params:
        best_params['batch_size'] = 32  # 默认批次大小

    print(f"\n📋 最终训练配置:")
    print(f"  数据集: {args.data_root}")
    print(f"  设备: {args.device}")
    print(f"  训练轮数: {best_params['num_epochs']}")
    print(f"  批次大小: {best_params['batch_size']}")
    print(f"  学习率: {best_params['learning_rate']}")
    print(f"  保存目录: {args.save_dir}")
    
    # 导入训练函数
    from train import train_wesad
    
    # 开始训练
    print(f"\n🚀 开始训练...")
    print("-" * 60)
    
    try:
        results = train_wesad(
            data_root=args.data_root,
            model_path=os.path.join(args.save_dir, 'best_model.pth'),
            num_epochs=best_params['num_epochs'],
            batch_size=best_params['batch_size'],
            learning_rate=best_params['learning_rate'],
            weight_decay=best_params.get('weight_decay', 1e-4),
            encoder_lr_ratio=best_params.get('encoder_lr_ratio', 0.1),
            device=args.device,
            save_dir=args.save_dir,
        )
        
        # 打印结果
        print(f"\n" + "=" * 60)
        print("📊 训练结果")
        print("=" * 60)
        print(f"最佳验证损失: {results['best_val_loss']:.6f}")
        print(f"MSE: {results['mse']:.6f}")
        print(f"MAE: {results['mae']:.6f}")
        print(f"R²: {results['r2']:.6f}")
        print(f"训练时间: {results['training_time']:.1f}秒")
        
        # 保存训练配置
        config_path = os.path.join(args.save_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'best_params_path': args.best_params_path,
                'best_params': best_params,
                'results': {
                    'best_val_loss': float(results['best_val_loss']),
                    'mse': float(results['mse']),
                    'mae': float(results['mae']),
                    'r2': float(results['r2']),
                    'training_time': float(results['training_time']),
                },
                'timestamp': str(datetime.now()),
            }, f, indent=2)
        
        print(f"\n💾 训练配置已保存到: {config_path}")
        print(f"\n✅ 训练完成！")
        print(f"结果已保存到: {args.save_dir}")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    from datetime import datetime
    main()