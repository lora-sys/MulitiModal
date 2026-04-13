"""
WESAD 数据集训练主程序

使用双门控融合模型在 WESAD 数据集上进行放松度预测（回归任务）

用法:
    python main_wesad.py --data_root /path/to/WESAD
"""

import os
import sys
import argparse
import torch

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'model'))
sys.path.insert(0, os.path.join(script_dir, 'dataset'))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='WESAD 数据集训练 - 双门控融合模型')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='WESAD 数据集根目录')
    
    # 模型参数
    parser.add_argument('--tcm_model_path', type=str,
                        default='data/tcm_ft_transformer/checkpoints/best_model.pth',
                        help='TCM 模型权重路径')
    parser.add_argument('--tcm_scaler_path', type=str,
                        default='data/tcm_ft_transformer/data/scaler_params.npz',
                        help='TCM 标准化参数路径')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--encoder_lr_ratio', type=float, default=0.1,
                        help='编码器学习率比例')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str,
                        default='experiment/results/wesad',
                        help='保存目录')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型保存路径（默认: save_dir/best_model.pth）')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("🚀 WESAD 数据集训练 - 双门控融合模型")
    print("=" * 60)
    
    # 打印配置
    print(f"\n📋 配置信息:")
    print(f"  数据集: {args.data_root}")
    print(f"  TCM 模型: {args.tcm_model_path}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  设备: {args.device}")
    print(f"  保存目录: {args.save_dir}")
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"\n⚠️  CUDA 不可用，切换到 CPU")
        args.device = 'cpu'
    
    # 检查数据集
    if not os.path.exists(args.data_root):
        print(f"\n❌ 错误: WESAD 数据集不存在: {args.data_root}")
        print("请下载 WESAD 数据集并解压到指定目录")
        return
    
    # 检查 TCM 模型
    if not os.path.exists(args.tcm_model_path):
        print(f"\n⚠️  警告: TCM 模型不存在: {args.tcm_model_path}")
        print("请确保 TCM 模型文件存在")
    
    # 导入训练函数
    from train import train_wesad
    
    # 开始训练
    print(f"\n🎯 开始训练...")
    print("-" * 60)
    
    try:
        results = train_wesad(
            data_root=args.data_root,
            model_path=args.model_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            encoder_lr_ratio=args.encoder_lr_ratio,
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
        
        print(f"\n✅ 训练完成！")
        print(f"结果已保存到: {args.save_dir}")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
