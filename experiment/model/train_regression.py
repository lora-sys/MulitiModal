"""
回归训练脚本
支持回归任务和噪声增强
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# 固定随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), 'dataset'))
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), 'recorder'))

from model import get_model
from config_regression import REGRESSION_CONFIG, REGRESSION_TRAIN_CONFIG, SCHEDULER_CONFIGS, CURRENT_SCHEDULER


# =========================================================================
# 噪声增强函数
# =========================================================================
def add_gaussian_noise(waveform, intensity=0.1):
    """添加高斯噪声"""
    noise = torch.randn_like(waveform) * intensity
    return waveform + noise


def add_drift_noise(waveform, intensity=0.05):
    """添加漂移噪声"""
    device = waveform.device
    drift = torch.linspace(0, intensity, waveform.shape[-1], device=device).view(1, 1, -1)
    return waveform + drift


def add_dropout_noise(waveform, dropout_prob=0.1):
    """添加数据丢失噪声"""
    mask = torch.rand_like(waveform) > dropout_prob
    return waveform * mask


def apply_noise_augmentation(waveform, noise_types=['gaussian', 'drift', 'dropout'], probability=0.5, intensity='medium'):
    """应用噪声增强"""
    if random.random() > probability:
        return waveform

    intensity_map = {'low': 0.05, 'medium': 0.1, 'high': 0.2}
    noise_intensity = intensity_map.get(intensity, 0.1)

    noise_type = random.choice(noise_types)

    if noise_type == 'gaussian':
        return add_gaussian_noise(waveform, noise_intensity)
    elif noise_type == 'drift':
        return add_drift_noise(waveform, noise_intensity * 0.5)
    elif noise_type == 'dropout':
        return add_dropout_noise(waveform, noise_intensity)
    else:
        return waveform


# =========================================================================
# 数据加载
# =========================================================================
class RegressionDataset(torch.utils.data.Dataset):
    """回归数据集"""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.dynamic = torch.from_numpy(data['dynamic']).float()
        self.static_basic = torch.from_numpy(data['static_basic']).float()
        self.static_scores = torch.from_numpy(data['static_scores']).float()
        self.constitution = torch.from_numpy(data['constitution']).long()
        self.label = torch.from_numpy(data['label_original']).float()  # 使用原始标签（未归一化）
        self.len = len(self.label)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            'dynamic': self.dynamic[idx],
            'static_basic': self.static_basic[idx],
            'static_scores': self.static_scores[idx],
            'constitution': self.constitution[idx],
            'label': self.label[idx]
        }


def create_dataset(npz_path):
    """创建数据集"""
    dataset = RegressionDataset(npz_path)
    print(f"[*] 加载数据集: {len(dataset)} 样本 from {npz_path}")
    return dataset


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """划分数据集（分层抽样）"""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # 获取所有标签
    labels = [dataset[i]['label'].item() for i in range(total_size)]

    # 按标签分层抽样
    unique_labels = np.unique(labels)
    train_indices = []
    val_indices = []
    test_indices = []

    for label in unique_labels:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        random.Random(seed).shuffle(label_indices)

        n_train = int(len(label_indices) * train_ratio)
        n_val = int(len(label_indices) * val_ratio)

        train_indices.extend(label_indices[:n_train])
        val_indices.extend(label_indices[n_train:n_train + n_val])
        test_indices.extend(label_indices[n_train + n_val:])

    random.Random(seed).shuffle(train_indices)
    random.Random(seed).shuffle(val_indices)
    random.Random(seed).shuffle(test_indices)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"[*] 数据集划分: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


# =========================================================================
# 训练和评估
# =========================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, model_type, noise_augmentation=False):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        dynamic = batch['dynamic'].to(device)
        static_basic = batch['static_basic'].to(device)
        static_scores = batch['static_scores'].to(device)
        constitution = batch['constitution'].to(device)
        labels = batch['label'].to(device)

        # 噪声增强（只在训练时）
        if noise_augmentation:
            noise_types = REGRESSION_TRAIN_CONFIG.get('noise_types', ['gaussian', 'drift', 'dropout'])
            probability = REGRESSION_TRAIN_CONFIG.get('noise_probability', 0.5)
            intensity = REGRESSION_TRAIN_CONFIG.get('noise_intensity', 'medium')
            dynamic = apply_noise_augmentation(dynamic, noise_types, probability, intensity)

        optimizer.zero_grad()

        # 回归输出是单个值，需要squeeze
        outputs = model(dynamic, static_basic, static_scores, constitution)
        outputs = outputs.squeeze(-1)  # (B,) -> (B,)
        labels = labels.float().squeeze(-1)  # (B,) -> (B,)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    mae = mean_absolute_error(all_labels, all_preds)
    return total_loss / len(dataloader), mae


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            dynamic = batch['dynamic'].to(device)
            static_basic = batch['static_basic'].to(device)
            static_scores = batch['static_scores'].to(device)
            constitution = batch['constitution'].to(device)
            labels = batch['label'].to(device)

            outputs = model(dynamic, static_basic, static_scores, constitution)
            outputs = outputs.squeeze(-1)
            labels = labels.float().squeeze(-1)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算回归指标
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)
    pearson, _ = pearsonr(all_labels, all_preds)

    return total_loss / len(dataloader), mae, rmse, r2, pearson, all_labels, all_preds


# =========================================================================
# 主训练函数
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='回归任务训练')
    parser.add_argument('--model_type', type=str, default='baseline_a',
                        choices=['baseline_a', 'baseline_b', 'baseline_c'],
                        help='模型类型')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据集路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--noise_augmentation', action='store_true',
                        help='是否使用噪声增强')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

    # 设备
    device = torch.device(args.device)
    print(f"[*] 使用设备: {device}")

    # 加载数据
    print("\n" + "=" * 60)
    print("📊 加载数据")
    print("=" * 60)
    dataset = create_dataset(args.data_path)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 创建模型
    print("\n" + "=" * 60)
    print("🧠 创建模型")
    print("=" * 60)
    model = get_model(model_type=args.model_type, num_classes=1, num_constitutions=39).to(device)
    print(f"[*] 模型类型: {args.model_type}")

    # 损失函数
    loss_type = REGRESSION_TRAIN_CONFIG.get('loss', 'MSE')
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == 'MAE':
        criterion = nn.L1Loss()
    elif loss_type == 'HuberLoss':
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()
    print(f"[*] 损失函数: {loss_type}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=REGRESSION_TRAIN_CONFIG.get('weight_decay', 1e-4))

    # 学习率调度器
    scheduler_config = SCHEDULER_CONFIGS[CURRENT_SCHEDULER]
    if scheduler_config['type'] == 'CosineAnnealingWarmup':
        from warmup_scheduler import GradualWarmupScheduler
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_config['max_epochs'], eta_min=scheduler_config['eta_min'])
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=scheduler_config['warmup_epochs'], after_scheduler=scheduler_cosine)
    else:
        scheduler = None

    # 训练
    print("\n" + "=" * 60)
    print("🚀 开始训练")
    print("=" * 60)
    print(f"[*] 噪声增强: {'是' if args.noise_augmentation else '否'}")

    best_val_mae = float('inf')
    best_epoch = 0
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_maes = []

    for epoch in range(args.num_epochs):
        # 训练
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device, args.model_type, args.noise_augmentation)

        # 验证
        val_loss, val_mae, val_rmse, val_r2, val_pearson, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # 学习率调度
        if scheduler is not None:
            scheduler.step()

        # 打印进度
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, "
              f"Val R²: {val_r2:.4f}, Val Pearson: {val_pearson:.4f}")

        # 保存最佳模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoints', 'best_model.pth'))
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f"[*] 早停于 epoch {epoch+1}")
            break

    print("\n" + "=" * 60)
    print("✅ 训练完成")
    print("=" * 60)
    print(f"[*] 最佳验证MAE: {best_val_mae:.4f} (Epoch {best_epoch+1})")

    # 加载最佳模型并测试
    print("\n" + "=" * 60)
    print("📊 测试集评估")
    print("=" * 60)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')))
    test_loss, test_mae, test_rmse, test_r2, test_pearson, test_labels, test_preds = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test Pearson: {test_pearson:.4f}")

    # 保存配置和结果
    run_config = {
        'model_type': args.model_type,
        'noise_augmentation': args.noise_augmentation,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'device': args.device,
        'best_val_mae': best_val_mae,
        'best_epoch': best_epoch,
        'test_metrics': {
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'r2': float(test_r2),
            'pearson': float(test_pearson)
        }
    }

    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_maes, label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Validation MAE')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'figures', 'training_curves.png'), dpi=150)
    plt.close()

    # 绘制预测散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(test_labels, test_preds, alpha=0.5, s=10)
    plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'r--', lw=2)
    plt.xlabel('True Score')
    plt.ylabel('Predicted Score')
    plt.title(f'Test Predictions (MAE={test_mae:.2f})')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, 'figures', 'predictions.png'), dpi=150)
    plt.close()

    print(f"\n[*] 结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
