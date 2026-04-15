"""
训练器模块
实现 KL Divergence 损失、早停、梯度裁剪、学习率调度等训练逻辑
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import TRAIN_CONFIG, MODEL_CONFIG, OUTPUT_FILES
from ft_transformer import get_model


class KLDivLossWithLogSoftmax(nn.Module):
    """
    KL Divergence 损失函数
    
    注意：由于模型输出层已经包含 Softmax，输出的是概率分布，
    所以这里直接使用 torch.log 而不是 log_softmax
    
    KL(P||Q) = sum(P * log(P/Q))
    其中 P 是目标分布，Q 是预测分布
    """
    def __init__(self, reduction='batchmean'):
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction=reduction, log_target=False)
    
    def forward(self, probs, target):
        """
        Args:
            probs: (B, n_classes) 模型输出（经过 softmax 的概率分布）
            target: (B, n_classes) 目标概率分布
            
        Returns:
            loss: KL 散度损失
        """
        # 数值保护：防止 log(0)
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        
        # 计算 log(概率分布)
        log_probs = torch.log(probs)
        
        # 计算 KL 散度
        loss = self.kl_div(log_probs, target)
        
        return loss


class WarmupScheduler:
    """
    Warmup 学习率调度器
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # 创建 CosineAnnealingLR
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr
        )
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup 阶段：线性增长
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine 退火阶段
            self.cosine_scheduler.step()
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    """
    训练器类
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_ratio=0.05,
        num_epochs=50,
        patience=5,
        grad_clip_max_norm=1.0,
        checkpoint_dir='./checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.grad_clip_max_norm = grad_clip_max_norm
        self.checkpoint_dir = checkpoint_dir
        
        # 确保检查点目录存在
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 损失函数
        self.criterion = KLDivLossWithLogSoftmax()
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        warmup_epochs = int(num_epochs * warmup_ratio)
        self.scheduler = WarmupScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs,
            min_lr=1e-6
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # 早停
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
    def train_epoch(self, epoch):
        """
        训练一个 epoch
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_max_norm
            )
            
            # 更新参数
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """
        验证
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="[Val]")
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 记录损失
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """
        完整训练流程
        """
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"学习率: {self.optimizer.param_groups[0]['lr']}")
        print(f"权重衰减: {self.optimizer.param_groups[0]['weight_decay']}")
        print(f"Warmup 比例: {TRAIN_CONFIG['warmup_ratio']}")
        print(f"梯度裁剪: {self.grad_clip_max_norm}")
        print(f"早停耐心值: {self.patience}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # 更新学习率
            self.scheduler.step(epoch)
            current_lr = self.scheduler.get_lr()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # 打印进度
            print(f"\nEpoch {epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {current_lr:.6f}")
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                
                # 保存最佳模型
                self.save_checkpoint(os.path.join(self.checkpoint_dir, OUTPUT_FILES["best_model"]))
                print(f"  → 保存最佳模型 (Val Loss: {val_loss:.6f})")
            else:
                self.epochs_no_improve += 1
                if self.patience is not None:
                    print(f"  → 无改进 ({self.epochs_no_improve}/{self.patience})")
                    
                    if self.epochs_no_improve >= self.patience:
                        print(f"\n早停触发！在第 {epoch+1} 轮停止训练。")
                        break
                else:
                    print(f"  → 无改进 ({self.epochs_no_improve}/None) - 早停已禁用")
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {total_time/60:.2f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        
        return self.history
    
    def save_checkpoint(self, path):
        """
        保存检查点
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """
        加载检查点
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        print(f"已加载检查点: {path}")


def train_single_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    model_params,
    train_config,
    fold_idx=0
):
    """
    训练单个 fold
    
    Args:
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征
        y_val: 验证集标签
        model_params: 模型参数
        train_config: 训练配置
        fold_idx: fold 索引
        
    Returns:
        history: 训练历史
        best_val_loss: 最佳验证损失
    """
    from preprocess import create_dataloaders
    
    # 创建 DataLoader
    train_loader = create_dataloaders(
        X_train, y_train,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=0  # 避免多进程问题
    )
    
    val_loader = create_dataloaders(
        X_val, y_val,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    model = get_model(
        n_features=8,
        n_classes=9,
        **model_params
    )
    
    # 创建训练器
    checkpoint_dir = os.path.join(train_config.get('checkpoint_dir', './checkpoints'), f'fold_{fold_idx}')
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=train_config.get('device', 'cuda'),
        learning_rate=train_config.get('learning_rate', 1e-3),
        weight_decay=train_config.get('weight_decay', 0.01),
        warmup_ratio=train_config.get('warmup_ratio', 0.05),
        num_epochs=train_config.get('num_epochs', 50),
        patience=train_config.get('patience', 5),
        grad_clip_max_norm=train_config.get('grad_clip_max_norm', 1.0),
        checkpoint_dir=checkpoint_dir
    )
    
    # 训练
    history = trainer.train()
    
    return history, trainer.best_val_loss


if __name__ == "__main__":
    # 测试训练器
    print("测试训练器...")
    
    # 创建虚拟数据
    X_train = np.random.randn(1000, 8).astype(np.float32)
    y_train = np.random.dirichlet(np.ones(9), size=1000).astype(np.float32)
    X_val = np.random.randn(200, 8).astype(np.float32)
    y_val = np.random.dirichlet(np.ones(9), size=200).astype(np.float32)
    
    # 训练
    history, best_val_loss = train_single_fold(
        X_train, y_train,
        X_val, y_val,
        model_params=MODEL_CONFIG,
        train_config={**TRAIN_CONFIG, 'num_epochs': 5, 'patience': 3},
        fold_idx=0
    )
    
    print(f"\n训练历史:")
    print(f"  Train Loss: {history['train_loss']}")
    print(f"  Val Loss: {history['val_loss']}")
    print(f"  Best Val Loss: {best_val_loss:.6f}")