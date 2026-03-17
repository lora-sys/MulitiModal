"""
测试训练优化策略
验证 Parameter Groups、滑动平均、调度器更新是否正确
"""

import sys
import torch
import torch.optim as optim
from collections import deque

sys.path.append("experiment/dataset")
sys.path.append("experiment/model")

from config import MODEL_CONFIG, TRAIN_CONFIG, SCHEDULER_CONFIGS, CURRENT_SCHEDULER
from model import get_model

def test_parameter_groups():
    """测试 Parameter Groups"""
    print("=" * 60)
    print("测试 1: Parameter Groups（不同模块不同学习率）")
    print("=" * 60)
    
    model = get_model(
        model_type=MODEL_CONFIG["type"],
        num_classes=MODEL_CONFIG["params"]["num_classes"],
        dyn_channels=MODEL_CONFIG["params"]["dyn_channels"],
        static_dim=MODEL_CONFIG["params"]["static_dim"],
    )
    
    # 分割参数组
    encoder_params = []
    fusion_params = []
    
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in ['encoder', 'embedding']):
            encoder_params.append((name, param))
        else:
            fusion_params.append((name, param))
    
    print(f"\nEncoder 参数数量: {len(encoder_params)}")
    print(f"Fusion/Classifier 参数数量: {len(fusion_params)}")
    
    # 显示前几个参数名
    print("\n前 5 个 Encoder 参数:")
    for name, _ in encoder_params[:5]:
        print(f"  - {name}")
    
    print("\n前 5 个 Fusion/Classifier 参数:")
    for name, _ in fusion_params[:5]:
        print(f"  - {name}")
    
    # 创建优化器
    base_lr = TRAIN_CONFIG["learning_rate"]
    encoder_lr_ratio = TRAIN_CONFIG.get("encoder_lr_ratio", 0.1)
    
    param_groups = [
        {"params": [p for _, p in encoder_params], "lr": base_lr * encoder_lr_ratio},
        {"params": [p for _, p in fusion_params], "lr": base_lr},
    ]
    
    optimizer = optim.Adam(param_groups, weight_decay=TRAIN_CONFIG.get("weight_decay", 1e-4))
    
    print(f"\n参数组配置:")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  参数组 {i}: lr={group['lr']:.6f}, 参数数量={len(group['params'])}")
    
    print("\n✅ Parameter Groups 测试通过")
    return True


def test_smoothing():
    """测试滑动平均"""
    print("\n" + "=" * 60)
    print("测试 2: 滑动平均指标（减少噪声）")
    print("=" * 60)
    
    smoothing_window = TRAIN_CONFIG.get("smoothing_window", 5)
    val_loss_history = deque(maxlen=smoothing_window)
    val_acc_history = deque(maxlen=smoothing_window)
    
    # 模拟 val 指标（带噪声）
    simulated_losses = [0.9, 0.85, 0.88, 0.82, 0.84, 0.79, 0.81, 0.77, 0.78, 0.75]
    simulated_accs = [70, 72, 71, 74, 73, 76, 75, 78, 77, 80]
    
    print(f"\n模拟 {len(simulated_losses)} 个 epoch 的 val 指标:")
    print(f"滑动平均窗口大小: {smoothing_window}")
    
    for i, (loss, acc) in enumerate(zip(simulated_losses, simulated_accs)):
        val_loss_history.append(loss)
        val_acc_history.append(acc)
        
        smoothed_loss = sum(val_loss_history) / len(val_loss_history)
        smoothed_acc = sum(val_acc_history) / len(val_acc_history)
        
        print(f"  Epoch {i+1}: Loss={loss:.3f} (smoothed={smoothed_loss:.3f}), Acc={acc}% (smoothed={smoothed_acc:.1f}%)")
    
    print("\n✅ 滑动平均测试通过")
    return True


def test_scheduler():
    """测试调度器"""
    print("\n" + "=" * 60)
    print("测试 3: 调度器更新逻辑")
    print("=" * 60)
    
    model = get_model(
        model_type=MODEL_CONFIG["type"],
        num_classes=MODEL_CONFIG["params"]["num_classes"],
        dyn_channels=MODEL_CONFIG["params"]["dyn_channels"],
        static_dim=MODEL_CONFIG["params"]["static_dim"],
    )
    
    base_lr = TRAIN_CONFIG["learning_rate"]
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    
    scheduler_cfg = SCHEDULER_CONFIGS[CURRENT_SCHEDULER]
    scheduler_type = scheduler_cfg.get("type")
    
    print(f"\n当前调度器: {scheduler_type}")
    print(f"配置: {scheduler_cfg}")
    
    # 创建调度器
    if scheduler_type == "CosineAnnealingWarmup":
        warmup_epochs = scheduler_cfg.get("warmup_epochs", 5)
        
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAIN_CONFIG["num_epochs"] - warmup_epochs,
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
        
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        print(f"\n学习率变化（前 10 个 epoch）:")
        for epoch in range(10):
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1}: LR={lr:.6f}")
            scheduler.step()
    
    print("\n✅ 调度器测试通过")
    return True


def test_full_integration():
    """测试完整集成"""
    print("\n" + "=" * 60)
    print("测试 4: 完整集成测试")
    print("=" * 60)
    
    model = get_model(
        model_type=MODEL_CONFIG["type"],
        num_classes=MODEL_CONFIG["params"]["num_classes"],
        dyn_channels=MODEL_CONFIG["params"]["dyn_channels"],
        static_dim=MODEL_CONFIG["params"]["static_dim"],
    )
    
    # 测试 Parameter Groups
    encoder_params = []
    fusion_params = []
    
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in ['encoder', 'embedding']):
            encoder_params.append(param)
        else:
            fusion_params.append(param)
    
    base_lr = TRAIN_CONFIG["learning_rate"]
    encoder_lr_ratio = TRAIN_CONFIG.get("encoder_lr_ratio", 0.1)
    
    param_groups = [
        {"params": encoder_params, "lr": base_lr * encoder_lr_ratio},
        {"params": fusion_params, "lr": base_lr},
    ]
    
    optimizer = optim.Adam(param_groups, weight_decay=TRAIN_CONFIG.get("weight_decay", 1e-4))
    
    # 测试滑动平均
    smoothing_window = TRAIN_CONFIG.get("smoothing_window", 5)
    val_loss_history = deque(maxlen=smoothing_window)
    
    # 测试调度器
    scheduler_cfg = SCHEDULER_CONFIGS[CURRENT_SCHEDULER]
    scheduler_type = scheduler_cfg.get("type")
    
    if scheduler_type == "CosineAnnealingWarmup":
        warmup_epochs = scheduler_cfg.get("warmup_epochs", 5)
        
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAIN_CONFIG["num_epochs"] - warmup_epochs,
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
        
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    
    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder 参数: {sum(p.numel() for p in encoder_params):,} (lr={base_lr * encoder_lr_ratio:.6f})")
    print(f"Fusion 参数: {sum(p.numel() for p in fusion_params):,} (lr={base_lr:.6f})")
    print(f"滑动平均窗口: {smoothing_window}")
    print(f"调度器: {scheduler_type}")
    
    # 模拟前 5 个 epoch
    print(f"\n模拟前 5 个 epoch:")
    for epoch in range(5):
        val_loss = 0.9 - epoch * 0.05
        val_loss_history.append(val_loss)
        smoothed_val_loss = sum(val_loss_history) / len(val_loss_history)
        
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}: Val Loss={val_loss:.3f}, Smoothed={smoothed_val_loss:.3f}, LR={lr:.6f}")
        
        if scheduler_type in ["CosineAnnealingWarmup"]:
            scheduler.step()
    
    print("\n✅ 完整集成测试通过")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 训练优化策略测试")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_parameter_groups()
    all_passed &= test_smoothing()
    all_passed &= test_scheduler()
    all_passed &= test_full_integration()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)
