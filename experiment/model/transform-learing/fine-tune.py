"""
迁移学习微调脚本
- 加载预训练模型
- 冻结 CNN 层
- 用新数据微调分类器
- 保存为新模型
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
import yaml
from model import get_model
from config import MODEL_CONFIG, TRAIN_CONFIG




# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 预训练模型路径
PRETRAINED_PATH = "experiment/model/bfoundation_model_inception.pth"
# 新数据路径（50条带偏差数据）
NEW_DATA_PATH = "experiment/model/transform-learing/real_data_50_samples.npz"
# 保存路径
FINETUNED_PATH = "experiment/model/bfoundation_model_inception_finetuned.pth"
# ====== 缺失的常量 =======
FREEZE_BACKBONE = True      # 是否冻结 backbone
FINETUNE_LR = 0.0005       # 微调学习率
FINETUNE_EPOCHS = 15       # 微调轮数





def load_pretrain_data():
    """加载预训练数据（10k）"""
    print("📂 加载预训练数据...")
    data = np.load("experiment/model/pretrain_10k.npz")
    dynamic = torch.tensor(data['dynamic'], dtype=torch.float32)
    static = torch.tensor(data['static'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.long)
    return TensorDataset(dynamic, static, labels)



def load_new_data():
    """加载新数据（50条带偏差）"""
    print("📂 加载新数据（50条带偏差）...")
    if not os.path.exists(NEW_DATA_PATH):
        print(f"❌ 新数据文件不存在: {NEW_DATA_PATH}")
        print("请先运行 generate_flow_shot.py 生成数据")
        return None
    
    data = np.load(NEW_DATA_PATH)
    dynamic = torch.tensor(data['dynamic'], dtype=torch.float32)
    static = torch.tensor(data['static'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.long)
    return TensorDataset(dynamic, static, labels)


def freeze_backbone(model):
    """冻结 CNN/Transformer 层"""
    print("❄️ 冻结特征提取器...")
    for param in model.dynamic_encoder.parameters():
        param.requires_grad = False
    # 确认只有这些层被冻结
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.dynamic_encoder.parameters())
    print(f"   可训练参数: {trainable:,} | 冻结参数: {frozen:,}")
    
    
def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for dyn, stat, lbl in loader:
        dyn, stat, lbl = dyn.to(device), stat.to(device), lbl.to(device)
        
        optimizer.zero_grad()
        out = model(dyn, stat)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = torch.max(out, 1)
        total += lbl.size(0)
        correct += (pred == lbl).sum().item()
    
    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for dyn, stat, lbl in loader:
            dyn, stat, lbl = dyn.to(device), stat.to(device), lbl.to(device)
            out = model(dyn, stat)
            loss = criterion(out, lbl)
            
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += lbl.size(0)
            correct += (pred == lbl).sum().item()
    
    return total_loss / len(loader), 100 * correct / total



def run_finetuning():
    print("=" * 60)
    print("🚀 迁移学习 - 微调脚本")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"预训练模型: {PRETRAINED_PATH}")
    print(f"新数据: {NEW_DATA_PATH}")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n📦 加载预训练模型...")
    model = get_model(
        model_type=MODEL_CONFIG["type"],
        num_classes=MODEL_CONFIG["params"]["num_classes"],
        dyn_channels=MODEL_CONFIG["params"]["dyn_channels"],
        static_dim=MODEL_CONFIG["params"]["static_dim"],
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
        print("✅ 成功加载预训练模型!")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 2. 冻结 backbone
    if FREEZE_BACKBONE:
        freeze_backbone(model)
    
    # 3. 加载新数据
    new_data = load_new_data()
    if new_data is None:
        return
    
    # 划分训练/验证
    train_ds, val_ds = random_split(new_data, [40, 10])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=10, shuffle=False)
    
    # 4. 配置优化器（只优化可训练参数）
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=FINETUNE_LR)
    criterion = nn.CrossEntropyLoss()
    
    # 5. 微调训练
    print(f"\n⚡ 开始微调 ({FINETUNE_EPOCHS} Epochs)...")
    print("-" * 60)
    
    start_time = time.time()
    best_acc = 0
    
    for epoch in range(FINETUNE_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        
        if (epoch + 1) % 5 == 0 or val_acc > best_acc:
            print(f"Epoch {epoch+1:02d} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Loss: {train_loss:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), FINETUNED_PATH)
    
    elapsed = time.time() - start_time
    
    # 6. 结果总结
    print("-" * 60)
    print(f"✅ 微调完成!")
    print(f"   最佳验证准确率: {best_acc:.1f}%")
    print(f"   耗时: {elapsed:.1f} 秒")
    print(f"   模型保存至: {FINETUNED_PATH}")
    print("\n📊 对比测试：")
    print(f"   原模型: {PRETRAINED_PATH}")
    print(f"   微调模型: {FINETUNED_PATH}")
if __name__ == "__main__":
    run_finetuning()