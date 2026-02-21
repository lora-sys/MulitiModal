"""K-Fold 交叉验证训练脚本"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
sys.path.append("experiment/dataset")
sys.path.append("experiment/model")
from csv_source import NPZDataSource
from nk2_processor import NK2Preprocessor
from massage_dataset import MassageDataset
from model import get_model
import yaml
# 配置
MODEL_TYPE = "cnn"  # 可改为 cnn/lstm
N_FOLDS = 5
NUM_EPOCHS = 10
def load_config():
    with open("experiment/dataset/config.yaml", 'r') as f:
        return yaml.safe_load(f)
def create_dataset(config):
    npz_path = "experiment/model/processed_data.npz"
    source = NPZDataSource(npz_path)
    source.initialize()
    preprocessor = NK2Preprocessor(config)
    return MassageDataset(source, preprocessor)
def train_fold(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    best_acc = 0
    for epoch in range(epochs):
        # 训练
        model.train()
        for batch in train_loader:
            x_dyn, x_stat, y = batch["dynamic"].to(device), batch["static"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(x_dyn, x_stat)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x_dyn, x_stat, y = batch["dynamic"].to(device), batch["static"].to(device), batch["label"].to(device)
                outputs = model(x_dyn, x_stat)
                _, preds = outputs.max(1)
                total += y.size(0)
                correct += preds.eq(y).sum().item()
        
        acc = 100 * correct / total
        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
    
    return best_acc




def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载全部数据
    full_dataset = create_dataset(config)
    X = np.arange(len(full_dataset))
    y = np.array([full_dataset[i]["label"] for i in range(len(full_dataset))])
    
    # K-Fold
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    print(f"🚀 {N_FOLDS}-Fold 交叉验证 | 模型: {MODEL_TYPE}")
    print("-" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n📦 Fold {fold+1}/{N_FOLDS}")
        
        # 创建子数据集
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 创建模型
        model = get_model(model_type=MODEL_TYPE, num_classes=4, dyn_channels=2, static_dim=4).to(device)
        
        # 训练
        acc = train_fold(model, train_loader, val_loader, NUM_EPOCHS, device)
        fold_results.append(acc)
        
        print(f"  Fold {fold+1} 最佳准确率: {acc:.2f}%")
 
    # 结果
    print("\n" + "=" * 50)
    print(f"📊 {N_FOLDS}-Fold 交叉验证结果")
    print("=" * 50)
    for i, acc in enumerate(fold_results, 1):
        print(f"  Fold {i}: {acc:.2f}%")
    print(f"\n🎯 平均准确率: {np.mean(fold_results):.2f}%")
    print(f"📈 标准差: {np.std(fold_results):.2f}%")
if __name__ == "__main__":
    main()