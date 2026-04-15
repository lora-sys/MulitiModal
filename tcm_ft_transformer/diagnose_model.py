"""
模型诊断脚本
用于检查模型预测单一类别的原因
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from config import TRAIN_CONFIG, MODEL_CONFIG, DATA_CONFIG, CONSTITUTION_NAMES
from ft_transformer import get_model
from preprocess import load_and_preprocess_data, create_dataloaders


def check_data_distribution(y_train, y_val, y_test):
    """检查数据分布"""
    print("=" * 80)
    print("1. 数据分布检查")
    print("=" * 80)
    
    # 转换为类别标签
    y_train_labels = np.argmax(y_train, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    print("\n训练集类别分布:")
    for i, name in enumerate(CONSTITUTION_NAMES):
        count = np.sum(y_train_labels == i)
        print(f"  {name}: {count} ({count/len(y_train_labels)*100:.2f}%)")
    
    print("\n验证集类别分布:")
    for i, name in enumerate(CONSTITUTION_NAMES):
        count = np.sum(y_val_labels == i)
        print(f"  {name}: {count} ({count/len(y_val_labels)*100:.2f}%)")
    
    print("\n测试集类别分布:")
    for i, name in enumerate(CONSTITUTION_NAMES):
        count = np.sum(y_test_labels == i)
        print(f"  {name}: {count} ({count/len(y_test_labels)*100:.2f}%)")
    
    # 检查是否有类别缺失
    unique_train = np.unique(y_train_labels)
    unique_val = np.unique(y_val_labels)
    unique_test = np.unique(y_test_labels)
    
    print(f"\n训练集唯一类别: {len(unique_train)} 个")
    print(f"验证集唯一类别: {len(unique_val)} 个")
    print(f"测试集唯一类别: {len(unique_test)} 个")
    
    if len(unique_test) == 1:
        print("\n⚠️  警告：测试集只包含 1 个类别！")
        print("   这会导致模型只能预测这一类别。")
        return False
    
    return True


def check_model_architecture(model):
    """检查模型架构"""
    print("\n" + "=" * 80)
    print("2. 模型架构检查")
    print("=" * 80)
    
    print("\n模型结构:")
    print(model)
    
    # 检查参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 检查输出层
    print("\n输出层检查:")
    for name, module in model.named_modules():
        if 'output' in name or 'final' in name:
            print(f"  {name}: {module}")
    
    return True


def check_model_predictions(model, test_loader, device, criterion):
    """检查模型预测"""
    print("\n" + "=" * 80)
    print("3. 模型预测检查")
    print("=" * 80)
    
    model.eval()
    all_preds = []
    all_trues = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            
            total_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
            all_probs.append(pred.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    # 获取预测类别
    y_pred_labels = np.argmax(all_preds, axis=1)
    y_true_labels = np.argmax(all_trues, axis=1)
    
    print(f"\n测试集平均损失: {avg_loss:.6f}")
    
    # 检查预测分布
    print("\n预测类别分布:")
    for i, name in enumerate(CONSTITUTION_NAMES):
        count = np.sum(y_pred_labels == i)
        print(f"  {name}: {count} ({count/len(y_pred_labels)*100:.2f}%)")
    
    unique_pred = np.unique(y_pred_labels)
    print(f"\n唯一预测类别: {len(unique_pred)} 个")
    print(f"预测类别: {[CONSTITUTION_NAMES[i] for i in unique_pred]}")
    
    if len(unique_pred) == 1:
        print("\n⚠️  警告：模型只预测 1 个类别！")
        print("   可能原因：")
        print("   1. 模型过拟合训练集的多数类")
        print("   2. 学习率过高导致模型崩溃")
        print("   3. 模型架构问题")
        print("   4. 数据预处理问题")
    
    # 检查预测概率分布
    print("\n预测概率分布统计:")
    for i, name in enumerate(CONSTITUTION_NAMES):
        probs = all_probs[:, i]
        print(f"  {name}:")
        print(f"    平均概率: {np.mean(probs):.6f}")
        print(f"    最大概率: {np.max(probs):.6f}")
        print(f"    最小概率: {np.min(probs):.6f}")
        print(f"    标准差: {np.std(probs):.6f}")
    
    # 详细分类报告
    print("\n" + "=" * 80)
    print("分类报告")
    print("=" * 80)
    print(classification_report(y_true_labels, y_pred_labels, 
                                target_names=CONSTITUTION_NAMES, 
                                zero_division=0))
    
    return all_preds, all_trues


def check_overfitting(model, train_loader, val_loader, device, criterion):
    """检查过拟合"""
    print("\n" + "=" * 80)
    print("4. 过拟合检查")
    print("=" * 80)
    
    # 训练集评估
    model.eval()
    train_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # 验证集评估
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    print(f"\n训练集损失: {train_loss:.6f}")
    print(f"验证集损失: {val_loss:.6f}")
    print(f"差距: {abs(val_loss - train_loss):.6f}")
    
    if val_loss > train_loss * 1.5:
        print("\n⚠️  警告：可能存在严重过拟合！")
        print("   建议：")
        print("   1. 增加数据增强")
        print("   2. 增加 dropout")
        print("   3. 使用 L2 正则化（已启用）")
        print("   4. 减少模型复杂度")
    elif val_loss > train_loss * 1.2:
        print("\n⚠️  注意：可能存在轻微过拟合")
    else:
        print("\n✓ 过拟合程度可接受")


def visualize_predictions(y_true, y_pred, save_path):
    """可视化预测结果"""
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CONSTITUTION_NAMES,
                yticklabels=CONSTITUTION_NAMES)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n混淆矩阵已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("模型诊断工具")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    from sklearn.model_selection import train_test_split

    X_pool, y_pool, X_test, y_test, scaler_params = load_and_preprocess_data()

    # 从池中划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool, test_size=0.2, random_state=42, stratify=np.argmax(y_pool, axis=1)
    )

    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 检查数据分布
    data_ok = check_data_distribution(y_train, y_val, y_test)
    
    # 创建数据加载器
    train_loader = create_dataloaders(X_train, y_train, 
                                      batch_size=TRAIN_CONFIG['batch_size'],
                                      shuffle=True, num_workers=0)
    val_loader = create_dataloaders(X_val, y_val,
                                    batch_size=TRAIN_CONFIG['batch_size'],
                                    shuffle=False, num_workers=0)
    test_loader = create_dataloaders(X_test, y_test,
                                     batch_size=TRAIN_CONFIG['batch_size'],
                                     shuffle=False, num_workers=0)
    
    # 加载模型
    print("\n加载最佳模型...")
    model = get_model(n_features=DATA_CONFIG["n_features"], n_classes=DATA_CONFIG["n_classes"], **MODEL_CONFIG)
    
    checkpoint_path = './checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载检查点: {checkpoint_path}")
        if 'best_val_loss' in checkpoint:
            print(f"最佳验证损失: {checkpoint['best_val_loss']:.6f}")
        elif 'val_loss' in checkpoint:
            print(f"验证损失: {checkpoint['val_loss']:.6f}")
    else:
        print(f"⚠️  未找到检查点: {checkpoint_path}")
        print("   使用未训练的模型")
    
    model = model.to(device)
    
    # 检查模型架构
    check_model_architecture(model)
    
    # 定义损失函数
    criterion = torch.nn.KLDivLoss(reduction='batchmean', log_target=False)
    
    # 检查模型预测
    preds, trues = check_model_predictions(model, test_loader, device, criterion)
    
    # 检查过拟合
    check_overfitting(model, train_loader, val_loader, device, criterion)
    
    # 可视化预测
    visualize_predictions(trues, preds, './checkpoints/diagnosis_confusion_matrix.png')
    
    # 总结和建议
    print("\n" + "=" * 80)
    print("诊断总结和建议")
    print("=" * 80)
    
    unique_pred = np.unique(np.argmax(preds, axis=1))
    if len(unique_pred) == 1:
        print("\n🔴 严重问题：模型只预测 1 个类别")
        print("\n可能原因和解决方案：")
        print("\n1. 如果测试集只有 1 个类别：")
        print("   - 这不是模型的问题，而是数据问题")
        print("   - 需要重新划分数据集，确保测试集包含所有类别")
        print("\n2. 如果测试集有多个类别但模型只预测 1 个：")
        print("   - 降低学习率（当前: 1e-3 → 尝试 1e-4）")
        print("   - 增加训练 epoch 数")
        print("   - 检查数据预处理是否正确")
        print("   - 尝试不同的模型架构")
        print("   - 使用类别权重平衡损失函数")
    else:
        print("\n✓ 模型能够预测多个类别")
        print("  检查分类报告以了解每个类别的表现")


if __name__ == "__main__":
    main()
