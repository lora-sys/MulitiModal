"""
数据预处理模块
实现数据加载、清洗、类型转换、动态划分、标准化、标签处理
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_CONFIG, OUTPUT_FILES


class TCMConstitutionDataset(Dataset):
    """
    中医体质数据集
    """
    def __init__(self, X, y):
        """
        Args:
            X: (N, n_features) 特征矩阵
            y: (N, n_classes) 标签矩阵（概率分布）
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(data_path=None, test_split=0.1, random_state=42, epsilon=0.01):
    """
    加载和预处理数据
    
    步骤：
    1. 读取 CSV
    2. Gender 字段编码 (Male → 0, Female → 1)
    3. 数据类型转换（float32）
    4. 动态数据集划分（90% 训练验证池，10% 锁定测试集）
    5. 特征标准化（仅在训练验证池上计算参数）
    6. 标签预处理（Epsilon 平滑 + 行归一化）
    
    Args:
        data_path: 数据文件路径
        test_split: 测试集比例
        random_state: 随机种子
        epsilon: 标签平滑参数
        
    Returns:
        X_pool: (N_pool, n_features) 训练验证池特征
        y_pool: (N_pool, n_classes) 训练验证池标签
        X_test: (N_test, n_features) 测试集特征
        y_test: (N_test, n_classes) 测试集标签
        scaler_params: 标准化参数（mean, std）
    """
    if data_path is None:
        data_path = DATA_CONFIG["input_path"]
    
    print("=" * 60)
    print("数据预处理")
    print("=" * 60)
    
    # =====================================================================
    # 步骤 1: 数据加载与清洗
    # =====================================================================
    print(f"\n[步骤 1] 加载数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"      原始数据量: {len(df)} 行, {len(df.columns)} 列")
    
    # Gender 字段编码
    if 'Gender' in df.columns:
        print(f"\n[步骤 1.1] Gender 字段编码...")
        gender_mapping = {'Male': 0, 'Female': 1, 'male': 0, 'female': 1, 'M': 0, 'F': 1}
        df['Gender'] = df['Gender'].map(gender_mapping)
        df['Gender'] = df['Gender'].astype(np.float32)
        print(f"      Gender 编码完成: Male → 0, Female → 1")
    else:
        print(f"\n[步骤 1.1] 警告: 未找到 Gender 列，跳过编码")
    
    # 数据类型转换
    print(f"\n[步骤 1.2] 数据类型转换...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
    
    # 检查缺失值
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"      警告: 发现 {missing_count} 个缺失值，使用中位数填充")
        df = df.fillna(df.median())
    else:
        print(f"      无缺失值")
    
    # =====================================================================
    # 步骤 2: 动态数据集划分
    # =====================================================================
    print(f"\n[步骤 2] 动态数据集划分...")
    
    # 提取特征和标签
    n_features = DATA_CONFIG["n_features"]
    n_classes = DATA_CONFIG["n_classes"]

    # 前 n_features 列为特征，最后 n_classes 列为标签
    X = df.iloc[:, :n_features].values
    y = df.iloc[:, -n_classes:].values

    print(f"      特征矩阵 X: {X.shape}")
    print(f"      标签矩阵 y: {y.shape}")
    print(f"      特征列: {list(df.columns[:n_features])}")
    print(f"      标签列: {list(df.columns[-n_classes:])}")
    
    # 划分训练验证池和测试集
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y,
        test_size=test_split,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"      训练验证池: {len(X_pool)} 样本 ({(1-test_split)*100:.1f}%)")
    print(f"      锁定测试集: {len(X_test)} 样本 ({test_split*100:.1f}%)")
    
    # 保存测试集索引（用于后续验证）
    test_ids = np.arange(len(X_pool), len(X_pool) + len(X_test))
    np.save(os.path.join(os.path.dirname(data_path), OUTPUT_FILES["fixed_test_ids"]), test_ids)
    print(f"      已保存测试集索引: {OUTPUT_FILES['fixed_test_ids']}")
    
    # =====================================================================
    # 步骤 3: 特征标准化（防止数据泄露）
    # =====================================================================
    print(f"\n[步骤 3] 特征标准化...")
    
    # 仅在训练验证池上计算标准化参数
    scaler = StandardScaler()
    X_pool_scaled = scaler.fit_transform(X_pool)
    
    # 使用同样的参数对测试集进行标准化
    X_test_scaled = scaler.transform(X_test)
    
    # 保存标准化参数
    scaler_params = {
        'mean': scaler.mean_,
        'std': scaler.scale_
    }
    scaler_path = os.path.join(os.path.dirname(data_path), OUTPUT_FILES["scaler_params"])
    np.savez(scaler_path, **scaler_params)
    print(f"      已保存标准化参数: {OUTPUT_FILES['scaler_params']}")
    
    print(f"      训练验证池标准化完成")
    print(f"      测试集标准化完成（使用相同参数）")
    
    # =====================================================================
    # 步骤 4: 标签预处理（Epsilon 平滑 + 行归一化）
    # =====================================================================
    print(f"\n[步骤 4] 标签预处理...")
    
    # Epsilon 平滑
    print(f"      Epsilon 平滑 (epsilon={epsilon})...")
    y_pool_smooth = y_pool + epsilon
    y_test_smooth = y_test + epsilon
    
    # 行归一化
    print(f"      行归一化...")
    y_pool_normalized = y_pool_smooth / y_pool_smooth.sum(axis=1, keepdims=True)
    y_test_normalized = y_test_smooth / y_test_smooth.sum(axis=1, keepdims=True)
    
    # 验证归一化结果
    pool_sum = y_pool_normalized.sum(axis=1)
    test_sum = y_test_normalized.sum(axis=1)
    
    print(f"      训练验证池标签和: mean={pool_sum.mean():.6f}, std={pool_sum.std():.6f}")
    print(f"      测试集标签和: mean={test_sum.mean():.6f}, std={test_sum.std():.6f}")
    
    print(f"\n✅ 数据预处理完成!")
    print(f"   - X_pool: {X_pool_scaled.shape}")
    print(f"   - y_pool: {y_pool_normalized.shape}")
    print(f"   - X_test: {X_test_scaled.shape}")
    print(f"   - y_test: {y_test_normalized.shape}")
    
    return X_pool_scaled, y_pool_normalized, X_test_scaled, y_test_normalized, scaler_params


def create_dataloaders(X, y, batch_size=256, shuffle=True, num_workers=4):
    """
    创建 DataLoader
    
    Args:
        X: 特征矩阵
        y: 标签矩阵
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        
    Returns:
        dataloader: DataLoader 对象
    """
    dataset = TCMConstitutionDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def load_scaler_params(scaler_path=None):
    """
    加载标准化参数
    
    Args:
        scaler_path: 标准化参数文件路径
        
    Returns:
        scaler_params: 标准化参数字典
    """
    if scaler_path is None:
        scaler_path = os.path.join(
            os.path.dirname(DATA_CONFIG["input_path"]),
            OUTPUT_FILES["scaler_params"]
        )
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"标准化参数文件不存在: {scaler_path}")
    
    scaler_params = np.load(scaler_path)
    return {
        'mean': scaler_params['mean'],
        'std': scaler_params['std']
    }


if __name__ == "__main__":
    # 测试数据预处理
    X_pool, y_pool, X_test, y_test, scaler_params = load_and_preprocess_data()
    
    # 测试 DataLoader
    train_loader = create_dataloaders(X_pool, y_pool, batch_size=32, shuffle=True)
    
    for batch_X, batch_y in train_loader:
        print(f"\n批次测试:")
        print(f"  batch_X: {batch_X.shape}")
        print(f"  batch_y: {batch_y.shape}")
        print(f"  batch_y sum: {batch_y.sum(dim=1)}")
        break
