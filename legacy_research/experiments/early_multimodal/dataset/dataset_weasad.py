"""
WESAD 数据集处理脚本

WESAD (Wearable Stress and Affect Detection) 数据集
包含生理传感器数据（ECG, ACC等）和压力标签

数据格式：
- 静态特征：Age, Gender, BMI, HR, Temp, Resp, BP, SpO2 (8维)
- 动态特征：ECG 和 ACC 通道，重采样到 1000 点 (2, 1000)
- 标签：continuous_label (0=baseline, 1=stress, 2=amusement, 3=meditation)
- 回归目标：Relaxation = 1 - Stress (0-1)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json
from typing import Dict, List, Optional
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interfaces import BaseDataset, IDataSource


class WESADDataSource(IDataSource):
    """WESAD 数据源
    
    读取 WESAD 数据集的原始数据
    """
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: WESAD 数据集根目录
        """
        self.data_root = data_root
        self._sample_list = []
        self._data_cache = {}
        
    def initialize(self) -> bool:
        """初始化数据源，扫描所有受试者数据"""
        try:
            # WESAD 数据集通常包含 S1, S2, ..., S17 等受试者文件夹
            if not os.path.exists(self.data_root):
                print(f"❌ WESAD 数据根目录不存在: {self.data_root}")
                return False
            
            # 扫描所有受试者文件夹
            subjects = []
            for item in os.listdir(self.data_root):
                subject_dir = os.path.join(self.data_root, item)
                if os.path.isdir(subject_dir) and item.startswith('S'):
                    subjects.append(item)
            
            if not subjects:
                print(f"❌ 未找到受试者数据（预期格式: S1, S2, ...）")
                return False
            
            subjects.sort()
            print(f"📦 找到 {len(subjects)} 个受试者: {subjects}")
            
            # 为每个受试者收集样本
            sample_id = 0
            for subject in subjects:
                subject_dir = os.path.join(self.data_root, subject)
                subject_info_file = os.path.join(subject_dir, f"{subject}.json")
                
                if not os.path.exists(subject_info_file):
                    print(f"⚠️  跳过 {subject}（缺少 info 文件）")
                    continue
                
                # 读取受试者信息
                with open(subject_info_file, 'r') as f:
                    subject_info = json.load(f)
                
                # 读取传感器数据
                for session_data in subject_info.get('sessions', []):
                    session_label = session_data.get('label', 'unknown')
                    
                    # 只处理特定标签：baseline, stress, amusement, meditation
                    if session_label not in ['baseline', 'stress', 'amusement', 'meditation']:
                        continue
                    
                    # 数据文件路径
                    sensor_file = session_data.get('sensor_file', '')
                    if not sensor_file:
                        continue
                    
                    sensor_path = os.path.join(subject_dir, sensor_file)
                    if not os.path.exists(sensor_path):
                        continue
                    
                    # 添加到样本列表
                    self._sample_list.append({
                        'subject_id': subject,
                        'sample_id': sample_id,
                        'session_label': session_label,
                        'sensor_path': sensor_path,
                        'subject_info': subject_info,
                        'session_data': session_data,
                    })
                    
                    sample_id += 1
            
            print(f"✅ 加载了 {len(self._sample_list)} 个样本")
            return len(self._sample_list) > 0
            
        except Exception as e:
            print(f"❌ 初始化 WESAD 数据源失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_sample_list(self) -> List:
        """获取样本ID列表"""
        return list(range(len(self._sample_list)))
    
    def load_sample(self, sample_id: int) -> Dict:
        """加载单个样本"""
        try:
            if sample_id < 0 or sample_id >= len(self._sample_list):
                raise IndexError(f"样本ID超出范围: {sample_id}")
            
            sample_info = self._sample_list[sample_id]
            
            # 检查缓存
            if sample_id in self._data_cache:
                return self._data_cache[sample_id]

            # 加载传感器数据
            loaded = np.load(sample_info['sensor_path'], allow_pickle=True)

            # 转换为字典（处理 .npz 和 dict-in-array 两种格式）
            if isinstance(loaded, np.ndarray):
                sensor_data = loaded.item()
            elif isinstance(loaded, np.lib.npyio.NpzFile):
                sensor_data = {}
                for key in loaded.files:
                    val = loaded[key]
                    if val.dtype == np.object_:
                        sensor_data[key] = val.item()
                    else:
                        sensor_data[key] = val
            else:
                sensor_data = dict(loaded)

            # 提取 ECG 和 ACC 信号
            ecg = sensor_data.get('chest', {}).get('ECG', [])
            acc = sensor_data.get('chest', {}).get('ACC', [])
            
            # 重采样到 1000 点
            ecg_resampled = self._resample_signal(ecg, target_length=1000)
            acc_resampled = self._resample_signal(acc, target_length=1000)
            
            # 提取静态特征
            static_features = self._extract_static_features(
                sample_info['subject_info'],
                sensor_data
            )
            
            # 提取标签
            continuous_label = self._extract_label(
                sample_info['session_label'],
                sensor_data
            )
            
            # 构建样本
            sample = {
                'dynamic': np.stack([ecg_resampled, acc_resampled], axis=0),  # (2, 1000)
                'static_basic': static_features,  # (8,)
                'continuous_label': continuous_label,  # scalar
                'session_label': sample_info['session_label'],
                'subject_id': sample_info['subject_id'],
            }
            
            # 缓存
            self._data_cache[sample_id] = sample
            
            return sample
            
        except Exception as e:
            print(f"❌ 加载样本 {sample_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _resample_signal(self, signal: np.ndarray, target_length: int = 1000) -> np.ndarray:
        """重采样信号到目标长度"""
        if len(signal) == 0:
            return np.zeros(target_length)
        
        if len(signal) == target_length:
            return signal
        
        # 使用线性插值重采样
        indices = np.linspace(0, len(signal) - 1, target_length)
        return np.interp(indices, np.arange(len(signal)), signal)
    
    def _extract_static_features(self, subject_info: Dict, sensor_data: Dict) -> np.ndarray:
        """提取 8 维静态特征
        
        特征顺序：Age, Gender, BMI, HR, Temp, Resp, BP, SpO2
        """
        # 从受试者信息中提取
        age = subject_info.get('AGE', 30)
        gender = 1 if subject_info.get('GENDER', 'male') == 'male' else 0
        height = subject_info.get('HEIGHT', 170) / 100  # cm -> m
        weight = subject_info.get('WEIGHT', 70)
        bmi = weight / (height ** 2) if height > 0 else 22
        
        # 从传感器数据中提取生理指标
        chest_data = sensor_data.get('chest', {})
        
        # 心率（从 ECG 计算）
        ecg = chest_data.get('ECG', [])
        hr = self._calculate_heart_rate(ecg) if len(ecg) > 0 else 75
        
        # 体温
        temp = chest_data.get('TEMP', [37])[0] if 'TEMP' in chest_data else 37.0
        
        # 呼吸频率
        resp = chest_data.get('RESP', [16])[0] if 'RESP' in chest_data else 16
        
        # 血压（如果数据中没有，用 0 填充）
        bp = chest_data.get('BP', [120, 80])[0] if 'BP' in chest_data else 0
        
        # 血氧（如果数据中没有，用 0 填充）
        spo2 = chest_data.get('SpO2', [98])[0] if 'SpO2' in chest_data else 0
        
        return np.array([
            float(age),
            float(gender),
            float(bmi),
            float(hr),
            float(temp),
            float(resp),
            float(bp),
            float(spo2),
        ], dtype=np.float32)
    
    def _calculate_heart_rate(self, ecg: np.ndarray) -> float:
        """从 ECG 信号计算心率"""
        if len(ecg) < 100:
            return 75.0
        
        # 简单的峰值检测
        from scipy import signal
        peaks, _ = signal.find_peaks(ecg, distance=50)
        
        if len(peaks) < 2:
            return 75.0
        
        # 计算平均心率
        rr_intervals = np.diff(peaks)
        avg_rr = np.mean(rr_intervals)
        sampling_rate = 700  # WESAD ECG 采样率
        
        hr = 60 / (avg_rr / sampling_rate)
        
        return float(hr)
    
    def _extract_label(self, session_label: str, sensor_data: Dict) -> float:
        """提取连续标签并转换为放松度
        
        转换公式：Relaxation = 1 - Stress
        
        标签映射：
        - baseline (0) -> 高放松度 -> 0.9
        - stress (1) -> 低放松度 -> 0.1
        - amusement (2) -> 中等放松度 -> 0.6
        - meditation (3) -> 高放松度 -> 0.95
        """
        label_map = {
            'baseline': 0.9,
            'stress': 0.1,
            'amusement': 0.6,
            'meditation': 0.95,
        }
        
        return label_map.get(session_label, 0.5)


class WESADDataset(Dataset):
    """WESAD 数据集 PyTorch 实现"""
    
    def __init__(self, data_root: str, transform=None):
        """
        Args:
            data_root: WESAD 数据集根目录
            transform: 可选的变换
        """
        self.data_source = WESADDataSource(data_root)
        self.transform = transform
        
        # 初始化数据源
        if not self.data_source.initialize():
            raise RuntimeError("初始化 WESAD 数据源失败")
        
        self._sample_list = self.data_source.get_sample_list()
        print(f"📦 WESAD 数据集初始化完成: {len(self._sample_list)} 样本")
    
    def __len__(self) -> int:
        return len(self._sample_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.data_source.load_sample(idx)
        
        # 转换为 Tensor
        item = {
            'dynamic': torch.from_numpy(sample['dynamic']).float(),  # (2, 1000)
            'static_basic': torch.from_numpy(sample['static_basic']).float(),  # (8,)
            'label': torch.tensor(sample['continuous_label'], dtype=torch.float32),  # scalar
            'session_label': sample['session_label'],
            'subject_id': sample['subject_id'],
        }
        
        # 应用变换
        if self.transform is not None:
            item = self.transform(item)
        
        return item
    
    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        if len(self._sample_list) == 0:
            return {}
        
        # 收集所有样本的静态特征
        all_static = []
        all_labels = []
        
        for idx in self._sample_list:
            sample = self.data_source.load_sample(idx)
            all_static.append(sample['static_basic'])
            all_labels.append(sample['continuous_label'])
        
        all_static = np.array(all_static)
        all_labels = np.array(all_labels)
        
        return {
            'num_samples': len(self._sample_list),
            'static_features': {
                'mean': all_static.mean(axis=0).tolist(),
                'std': all_static.std(axis=0).tolist(),
            },
            'labels': {
                'mean': float(all_labels.mean()),
                'std': float(all_labels.std()),
                'min': float(all_labels.min()),
                'max': float(all_labels.max()),
            },
        }


def create_wesad_dataset(data_root: str, train_ratio: float = 0.8) -> tuple:
    """
    创建训练和验证数据集
    
    Args:
        data_root: WESAD 数据集根目录
        train_ratio: 训练集比例
        
    Returns:
        (train_dataset, val_dataset)
    """
    from torch.utils.data import random_split
    
    # 创建完整数据集
    full_dataset = WESADDataset(data_root)

    # 按subject_id分组样本索引
    subject_to_indices = {}
    for idx in range(len(full_dataset)):
        sample = full_dataset._sample_list[idx]
        subject_id = sample['subject_id']
        if subject_id not in subject_to_indices:
            subject_to_indices[subject_id] = []
        subject_to_indices[subject_id].append(idx)

    # 随机打乱subject顺序
    subject_ids = list(subject_to_indices.keys())
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(subject_ids)).tolist()
    shuffled_subject_ids = [subject_ids[i] for i in shuffled_indices]

    # 按比例分配subjects到train/val
    n_samples = len(full_dataset)
    n_train = int(train_ratio * n_samples)
    n_val = n_samples - n_train

    train_indices = []
    val_indices = []

    for subject_id in shuffled_subject_ids:
        indices = subject_to_indices[subject_id]
        if len(train_indices) < n_train:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)

    # 确保train_indices不超过n_train
    train_indices = train_indices[:n_train]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"📊 数据集划分:")
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本")

    return train_dataset, val_dataset


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 WESAD 数据集")
    print("=" * 60)
    
    # 假设数据集路径
    data_root = "data/WESAD"
    
    if not os.path.exists(data_root):
        print(f"❌ WESAD 数据集不存在: {data_root}")
        print("请下载 WESAD 数据集并解压到该目录")
        sys.exit(1)
    
    # 创建数据集
    dataset = WESADDataset(data_root)
    
    # 打印统计信息
    stats = dataset.get_statistics()
    print(f"\n📊 数据统计:")
    print(f"  样本数: {stats['num_samples']}")
    print(f"  静态特征均值: {stats['static_features']['mean']}")
    print(f"  标签范围: [{stats['labels']['min']:.3f}, {stats['labels']['max']:.3f}]")
    
    # 测试加载
    print(f"\n🔄 测试加载样本:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  样本 {i}:")
        print(f"    动态特征: {sample['dynamic'].shape}")
        print(f"    静态特征: {sample['static_basic'].shape}")
        print(f"    标签: {sample['label'].item():.3f}")
        print(f"    会话标签: {sample['session_label']}")
        print(f"    受试者: {sample['subject_id']}")
    
    print("\n✅ 测试完成！")