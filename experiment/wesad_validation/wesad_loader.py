"""
WESAD数据加载器
用于验证多模态融合架构的泛化能力
"""
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import zipfile
import os

class WESADLoader:
    """
    WESAD数据集加载器
    
    数据映射：
    - ACC (三轴加速度) → pressure (动态波形)
    - HR (心率) + EDA (皮电) → static_basic (静态特征)
    - labels (3类情感状态) → labels (3类舒适度)
    """
    
    def __init__(self, data_path='wesad_data'):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        
    def download_wesad(self):
        """
        下载WESAD数据集
        数据集大小：约2GB
        """
        print("=" * 60)
        print("下载WESAD数据集")
        print("=" * 60)
        
        # Kaggle下载链接
        kaggle_url = "https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-and-affect-detection-dataset/download"
        
        print(f"\n请手动下载数据集:")
        print(f"1. 访问: {kaggle_url}")
        print(f"2. 登录Kaggle账号")
        print(f"3. 下载数据集")
        print(f"4. 解压到: {self.data_path.absolute()}")
        print(f"\n数据集结构:")
        print(f"  S{N}/  # 受试者N")
        print(f"    ├── S{N}.pkl  # 主要数据文件")
        print(f"    └── ...")
        
        return False
    
    def load_subject_data(self, subject_id):
        """
        加载单个受试者的数据
        
        参数：
        - subject_id: 受试者ID（S2, S3, ..., S17）
        
        返回：
        - data: 包含ACC、EDA、HR、labels的字典
        """
        print(f"\n加载受试者 {subject_id} 的数据...")
        
        try:
            import pickle
            
            subject_file = self.data_path / subject_id / f"{subject_id}.pkl"
            
            if not subject_file.exists():
                print(f"  警告：找不到文件 {subject_file}")
                return None
            
            with open(subject_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            print(f"  数据键: {list(data.keys())}")
            
            return data
            
        except Exception as e:
            print(f"  错误：{e}")
            return None
    
    def extract_features(self, data):
        """
        提取特征
        
        参数：
        - data: WESAD原始数据
        
        返回：
        - features: 包含ACC特征、HR、EDA、labels的字典
        """
        try:
            # 提取ACC数据（三轴加速度）
            # 数据格式：[n_samples, 3]
            if 'acc' in data:
                acc = data['acc']
            elif 'ACC' in data:
                acc = data['ACC']
            else:
                print("  警告：找不到ACC数据")
                return None
            
            print(f"  ACC形状: {acc.shape}")
            
            # 提取EDA数据
            if 'eda' in data:
                eda = data['eda']
            elif 'EDA' in data:
                eda = data['EDA']
            else:
                print("  警告：找不到EDA数据")
                return None
            
            print(f"  EDA形状: {eda.shape}")
            
            # 提取标签
            if 'label' in data:
                labels = data['label']
            elif 'labels' in data:
                labels = data['labels']
            else:
                print("  警告：找不到标签数据")
                return None
            
            print(f"  标签形状: {labels.shape}")
            print(f"  标签类型: {np.unique(labels)}")
            
            # 从ACC提取时序特征（模拟pressure）
            # 计算每秒的特征
            fs_acc = 32  # Empatica E4的ACC采样率
            window_size = fs_acc  # 1秒窗口
            step_size = fs_acc  # 1秒步长
            
            acc_features = self._extract_acc_features(acc, window_size, step_size)
            print(f"  提取的ACC特征形状: {acc_features.shape}")
            
            # 从EDA提取统计特征（模拟static）
            eda_features = self._extract_eda_features(eda, fs=4, window_size=window_size)
            print(f"  提取的EDA特征形状: {eda_features.shape}")
            
            # 从ACC提取心率（模拟static中的HR）
            # 这里简化处理，使用ACC的频率特征作为HR代理
            hr_features = self._extract_hr_proxy_from_acc(acc, fs=32, window_size=window_size)
            print(f"  提取的HR特征形状: {hr_features.shape}")
            
            # 组合静态特征
            static_features = np.concatenate([hr_features, eda_features], axis=1)
            print(f"  组合的静态特征形状: {static_features.shape}")
            
            # 提取标签（下采样到特征长度）
            fs_labels = 700  # RespiBAN采样率
            labels_features = self._downsample_labels(labels, fs_labels, window_size)
            print(f"  提取的标签形状: {labels_features.shape}")
            
            return {
                'acc_features': acc_features,  # 模拟pressure
                'static_features': static_features,  # HR + EDA
                'labels': labels_features,
            }
            
        except Exception as e:
            print(f"  错误：{e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_acc_features(self, acc, window_size, step_size):
        """
        从ACC数据提取时序特征
        
        参数：
        - acc: ACC数据 [n_samples, 3]
        - window_size: 窗口大小
        - step_size: 步长
        
        返回：
        - features: [n_windows, n_features]
        """
        n_samples = acc.shape[0]
        n_windows = (n_samples - window_size) // step_size + 1
        
        features = []
        
        for i in range(n_windows):
            start = i * step_size
            end = start + window_size
            
            window = acc[start:end]
            
            # 计算特征
            acc_x = window[:, 0]
            acc_y = window[:, 1]
            acc_z = window[:, 2]
            
            # 特征1：x轴均值
            mean_x = np.mean(acc_x)
            # 特征2：y轴均值
            mean_y = np.mean(acc_y)
            # 特征3：z轴均值
            mean_z = np.mean(acc_z)
            # 特征4：总加速度均值
            mean_acc = np.mean(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2))
            # 特征5：x轴标准差
            std_x = np.std(acc_x)
            # 特征6：y轴标准差
            std_y = np.std(acc_y)
            # 特征7：z轴标准差
            std_z = np.std(acc_z)
            # 特征8：总加速度标准差
            std_acc = np.std(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2))
            # 特征9：x轴能量
            energy_x = np.sum(acc_x**2)
            # 特征10：y轴能量
            energy_y = np.sum(acc_y**2)
            # 特征11：z轴能量
            energy_z = np.sum(acc_z**2)
            
            features.append([
                mean_x, mean_y, mean_z, mean_acc,
                std_x, std_y, std_z, std_acc,
                energy_x, energy_y, energy_z
            ])
        
        features = np.array(features)
        
        # 归一化
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
        
        return features
    
    def _extract_eda_features(self, eda, fs, window_size):
        """
        从EDA数据提取特征
        
        参数：
        - eda: EDA数据 [n_samples]
        - fs: 采样率
        - window_size: 窗口大小（秒）
        
        返回：
        - features: [n_windows, n_features]
        """
        n_samples = len(eda)
        n_windows = (n_samples - window_size) // window_size + 1
        
        features = []
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            
            window = eda[start:end]
            
            # 特征1：均值
            mean_eda = np.mean(window)
            # 特征2：标准差
            std_eda = np.std(window)
            # 特征3：最大值
            max_eda = np.max(window)
            # 特征4：最小值
            min_eda = np.min(window)
            # 特征5：能量
            energy_eda = np.sum(window**2)
            # 特征6：斜率（变化率）
            if len(window) > 1:
                slope = np.mean(np.diff(window))
            else:
                slope = 0
            
            features.append([
                mean_eda, std_eda, max_eda, min_eda, energy_eda, slope
            ])
        
        features = np.array(features)
        
        # 归一化
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
        
        return features
    
    def _extract_hr_proxy_from_acc(self, acc, fs, window_size):
        """
        从ACC数据提取心率代理特征
        
        参数：
        - acc: ACC数据 [n_samples, 3]
        - fs: 采样率
        - window_size: 窗口大小（秒）
        
        返回：
        - features: [n_windows, n_features]
        """
        n_samples = acc.shape[0]
        n_windows = (n_samples - window_size) // window_size + 1
        
        features = []
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            
            window = acc[start:end]
            
            # 计算总加速度
            total_acc = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
            
            # 特征1：均值（心率的代理）
            mean_acc = np.mean(total_acc)
            # 特征2：标准差
            std_acc = np.std(total_acc)
            # 特征3：峰值频率（心率的代理）
            if len(total_acc) > 1:
                fft = np.fft.fft(total_acc)
                freqs = np.fft.fftfreq(len(total_acc), 1/fs)
                peak_freq = freqs[np.argmax(np.abs(fft))]
            else:
                peak_freq = 0
            
            features.append([mean_acc, std_acc, peak_freq])
        
        features = np.array(features)
        
        # 归一化
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
        
        return features
    
    def _downsample_labels(self, labels, fs, window_size):
        """
        下采样标签
        
        参数：
        - labels: 标签数据 [n_samples]
        - fs: 采样率
        - window_size: 窗口大小（秒）
        
        返回：
        - downsampled_labels: [n_windows]
        """
        n_samples = len(labels)
        n_windows = (n_samples - window_size) // window_size + 1
        
        downsampled_labels = []
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            
            window = labels[start:end]
            
            # 使用众数作为窗口的标签
            unique, counts = np.unique(window, return_counts=True)
            most_common = unique[np.argmax(counts)]
            
            downsampled_labels.append(most_common)
        
        return np.array(downsampled_labels)
    
    def load_all_subjects(self, subject_ids=['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']):
        """
        加载所有受试者的数据
        
        参数：
        - subject_ids: 受试者ID列表
        
        返回：
        - dataset: 包含所有受试者数据的字典
        """
        print("=" * 60)
        print("加载所有受试者数据")
        print("=" * 60)
        
        all_acc_features = []
        all_static_features = []
        all_labels = []
        
        for subject_id in subject_ids:
            print(f"\n处理受试者 {subject_id}...")
            
            # 加载数据
            raw_data = self.load_subject_data(subject_id)
            
            if raw_data is None:
                continue
            
            # 提取特征
            features = self.extract_features(raw_data)
            
            if features is None:
                continue
            
            all_acc_features.append(features['acc_features'])
            all_static_features.append(features['static_features'])
            all_labels.append(features['labels'])
        
        # 合并所有受试者的数据
        if len(all_acc_features) > 0:
            dataset = {
                'dynamic': np.concatenate(all_acc_features, axis=0),  # 模拟pressure
                'static_basic': np.concatenate(all_static_features, axis=0),  # HR + EDA
                'labels': np.concatenate(all_labels, axis=0),
                'n_samples': sum(len(labels) for labels in all_labels)
            }
            
            print(f"\n数据集统计:")
            print(f"  样本总数: {dataset['n_samples']}")
            print(f"  动态特征形状: {dataset['dynamic'].shape}")
            print(f"  静态特征形状: {dataset['static_basic'].shape}")
            print(f"  标签形状: {dataset['labels'].shape}")
            print(f"  标签分布: {np.unique(dataset['labels'], return_counts=True)}")
            
            return dataset
        else:
            print("\n错误：无法加载任何数据")
            return None


if __name__ == "__main__":
    print("=" * 60)
    print("WESAD数据加载器测试")
    print("=" * 60)
    
    loader = WESADLoader()
    
    # 提示用户下载数据
    loader.download_wesad()
    
    print("\n" + "=" * 60)
    print("数据加载器测试完成")
    print("=" * 60)
