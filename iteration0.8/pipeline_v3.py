"""
多模态按摩椅舒适度分析 Pipeline
Iteration 2.5: 时序补偿 → 校验 → 特征 → 融合 → 模型 → 验证
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class DataGenerator:
    """数据生成器：生成仿真因果数据"""

    def __init__(self, fs=50, duration=120):
        self.fs = fs
        self.duration = duration
        self.t = np.linspace(0, duration, duration * fs)

    def generate_causal_data(self):
        """生成有因果关系的压力和心率数据"""
        # 压力信号：正弦波 + 两个脉冲
        pressure = 30 + 10 * np.sin(2 * np.pi * 0.1 * self.t)
        pressure[20*self.fs:30*self.fs] += 30  # 20-30秒：强力按压
        pressure[60*self.fs:70*self.fs] += 35  # 60-70秒：更强力按压

        # 心率：压力的延迟响应（3秒延迟）
        delay_samples = int(3.0 * self.fs)
        pressure_delayed = np.roll(pressure, delay_samples)
        pressure_delayed[-delay_samples:] = pressure[0]
        hr = 75 - 0.3 * pressure_delayed + np.random.normal(0, 0.5, len(self.t))

        # 下采样到1Hz，加抖动
        t_h = np.arange(0, self.duration, 1.0) + np.random.uniform(-0.1, 0.1, self.duration)
        hr_1hz = hr[::self.fs]

        # 生成综合标签：基于压力+心率+耦合关系
        # 扩展心率到50Hz用于标签计算
        hr_50hz = np.repeat(hr_1hz, self.fs)[:len(pressure)]

        # 对每个2秒窗口计算标签
        window_sec = 2
        window_size = int(window_sec * self.fs)
        n_windows = len(pressure) // window_size

        labels_50hz = np.zeros(len(pressure))
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            p_window = pressure[start:end]
            hr_window = hr_50hz[start:end]
            label = self._compute_comfort_label(p_window, hr_window)
            labels_50hz[start:end] = label

        return self.t, pressure, t_h, hr_1hz, labels_50hz

    def _compute_comfort_label(self, pressure_window, hr_window):
        """
        综合计算舒适度标签

        考虑因素：
        1. 压力均值和范围
        2. 心率均值和变化
        3. 压力-心率耦合关系
        4. 稳定性
        """
        # === 压力指标 ===
        p_mean = np.mean(pressure_window)
        p_std = np.std(pressure_window)
        p_max = np.max(pressure_window)

        # 压力评分：30-60舒适，<30太轻，>70太重
        if p_mean < 30:
            p_score = 0.5 + (30 - p_mean) / 30 * 0.5
        elif p_mean > 70:
            p_score = max(0.0, 1.0 - (p_mean - 70) / 30)
        else:
            p_score = 1.0 - abs(p_mean - 50) / 40 * 0.5

        # 压力稳定性
        p_stability = max(0.0, 1.0 - p_std / 20)

        # === 心率指标 ===
        hr_mean = np.mean(hr_window)

        # 心率评分：60-80正常
        if hr_mean < 60:
            hr_score = 0.5 + (60 - hr_mean) / 20 * 0.5
        elif hr_mean > 100:
            hr_score = max(0.0, 1.0 - (hr_mean - 100) / 20)
        else:
            hr_score = 0.9

        # 心率稳定性
        hr_std = np.std(hr_window)
        hr_stability = max(0.0, 1.0 - hr_std / 10)

        # === 压力-心率耦合关系 ===
        # 理想：压力增加，心率下降（放松反应）
        # 异常：压力增加，心率上升（疼痛/紧张）
        p_trend = np.polyfit(np.arange(len(pressure_window)), pressure_window, 1)[0]
        hr_trend = np.polyfit(np.arange(len(hr_window)), hr_window, 1)[0]

        if p_trend > 2:  # 压力明显上升
            coupling_score = 1.0 if hr_trend < 0 else 0.2
        elif p_trend < -2:  # 压力明显下降
            coupling_score = 0.8  # 压力下降总是好的
        else:  # 压力稳定
            coupling_score = 0.7 if abs(hr_trend) < 1 else 0.5

        # === 综合评分 ===
        # 权重：压力35% + 心率25% + 稳定性15% + 耦合25%
        comfort_score = (
            0.35 * p_score +
            0.25 * hr_score +
            0.15 * (p_stability + hr_stability) / 2 +
            0.25 * coupling_score
        )

        return np.clip(comfort_score, 0, 1)

    def generate_test_data(self):
        """生成测试数据（6种场景）"""
        scenarios = []

        # 场景1: 舒适（低压+心率下降）
        t, p, _, hr, _ = self.generate_causal_data()
        p_test = np.clip(p, 20, 40)
        hr_test = 75 - 0.5 * np.roll(p_test, 150) + np.random.normal(0, 0.3, len(p))
        hr_test = np.clip(hr_test, 60, 85)
        for i in range(30):
            scenarios.append({
                'pressure': p_test + np.random.normal(0, 2, len(p)),
                'hr': hr_test + np.random.normal(0, 1, len(p)),
                'scenario': '舒适',
                'true_label': 1.0
            })

        # 场景2: 一般（中等压力+心率平稳）
        p_test = 50 + 5 * np.sin(2 * np.pi * 0.1 * self.t)
        hr_test = 70 + np.random.normal(0, 1, len(p))
        for i in range(30):
            scenarios.append({
                'pressure': p_test + np.random.normal(0, 3, len(p)),
                'hr': hr_test + np.random.normal(0, 1, len(p)),
                'scenario': '一般',
                'true_label': 0.5
            })

        # 场景3: 不舒适（高压+心率上升）
        p_test = np.clip(70 + 10 * np.sin(2 * np.pi * 0.1 * self.t), 60, 85)
        hr_test = 80 + 0.3 * np.roll(p_test, 150) + np.random.normal(0, 0.5, len(p))
        hr_test = np.clip(hr_test, 70, 95)
        for i in range(30):
            scenarios.append({
                'pressure': p_test + np.random.normal(0, 3, len(p)),
                'hr': hr_test + np.random.normal(0, 1, len(p)),
                'scenario': '不舒适',
                'true_label': 0.0
            })

        # 场景4: 传感器噪声（压力尖峰+心率无变化）
        p_test = 40 + 5 * np.sin(2 * np.pi * 0.1 * self.t)
        spikes = np.random.choice(len(p_test), 10, replace=False)
        p_test[spikes] += np.random.uniform(30, 50, len(spikes))
        hr_test = 70 + np.random.normal(0, 0.5, len(p))
        for i in range(10):
            scenarios.append({
                'pressure': p_test + np.random.normal(0, 2, len(p)),
                'hr': hr_test + np.random.normal(0, 0.5, len(p)),
                'scenario': '噪声',
                'true_label': 0.5  # 噪声样本标记为一般
            })

        # 场景5: 痛感响应（压力尖峰+心率上升）
        p_test = 40 + 5 * np.sin(2 * np.pi * 0.1 * self.t)
        spikes = np.random.choice(len(p_test), 10, replace=False)
        p_test[spikes] += np.random.uniform(25, 45, len(spikes))
        hr_test = 70 + 0.2 * np.roll(np.maximum(0, p_test - 50), 150) + np.random.normal(0, 0.5, len(p))
        for i in range(10):
            scenarios.append({
                'pressure': p_test + np.random.normal(0, 2, len(p)),
                'hr': hr_test + np.random.normal(0, 1, len(p)),
                'scenario': '痛感响应',
                'true_label': 0.2  # 痛感响应接近不舒适
            })

        # 场景6: 延迟异常（延迟不是3秒）
        p_test = 40 + 10 * np.sin(2 * np.pi * 0.1 * self.t)
        p_test[20*self.fs:30*self.fs] += 30
        delay_wrong = int(5.0 * self.fs)  # 错误延迟5秒
        pressure_delayed = np.roll(p_test, delay_wrong)
        pressure_delayed[-delay_wrong:] = p_test[0]
        hr_test = 75 - 0.3 * pressure_delayed + np.random.normal(0, 0.5, len(p))
        for i in range(10):
            scenarios.append({
                'pressure': p_test + np.random.normal(0, 2, len(p)),
                'hr': hr_test + np.random.normal(0, 1, len(p)),
                'scenario': '延迟异常',
                'true_label': 0.5  # 标签不变，测试模型对延迟异常的敏感度
            })

        return scenarios


class SignalAligner:
    """信号对齐器：merge_asof + 插值"""

    def __init__(self, fs=50):
        self.fs = fs

    def align(self, df_p, df_h):
        """对齐压力和心率信号"""
        # 排序
        df_p = df_p.sort_values('ts')
        df_h = df_h.sort_values('ts')

        # merge_asof 对齐
        aligned = pd.merge_asof(df_p, df_h, on='ts', direction='backward')

        # cubic插值填补NaN
        aligned['hr'] = aligned['hr'].interpolate(method='cubic').bfill().ffill()

        return aligned


class TemporalCompensator:
    """时序补偿器：按检测到的延迟平移心率信号"""

    def __init__(self, fs=50, lag_sec=-3.06):
        self.fs = fs
        self.lag_sec = lag_sec

    def compensate(self, hr_signal, pressure_signal=None):
        """时序补偿：将心率信号按延迟平移"""
        shift_samples = int(self.lag_sec * self.fs)
        hr_compensated = np.roll(hr_signal, -shift_samples)

        # 边界处理
        if pressure_signal is not None:
            hr_compensated[-abs(shift_samples):] = hr_compensated[abs(shift_samples)]

        return hr_compensated


class CrossModalValidator:
    """多模态交叉校验器"""

    def __init__(self, fs=50):
        self.fs = fs

    def validate(self, pressure, hr):
        """校验压力和心率的一致性"""
        # 检测压力尖峰
        p_mean = np.mean(pressure)
        p_std = np.std(pressure)
        p_threshold = p_mean + 2 * p_std

        spike_indices = np.where(pressure > p_threshold)[0]

        # 检测心率响应
        hr_mean = np.mean(hr)
        hr_change = np.diff(hr)
        hr_response = np.abs(hr_change) > 2 * np.std(hr_change)

        # 分析尖峰与响应的关系
        spike_with_response = 0
        spike_without_response = 0

        for idx in spike_indices:
            # 检查尖峰后3秒内心率是否明显变化
            window_end = min(idx + 3 * self.fs, len(hr))
            if np.any(hr_response[idx:window_end]):
                spike_with_response += 1
            else:
                spike_without_response += 1

        # 判定
        if spike_without_response > spike_with_response * 2:
            is_valid = False
            anomaly_type = 'sensor_noise'
            stress_score = 0.0
        elif spike_with_response > spike_without_response:
            is_valid = True
            anomaly_type = 'pain_response'
            stress_score = min(1.0, spike_with_response / max(1, len(spike_indices)))
        else:
            is_valid = True
            anomaly_type = 'normal'
            stress_score = 0.3

        return is_valid, anomaly_type, stress_score


class HeartRateFeatureExtractor:
    """心率特征提取器（增强版：9个特征）"""

    def __init__(self, fs=50):
        self.fs = fs

    def extract(self, hr_signal):
        """提取心率特征（增强版）"""
        hr_diff = np.diff(hr_signal)
        hr_diff2 = np.diff(hr_diff)

        return {
            # 基础特征（5个）
            'hr_mean': np.mean(hr_signal),
            'hr_std': np.std(hr_signal),
            'hrv': np.std(hr_diff),  # 心率变异性（一阶差分标准差）
            'hr_range': np.ptp(hr_signal),  # 心率范围
            'hr_slope': np.polyfit(np.arange(len(hr_signal)), hr_signal, 1)[0] if len(hr_signal) > 10 else 0,
            # 新增特征（4个）
            'hrv_change': np.std(hr_diff2),  # HRV变化率（二阶差分标准差）
            'hr_local_std': np.std(hr_signal[-50:]) if len(hr_signal) >= 50 else np.std(hr_signal),  # 局部标准差
            'hr_response_amp': np.max(hr_signal) - np.min(hr_signal),  # 响应幅度
            'hr_deviation': np.abs(hr_signal - np.mean(hr_signal)).mean(),  # 平均偏离度
        }


class FeatureFusion:
    """特征融合器"""

    def __init__(self):
        pass

    def fuse(self, p_features, hr_features, cpi):
        """融合特征"""
        # 拼接所有特征
        all_features = {**p_features, **hr_features, 'cpi': cpi}
        return all_features


class ComfortPredictor:
    """舒适度预测模型"""

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = None
        self.is_trained = False

    def train(self, X, y):
        """训练模型"""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_trained = True
        print(f"模型训练完成！特征数: {len(self.feature_names)}")

    def predict(self, X):
        """预测"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        return self.model.predict(X)

    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)

        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }

        return metrics, y_pred

    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        return dict(zip(self.feature_names, self.model.feature_importances_))


class Pipeline:
    """完整Pipeline"""

    def __init__(self):
        self.data_gen = DataGenerator()
        self.aligner = SignalAligner()
        self.compensator = TemporalCompensator(lag_sec=-3.06)
        self.validator = CrossModalValidator()
        self.hr_extractor = HeartRateFeatureExtractor()
        self.p_extractor = None  # 外部传入
        self.fusion = FeatureFusion()
        self.model = ComfortPredictor()
        self.feature_names = None

    def load_pressure_extractor(self, extractor):
        """加载压力特征提取器"""
        self.p_extractor = extractor

    def run(self, use_existing_data=False):
        """运行完整Pipeline"""
    print("=" * 60)
    print("Pipeline V3 完成！（综合标签版：压力+心率+耦合判断）")
    print("=" * 60)

        # Phase 1: 数据准备
        print("\n[Phase 1] 数据准备...")
        if use_existing_data and os.path.exists('data/aligned_data.csv'):
            print("  加载已有数据...")
            aligned = pd.read_csv('data/aligned_data.csv')
        else:
            print("  生成仿真因果数据...")
            t, pressure, t_h, hr_1hz, labels = self.data_gen.generate_causal_data()

            df_p = pd.DataFrame({'ts': t, 'pressure': pressure})
            df_h = pd.DataFrame({'ts': t_h, 'hr': hr_1hz})

            print("  对齐信号...")
            aligned = self.aligner.align(df_p, df_h)
            aligned['label'] = labels[:len(aligned)]

            aligned.to_csv('data/aligned_data.csv', index=False)

        print(f"  对齐后数据: {len(aligned)} 行")

        # Phase 2: 时序补偿 + 异常校验
        print("\n[Phase 2] 时序补偿 + 异常校验...")
        aligned['hr_compensated'] = self.compensator.compensate(
            aligned['hr'].values, aligned['pressure'].values
        )

        is_valid, anomaly_type, stress = self.validator.validate(
            aligned['pressure'].values, aligned['hr_compensated'].values
        )
        aligned['is_anomaly'] = not is_valid
        aligned['anomaly_type'] = anomaly_type
        aligned['stress_score'] = stress

        print(f"  异常样本数: {aligned['is_anomaly'].sum()}")
        print(f"  异常类型: {anomaly_type}")

        # Phase 3: 特征提取
        print("\n[Phase 3] 特征提取...")

        # 压力特征（滑动窗口）
        window_sec = 2
        step_sec = 1
        window_size = int(window_sec * 50)
        step_size = int(step_sec * 50)

        feature_list = []
        for start in range(0, len(aligned) - window_size, step_size):
            end = start + window_size

            window = aligned.iloc[start:end]
            p_signal = window['pressure'].values
            hr_signal = window['hr_compensated'].values

            # 压力特征
            p_features = self.p_extractor.extract_from_window(p_signal)

            # 心率特征
            hr_features = self.hr_extractor.extract(hr_signal)

            # CPI指标
            p_norm = (p_signal - np.mean(p_signal)) / (np.std(p_signal) + 1e-6)
            hr_norm = (hr_signal - np.mean(hr_signal)) / (np.std(hr_signal) + 1e-6)
            cpi = np.mean(p_norm * hr_norm)

            # 融合特征
            fused = self.fusion.fuse(p_features, hr_features, cpi)
            fused['start_time'] = start / 50.0
            fused['label'] = window['label'].mean()
            fused['is_anomaly'] = window['is_anomaly'].any()
            fused['stress_score'] = window['stress_score'].mean()

            feature_list.append(fused)

        features_df = pd.DataFrame(feature_list)
        self.feature_names = [c for c in features_df.columns if c not in ['start_time', 'label', 'is_anomaly']]
        features_df.to_csv('features/fused_features_v3.csv', index=False)

        print(f"  特征矩阵: {len(features_df)} 行 × {len(self.feature_names)} 特征")

        # Phase 4: 模型训练
        print("\n[Phase 4] 模型训练...")

        # 移除异常样本训练
        clean_df = features_df[~features_df['is_anomaly']]
        X = clean_df[self.feature_names]
        y = clean_df['label']

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_val)} 样本")

        self.model.train(X_train, y_train)

        train_metrics, _ = self.model.evaluate(X_train, y_train)
        val_metrics, y_val_pred = self.model.evaluate(X_val, y_val)

        print(f"\n  训练集指标:")
        print(f"    MSE: {train_metrics['mse']:.4f}")
        print(f"    RMSE: {train_metrics['rmse']:.4f}")
        print(f"    MAE: {train_metrics['mae']:.4f}")
        print(f"    R²: {train_metrics['r2']:.4f}")

        print(f"\n  验证集指标:")
        print(f"    MSE: {val_metrics['mse']:.4f}")
        print(f"    RMSE: {val_metrics['rmse']:.4f}")
        print(f"    MAE: {val_metrics['mae']:.4f}")
        print(f"    R²: {val_metrics['r2']:.4f}")

        # Phase 5: 测试数据验证
        print("\n[Phase 5] 测试数据验证...")
        self.evaluate_test_data()

        # 保存特征重要性
        importance = self.model.get_feature_importance()
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        print("\n  特征重要性 Top 5:")
        for i, (name, imp) in enumerate(sorted_importance.items()):
            if i < 5:
                print(f"    {name}: {imp:.4f}")

        return features_df, val_metrics, sorted_importance

    def evaluate_test_data(self):
        """评估测试数据"""
        scenarios = self.data_gen.generate_test_data()

        results = []
        for scenario in scenarios:
            p_signal = scenario['pressure']
            hr_signal = scenario['hr']

            # 时序补偿
            hr_comp = self.compensator.compensate(hr_signal, p_signal)

            # 特征提取
            p_features = self.p_extractor.extract_from_window(p_signal)
            hr_features = self.hr_extractor.extract(hr_comp)

            # CPI
            p_norm = (p_signal - np.mean(p_signal)) / (np.std(p_signal) + 1e-6)
            hr_norm = (hr_signal - np.mean(hr_signal)) / (np.std(hr_signal) + 1e-6)
            cpi = np.mean(p_norm * hr_norm)

            # 融合
            features = self.fusion.fuse(p_features, hr_features, cpi)
            features['stress_score'] = 0.3  # 测试数据默认

            # 只选择训练时使用的特征
            X_test = pd.DataFrame([features])[self.feature_names]

            # 预测
            pred = self.model.predict(X_test)[0]

            results.append({
                'scenario': scenario['scenario'],
                'true_label': scenario['true_label'],
                'predicted': pred,
                'error': abs(pred - scenario['true_label'])
            })

        results_df = pd.DataFrame(results)

        # 按场景汇总
        print("\n  各场景预测结果:")
        for scenario in results_df['scenario'].unique():
            subset = results_df[results_df['scenario'] == scenario]
            mae = subset['error'].mean()
            print(f"    {scenario}: MAE = {mae:.4f}")

        results_df.to_csv('results/test_results_v3.csv', index=False)
        print(f"\n  测试结果已保存: results/test_results_v3.csv")

        return results_df


def main():
    """主函数"""
    # 加载压力特征提取器
    from feature_extractor import FeatureExtractor
    p_extractor = FeatureExtractor(sampling_rate=50)

    # 运行Pipeline
    pipeline = Pipeline()
    pipeline.load_pressure_extractor(p_extractor)
    features_df, metrics, importance = pipeline.run(use_existing_data=False)

    # 可视化
    print("\n[Visualization] 生成可视化报告...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: 特征重要性
    ax1 = axes[0, 0]
    sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8])
    ax1.barh(list(sorted_imp.keys()), list(sorted_imp.values()))
    ax1.set_xlabel('Importance')
    ax1.set_title('Feature Importance (Top 8)')
    ax1.invert_yaxis()

    # 图2: 预测 vs 真实
    ax2 = axes[0, 1]
    # 重新获取验证集数据用于可视化
    clean_df = features_df[~features_df['is_anomaly']]
    X = clean_df[pipeline.feature_names]
    y = clean_df['label']
    _, y_pred = pipeline.model.evaluate(X, y)
    ax2.scatter(y, y_pred, alpha=0.5, s=10)
    ax2.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    ax2.set_xlabel('True Label')
    ax2.set_ylabel('Predicted Label')
    ax2.set_title('Predicted vs True (Validation Set)')
    ax2.legend()

    # 图3: CPI vs 标签
    ax3 = axes[1, 0]
    ax3.scatter(features_df['cpi'], features_df['label'], alpha=0.3, s=10)
    ax3.set_xlabel('CPI (Comfort Pressure Index)')
    ax3.set_ylabel('Comfort Label')
    ax3.set_title('CPI vs Comfort Label')

    # 图4: 特征相关性热力图
    ax4 = axes[1, 1]
    corr_matrix = features_df[pipeline.feature_names[:5] + ['cpi']].corr()
    im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax4.set_yticklabels(corr_matrix.columns)
    ax4.set_title('Feature Correlation Heatmap')
    plt.colorbar(im, ax=ax4)

    plt.tight_layout()
    plt.savefig('results/v3_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("Pipeline V3 完成！（综合标签版：压力+心率+耦合判断）")
    print("=" * 60)
    print("\n输出文件:")
    print("  - features/fused_features_v3.csv: 融合特征矩阵(V3)")
    print("  - results/test_results_v3.csv: 测试结果(V3)")
    print("  - results/v3_visualization.png: 可视化报告(V3)")


if __name__ == "__main__":
    main()
