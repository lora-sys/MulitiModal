import numpy as np
import pandas as pd
from scipy.signal import medfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# --- 1. 模拟数据生成函数 ---
def generate_labeled_data(mode, duration=100, fs=50):
    t = np.linspace(0, duration, duration * fs)
    if mode == 0:  # 空载: 只有低力度和纯底噪
        signal = 20 + np.random.normal(0, 1.5, len(t))
    elif mode == 1:  # 柔和: 频率慢(0.1Hz), 振幅小(10)
        signal = (
            40 + 10 * np.sin(2 * np.pi * 0.1 * t) + np.random.normal(0, 1.5, len(t))
        )
    elif mode == 2:  # 深度: 频率快(0.5Hz), 振幅大(30)
        signal = (
            60 + 30 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 1.5, len(t))
        )

    # 模拟投毒 (注入 0.5% 的异常点)
    poison_idx = np.random.choice(
        len(signal), size=int(len(signal) * 0.005), replace=False
    )
    signal[poison_idx] += np.random.choice([-40, 40], size=len(poison_idx))

    return signal


# --- 2. 核心清洗 Pipeline (复用之前的专业修复逻辑) ---
def professional_cleaning_pipeline(raw_signal, fs=50):
    df = pd.DataFrame({"raw": raw_signal})
    # 3-Sigma 检测
    rolling = df["raw"].rolling(window=15, center=True)
    mu, std = rolling.mean(), rolling.std()
    is_anomaly = (df["raw"] > mu + 3 * std) | (df["raw"] < mu - 3 * std)

    # 修复：异常点设为 NaN 并插值
    df["clean"] = df["raw"].copy()
    df.loc[is_anomaly, "clean"] = np.nan
    df["clean"] = (
        df["clean"].interpolate().fillna(method="bfill").fillna(method="ffill")
    )

    # 平滑
    return df["clean"].rolling(window=15, center=True, min_periods=1).mean().values


# --- 3. 特征提取函数 ---
def get_features(signal, window_size=100):
    feats = []
    for i in range(0, len(signal) - window_size, window_size):
        seg = signal[i : i + window_size]
        feats.append(
            [
                np.mean(seg),  # 均值 -> 力度
                np.std(seg),  # 标准差 -> 波动
                np.ptp(seg),  # 峰峰值 -> 振幅
                # 简单过零率模拟频率：信号穿过均值的次数
                np.sum(np.diff(seg > np.mean(seg)) != 0),
            ]
        )
    return np.array(feats)


# --- 4. 开始主实验 ---

print("🛠️  正在生成并清洗多模态数据...")
# 生成三类数据
s0 = generate_labeled_data(0)
s1 = generate_labeled_data(1)
s2 = generate_labeled_data(2)

# 清洗数据
c0 = professional_cleaning_pipeline(s0)
c1 = professional_cleaning_pipeline(s1)
c2 = professional_cleaning_pipeline(s2)

# 提取特征
f0 = get_features(c0)
f1 = get_features(c1)
f2 = get_features(c2)

# 合并数据集
X = np.vstack([f0, f1, f2])
y = np.array([0] * len(f0) + [1] * len(f1) + [2] * len(f2))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练随机森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
print("\n🎯 按摩模式识别结果:")
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.2%}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=["空载", "柔和", "深度"]))

# 打印特征重要性 (看模型最看重哪个指标)
importances = clf.feature_importances_
feat_names = ["平均力度", "力度波动", "压力振幅", "波动频率"]
for name, imp in zip(feat_names, importances):
    print(f"特征 [{name}] 对预测的贡献度: {imp:.2%}")
