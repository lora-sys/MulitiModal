import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 读取处理好的特征
df = pd.read_pickle('/Users/loralora/repos/MulitiModal/processed_features.pickle')

# 设置样式
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False

# 创建4个子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: weight vs p1_mean
axes[0, 0].scatter(df['weight'], df['sensor1_mean'], alpha=0.6, s=50)
axes[0, 0].set_xlabel('体重')
axes[0, 0].set_ylabel('传感器1 平均压力')
axes[0, 0].set_title('图1: 体重 vs 平均压力')
axes[0, 0].axhline(y=df['sensor1_mean'].mean(), color='r', linestyle='--', alpha=0.5)

# 图2: label vs p1_std
axes[0, 1].scatter(df['label'], df['sensor1_std'], alpha=0.6, s=50, c=df['label'], cmap='viridis')
axes[0, 1].set_xlabel('身体表征等级')
axes[0, 1].set_ylabel('传感器1 标准差')
axes[0, 1].set_title('图2: 等级 vs 稳定性 (std越小越稳)')

# 图3: hr vs p1_ptp
axes[1, 0].scatter(df['hr'], df['sensor1_ptp'], alpha=0.6, s=50)
axes[1, 0].set_xlabel('心率')
axes[1, 0].set_ylabel('传感器1 峰峰值')
axes[1, 0].set_title('图3: 心率 vs 压力振幅 (心率越低，振幅越大)')

# 图4: spo2 vs p1_mean
axes[1, 1].scatter(df['spo2'], df['sensor1_mean'], alpha=0.6, s=50)
axes[1, 1].set_xlabel('血氧')
axes[1, 1].set_ylabel('传感器1 平均压力')
axes[1, 1].set_title('图4: 血氧 vs 平均压力')

plt.tight_layout()
plt.savefig('sanity_check.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 可视化完成！保存到 sanity_check.png")
