#。手动在数据中插入 10 个巨大的脉冲（Spikes），模拟传感器突然失灵或受到强烈干扰的情况。

# 1. 加载数据
df = load_signal_from_csv("pressure_sim.csv")

if df is not None:
    # --- 任务 0.3b 投毒逻辑 (Outlier Injection) ---
    # 随机选择 10 个位置，注入远超正常范围的脉冲噪声
    np.random.seed(42) # 保证实验可复现
    poison_indices = np.random.choice(df.index[50:-50], size=10, replace=False) # 避开边缘

    # 注入偏离均值 20-50 个单位的巨型跳变
    for idx in poison_indices:
        spike = np.random.uniform(20, 50) * np.random.choice([-1, 1])
        df.loc[idx, 'noisy_signal'] += spike

    print(f"☣️ 投毒成功：已手动注入 {len(poison_indices)} 个异常脉冲")

    # --- 0.2 阶段：平滑滤波 ---
    window_size = 15
    df['filter_ma'] = df['noisy_signal'].rolling(window=window_size, center=True, min_periods=1).mean()

    # --- 0.3 阶段：异常检测 (3-Sigma 准则) ---
    # 计算滑动统计量
    df['rolling_std'] = df['noisy_signal'].rolling(window=window_size, center=True, min_periods=1).std()

    # 定义 3-Sigma 边界
    df['upper_bound'] = df['filter_ma'] + 3 * df['rolling_std']
    df['lower_bound'] = df['filter_ma'] - 3 * df['rolling_std']

    # 判定异常点
    df['is_anomaly'] = (df['noisy_signal'] > df['upper_bound']) | (df['noisy_signal'] < df['lower_bound'])
    anomaly_count = df['is_anomaly'].sum()
    # ... 后续绘图逻辑保持不变 ...
