import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 1. 准备数据
# 加载原始数据（包含真值和投毒后的噪声）
df = pd.read_csv("../iteration0.5/pressure_sim.csv")

# 模拟之前的投毒逻辑（确保 A 组有毒）
np.random.seed(42)
poison_indices = np.random.choice(df.index[50:-50], size=10, replace=False)
for idx in poison_indices:
    # 注入一个足以误导决策的巨型正向脉冲 (让 20 变成 70，从而触发错误报警)
    df.loc[idx, "noisy_signal"] += 50

# 加载你修复后的数据 (B 组)
# 注意：这里我们直接从之前的修复逻辑中提取结果，或者重新运行修复
window_size = 15
df["filter_ma"] = df["noisy_signal"].rolling(window=window_size, center=True).mean()
df["rolling_std"] = df["noisy_signal"].rolling(window=window_size, center=True).std()
is_anomaly = (df["noisy_signal"] > (df["filter_ma"] + 3 * df["rolling_std"])) | (
    df["noisy_signal"] < (df["filter_ma"] - 3 * df["rolling_std"])
)

df["repaired"] = df["noisy_signal"].copy()
df.loc[is_anomaly, "repaired"] = np.nan
df["repaired"] = df["repaired"].interpolate().ffill().bfill()
df["final_recovered"] = df["repaired"].rolling(window=window_size, center=True).mean()

# 2. 定义 AB Test 任务：压力是否 > 50？
threshold = 50

# 地面真值 (我们希望模型达到的终极目标)
y_true = (df["clean_signal"] > threshold).astype(int)

# A 组预测 (直接用脏数据)
y_pred_A = (df["noisy_signal"] > threshold).astype(int)

# B 组预测 (用修复后的数据)
y_pred_B = (df["final_recovered"] > threshold).astype(int)


# 3. 跑分评价
def get_report(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (报警准确率)": precision_score(y_true, y_pred),
        "Recall (漏报率)": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
    }


report_A = get_report(y_true, y_pred_A)
report_B = get_report(y_true, y_pred_B)

# 4. 打印最终战报
print("=" * 50)
print("       🚀 终极 AB Test 跑分报告 🚀")
print("=" * 50)
print(f"{'指标':<20} | {'A 组 (带毒数据)':<15} | {'B 组 (专业修复)':<15}")
print("-" * 55)

for metric in report_A.keys():
    valA = report_A[metric]
    valB = report_B[metric]
    mark = "⭐" if valB > valA else ""
    print(f"{metric:<18} | {valA:15.4f} | {valB:15.4f} {mark}")

print("-" * 55)
# 计算误报（False Positives）
fp_A = ((y_pred_A == 1) & (y_true == 0)).sum()
fp_B = ((y_pred_B == 1) & (y_true == 0)).sum()
print(f"误报次数 (False Alarms):  A 组 = {fp_A} 次 | B 组 = {fp_B} 次")
print(f"✅ B 组将误报降低了: {((fp_A - fp_B) / fp_A) * 100:.1f}%")
print("=" * 50)
