import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['STHeiti']  # Windows常用：SimHei（黑体），SimSun（宋体）
# 也可以尝试：['Microsoft YaHei'] (微软雅黑) 或 ['STHeiti'] (Mac常用)

# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False 
# 读取已对齐的数据
aligned = pd.read_csv("sensor_aligned.csv")
# 两种插值
aligned['hr_linear'] = aligned['hr'].interpolate(method='linear')
aligned['hr_cubic'] = aligned['hr'].interpolate(method='cubic')
# 可视化对比
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(aligned['ts'], aligned['pressure'], label='压力', alpha=0.7)
plt.xlabel('时间(s)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(aligned['ts'], aligned['hr_linear'], label='线性插值', alpha=0.7)
plt.plot(aligned['ts'], aligned['hr_cubic'], label='三次样条', alpha=0.7)
plt.xlabel('时间(s)')
plt.legend()
plt.tight_layout()
plt.savefig("interpolation_comparison.png")
plt.show()
aligned['hr'] = aligned['hr_cubic']
aligned.to_csv("sensor_aligned_filled.csv", index=False)
print("插值完成，结果已保存到 sensor_aligned_filled.csv")