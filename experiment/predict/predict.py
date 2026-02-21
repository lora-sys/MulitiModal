
import  sys
sys.path.append("experiment/model")
sys.path.append("experiment/dataset")
import torch
import pandas as pd
import numpy as np
import neurokit2 as nk
import re
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from model import MassageFusionNet # 导入你的模型


# ==========================================
# 1. 基础配置
# ==========================================
MODEL_TYPE = 'inception' # 或者 'inception'，取决于你保存了哪个权重
WEIGHT_PATH = f"experiment/model/best_model_{MODEL_TYPE}.pth" 
CSV_FILE = "experiment/data/身体表征正常/501_71_86_95_171.csv"

LABELS = {0: "Very Bad", 1: "Bad", 2: "Normal", 3: "Good"}
FS = 50
TARGET_LEN = 1000

# ==========================================
# 2. 单文件预处理流水线 (与训练完全一致)
# ==========================================
def preprocess(csv_path):
    """单文件预处理流水线"""
    # 1. 提取静态特征（从文件名）
    filename = csv_path.split('/')[-1]  # 取文件名
    numbers = re.findall(r'\d+', filename)
    
    if len(numbers) >= 5:
        weight = float(numbers[1])    # 体重
        hr = float(numbers[2])       # 心率
        spo2 = float(numbers[3])     # 血氧
        height = float(numbers[4])   # 身高
    else:
        weight, hr, spo2, height = 65, 75, 97, 170
    
    # 静态特征归一化
    static = np.array([
        (weight - 65) / 15,
        (hr - 75) / 15,
        (spo2 - 97) / 2,
        (height - 170) / 10
    ])
    static_tensor = torch.tensor(static, dtype=torch.float32)
    
    # 2. 提取动态波形
    df = pd.read_csv(csv_path)
    
    s1 = df["压力传感器1"].values
    s2 = df["压力传感器2"].values
    
    # 3. NK2 处理（与训练一致）
    s1 = nk.signal_filter(s1, sampling_rate=FS, highcut=15, method='butterworth')
    s2 = nk.signal_filter(s2, sampling_rate=FS, highcut=15, method='butterworth')
    
    s1 = nk.signal_resample(s1, sampling_rate=FS, desired_length=TARGET_LEN)
    s2 = nk.signal_resample(s2, sampling_rate=FS, desired_length=TARGET_LEN)
    
    # 4. Z-Score 归一化
    s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-8)
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-8)
    
    # 5. 合并为张量
    dynamic = np.stack([s1, s2])
    dynamic_tensor = torch.tensor(dynamic, dtype=torch.float32)
    
    return dynamic_tensor, static_tensor
# ==========================================
# 3. 核心可视化函数：画出模型的"注意力"
# ==========================================
def plot_saliency(signal, saliency, channel_name, ax):
    """用热力背景色表示模型关注的区域"""
    time_axis = np.linspace(0, 20, len(signal)) # 20秒
    
    # 归一化显著性 (0到1之间)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # 画原始信号
    ax.plot(time_axis, signal, color='black', linewidth=1.5, zorder=2)
    
    # 用红色渐变背景表示注意力
    extent = [time_axis, time_axis, signal.min()-0.5, signal.max()+0.5]
    saliency_2d = saliency.reshape(1, -1)
    extent = [0,20,signal.min()-0.5,signal.max()+0.5] 
    ax.imshow(saliency_2d, cmap='Reds', aspect='auto', alpha=0.6, extent=extent, zorder=1)
    
    ax.set_title(f"{channel_name} (Red background = Model's Attention)", fontsize=12)
    ax.set_ylabel("Normalized Pressure")
    ax.grid(True, alpha=0.3)

# ==========================================
# 4. 推理与反向传播 (探究大脑黑盒)
# ==========================================
def predict_and_explain(csv_path):
    print(f"🔍 正在诊断文件: {csv_path}")
    
    # 加载模型
    model = MassageFusionNet(model_type=MODEL_TYPE, num_classes=4)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    model.eval() # 开启评估模式
    
    # 获取数据
    dyn, stat = preprocess(csv_path)
    dyn = dyn.unsqueeze(0)   
    stat = stat.unsqueeze(0)  
    #: 告诉 PyTorch，我们要追踪动态波形的梯度 (为了画热力图)
    dyn.requires_grad_()
    
    # 推理
    output = model(dyn, stat)
    probs = torch.softmax(output, dim=1).squeeze()
    pred_idx = torch.argmax(probs,dim=0).item()
    confidence = probs[pred_idx].item()

    print(f"\n✅ 诊断完成:")
    print(f"👉 预测结果: 【{LABELS}】 (置信度: {confidence:.2%})")
    
    #: 对预测的类别进行反向传播，求出波形中哪些点对这个预测贡献最大
    score = output
    model.zero_grad()
    target_score = output[0, pred_idx]
    target_score.backward()
    
    # 提取梯度绝对值作为“注意力权重”
    saliency_map = dyn.grad.abs().squeeze().numpy() # 形状: (2, 1000)
    
    # ==========================================
    # 5. 生成可视化报告
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # 平滑一下热力图，让它更好看 (滑动平均)
    smooth_window = 20
    sal_c1 = np.convolve(saliency_map[0], np.ones(smooth_window)/smooth_window, mode='same')
    sal_c2 = np.convolve(saliency_map[1], np.ones(smooth_window)/smooth_window, mode='same')
    dyn_np = dyn.detach().numpy()[0]  # (2, 1000)
    sal_c1_np = np.atleast_1d(sal_c1).flatten()
    sal_c2_np = np.atleast_1d(sal_c2).flatten()
    plot_saliency(dyn_np[0], sal_c1_np, "Sensor 1 Waveform", ax1)
    plot_saliency(dyn_np[1], sal_c2_np, "Sensor 2 Waveform", ax2)  

    ax2.set_xlabel("Time (Seconds)")
    fig.suptitle(f"AI Explainability Report - Predicted: {LABELS} ({confidence:.1%})", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("experiment/test/result/model_attention_report.png", dpi=150)
    print("\n📸 模型注意力热力图已生成: model_attention_report.png")
    plt.close()

if __name__ == "__main__":
    predict_and_explain(CSV_FILE)