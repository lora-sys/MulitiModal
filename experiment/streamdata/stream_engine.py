import sys
sys.path.append("experiment/model")
sys.path.append("experiment/dataset")

import time
import os
import re
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, lfilter
from model import MassageFusionNet



# ==========================================
# 1. 核心组件：单片机内存模拟器 (Ring Buffer)
# ==========================================
class RingBuffer:
    def __init__(self,channels=2,capacity=1000):
        self.capacity = capacity
        self.buffer = np.zeros((channels,capacity),dtype=np.float32)
        self.is_ready= False
        self.current_size =0
    
    
    def append(self,new_data):
        """
        new_data.shape : (2,chunk_size)
        """
        chunk_size= new_data.shape[1]
        self.buffer = np.roll(self.buffer,-chunk_size,axis=1)
        self.buffer[:,-chunk_size:] = new_data
        
        self.current_size += chunk_size
        
        if self.current_size >= self.capacity:
            self.is_ready = True
            
        return self.buffer
    
    
# ==========================================
# 2. 核心组件：实时信号清洗器 (Online Preprocessor)
# ==========================================
class RealtimePreprocessor:
    def __init__(self,fs=50,cutoff=10.0):
     nyq = 0.5 * fs
     normal_cutoff = cutoff / nyq
     self.b,self.a = butter(4,normal_cutoff,btype="low",analog=False)
     
    def process(self,buffer_data):
        """
        输入 shape : (2,1000)
        """
        process_channels = []
        for i in range(buffer_data.shape[0]):
            signal = buffer_data[i,:]
            
            # 1. 工业级因果滤波 (Causal Filtering)
            # 绝对不能用 filtfilt，因为那是双向滤波(会透视未来)，必须用单向的 lfilter
            sig_filtered = lfilter(self.b,self.a,signal)                
            
            # 2. 滚动 Z-Score (Rolling Normalization)
            # 只用当前 20 秒的数据算均值和方差，防止长时间按压导致的基线漂移
            mean_val =np.mean(sig_filtered)
            std_val = np.std(sig_filtered)+1e-6
            sig_norm = (sig_filtered-mean_val)/ std_val
            process_channels.append(sig_norm)
        return np.array(process_channels,dtype=np.float32)
    
    
# ==========================================
# 3. 流式引擎主程序 (Streaming Engine)
# ==========================================
def run_streaming_engine():
    # --- 1. 配置与加载模型 ---
    MODEL_TYPE = 'inception'  # 根据你训练的最好模型修改
    WEIGHT_PATH = f"experiment/model/best_model_inception.pth"
    CSV_PATH = "experiment/streamdata/stream_003_70_75_98_175.csv" # 你的 5 分钟剧本数据

    print("⚙️ 正在初始化流式推理引擎...")
    device = torch.device("cpu")
    model =MassageFusionNet(model_type=MODEL_TYPE,num_classes=4).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH,map_location=device))
    model.eval()
    
    
    # --- 2. 提取静态特征 (模拟用户扫码登录) ---
    filename = os.path.basename(CSV_PATH)
    stats = re.findall(r'\d+', filename)
    weight, hr, spo2, height = float(stats[1])/100.0, float(stats[2])/120.0, float(stats[3])/100.0, float(stats[4])/200.0
    static_tensor = torch.tensor([[weight, hr, spo2, height]], dtype=torch.float32).to(device) 
    
    # 3. 初始化缓冲区和清洗器
    buffer = RingBuffer(channels=2,capacity=1000)
    preprocessor = RealtimePreprocessor(fs=50,cutoff=10.0)
    
     # 读取剧本数据
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    chunk_size = 50 # 模拟传感器每秒发送 50 个点 (1Hz 预测频率)          
        
    labels_map = {0: "🟥 很差(警报)", 1: "🟨 一般(注意)", 2: "🟦 正常(平稳)", 3: "🟩 良好(享受)"}
    
    print("\n🚀 传感器连接成功！开始实时监测...\n")
    print("="*60)
    print(f"{'时间':<10} | {'系统判定状态':<20} | {'置信度':<10} | {'真实剧本(上帝视角)'}")
    print("="*60)
    
   # 开始时间循环
    for i in range(0,total_rows,chunk_size):
        # 截取1秒的数据
        end_idx = min(i+chunk_size,total_rows)
        new_p1 = df['压力传感器1'].values[i:end_idx]
        new_p2 = df['压力传感器2'].values[i:end_idx]
        true_label_chunk = df['True_Label'].values[i:end_idx]
        
        # 压入缓冲区
        new_data = np.vstack((new_p1,new_p2))
        current_buffer = buffer.append(new_data)
        
        current_sec = end_idx // 50
        
        # 如果缓冲区没满20秒，处于数据蓄水期，跳过推理
        if not buffer.is_ready:
            print(f"⏱️ 00:{current_sec:02d} | ⏳ 正在收集初始数据 (需20秒)...")
            time.sleep(0.05)
            continue
        # ==================================
        # ⚡ 实时触发推理
        # ==================================
        # 1. 实时清洗
        processed_signal = preprocessor.process(current_buffer)
        
        # 2. 转为张量（batch =1）
        dynamic_tensor = torch.tensor(processed_signal).unsqueeze(0).to(device)
        
        # 3 .ai 预测
        with torch.no_grad():
            output = model(dynamic_tensor,static_tensor)
            probs = torch.softmax(output,dim=1).squeeze()
            confidence ,pred_class = torch.max(probs,0)
       
        # 打印实时大屏
        pred_id = pred_class.item()
        true_id = true_label_chunk[-1]  # 取一秒最后一刻的真实状态
        print(f"⏱️ 00:{current_sec:02d} | {labels_map[pred_id]:<18} | {confidence.item():>6.2%}   | 真实: {true_id}")          
        
        time.sleep(0.1)

if __name__ == "__main__":
    run_streaming_engine()
                        
            
                    