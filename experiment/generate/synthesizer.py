import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WaveformGenerator:
    """
    机械波形生成器
    模拟真实按摩椅机械手的往复运动：
    - 推压期(35%):sin²曲线,模拟滚轮下压
    - 保持期(25%）：平顶，模拟停顿施压
    - 回弹期(30%）：指数衰减，模拟快速回弹
    - 复位期(10%）：基线，模拟机械复位
    """
    def __init__(self, base_freq: float = 0.5):
        self.base_freq = base_freq
        self.period = 1.0 / base_freq
        self.duty_cycle = {
            "push": 0.35,
            "hold": 0.25,
            "release": 0.30,
            "reset": 0.10   
        }
        # 预计算边界，避免重复计算
        self._bound_push_end = self.duty_cycle['push']
        self._bound_hold_end = self.duty_cycle['push'] + self.duty_cycle['hold']
        self._bound_release_end = 1.0 - self.duty_cycle['reset']

    def generate(self, t: np.ndarray, base_pressure: float, amplitude: float, phase_shift: float = 0.0) -> np.ndarray:
        """
        生成机械波形
        """
       
        # phase_shift / (2*np.pi) 将弧度转换为归一化周期比例
        phase = ((t / self.period) + (phase_shift / (2 * np.pi))) % 1.0
        
        # 初始化全0数组
        waveform = np.zeros_like(phase)
        
        # --- 1. 推压期 ---
        mask_push = phase < self._bound_push_end
        if self.duty_cycle['push'] > 0:
            # 归一化到 [0, 1]
            p_norm = phase[mask_push] / self.duty_cycle['push']
            # sin² 曲线
            waveform[mask_push] = np.sin(np.pi * p_norm / 2) ** 2
            
        # --- 2. 保持期 ---
        mask_hold = (phase >= self._bound_push_end) & (phase < self._bound_hold_end)
        waveform[mask_hold] = 1.0
        
        # --- 3. 回弹期 ---
        mask_release = (phase >= self._bound_hold_end) & (phase < self._bound_release_end)
        if self.duty_cycle['release'] > 0:
            # 归一化到 [0, 1]
            p_norm = (phase[mask_release] - self._bound_hold_end) / self.duty_cycle['release']
            # 指数衰减: exp(-5x), x从0到1，值从1衰减到exp(-5)≈0.006
            waveform[mask_release] = np.exp(-5 * p_norm)
            
        # --- 4. 复位期 ---
        # mask_reset = phase >= self._bound_release_end
        # waveform 已经初始化为 0，所以这一步可以省略
        
        # 映射实际压力值
        pressure = base_pressure + amplitude * waveform
        return pressure
class TemporalJitter :
    """
    时序抖动引擎 - 让机械运动更真实
    
    模拟三种电机不稳定性：
    1. 长期漂移:电机发热,频率缓慢变化(±10%,周期100秒)
    2. 短期抖动:齿轮间隙,相位随机偏移(±3%，平滑）
    3. 瞬时事件：阻力变化，随机加速/减速(每30秒,持续4-6秒）
    
    保守参数，确保波形仍可识别但不再"太规律"
    """    
    def __init__(self, 
                 long_term_amp: float = 0.10,   # 10%
                 short_term_amp: float = 0.03,   # 3%
                 transient_freq: float = 0.033):  # 每30秒1次
        self.long_term_amp = long_term_amp
        self.short_term_amp = short_term_amp
        self.transient_freq = transient_freq
        
    def apply(self,t: np.ndarray,base_freq : float) -> np.ndarray:
        """
        对时间轴应用抖动，返回调整后的时间轴
        
        Args:
            t: 原始时间数组（秒）
            base_freq: 基础频率（Hz）
            
        Returns:
            t_jittered: 抖动后的时间轴（用于波形采样）
        """
        dt = t[1] - t[0]  
        # 采样间隔（50Hz = 0.02s）
        
        # 长期偏移： 慢变正弦调制
        # 模拟电机发热导致的转速周期性变化
        drift = 1.0 + self.long_term_amp * np.sin(2 * np.pi * t / 100.0)
        
        # --- 2. 短期抖动：平滑随机游走 ---
        # 模拟齿轮间隙导致的微小相位抖动
        # 生成随机噪声，然后用移动平均平滑（50点 = 1秒）
        raw_noise = np.random.normal(0, self.short_term_amp, len(t))
        jitter = np.convolve(raw_noise, np.ones(50)/50, mode='same') 
        
        # --- 3. 瞬时事件：随机变速 ---
        # 模拟阻力变化导致的瞬时加速或减速
        transient = np.ones_like(t)
        if self.transient_freq > 0:
            # 每30秒左右发生一次
            event_interval = int(30.0 / dt)  # 30秒对应的采样点数
            num_events = len(t) // event_interval
            
            for _ in range(num_events):
                # 随机起始点
                start = np.random.randint(0, len(t) - 300)
                duration = np.random.randint(200, 300)  # 持续4-6秒
                end = min(start + duration, len(t))
                
                # 随机选择加速或减速（1.1-1.3倍 或 0.7-0.9倍）随机因子
                factor = np.random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
                transient[start:end] = factor
        # --- 组合所有抖动 ---
        # 瞬时频率 = 基础频率 × 长期漂移 × 瞬时事件 
        instantaneous_freq = base_freq * drift * transient  
        
        
      # 从频率重建时间轴：累积积分
        # 频率是相位的变化率，所以相位 = 积分(频率) × 2π
        # instantaneous_freq：决定这一刻跑多快。
        # np.cumsum：算出总共跑了多远（真实相位）。
        # `adjusted_t：算出“这段距离如果按正常跑，需要多少时间”。
        phase = np.cumsum(instantaneous_freq) * dt * 2 * np.pi
        t_jittered = phase / (2 * np.pi * base_freq)
        
        return t_jittered 
 # ========== Part 3: 软边界参数生成器 ==========
class FuzzyBoundaryGenerator:
    """
    软边界参数生成器
    
    解决iteration1.1的问题：类别边界过于清晰（RandomForest一眼看穿）
    
    策略：
    - 85%样本：清晰类别特征（硬边界）
    - 15%样本：跨类别混合（软边界）
    
    混合方式：使用Beta分布生成混合比例，偏向中间但不完全平均
    """        
    def __init__(self,overlap_ratio:float = 0.15): #15% 保守
        self.overlap_ratio = overlap_ratio
        
        # 类别参数定义
        self.category_profiles = {
            0 : {
                # 身体表征很差
                'hr_base' : 95, # 心率高
                'hr_var' : 10., # 波动大
                'pressure_factor' : 0.8, # 承受力低
                'amplitude_var' : 0.2, # 振幅变化大
            },
            1 : {
                # 身体表征一般
                'hr_base':85,
                'hr_var' : 8,
                'pressure_factor':1.0,
                'amplitude_var' : 0.15,
          },
            2 : {
                ## 身体表征正常
                'hr_base':75,
                'hr_var' : 6,
                'pressure_factor': 1.2,
                'amplitude_var' : 0.1,
            },
            3 : {
                # 身体表征很好
                'hr_base' : 65, # 心率低
                'hr_var' : 5,  # 波动小
                'pressure_factor' : 1.5 , # 承受力高
                'amplitude_var' : 0.08,   # 振幅稳定
            }
            
        }
    def generate_params(self,category : int) -> dict :
        """
        生成带软边界的物理参数
        
        Args:
            category: 目标类别（0-3）
            
        Returns:
            params: 包含所有物理参数的字典
                  如果是软边界样本，会包含'mixed_with'和'mix_ratio'
        """
        # 决定是否生成为软边界样本（15%概率）
        is_fuzzy = np.random.random() < self.overlap_ratio
        
        if is_fuzzy and category in [1, 2]:  # 只有中间类别才有软边界
            # 选择相邻类别进行混合
            neighbor = category - 1 if np.random.random() < 0.5 else category + 1
            neighbor = np.clip(neighbor, 0, 3)
            
            # 使用Beta分布生成混合比例（α=2, β=2，中间偏向）
            mix_ratio = np.random.beta(2, 2)
            
            # 混合参数
            params = self._mix_params(category, neighbor, mix_ratio)
            params['is_fuzzy'] = True
            params['mixed_with'] = neighbor
            params['mix_ratio'] = mix_ratio
            
        else:
            # 硬边界：从目标类别直接采样
            params = self._sample_from_category(category)
            params['is_fuzzy'] = False
            
        return params
    def _sample_from_category(self,cat: int) -> dict:
         """从单个类别采样参数"""   
         profile = self.category_profiles[cat]
         
         # 身高体重
         height = np.random.randint(155,198)
         weight = int(height-105 + np.random.normal(0,5))
         
         # 心率
         hr = profile['hr_base'] + np.random.randint(-profile['hr_var'],profile['hr_var'])
         hr= np.clip(hr,50,120)
         
         # 血氧（与类别相关）
         spo2 = 98 - (3 - cat) + np.random.randint(-1, 1)
         spo2 = np.clip(spo2, 90, 100)
         
         pressure_factor = profile['pressure_factor'] + np.random.normal(0,0.1)
         base_pressure = weight*0.6*pressure_factor
         amplitude = 15 * cat *8 + np.random.normal(0,profile['amplitude_var']+10)
         
         return {
             'category' : cat,
             'height' : height,
             'weight' : weight,
             'hr' : hr,
             'spo2' : spo2,
             'base_pressure' : base_pressure,
             'amplitude' : amplitude 
         }
    
    def _mix_params(self,cat1 : int ,cat2 : int,ratio : float) -> dict:
        """混合两个类别的参数"""
        param1 = self._sample_from_category(cat1)
        param2 = self._sample_from_category(cat2)
        
        mixed = {
            
        }
        for key in ["height",'weight','hr','spo2','base_pressure','amplitude']:
            # 线性插值混合
            mixed[key] = ratio * param1[key] + (1-ratio) * param2[key]
        
        # 类别标签使用混合比列最大的
        mixed['category'] = cat1 if ratio > 0.5 else cat2
        mixed['time_category'] = cat1 # 记录原始目标类别
        
        return  mixed
# ========== Part 1+2+3 联合测试 ==========
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    np.random.seed(42)  # 固定种子
    
    # 创建组件
    wave_gen = WaveformGenerator(base_freq=0.5)
    jitter = TemporalJitter(long_term_amp=0.10, short_term_amp=0.03, transient_freq=0.033)
    boundary_gen = FuzzyBoundaryGenerator(overlap_ratio=0.15)
    
    # 生成4个类别的样本（每类2个：1个硬边界，1个软边界）
    categories = [0, 1, 2, 3]
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    
    for i, cat in enumerate(categories):
        for j in range(2):  # 每类生成2个样本
            # 生成参数
            params = boundary_gen.generate_params(cat)
            
            # 生成时间轴
            t = np.linspace(0, 10, 500)  # 10秒展示
            
            # 应用抖动
            t_jittered = jitter.apply(t, 0.5)
            
            # 生成波形
            waveform = wave_gen.generate(t_jittered, params['base_pressure'], params['amplitude'])
            
            # 绘制
            ax = axes[i, j]
            color = 'red' if params.get('is_fuzzy', False) else 'blue'
            label = f"Cat{cat} {'(Fuzzy)' if params.get('is_fuzzy') else '(Hard)'}"
            
            ax.plot(t, waveform, color=color, linewidth=1.5)
            ax.set_title(f"{label} HR:{int(params['hr'])} Amp:{params['amplitude']:.1f}")
            ax.set_ylim(40, 120)
            ax.grid(True, alpha=0.3)
            
            if i == 3:  # 最后一行
                ax.set_xlabel('Time (s)')
            if j == 0:  # 第一列
                ax.set_ylabel('Pressure (Pa)')
    
    plt.suptitle('Waveform Samples by Category (Red=Fuzzy Boundary, Blue=Hard Boundary)', fontsize=14)
    plt.tight_layout()
    plt.savefig('experiment/test/generate/test_part3.png', dpi=150)
    plt.close()
    
    # 统计软边界比例
    fuzzy_count = 0
    total = 1000
    for _ in range(total):
        p = boundary_gen.generate_params(np.random.randint(0, 4))
        if p.get('is_fuzzy', False):
            fuzzy_count += 1
    
    print(f"\n✓ Part 3 完成！")
    print(f"  总测试样本: {total}")
    print(f"  软边界样本: {fuzzy_count} ({fuzzy_count/total*100:.1f}%)")
    print(f"  预期比例: 15%")    
    
    

        
         
             
