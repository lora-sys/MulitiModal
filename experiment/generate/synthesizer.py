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
# test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    wave_gen = WaveformGenerator(
        base_freq=0.5
    )
    jitter  = TemporalJitter(
        long_term_amp=0.10,  
        short_term_amp=0.03,
        transient_freq=0.033,
    ) 
    t =np.linspace(0,20,1000)
    base_pressure= 50
    amplitude = 20
    wave_clean = wave_gen.generate(t,base_pressure,amplitude)
    
    t_jittered= jitter.apply(t,0.5)
    wave_jittered = wave_gen.generate(t_jittered,base_pressure,amplitude)
    fig,axes = plt.subplots(2,1,figsize=(14,8))
    # 图1 波形对比
    axes[0].plot(t,wave_clean,'b-',alpha=0.5,linewidth =1.5,label = "original (too regular)")
    axes[0].plot(t,wave_jittered,'r-',alpha=0.8,linewidth=1.5,label = "with jitterer realastic") 
    axes[0].set_ylabel('Pressure (Pa)')
    axes[0].set_title('Waveform: Original vs Jittered')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 图2 时间轴偏差累积
    time_devation = t_jittered-t
    axes[1].plot(t,time_devation,'g-',linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Time Deviation (s)')
    axes[1].set_title('Cumulative Time Jitter (Shows Frequency Drift)')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment/test/generate/test_part2.png', dpi=150)
    plt.close()                    