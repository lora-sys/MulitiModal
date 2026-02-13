""""
单一按摩椅数据合成器
合成的数据格式规范
 "{global_id:03d}_{weight}_{hr}_{spo2}_{height}.csv"
按人员组织
数据合成和增强
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List,Dict,Tuple
import logging


# -----    机械波形生成             -------
class WaveformGenerator:
    """
    机械波形生成器
    模拟真实按摩椅机械手的往复运动：
    - 推压期（35%）：sin²曲线，模拟滚轮下压
    - 保持期（25%）：平顶，模拟停顿施压
    - 回弹期（30%）：指数衰减，模拟快速回弹
    - 复位期（10%）：基线，模拟机械复位
    
    设计依据：按摩椅机械结构（滚轮+导轨+弹簧）
    """
    def __init__(self,base_freq:float = 0.5):
        """
        初始化波形生成器
        Args:
            base_freq: 基础频率(Hz),默认0.5Hz = 2秒/周期
        """
        self.base_freq= base_freq
        self.period = 1.0/base_freq
        self.duty_cycle = {
            "push" : 0.35,    #指压期
            "hold" : 0.25 ,   # 保持期
            "release" : 0.30, # 回弹期
            "reset" : 0.10    # 复位期   
        }
    def generate(self,t : np.ndarray,base_pressure : float, amplitude : float , phase_shift : float = 0.0) -> np.ndarray :
        """
        生成机械波形
        
        Args:
            t: 时间数组（秒）
            base_pressure: 基础压力( Pa)，与体重相关
            amplitude: 振幅(Pa)，与身体表征相关
            phase_shift: 相位偏移（弧度），用于双传感器区分
            
        Returns:
            waveform: 压力波形数组(Pa)
        """
         # 计算每个时间点的相位（0-1表示一个周期内的位置）
        # 将连续的时间点t映射到一个 0到1的归一化相位值
        phase = ((t/self.period)+phase_shift/ (2*np.pi)) % 1.0
        
        # 分段函数，根据相位生成波形
        waveform = np.piecewise(
           phase,
           [
               phase < self.duty_cycle['push'],      # 推压期
               (phase >= self.duty_cycle['push']) &
               (phase < self.duty_cycle['push'] + self.duty_cycle['hold']) , # 保持期
               (phase >= self.duty_cycle['push'] + self.duty_cycle['hold']) & 
                (phase < 1 - self.duty_cycle['reset']),      # 回弹期
                phase >= 1- self.duty_cycle['reset']  # 复位期
           ] ,
            [
                lambda p: self._push_phase(p),                                      # 推压期
                lambda p: 1.0,                                                       # 保持期（平顶）
                lambda p: self._release_phase(p),                                   # 回弹期
                lambda p: 0.0                                                        # 复位期（基线）
            ]
        )
        
        # 映射实际压力值，归一化后的比例， base_pressure+ amplitude *waveform
        pressure = base_pressure+amplitude * waveform
        
        return pressure
    
    def _push_phase(self,phase : np.ndarray) -> np.ndarray :
        """
        推压期:sin²曲线模拟滚轮下压
        
        从0平滑上升到1
        """
        normalized_phase = phase / self.duty_cycle['push']
        # sin **2 曲线 ，平滑加速上升
        return np.sin(np.pi * normalized_phase / 2) ** 2
    
    def _release_phase(self, phase: np.ndarray) -> np.ndarray:
        """
        回弹期：指数衰减模拟快速回弹
        
        从1快速衰减到0
        """
        # 计算在回弹期的位置
        release_start = self.duty_cycle['push'] + self.duty_cycle['hold']
        normalized_phase = (phase - release_start) / self.duty_cycle['release']
        # 指数衰减：快速回弹
        return np.exp(-5 * normalized_phase)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 创建波形生成器
    gen = WaveformGenerator(base_freq=0.5)
    
    # 生成10秒数据（50Hz采样率）
    t = np.linspace(0, 10, 500)  # 10秒，500个点
    base_pressure = 50  # 基础压力50Pa
    amplitude = 20      # 振幅20Pa
    
    # 生成波形
    waveform = gen.generate(t, base_pressure, amplitude)
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(t, waveform, linewidth=2, label='Mechanical Waveform')
    plt.axhline(y=base_pressure, color='r', linestyle='--', alpha=0.5, label='Base Pressure')
    plt.axhline(y=base_pressure + amplitude, color='g', linestyle='--', alpha=0.5, label='Peak Pressure')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (Pa)')
    plt.title('Massage Chair Mechanical Waveform (10s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('experiment/test/generate/test_waveform_part1.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Part 1 测试完成！")
    print(f"  - 生成 {len(t)} 个数据点")
    print(f"  - 基础压力: {base_pressure}Pa")
    print(f"  - 振幅: {amplitude}Pa")
    print(f"  - 频率: 0.5Hz (2秒/周期)")
        
        
        
    

