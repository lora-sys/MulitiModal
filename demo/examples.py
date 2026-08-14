"""MulitiModal Demo — 预设示例数据
=====================================

每条样本包含:
  - profile:    年龄/性别/身高/体重/BMI/心率/血氧
  - tcm:        舌诊/舌苔/脉象/面诊 评分 (1-10)
  - vitals:     心率/血氧/呼吸率/体温
  - eeg:        模拟脑电波形 (1000 点, 模拟正念指数变化)
  - ecg:        模拟 ECG 波形 (1000 点)
  - eda:        模拟 EDA 波形 (1000 点)
  - mindfulness: 正念指数 (0-1)
  - label:      场景标签
"""

from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────
# 波形生成工具
# ──────────────────────────────────────────────────────────────

def _ecg_waveform(heart_rate: float = 72.0, length: int = 1000, seed: int = 42) -> np.ndarray:
    """生成近似 ECG PQRST 波形."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8, length)
    period = 60.0 / heart_rate
    phase = (t % period) / period

    ecg = np.zeros(length)
    # P wave
    mask = (phase >= 0.1) & (phase < 0.25)
    ecg[mask] = 0.15 * np.sin((phase[mask] - 0.1) * np.pi / 0.15)
    # QRS complex
    mask = (phase >= 0.30) & (phase < 0.35)
    ecg[mask] = -0.1 * np.sin((phase[mask] - 0.30) * np.pi / 0.05)
    mask = (phase >= 0.35) & (phase < 0.42)
    ecg[mask] = 1.0 * np.sin((phase[mask] - 0.35) * np.pi / 0.07)
    mask = (phase >= 0.42) & (phase < 0.48)
    ecg[mask] = -0.2 * np.sin((phase[mask] - 0.42) * np.pi / 0.06)
    # T wave
    mask = (phase >= 0.55) & (phase < 0.75)
    ecg[mask] = 0.25 * np.sin((phase[mask] - 0.55) * np.pi / 0.20)

    noise = 0.03 * rng.standard_normal(length)
    return (ecg + noise).astype(np.float32)


def _eda_waveform(stress_level: float = 0.3, length: int = 1000, seed: int = 42) -> np.ndarray:
    """生成 EDA 波形，tone_level 越高越活跃."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8, length)
    base = 2.0
    tonic = stress_level * 1.5
    phasic = 0.3 * np.sin(2 * np.pi * 0.2 * t) + 0.15 * np.sin(2 * np.pi * 0.47 * t)
    scr = 0.08 * np.abs(np.sin(2 * np.pi * 0.13 * t + rng.uniform(0, np.pi)))
    noise = 0.05 * rng.standard_normal(length)
    return (base + tonic + phasic + scr + noise).astype(np.float32)


def _eeg_waveform(mindfulness: float, length: int = 1000, seed: int = 42) -> np.ndarray:
    """模拟脑电波形，正念指数高 → 更多 alpha 节律 (8-13Hz)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8, length)
    fs = length / 8

    # alpha 功率 (8-13 Hz)
    alpha_power = 0.3 + mindfulness * 0.7
    alpha = alpha_power * np.sin(2 * np.pi * 10 * t)
    # beta (13-30 Hz)
    beta = 0.15 * np.sin(2 * np.pi * 20 * t) * (1.0 - mindfulness * 0.5)
    # theta (4-8 Hz)
    theta = 0.2 * np.sin(2 * np.pi * 6 * t) * (0.3 + mindfulness * 0.7)

    noise = 0.1 * rng.standard_normal(length)
    return (alpha + beta + theta + noise).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 预设样本
# ──────────────────────────────────────────────────────────────

def get_preset(name: str) -> dict:
    presets = {
        "balanced_health": _preset_balanced_health(),
        "stress_recovery": _preset_stress_recovery(),
        "low_mindfulness": _preset_low_mindfulness(),
        "deep_relaxation": _preset_deep_relaxation(),
    }
    return presets.get(name, presets["balanced_health"])


def get_preset_list() -> list[dict]:
    """返回 [(key, label, description), ...]"""
    return [
        ("balanced_health", "均衡健康", "平和质体质，各项指标正常，建议日常保健"),
        ("stress_recovery", "压力恢复", "气虚质，工作压力大，心率偏高，建议深度放松"),
        ("low_mindfulness", "专注力不足", "气郁质，脑电正念指数低，建议身心调节"),
        ("deep_relaxation", "深度放松", "湿热质，血氧正常，建议深层理疗舒缓"),
    ]


def _preset_balanced_health() -> dict:
    """场景 1: 均衡健康 — 平和质, 各项指标正常"""
    return {
        "label": "均衡健康 — 平和质",
        "description": "30 岁男性，日常健康状态，中医诊断为平和质",
        "profile": {
            "age": 30, "gender": "男", "height": 175, "weight": 70,
            "bmi": 22.9, "heart_rate": 68, "spo2": 98.5,
        },
        "tcm": {
            "tongue": 6.5,     # 舌色红润
            "coating": 6.0,    # 舌苔薄白
            "pulse": 6.5,      # 脉象平和
            "face": 7.0,       # 面色红润
        },
        "vitals": {
            "heart_rate": 68,
            "spo2": 98.5,
            "resp_rate": 16,
            "temperature": 36.5,
        },
        "mindfulness": 0.78,
        "ecg": _ecg_waveform(68, seed=100),
        "eda": _eda_waveform(0.15, seed=100),
        "eeg": _eeg_waveform(0.78, seed=100),
    }


def _preset_stress_recovery() -> dict:
    """场景 2: 压力恢复 — 气虚质, 工作压力大"""
    return {
        "label": "压力恢复 — 气虚质",
        "description": "38 岁女性，近期高强度工作，疲劳累积，中医诊断为气虚质",
        "profile": {
            "age": 38, "gender": "女", "height": 162, "weight": 55,
            "bmi": 20.9, "heart_rate": 88, "spo2": 96.8,
        },
        "tcm": {
            "tongue": 4.0,     # 舌色淡白
            "coating": 3.5,    # 舌苔薄少
            "pulse": 3.5,      # 脉象细弱
            "face": 4.0,       # 面色萎黄
        },
        "vitals": {
            "heart_rate": 88,
            "spo2": 96.8,
            "resp_rate": 20,
            "temperature": 36.4,
        },
        "mindfulness": 0.45,
        "ecg": _ecg_waveform(88, seed=200),
        "eda": _eda_waveform(0.55, seed=200),
        "eeg": _eeg_waveform(0.45, seed=200),
    }


def _preset_low_mindfulness() -> dict:
    """场景 3: 专注力不足 — 气郁质, 正念指数低"""
    return {
        "label": "专注力不足 — 气郁质",
        "description": "26 岁男性，长期焦虑，脑电正念指数偏低，中医诊断为气郁质",
        "profile": {
            "age": 26, "gender": "男", "height": 178, "weight": 72,
            "bmi": 22.7, "heart_rate": 78, "spo2": 97.2,
        },
        "tcm": {
            "tongue": 5.0,     # 舌色暗
            "coating": 4.5,    # 舌苔薄白
            "pulse": 4.0,      # 脉象弦
            "face": 4.5,       # 面色晦暗
        },
        "vitals": {
            "heart_rate": 78,
            "spo2": 97.2,
            "resp_rate": 18,
            "temperature": 36.5,
        },
        "mindfulness": 0.25,
        "ecg": _ecg_waveform(78, seed=300),
        "eda": _eda_waveform(0.40, seed=300),
        "eeg": _eeg_waveform(0.25, seed=300),
    }


def _preset_deep_relaxation() -> dict:
    """场景 4: 深层放松 — 湿热质, 适合深层理疗"""
    return {
        "label": "深层放松 — 湿热质",
        "description": "42 岁男性，湿热体质，肌肉紧张，建议深层理疗方案",
        "profile": {
            "age": 42, "gender": "男", "height": 180, "weight": 85,
            "bmi": 26.2, "heart_rate": 76, "spo2": 97.0,
        },
        "tcm": {
            "tongue": 7.5,     # 舌红苔黄
            "coating": 7.5,    # 舌苔黄腻
            "pulse": 7.0,      # 脉象滑数
            "face": 6.5,       # 面色油光
        },
        "vitals": {
            "heart_rate": 76,
            "spo2": 97.0,
            "resp_rate": 17,
            "temperature": 36.6,
        },
        "mindfulness": 0.55,
        "ecg": _ecg_waveform(76, seed=400),
        "eda": _eda_waveform(0.35, seed=400),
        "eeg": _eeg_waveform(0.55, seed=400),
    }
