# MulitiModal Demo

多模态人体状态感知 → 中医体质分析 → 智能按摩方案推荐 — 交互式演示系统

## 快速启动

```bash
# 1. 克隆仓库
git clone https://github.com/lora-sys/MulitiModal.git
cd MulitiModal

# 2. 安装依赖
pip install -r demo/requirements.txt

# 3. 启动演示
python demo/ui.py
```

浏览器打开 `http://localhost:7860`

## 三条检测链路

| 链路 | 输入 | 编码器 | 输出 |
|------|------|--------|------|
| TCM 诊断 | 舌诊/舌苔/脉象/面诊 4-D 评分 | FT-Transformer | 九型体质概率 + 128-D 特征 |
| 生理信号 | ECG + EDA 波形 (1000 点) | TCN 动态编码器 | 128-D 动态表征 |
| 脑电正念 | EEG 波形 (1000 点) | CNN 编码器 | 8-D 神经表征 + 正念指数 |

## 预设场景

- **均衡健康** — 平和质，日常保健
- **压力恢复** — 气虚质，高强度工作后
- **专注力不足** — 气郁质，正念指数偏低
- **深层放松** — 湿热质，深层理疗方案

## 模型资产

- OPLRI Backbone (SHA-256: `89f75e66…e2f65ba`) — 生理信号编码 + Gate A/B
- TCM Encoder (SHA-256: `b5c92665…e9422f0`) — FT-Transformer 中医体质分类
- Scaler (SHA-256: `41b5af43…a81b5b`) — 8-D 标准化参数
