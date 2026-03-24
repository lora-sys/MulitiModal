#!/usr/bin/env python
"""测试 baseline_c 模型是否能正常工作"""
import sys
sys.path.insert(0, '/home/lora/repos/MulitiModal/experiment/model')
sys.path.insert(0, '/home/lora/repos/MulitiModal/experiment')

import torch
from model import get_model

# 创建模型
model = get_model(
    model_type="baseline_c",
    num_classes=3,
    dyn_channels=2,
    static_dim=4
)

print("模型创建成功！")
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# 测试前向传播
batch_size = 4
dynamic = torch.randn(batch_size, 2, 1000)
static_basic = torch.randn(batch_size, 4)
static_scores = torch.randn(batch_size, 2)
constitution = torch.randint(0, 38, (batch_size,))

try:
    with torch.no_grad():
        output, attn_weights = model(dynamic, static_basic, static_scores, constitution, return_attention=True)
    print(f"前向传播成功！")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"输出值范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"注意力权重范围: [{attn_weights.min():.4f}, {attn_weights.max():.4f}]")
except Exception as e:
    print(f"前向传播失败: {e}")
    import traceback
    traceback.print_exc()

# 测试多次前向传播
print("\n测试多次前向传播...")
for i in range(5):
    try:
        with torch.no_grad():
            output, attn_weights = model(dynamic, static_basic, static_scores, constitution, return_attention=True)
        print(f"  第 {i+1} 次成功: 输出范围 [{output.min():.4f}, {output.max():.4f}]")
    except Exception as e:
        print(f"  第 {i+1} 次失败: {e}")
        break

print("\n测试完成！")