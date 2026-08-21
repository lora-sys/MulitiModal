#!/usr/bin/env python3
"""
演示 checkpoint 结构
"""

import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path('/home/lora/repos/work/MulitiModal')
LEGACY_ROOT = PROJECT_ROOT / "legacy_research"

sys.path.insert(0, str(LEGACY_ROOT / "source/tcm/tcm_ft_transformer"))
sys.path.insert(0, str(LEGACY_ROOT / "source/oplri/src"))
sys.path.insert(0, str(LEGACY_ROOT / "source/oplri/src/models"))

# 加载 checkpoint
ckpt = torch.load(
    LEGACY_ROOT / "checkpoints/oplri/server_f743da3/best_model.pth",
    map_location="cpu",
    weights_only=True
)

state_dict = ckpt["model_state_dict"]

print("=" * 60)
print("Checkpoint 结构分析")
print("=" * 60)

print(f"\n顶层键: {list(ckpt.keys())}")
print(f"  - model_state_dict: {len(state_dict)} 个参数")
print(f"  - best_encoder: {ckpt['best_encoder']}")
print(f"  - metrics: {ckpt['metrics']}")

# 分析 state_dict 的顶层命名空间
namespaces = {}
for key in state_dict.keys():
    parts = key.split('.')
    ns = parts[0]
    if ns not in namespaces:
        namespaces[ns] = []
    namespaces[ns].append(key)

print(f"\n命名空间统计:")
for ns, keys in namespaces.items():
    print(f"  {ns}.xxx: {len(keys)} 个参数")

# 动态编码器详细信息
print(f"\n动态编码器 ({namespaces.get('dynamic_encoder', [])[0].split('.')[1]}):")
for key in sorted([k for k in state_dict.keys() if k.startswith('dynamic_encoder.')])[:10]:
    print(f"  {key}: {state_dict[key].shape}")

# TCM 编码器详细信息
print(f"\nTCM 编码器:")
for key in sorted([k for k in state_dict.keys() if k.startswith('tcm_encoder.')])[:5]:
    print(f"  {key}: {state_dict[key].shape}")
