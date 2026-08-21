#!/usr/bin/env python3
"""
MulitiModal Demo 验证脚本
测试 demo 模块导入和模型初始化
"""

import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 路径设置 (与 demo/app.py 保持一致)
# ──────────────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
LEGACY_ROOT = PROJECT_ROOT / "legacy_research"

sys.path.insert(0, str(DEMO_DIR))
sys.path.insert(0, str(LEGACY_ROOT / "source/tcm/tcm_ft_transformer"))
sys.path.insert(0, str(LEGACY_ROOT / "source/oplri/src"))
sys.path.insert(0, str(LEGACY_ROOT / "source/oplri/src/models"))

print("=" * 60)
print("MulitiModal Demo 验证")
print("=" * 60)

# 1. 测试基础导入
print("\n📦 测试模块导入...")

try:
    import numpy as np
    print(f"  ✓ numpy {np.__version__}")
except ImportError as e:
    print(f"  ✗ numpy: {e}")
    sys.exit(1)

try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except ImportError as e:
    print(f"  ✗ torch: {e}")
    sys.exit(1)

try:
    import gradio as gr
    print(f"  ✓ gradio {gr.__version__}")
except ImportError as e:
    print(f"  ✗ gradio: {e}")
    sys.exit(1)

# 2. 测试历史代码导入
print("\n📚 测试历史代码导入...")

try:
    from ft_transformer import get_model
    print("  ✓ ft_transformer.get_model")
except Exception as e:
    print(f"  ✗ ft_transformer: {e}")
    sys.exit(1)

try:
    from models.encoders import get_dynamic_encoder
    print("  ✓ models.encoders.get_dynamic_encoder")
except Exception as e:
    print(f"  ✗ models.encoders: {e}")
    sys.exit(1)

# 3. 测试 demo 模块导入
print("\n🎯 测试 demo 模块...")

try:
    from app import get_manager, CONSTITUTION_NAMES
    print("  ✓ app.get_manager")
    print(f"  ✓ app.CONSTITUTION_NAMES ({len(CONSTITUTION_NAMES)} 个体质)")
except Exception as e:
    print(f"  ✗ app 模块: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from examples import get_preset, get_preset_list
    print("  ✓ examples.get_preset")
    print(f"  ✓ examples.get_preset_list ({len(get_preset_list())} 个预设)")
except Exception as e:
    print(f"  ✗ examples: {e}")
    sys.exit(1)

try:
    from ui import draw_waveform, draw_tcm_bars
    print("  ✓ ui 绘图函数")
except Exception as e:
    print(f"  ✗ ui: {e}")
    sys.exit(1)

# 4. 测试模型加载
print("\n🧠 测试模型加载 (这可能需要 10-30 秒)...")

try:
    print("  正在加载 ModelManager...")
    manager = get_manager()
    print(f"  ✓ ModelManager 初始化成功")
    print(f"  ✓ 设备: {manager.device}")
except Exception as e:
    print(f"  ✗ ModelManager 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 测试推理
print("\n⚡ 测试推理管线...")

try:
    preset_name = get_preset_list()[0]
    sample = get_preset(preset_name)
    print(f"  使用预设样本: {preset_name}")

    result = manager.run_inference(sample)
    print("  ✓ 推理成功!")
    print(f"  ✓ 体质识别: {result['constitution']['name']} (置信度: {result['constitution']['confidence']:.2%})")
    print(f"  ✓ 推荐方案: {result['recommendation']['program']['name']}")
    print(f"  ✓ 力度等级: {result['recommendation']['intensity']['name']}")

except Exception as e:
    print(f"  ✗ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 总结
print("\n" + "=" * 60)
print("✅ Demo 验证完成! 所有测试通过!")
print("=" * 60)
print("\n启动演示:")
print("  cd demo && python app.py")
print("\n或运行此脚本所在目录的 Gradio 界面:")
print("  cd demo && .venv/bin/python app.py")
