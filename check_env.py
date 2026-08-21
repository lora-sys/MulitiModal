#!/usr/bin/env python3
"""
MulitiModal 环境验证脚本
检查虚拟环境和依赖是否正确安装
"""

import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    assert version >= (3, 10), "需要 Python 3.10+"
    return True

def check_venv():
    """检查是否在虚拟环境中"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print(f"✅ 虚拟环境激活: {sys.prefix}")
    else:
        print("⚠️  未检测到虚拟环境 (建议使用 .venv/)")
    return True

def check_package(package_name, import_name=None):
    """检查单个包"""
    if import_name is None:
        import_name = package_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {package_name:20s} {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {package_name:20s} 未安装")
        return False

def main():
    print("=" * 60)
    print("MulitiModal 环境验证")
    print("=" * 60)

    # Python 版本
    print("\n📌 Python 环境")
    check_python_version()
    check_venv()

    # 核心依赖
    print("\n📦 核心科学计算")
    core_packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
    ]
    core_ok = all(check_package(*pkg) for pkg in core_packages)

    # 深度学习
    print("\n🧠 深度学习框架")
    dl_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
    ]
    dl_ok = all(check_package(*pkg) for pkg in dl_packages)

    # 可视化
    print("\n📊 可视化")
    viz_packages = [
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    viz_ok = all(check_package(*pkg) for pkg in viz_packages)

    # 机器学习
    print("\n🤖 机器学习工具")
    ml_packages = [
        ("scikit-learn", "sklearn"),
        ("optuna", "optuna"),
    ]
    ml_ok = all(check_package(*pkg) for pkg in ml_packages)

    # 演示与工具
    print("\n🖥️  演示与工具")
    tool_packages = [
        ("gradio", "gradio"),
        ("tqdm", "tqdm"),
        ("h5py", "h5py"),
        ("pyyaml", "yaml"),
    ]
    tool_ok = all(check_package(*pkg) for pkg in tool_packages)

    # 总结
    print("\n" + "=" * 60)
    all_ok = core_ok and dl_ok and viz_ok and ml_ok and tool_ok
    if all_ok:
        print("✅ 所有依赖检查通过!")
    else:
        print("❌ 部分依赖缺失,请运行: pip install -r requirements.txt")
    print("=" * 60)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
