#!/bin/bash
# MulitiModal 开发环境快速检查脚本

set -e

echo "========================================"
echo "MulitiModal 开发环境检查"
echo "========================================"
echo

# 检查 Python 版本
echo "📌 Python 环境"
python --version
if [ -d ".venv" ]; then
    echo "✅ 虚拟环境已创建"
else
    echo "❌ 虚拟环境未找到"
    exit 1
fi

# 检查核心依赖
echo
echo "📦 核心依赖检查"
python -c "
import sys
packages = [
    ('numpy', 'NumPy'),
    ('torch', 'PyTorch'),
    ('gradio', 'Gradio'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
    ('sklearn', 'scikit-learn'),
]
for mod, name in packages:
    try:
        __import__(mod)
        print(f'  ✓ {name}')
    except ImportError:
        print(f'  ✗ {name}')
        sys.exit(1)
"
echo "✅ 所有核心依赖就绪"

# 检查开发工具
echo
echo "🛠️  开发工具检查"
python -c "
import sys
tools = [
    ('black', 'Black'),
    ('flake8', 'Flake8'),
    ('mypy', 'MyPy'),
    ('pytest', 'pytest'),
    ('jupyter', 'Jupyter'),
]
for mod, name in tools:
    try:
        __import__(mod)
        print(f'  ✓ {name}')
    except ImportError:
        print(f'  ✗ {name}')
        sys.exit(1)
"
echo "✅ 所有开发工具就绪"

echo
echo "========================================"
echo "✅ 开发环境配置完成!"
echo "========================================"
echo
echo "激活虚拟环境: source .venv/bin/activate"
echo "运行演示: cd demo && python app.py"
echo "验证环境: python check_env.py"
echo
