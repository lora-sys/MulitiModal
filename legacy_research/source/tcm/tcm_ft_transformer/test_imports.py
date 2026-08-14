"""
测试脚本 - 验证所有模块是否可以正常导入
"""

import sys

def test_imports():
    """测试所有模块导入"""
    print("=" * 60)
    print("测试模块导入")
    print("=" * 60)
    
    modules = [
        ('config', '配置模块'),
        ('ft_transformer', 'FT-Transformer 模型'),
        ('preprocess', '数据预处理'),
        ('train', '训练器'),
        ('optuna_search', 'Optuna 搜索'),
        ('visualize', '可视化工具'),
    ]
    
    failed = []
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✅ {description} ({module_name})")
        except ImportError as e:
            print(f"❌ {description} ({module_name}): {e}")
            failed.append(module_name)
    
    print("=" * 60)
    
    if failed:
        print(f"\n❌ {len(failed)} 个模块导入失败:")
        for module in failed:
            print(f"  - {module}")
        return False
    else:
        print("\n✅ 所有模块导入成功！")
        return True


def test_dependencies():
    """测试依赖包"""
    print("\n" + "=" * 60)
    print("测试依赖包")
    print("=" * 60)
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('optuna', 'Optuna'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    
    failed = []
    
    for module_name, description in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {description} ({module_name})")
        except ImportError as e:
            print(f"❌ {description} ({module_name}): {e}")
            failed.append(module_name)
    
    print("=" * 60)
    
    if failed:
        print(f"\n❌ {len(failed)} 个依赖包缺失:")
        for module in failed:
            print(f"  - {module}")
        print("\n安装命令: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 所有依赖包已安装！")
        return True


def test_cuda():
    """测试 CUDA"""
    print("\n" + "=" * 60)
    print("测试 CUDA")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用")
            print(f"   设备数量: {torch.cuda.device_count()}")
            print(f"   当前设备: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 版本: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA 不可用，将使用 CPU")
            return True
    except Exception as e:
        print(f"❌ CUDA 检测失败: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FT-Transformer 中医体质分类 - 环境测试")
    print("=" * 80 + "\n")
    
    # 测试依赖
    deps_ok = test_dependencies()
    
    # 测试模块
    modules_ok = test_imports()
    
    # 测试 CUDA
    cuda_ok = test_cuda()
    
    print("\n" + "=" * 80)
    if deps_ok and modules_ok and cuda_ok:
        print("✅ 环境测试通过！可以开始训练。")
        print("=" * 80 + "\n")
        sys.exit(0)
    else:
        print("❌ 环境测试失败！请检查上述错误。")
        print("=" * 80 + "\n")
        sys.exit(1)
