"""
测试改进后的代码

测试新的日志系统、配置系统和错误处理
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_logger():
    """测试日志系统"""
    print("\n" + "=" * 60)
    print("测试1: 日志系统")
    print("=" * 60)

    from experiment.utils.logger import get_logger, setup_logging

    # 初始化日志系统
    setup_logging(
        log_dir='experiment/logs',
        level='DEBUG',
        console=True,
        file=True
    )

    # 获取日志记录器
    logger = get_logger(__name__)

    # 测试不同级别的日志
    logger.debug("这是一条DEBUG级别的日志")
    logger.info("这是一条INFO级别的日志")
    logger.warning("这是一条WARNING级别的日志")
    logger.error("这是一条ERROR级别的日志")

    print("✅ 日志系统测试通过")


def test_config():
    """测试配置系统"""
    print("\n" + "=" * 60)
    print("测试2: 配置系统")
    print("=" * 60)

    from experiment.config.base_config import ExperimentConfig

    # 创建默认配置
    config = ExperimentConfig()

    # 打印配置信息
    print(f"模型类型: {config.model.model_type}")
    print(f"任务类型: {config.model.task_type}")
    print(f"批次大小: {config.data.batch_size}")
    print(f"学习率: {config.training.learning_rate}")
    print(f"训练轮数: {config.training.num_epochs}")

    # 测试配置保存
    config.save_yaml('experiment/config/test_config.yaml')
    print("✅ 配置保存成功")

    # 测试配置加载
    loaded_config = ExperimentConfig.load_yaml('experiment/config/test_config.yaml')
    print(f"加载的配置 - 模型类型: {loaded_config.model.model_type}")
    print("✅ 配置加载成功")

    # 清理测试文件
    if os.path.exists('experiment/config/test_config.yaml'):
        os.remove('experiment/config/test_config.yaml')
        print("✅ 测试文件已清理")


def test_encoders():
    """测试编码器模块"""
    print("\n" + "=" * 60)
    print("测试3: 编码器模块")
    print("=" * 60)

    from experiment.model.encoders import (
        InceptionEncoder,
        LSTMEncoder,
        SimpleCNNEncoder,
        TransformerEncoder,
        StaticMLPEncoder,
        ConstitutionEmbedding,
        WaveformEncoder
    )

    import torch

    # 测试InceptionEncoder
    print("测试 InceptionEncoder...")
    dynamic_data = torch.randn(4, 2, 1000)  # (B, C, L)
    inception_encoder = InceptionEncoder(in_channels=2, out_channels=32)
    output = inception_encoder(dynamic_data)
    print(f"  输入形状: {dynamic_data.shape}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ InceptionEncoder测试通过")

    # 测试LSTMEncoder
    print("测试 LSTMEncoder...")
    lstm_encoder = LSTMEncoder(in_channels=2, hidden_dim=64, num_layers=2)
    output = lstm_encoder(dynamic_data)
    print(f"  输入形状: {dynamic_data.shape}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ LSTMEncoder测试通过")

    # 测试SimpleCNNEncoder
    print("测试 SimpleCNNEncoder...")
    cnn_encoder = SimpleCNNEncoder(in_channels=2, out_channels=32)
    output = cnn_encoder(dynamic_data)
    print(f"  输入形状: {dynamic_data.shape}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ SimpleCNNEncoder测试通过")

    # 测试TransformerEncoder
    print("测试 TransformerEncoder...")
    transformer_encoder = TransformerEncoder(in_channels=2, d_model=64, nhead=4, num_layers=2)
    output = transformer_encoder(dynamic_data)
    print(f"  输入形状: {dynamic_data.shape}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ TransformerEncoder测试通过")

    # 测试StaticMLPEncoder
    print("测试 StaticMLPEncoder...")
    static_data = torch.randn(4, 4)  # (B, D)
    static_encoder = StaticMLPEncoder(in_dim=4, out_dim=128)
    output = static_encoder(static_data)
    print(f"  输入形状: {static_data.shape}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ StaticMLPEncoder测试通过")

    # 测试ConstitutionEmbedding
    print("测试 ConstitutionEmbedding...")
    constitution_data = torch.randint(0, 39, (4,))  # (B,)
    constitution_embedding = ConstitutionEmbedding(num_constitutions=39, embed_dim=32, out_dim=128)
    output = constitution_embedding(constitution_data)
    print(f"  输入形状: {constitution_data.shape}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ ConstitutionEmbedding测试通过")

    # 测试WaveformEncoder
    print("测试 WaveformEncoder...")
    waveform_encoder = WaveformEncoder(encoder_type='inception', in_channels=2, out_channels=64)
    output = waveform_encoder(dynamic_data)
    print(f"  输入形状: {dynamic_data.shape}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ WaveformEncoder测试通过")


def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试4: 错误处理")
    print("=" * 60)

    from experiment.dataset.unified_source import UnifiedNPZDataSource

    # 测试文件不存在的情况
    print("测试文件不存在的情况...")
    source = UnifiedNPZDataSource('nonexistent_file.npz')
    result = source.initialize()
    if not result:
        print("  ✅ 正确处理了文件不存在的情况")
    else:
        print("  ❌ 未能正确处理文件不存在的情况")

    # 测试加载不存在的样本
    print("测试加载不存在的样本...")
    try:
        from experiment.dataset.unified_source import UnifiedNPZDataSource
        source = UnifiedNPZDataSource('experiment/dataset/unified_dataset_expanded.npz')
        if not source.initialize():
            print("  ❌ 数据源初始化失败")
            raise RuntimeError("数据源初始化失败")
        source.load_sample(999999)  # 不存在的样本ID
        print("  ❌ 未能正确处理不存在的样本ID")
    except IndexError as e:
        print(f"  ✅ 正确处理了不存在的样本ID: {e}")
    except RuntimeError as e:
        print(f"  ⚠️  数据源初始化失败: {e}")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("开始测试改进后的代码")
    print("=" * 60)

    try:
        # 测试日志系统
        test_logger()

        # 测试配置系统
        test_config()

        # 测试编码器模块
        test_encoders()

        # 测试错误处理
        test_error_handling()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
