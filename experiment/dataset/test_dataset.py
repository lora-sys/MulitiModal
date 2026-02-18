"""
Dataset测试脚本
"""

import yaml
import torch
from torch.utils.data import DataLoader

# 导入模块
from csv_source import CSVDataSource
from nk2_processor import NK2Preprocessor
from massage_dataset import MassageDataset


def load_config(config_path: str) -> dict:
    """加载配置"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dataset(config: dict):
    """根据配置创建数据集"""
    # 1. 创建数据源
    source_config = config["source"]
    source = CSVDataSource(
        base_path=source_config["params"]["base_path"],
        label_map=source_config["params"]["label_map"],
    )

    # 2. 创建预处理器
    processor_config = config["preprocessor"]
    processor = NK2Preprocessor(processor_config["params"])

    # 3. 创建数据集
    dataset = MassageDataset(source=source, preprocessor=processor)

    return dataset


def test_dataset():
    """测试数据集"""
    # 加载配置
    config = load_config("experiment/dataset/config.yaml")

    print("=" * 60)
    print("Dataset测试")
    print("=" * 60)

    # 创建数据集
    dataset = create_dataset(config)
    print(f"✅ Dataset创建成功: {len(dataset)} 样本")

    # 创建DataLoader
    loader_config = config["dataset"]
    loader = DataLoader(
        dataset,
        batch_size=loader_config["batch_size"],
        shuffle=loader_config["shuffle"],
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
    )

    # 测试加载一个batch
    print(f"\n🚀 测试DataLoader (batch_size={loader_config['batch_size']})...")

    for batch in loader:
        print(f"\n✅ Batch获取成功!")
        print(f"  动态张量 shape: {batch['dynamic'].shape}")  # [32, 2, 1000]
        print(f"  静态张量 shape: {batch['static'].shape}")  # [32, 4]
        print(f"  标签 shape: {batch['label'].shape}")  # [32]
        break

    print("\n✅ Dataset测试通过!")


if __name__ == "__main__":
    test_dataset()
