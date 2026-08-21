#!/usr/bin/env python3
"""
消融实验模板: 评估 DualGatingModel 各组件的必要性

实验变体:
  - M0: 基线 (仅动态编码器)
  - M1: 仅 TCM
  - M2: 仅动态 (无 TCM, 无 Gate)
  - M3: 无 Gate (TCM + 动态, 直接拼接)
  - M4: 仅 Gate A
  - M5: 仅 Gate B
  - M6: 完整模型 (Gate A + B)

使用方法:
  python experiments/ablation_study.py \
    --data_path /path/to/data \
    --checkpoint /path/to/checkpoint \
    --output results/ablation/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "legacy_research/source/oplri/src"))

from models.fusion import DualGatingModel
from models.encoders import get_dynamic_encoder


class M0Baseline(nn.Module):
    """M0: 仅动态编码器 (基线)"""

    def __init__(self, encoder_name: str = "resnet"):
        super().__init__()
        self.dynamic_encoder = get_dynamic_encoder(encoder_name, in_channels=2)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, dynamic: torch.Tensor, static_4d: torch.Tensor) -> torch.Tensor:
        z = self.dynamic_encoder(dynamic)
        return self.head(z)


class M1TCMOnly(nn.Module):
    """M1: 仅 TCM (静态诊断)"""

    def __init__(self, tcm_checkpoint_path: str, tcm_scaler_path: str):
        super().__init__()
        sys.path.insert(0, str(Path(__file__).parent.parent / "legacy_research/source/tcm/tcm_ft_transformer"))
        from ft_transformer import get_model

        self.tcm_model = get_model(n_features=4, n_classes=9)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, dynamic: torch.Tensor, static_4d: torch.Tensor) -> torch.Tensor:
        tcm_internal, _ = self.tcm_encoder.extract_features_and_probs(static_4d)
        return self.head(tcm_internal)


class M2DynamicOnly(nn.Module):
    """M2: 仅动态 (无 TCM, 无 Gate)"""

    def __init__(self, encoder_name: str = "resnet"):
        super().__init__()
        self.dynamic_encoder = get_dynamic_encoder(encoder_name, in_channels=2)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, dynamic: torch.Tensor, static_4d: torch.Tensor) -> torch.Tensor:
        z = self.dynamic_encoder(dynamic)
        return self.head(z)


class M3NoGate(nn.Module):
    """M3: TCM + 动态, 无 Gate (简单拼接)"""

    def __init__(self, tcm_checkpoint_path: str, tcm_scaler_path: str, encoder_name: str = "resnet"):
        super().__init__()
        self.dynamic_encoder = get_dynamic_encoder(encoder_name, in_channels=2)
        self.tcm_encoder = ...  # 加载 TCM
        self.fusion = nn.Linear(256, 128)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, dynamic: torch.Tensor, static_4d: torch.Tensor) -> torch.Tensor:
        dyn_z = self.dynamic_encoder(dynamic)  # [B, 128]
        tcm_z, _ = self.tcm_encoder.extract_features_and_probs(static_4d)  # [B, 128]
        fused = torch.cat([dyn_z, tcm_z], dim=1)  # [B, 256]
        fused = self.fusion(fused)  # [B, 128]
        return self.head(fused)


def create_model(mode: str, checkpoint_path: str = None, **kwargs) -> nn.Module:
    """创建消融实验模型"""
    if mode == "M0":
        return M0Baseline(**kwargs)
    elif mode == "M1":
        return M1TCMOnly(**kwargs)
    elif mode == "M2":
        return M2DynamicOnly(**kwargs)
    elif mode == "M3":
        return M3NoGate(**kwargs)
    elif mode in ["M4", "M5", "M6"]:
        use_gate_a = mode in ["M4", "M6"]
        use_gate_b = mode in ["M5", "M6"]
        model = DualGatingModel(
            use_tcm=True,
            use_gate_a=use_gate_a,
            use_gate_b=use_gate_b,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 加载 checkpoint (如果提供)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state_dict"]

        # 提取对应组件的 state_dict
        if mode == "M6":
            # 完整模型: 直接加载
            model.load_state_dict(state_dict, strict=False)
        else:
            # 部分模型: 只加载匹配的键
            model_dict = model.state_dict()
            matched_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict, strict=False)

    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """评估模型性能"""
    model.eval()
    model.to(device)

    criterion = nn.L1Loss()  # MAE
    mse_criterion = nn.MSELoss()

    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0
    all_targets = []

    with torch.no_grad():
        for dynamic, static, target in dataloader:
            dynamic = dynamic.to(device)
            static = static.to(device)
            target = target.to(device)
            all_targets.append(target.cpu())

            output = model(dynamic, static)
            total_mae += criterion(output, target).item() * dynamic.size(0)
            total_mse += mse_criterion(output, target).item() * dynamic.size(0)
            total_samples += dynamic.size(0)

    mae = total_mae / total_samples
    rmse = torch.sqrt(torch.tensor(total_mse / total_samples)).item()

    # R² 计算
    all_targets = torch.cat(all_targets, dim=0)
    ss_res = total_mse * total_samples
    ss_tot = torch.sum((all_targets - all_targets.mean()) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "num_samples": total_samples,
    }


def run_ablation_study(
    data_path: str,
    checkpoint_path: str,
    output_dir: str,
    modes: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    运行消融实验

    Args:
        data_path: 数据路径
        checkpoint_path: 预训练 checkpoint 路径
        output_dir: 结果输出目录
        modes: 要测试的模型变体列表

    Returns:
        results: {mode: {metric: value}}
    """
    if modes is None:
        modes = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据 (示例,实际需要根据数据格式调整)
    print("⚠️  使用随机数据测试流程,请替换为真实数据!")
    dummy_data = TensorDataset(
        torch.randn(100, 2, 1000),  # dynamic
        torch.randn(100, 4),         # static
        torch.randn(100, 1),         # target
    )
    dataloader = DataLoader(dummy_data, batch_size=32)

    results = {}

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"测试模型: {mode}")
        print(f"{'=' * 60}")

        try:
            # 创建模型
            model = create_model(
                mode,
                checkpoint_path=checkpoint_path,
                encoder_name="resnet",
                tcm_checkpoint_path="legacy_research/checkpoints/tcm/server_f743da3/best_model.pth",
                tcm_scaler_path="legacy_research/checkpoints/tcm/server_f743da3/scaler_params_8d.npz",
            )

            # 评估
            metrics = evaluate_model(model, dataloader, device)
            results[mode] = metrics

            print(f"  MAE:  {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R²:   {metrics['r2']:.4f}")

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results[mode] = {"error": str(e)}

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 结果已保存: {output_path / 'ablation_results.json'}")

    # 打印对比表
    print(f"\n{'=' * 60}")
    print("消融实验对比")
    print(f"{'=' * 60}")
    print(f"{'模型':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-" * 60)
    for mode in modes:
        if mode in results and "mae" in results[mode]:
            m = results[mode]
            print(f"{mode:<10} {m['mae']:<10.4f} {m['rmse']:<10.4f} {m['r2']:<10.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DualGatingModel 消融实验")
    parser.add_argument("--data_path", type=str, required=True, help="数据路径")
    parser.add_argument("--checkpoint", type=str, help="预训练 checkpoint 路径")
    parser.add_argument("--output", type=str, default="results/ablation/", help="输出目录")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["M0", "M1", "M2", "M3", "M6"],
        help="要测试的模型变体",
    )

    args = parser.parse_args()
    run_ablation_study(args.data_path, args.checkpoint, args.output, args.modes)
