"""
双门控融合模型

集成 TCM Encoder 和压力编码器，通过双门控机制进行特征融合
用于 WESAD 数据集的放松度预测（回归任务）

架构：
1. TCM Encoder: 8维静态体征 -> 128维特征 + 9维体质概率
2. InceptionEncoder: 2通道动态信号 -> 512维特征 -> 128维投影
3. 门控 A: TCM 概率 -> 压力特征权重
4. 门控 B: 压力特征 -> 静态特征权重
5. 融合 + 回归: 256维 -> 2维输出（放松度、疲劳缓解度）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加路径以导入现有编码器
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset'))

from model import InceptionEncoder
from encoders import create_tcm_encoder


class DualGatingModel(nn.Module):
    """
    双门控融合模型
    
    用于 WESAD 数据集的放松度预测
    
    特点：
    - 集成训练好的 TCM Encoder（参数冻结）
    - 使用 InceptionEncoder 处理 ECG 和 ACC 信号
    - 双向门控机制：中医指导压力，压力修正静态
    - 回归任务：预测放松度和疲劳缓解度（0-1）
    """
    
    def __init__(
        self,
        tcm_model_path: str = 'data/tcm_ft_transformer/checkpoints/best_model.pth',
        tcm_scaler_path: str = 'data/tcm_ft_transformer/data/scaler_params.npz',
        device: str = 'cuda',
        shared_dim: int = 128,
        gate_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            tcm_model_path: TCM 模型权重路径
            tcm_scaler_path: TCM 标准化参数路径
            device: 设备
            shared_dim: 共享特征维度
            gate_dim: 门控网络维度
            hidden_dim: 隐藏层维度
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.device = device
        self.shared_dim = shared_dim
        self.num_constitutions = 9  # 9 种中医体质
        
        print("=" * 60)
        print("初始化 DualGatingModel")
        print("=" * 60)
        
        # 1. 加载 TCM Encoder（参数已冻结）
        print(f"\n[1/5] 加载 TCM Encoder...")
        self.tcm_encoder = create_tcm_encoder(
            model_path=tcm_model_path,
            scaler_path=tcm_scaler_path,
            device='cpu'  # 先在 CPU 初始化，后续移动到正确设备
        )
        
        # 验证 TCM Encoder 参数冻结状态
        tcm_frozen = all(not p.requires_grad for p in self.tcm_encoder.model.parameters())
        print(f"  ✅ TCM Encoder 参数冻结: {'是' if tcm_frozen else '否'}")
        
        # 2. 创建压力编码器（Inception）
        print(f"\n[2/5] 创建 InceptionEncoder...")
        self.pressure_encoder = InceptionEncoder(
            in_channels=2,  # ECG + ACC
            out_channels=512,  # 输出 512 维
            depth=3
        )
        print(f"  ✅ InceptionEncoder: 2 通道 -> 512 维")
        
        # 3. 压力特征投影层
        print(f"\n[3/5] 创建压力特征投影层...")
        self.pressure_projection = nn.Sequential(
            nn.Linear(512, shared_dim),
            nn.ReLU(),
            nn.LayerNorm(shared_dim),
            nn.Dropout(dropout)
        )
        print(f"  ✅ 投影层: 512 维 -> {shared_dim} 维")
        
        # 4. 门控网络
        print(f"\n[4/5] 创建双门控网络...")
        
        # 门控 A: 中医概率 -> 压力特征权重
        self.gate_a = nn.Sequential(
            nn.Linear(self.num_constitutions, gate_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gate_dim, shared_dim),
            nn.Sigmoid()
        )
        print(f"  ✅ 门控 A: {self.num_constitutions} 维 -> {shared_dim} 维")
        
        # 门控 B: 压力特征 -> 静态特征权重
        self.gate_b = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim, shared_dim),
            nn.Sigmoid()
        )
        print(f"  ✅ 门控 B: {shared_dim} 维 -> {shared_dim} 维")
        
        # 5. 回归头
        print(f"\n[5/5] 创建回归头...")
        self.regressor = nn.Sequential(
            nn.Linear(shared_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # 放松度 + 疲劳缓解度
            nn.Sigmoid()  # 输出 0-1
        )
        print(f"  ✅ 回归头: {shared_dim * 2} 维 -> 256 -> 128 -> 2 维")
        
        print(f"\n" + "=" * 60)
        print("✅ DualGatingModel 初始化完成")
        print("=" * 60)
        
        # 存储结构信息
        self.model_name = "DualGating"
        self.fusion_type = "dual_gating"
    
    def forward(self, dynamic: torch.Tensor, static_basic: torch.Tensor, return_intermediate: bool = False):
        """
        前向传播
        
        Args:
            dynamic: (B, 2, 1000) 动态信号（ECG + ACC）
            static_basic: (B, 8) 静态体征
            return_intermediate: 是否返回中间结果
            
        Returns:
            outputs: (B, 2) [放松度, 疲劳缓解度]
            intermediate: (可选) 中间结果字典
        """
        batch_size = dynamic.size(0)
        
        # 1. TCM 编码
        tcm_features, tcm_probs = self.tcm_encoder(static_basic)
        # tcm_features: (B, 128)
        # tcm_probs: (B, 9)
        
        # 2. 压力编码
        pressure_features_raw = self.pressure_encoder(dynamic)
        # pressure_features_raw: (B, 512)
        
        # 压力特征投影
        pressure_features = self.pressure_projection(pressure_features_raw)
        # pressure_features: (B, 128)
        
        # 3. 门控 A: 中医指导压力
        gate_a_weights = self.gate_a(tcm_probs)  # (B, 128)
        gated_pressure = pressure_features * gate_a_weights  # (B, 128)
        
        # 4. 门控 B: 压力修正静态
        gate_b_weights = self.gate_b(pressure_features)  # (B, 128)
        gated_static = tcm_features * gate_b_weights  # (B, 128)
        
        # 5. 特征融合
        fused = torch.cat([gated_static, gated_pressure], dim=1)  # (B, 256)
        
        # 6. 回归预测
        outputs = self.regressor(fused)  # (B, 2)
        
        if return_intermediate:
            return outputs, {
                'tcm_features': tcm_features,           # (B, 128)
                'tcm_probs': tcm_probs,                  # (B, 9)
                'pressure_features_raw': pressure_features_raw,  # (B, 512)
                'pressure_features': pressure_features,  # (B, 128)
                'gate_a_weights': gate_a_weights,        # (B, 128)
                'gate_b_weights': gate_b_weights,        # (B, 128)
                'gated_pressure': gated_pressure,        # (B, 128)
                'gated_static': gated_static,            # (B, 128)
                'fused_features': fused,                 # (B, 256)
            }
        
        return outputs
    
    def predict(self, dynamic: torch.Tensor, static_basic: torch.Tensor):
        """
        预测接口
        
        Args:
            dynamic: (B, 2, 1000) 动态信号
            static_basic: (B, 8) 静态体征
            
        Returns:
            outputs: (B, 2) [放松度, 疲劳缓解度]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(dynamic, static_basic)
    
    def get_trainable_params(self):
        """获取可训练参数（排除 TCM Encoder）"""
        trainable_params = []
        for name, param in self.named_parameters():
            if 'tcm_encoder.model' not in name:  # 排除 TCM Encoder 的模型参数
                trainable_params.append(param)
        return trainable_params
    
    def get_param_groups(self, base_lr: float = 1e-3, encoder_lr_ratio: float = 0.1):
        """
        获取参数组（用于不同学习率）
        
        Args:
            base_lr: 基础学习率
            encoder_lr_ratio: 编码器学习率比例
            
        Returns:
            param_groups: 参数组列表
        """
        # TCM Encoder 参数（冻结，学习率为 0）
        tcm_params = []
        
        # 压力编码器参数（低学习率）
        pressure_params = []
        
        # 融合和回归头参数（正常学习率）
        fusion_params = []
        
        for name, param in self.named_parameters():
            if 'tcm_encoder' in name:
                tcm_params.append(param)
            elif 'pressure_encoder' in name or 'pressure_projection' in name:
                pressure_params.append(param)
            else:
                fusion_params.append(param)
        
        param_groups = [
            {
                'params': tcm_params,
                'lr': 0.0,  # TCM 参数冻结
            },
            {
                'params': pressure_params,
                'lr': base_lr * encoder_lr_ratio,
            },
            {
                'params': fusion_params,
                'lr': base_lr,
            },
        ]
        
        print(f"\n[参数分组]")
        print(f"  TCM Encoder: {len(tcm_params)} 参数 (lr=0.0)")
        print(f"  压力编码器: {len(pressure_params)} 参数 (lr={base_lr * encoder_lr_ratio:.6f})")
        print(f"  融合/回归: {len(fusion_params)} 参数 (lr={base_lr:.6f})")
        
        return param_groups


def create_dual_gating_model(
    tcm_model_path: str = 'data/tcm_ft_transformer/checkpoints/best_model.pth',
    tcm_scaler_path: str = 'data/tcm_ft_transformer/data/scaler_params.npz',
    device: str = 'cuda',
    **kwargs
) -> DualGatingModel:
    """
    工厂函数：创建双门控融合模型
    
    Args:
        tcm_model_path: TCM 模型权重路径
        tcm_scaler_path: TCM 标准化参数路径
        device: 设备
        **kwargs: 其他模型参数
        
    Returns:
        model: DualGatingModel 实例
    """
    model = DualGatingModel(
        tcm_model_path=tcm_model_path,
        tcm_scaler_path=tcm_scaler_path,
        device=device,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 DualGatingModel")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 创建模型
    model = create_dual_gating_model(device=device)
    model = model.to(device)
    
    # 测试数据
    batch_size = 4
    dynamic = torch.randn(batch_size, 2, 1000).to(device)
    static_basic = torch.randn(batch_size, 8).to(device)
    
    print(f"\n输入数据:")
    print(f"  动态特征: {dynamic.shape}")
    print(f"  静态特征: {static_basic.shape}")
    
    # 前向传播
    outputs, intermediate = model(dynamic, static_basic, return_intermediate=True)
    
    print(f"\n输出数据:")
    print(f"  预测输出: {outputs.shape}")
    print(f"  放松度: {outputs[:, 0].detach().cpu().numpy()}")
    print(f"  疲劳缓解度: {outputs[:, 1].detach().cpu().numpy()}")
    
    print(f"\n中间结果:")
    for key, value in intermediate.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # 测试参数分组
    print(f"\n参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {frozen_params:,}")
    
    print(f"\n✅ 测试完成！")