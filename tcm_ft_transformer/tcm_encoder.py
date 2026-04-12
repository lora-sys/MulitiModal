"""
TCM Encoder - 中医体质特征编码器
将训练好的 FT-Transformer 分类器改造为特征编码器
用于多模态融合系统
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

from ft_transformer import get_model


class TCM_Encoder(nn.Module):
    """
    中医体质编码器
    
    功能:
    1. 加载训练好的 FT-Transformer 模型权重
    2. 加载训练时的标准化参数（均值和标准差）
    3. 对输入进行标准化处理
    4. 提取深层特征（[CLS] Token 向量）
    5. 输出特征向量和概率分布
    
    输出:
    - features: (B, 128) 深层特征向量（用于多模态融合）
    - probs: (B, 9) 中医体质概率分布（用于门控机制）
    """
    
    def __init__(
        self,
        model_path='checkpoints/best_model.pth',
        scaler_path='data/scaler_params.npz',
        device='cuda'
    ):
        super().__init__()
        
        self.device = device
        
        # 加载标准化参数
        self.scaler_params = self._load_scaler(scaler_path)
        self.mean = torch.tensor(self.scaler_params['mean'], dtype=torch.float32).to(device)
        self.std = torch.tensor(self.scaler_params['std'], dtype=torch.float32).to(device)
        
        # 加载训练好的模型
        self.model = get_model(n_features=8, n_classes=9)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # 冻结模型参数（推理时不需要梯度）
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 特征投影层（将 64 维 CLS token 投影到 128 维）
        self.feature_projection = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(device)
        
        # 加载最佳参数（如果有）
        if 'best_params' in checkpoint:
            self.best_params = checkpoint['best_params']
        else:
            self.best_params = None
    
    def _load_scaler(self, scaler_path):
        """加载标准化参数"""
        scaler_data = np.load(scaler_path)
        return {
            'mean': scaler_data['mean'],
            'std': scaler_data['std']
        }
    
    def normalize(self, x):
        """
        标准化输入特征
        
        Args:
            x: (B, n_features) 原始特征
            
        Returns:
            x_normalized: (B, n_features) 标准化后的特征
        """
        # 避免除以零
        std_safe = torch.where(self.std < 1e-8, torch.ones_like(self.std), self.std)
        return (x - self.mean) / std_safe
    
    def extract_cls_token(self, x):
        """
        提取 [CLS] Token 向量
        
        Args:
            x: (B, n_features) 标准化后的特征
            
        Returns:
            cls_token: (B, 64) [CLS] Token 向量
        """
        batch_size = x.size(0)
        
        # 1. 特征 Tokenization
        tokens = self.model.feature_tokenizer(x)  # (B, n_features, 64)
        
        # 2. 添加 [CLS] Token
        cls_token = self.model.cls_token(batch_size)  # (B, 1, 64)
        tokens = torch.cat([cls_token, tokens], dim=1)  # (B, n_features+1, 64)
        
        # 3. Dropout
        tokens = self.model.dropout_layer(tokens)
        
        # 4. Transformer Encoder
        encoded = self.model.transformer_encoder(tokens)  # (B, n_features+1, 64)
        
        # 5. 提取 [CLS] Token
        cls_token = encoded[:, 0, :]  # (B, 64)
        
        # 6. Layer Normalization
        cls_token = self.model.layer_norm(cls_token)
        
        return cls_token
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, n_features) 原始特征（未标准化）
            
        Returns:
            features: (B, 128) 深层特征向量
            probs: (B, 9) 中医体质概率分布
        """
        with torch.no_grad():
            # 1. 标准化
            x_normalized = self.normalize(x)
            
            # 2. 提取 [CLS] Token
            cls_token = self.extract_cls_token(x_normalized)  # (B, 64)
            
            # 3. 投影到 128 维
            features = self.feature_projection(cls_token)  # (B, 128)
            
            # 4. 获取概率分布
            probs = self.model(x_normalized)  # (B, 9)
        
        return features, probs
    
    def encode(self, x):
        """
        编码接口（与 forward 相同）
        
        Args:
            x: (B, n_features) 原始特征
            
        Returns:
            features: (B, 128) 深层特征向量
            probs: (B, 9) 中医体质概率分布
        """
        return self.forward(x)
    
    def get_constitution_names(self):
        """获取体质名称列表"""
        return [
            "平和质",
            "气虚质",
            "阳虚质",
            "阴虚质",
            "痰湿质",
            "湿热质",
            "血瘀质",
            "气郁质",
            "特禀质"
        ]
    
    def predict_constitution(self, x):
        """
        预测体质类型
        
        Args:
            x: (B, n_features) 原始特征
            
        Returns:
            constitution: (B,) 预测的体质类型索引
            constitution_names: (B,) 预测的体质类型名称
            probs: (B, 9) 概率分布
        """
        _, probs = self.forward(x)
        constitution = torch.argmax(probs, dim=1)
        names = self.get_constitution_names()
        constitution_names = [names[idx] for idx in constitution.cpu().numpy()]
        
        return constitution, constitution_names, probs
    
    def save_encoder(self, save_path='checkpoints/tcm_encoder.pth'):
        """
        保存编码器
        
        Args:
            save_path: 保存路径
        """
        encoder_state = {
            'model_state_dict': self.model.state_dict(),
            'feature_projection_state_dict': self.feature_projection.state_dict(),
            'scaler_params': self.scaler_params,
            'best_params': self.best_params
        }
        torch.save(encoder_state, save_path)
        print(f"编码器已保存: {save_path}")
    
    def load_encoder(self, load_path='checkpoints/tcm_encoder.pth'):
        """
        加载编码器
        
        Args:
            load_path: 加载路径
        """
        encoder_state = torch.load(load_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(encoder_state['model_state_dict'])
        self.feature_projection.load_state_dict(encoder_state['feature_projection_state_dict'])
        self.scaler_params = encoder_state['scaler_params']
        self.best_params = encoder_state.get('best_params', None)
        print(f"编码器已加载: {load_path}")


def create_tcm_encoder(
    model_path='checkpoints/best_model.pth',
    scaler_path='data/scaler_params.npz',
    device='cuda'
):
    """
    工厂函数：创建 TCM Encoder
    
    Args:
        model_path: 模型权重路径
        scaler_path: 标准化参数路径
        device: 设备
        
    Returns:
        encoder: TCM_Encoder 实例
    """
    encoder = TCM_Encoder(
        model_path=model_path,
        scaler_path=scaler_path,
        device=device
    )
    return encoder


if __name__ == "__main__":
    # 测试编码器
    print("=" * 60)
    print("TCM Encoder 测试")
    print("=" * 60)
    
    # 创建编码器
    encoder = create_tcm_encoder(
        model_path='checkpoints/best_model.pth',
        scaler_path='data/scaler_params.npz',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 测试数据
    batch_size = 4
    x = torch.randn(batch_size, 8).to(encoder.device)
    
    print(f"\n输入形状: {x.shape}")
    print(f"输入样本: {x[0].cpu().numpy()}")
    
    # 编码
    features, probs = encoder(x)
    
    print(f"\n特征向量形状: {features.shape}")
    print(f"概率分布形状: {probs.shape}")
    print(f"概率分布和: {probs.sum(dim=1)}")
    
    # 预测体质
    constitution, names, probs = encoder.predict_constitution(x)
    
    print(f"\n预测体质: {names}")
    print(f"体质索引: {constitution.cpu().numpy()}")
    print(f"\n概率分布:")
    for i, (name, prob) in enumerate(zip(names, probs)):
        print(f"  {name}: {prob[constitution[i]].item():.4f}")
    
    # 保存编码器
    encoder.save_encoder('checkpoints/tcm_encoder.pth')
    
    print("\n✅ 测试完成！")