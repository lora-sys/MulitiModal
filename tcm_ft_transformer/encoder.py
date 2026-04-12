"""
FT-Transformer 编码器
用于提取 64 维健康表征向量，作为多模态大模型的输入
"""

import torch
import torch.nn as nn
from ft_transformer import FTTransformer
from config import MODEL_CONFIG


class TCMConstitutionEncoder(nn.Module):
    """
    中医体质编码器
    从 FT-Transformer 中提取 64 维特征向量
    """
    def __init__(self, pretrained_path='checkpoints/best_model.pth'):
        super().__init__()
        
        # 加载预训练模型
        self.backbone = FTTransformer(
            n_features=8,
            d_token=MODEL_CONFIG['d_token'],
            n_heads=MODEL_CONFIG['n_heads'],
            n_layers=3,
            dropout=0.0,  # 推理时关闭 dropout
            n_classes=9
        )
        
        # 加载权重
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 已加载预训练权重: {pretrained_path}")
        
        # 移除分类头，只保留编码部分
        self.feature_tokenizer = self.backbone.feature_tokenizer
        self.cls_token = self.backbone.cls_token
        self.transformer_encoder = self.backbone.transformer_encoder
        self.layer_norm = self.backbone.layer_norm
        
        # 冻结参数（可选）
        # self.freeze_parameters()
        
    def freeze_parameters(self):
        """冻结所有参数，只用于特征提取"""
        for param in self.parameters():
            param.requires_grad = False
        print("⚠️  已冻结所有参数")
    
    def unfreeze_parameters(self):
        """解冻参数，允许微调"""
        for param in self.parameters():
            param.requires_grad = True
        print("✅ 已解冻所有参数")
    
    def forward(self, x):
        """
        前向传播，提取 64 维特征向量
        
        Args:
            x: (B, 8) 输入特征（Age, Gender, BMI, HeartRate, SBP, DBP, SpO2, Temperature）
            
        Returns:
            features: (B, 64) 健康表征向量
        """
        batch_size = x.size(0)
        
        # 1. 特征 Tokenization
        tokens = self.feature_tokenizer(x)  # (B, 8, 64)
        
        # 2. 添加 [CLS] Token
        cls_token = self.cls_token(batch_size)  # (B, 1, 64)
        tokens = torch.cat([cls_token, tokens], dim=1)  # (B, 9, 64)
        
        # 3. Transformer Encoder
        encoded = self.transformer_encoder(tokens)  # (B, 9, 64)
        
        # 4. 提取 [CLS] Token
        cls_output = encoded[:, 0, :]  # (B, 64)
        
        # 5. Layer Normalization
        features = self.layer_norm(cls_output)  # (B, 64)
        
        return features
    
    def get_probabilities(self, x):
        """
        获取体质概率分布（可选功能）
        
        Args:
            x: (B, 8) 输入特征
            
        Returns:
            probs: (B, 9) 体质概率分布
        """
        features = self.forward(x)
        probs = self.backbone.classifier(features)
        return probs


class MultiModalHealthEncoder(nn.Module):
    """
    多模态健康表征编码器
    整合中医体质编码器和其他模态特征
    """
    def __init__(
        self,
        tcm_encoder_path='checkpoints/best_model.pth',
        other_modalities_dim=0,
        fusion_dim=128,
        output_dim=256
    ):
        super().__init__()
        
        # 1. 中医体质编码器 (64 维)
        self.tcm_encoder = TCMConstitutionEncoder(tcm_encoder_path)
        tcm_dim = 64
        
        # 2. 其他模态特征（如果有）
        self.other_modalities_dim = other_modalities_dim
        if other_modalities_dim > 0:
            self.other_projection = nn.Linear(other_modalities_dim, tcm_dim)
        
        # 3. 融合层
        input_dim = tcm_dim + (tcm_dim if other_modalities_dim > 0 else 0)
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, output_dim)
        )
        
    def forward(self, tcm_features, other_features=None):
        """
        前向传播
        
        Args:
            tcm_features: (B, 8) 中医体质输入特征
            other_features: (B, other_dim) 其他模态特征（可选）
            
        Returns:
            health_representation: (B, output_dim) 多模态健康表征
        """
        # 1. 提取中医体质特征
        tcm_emb = self.tcm_encoder(tcm_features)  # (B, 64)
        
        # 2. 处理其他模态特征
        if other_features is not None and self.other_modalities_dim > 0:
            other_emb = self.other_projection(other_features)  # (B, 64)
            combined = torch.cat([tcm_emb, other_emb], dim=1)  # (B, 128)
        else:
            combined = tcm_emb  # (B, 64)
        
        # 3. 融合
        health_representation = self.fusion(combined)  # (B, output_dim)
        
        return health_representation


if __name__ == "__main__":
    # 测试编码器
    print("=" * 60)
    print("测试中医体质编码器")
    print("=" * 60)
    
    # 创建编码器
    encoder = TCMConstitutionEncoder()
    encoder.eval()
    
    # 测试输入
    batch_size = 4
    x = torch.randn(batch_size, 8)  # (B, 8)
    
    # 前向传播
    with torch.no_grad():
        features = encoder(x)
        probs = encoder.get_probabilities(x)
    
    print(f"输入形状: {x.shape}")
    print(f"特征向量形状: {features.shape}")
    print(f"概率分布形状: {probs.shape}")
    print(f"特征向量示例: {features[0][:5].tolist()}...")
    print(f"概率分布示例: {probs[0].tolist()}")
    print(f"概率和: {probs.sum(dim=1).tolist()}")
    
    print("\n" + "=" * 60)
    print("测试多模态健康表征编码器")
    print("=" * 60)
    
    # 创建多模态编码器
    multimodal_encoder = MultiModalHealthEncoder(
        tcm_encoder_path=None,  # 不加载权重
        other_modalities_dim=32,
        fusion_dim=128,
        output_dim=256
    )
    multimodal_encoder.eval()
    
    # 测试输入
    tcm_features = torch.randn(batch_size, 8)
    other_features = torch.randn(batch_size, 32)
    
    # 前向传播
    with torch.no_grad():
        health_repr = multimodal_encoder(tcm_features, other_features)
    
    print(f"中医特征形状: {tcm_features.shape}")
    print(f"其他模态特征形状: {other_features.shape}")
    print(f"健康表征形状: {health_repr.shape}")
    print(f"健康表征示例: {health_repr[0][:5].tolist()}...")