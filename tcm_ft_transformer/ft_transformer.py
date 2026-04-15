"""
FT-Transformer 模型架构
基于 Feature Tokenizer Transformer 的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NumericalFeatureTokenizer(nn.Module):
    """
    数值特征 Tokenizer
    将每个数值特征映射为 d_token 维向量
    """
    def __init__(self, n_features, d_token):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        
        # 为每个特征创建独立的线性层
        self.weight = nn.Parameter(torch.Tensor(n_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(n_features, d_token))
        
        # 初始化参数
        with torch.no_grad():
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, n_features) 输入特征
        Returns:
            tokens: (B, n_features, d_token) 特征 tokens
        """
        # x: (B, n_features) -> (B, n_features, 1)
        x = x.unsqueeze(-1)
        
        # 广播计算: (B, n_features, d_token)
        tokens = x * self.weight + self.bias
        return tokens


class CLSToken(nn.Module):
    """
    [CLS] Token
    用于聚合全局信息的特殊 token
    """
    def __init__(self, d_token):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, batch_size):
        """
        Args:
            batch_size: 批次大小
        Returns:
            cls_token: (B, 1, d_token)
        """
        return self.cls_token.expand(batch_size, -1, -1)


class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer 模型
    
    架构:
    1. NumericalFeatureTokenizer: 将 8 个数值特征映射为 64 维向量
    2. CLS Token: 添加 [CLS] token 用于聚合全局信息
    3. Transformer Encoder: 3 层，4 个注意力头
    4. 输出层: Linear + Softmax
    """
    def __init__(
        self,
        n_features=8,
        d_token=64,
        n_heads=4,
        n_layers=3,
        dropout=0.3,
        n_classes=9
    ):
        super().__init__()
        
        self.n_features = n_features
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_classes = n_classes
        
        # 1. 数值特征 Tokenizer
        self.feature_tokenizer = NumericalFeatureTokenizer(n_features, d_token)
        
        # 2. [CLS] Token
        self.cls_token = CLSToken(d_token)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # 4. Layer Normalization
        self.layer_norm = nn.LayerNorm(d_token)
        
        # 5. 输出层
        self.classifier = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, n_classes),
            nn.Softmax(dim=-1)  # 强制输出和为 1
        )
        
        # 6. Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, n_features) 输入特征
        Returns:
            output: (B, n_classes) 概率分布
        """
        batch_size = x.size(0)
        
        # 1. 特征 Tokenization
        tokens = self.feature_tokenizer(x)  # (B, n_features, d_token)
        
        # 2. 添加 [CLS] Token
        cls_token = self.cls_token(batch_size)  # (B, 1, d_token)
        tokens = torch.cat([cls_token, tokens], dim=1)  # (B, n_features+1, d_token)
        
        # 3. Dropout
        tokens = self.dropout_layer(tokens)
        
        # 4. Transformer Encoder
        encoded = self.transformer_encoder(tokens)  # (B, n_features+1, d_token)
        
        # 5. 提取 [CLS] Token
        cls_output = encoded[:, 0, :]  # (B, d_token)
        
        # 6. Layer Normalization
        cls_output = self.layer_norm(cls_output)
        
        # 7. 分类
        output = self.classifier(cls_output)  # (B, n_classes)
        
        return output


def get_model(n_features=8, n_classes=9, **kwargs):
    """
    工厂函数：创建 FT-Transformer 模型
    
    Args:
        n_features: 输入特征维度
        n_classes: 输出类别数
        **kwargs: 其他超参数
        
    Returns:
        model: FT-Transformer 模型
    """
    model = FTTransformer(
        n_features=n_features,
        n_classes=n_classes,
        **kwargs
    )
    return model


if __name__ == "__main__":
    # 测试模型
    model = get_model(
        n_features=8,
        n_classes=9,
        d_token=64,
        n_heads=4,
        n_layers=3,
        dropout=0.3
    )
    
    # 测试前向传播
    x = torch.randn(32, 4)  # (B, n_features)
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出和: {output.sum(dim=1)[:5]}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")