import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================================
# 核心组件 1: InceptionTime 模块 (SOTA 标杆)
# 作用: 通过多尺度卷积核并行提取特征，既看细节(9)，又看趋势(39)。
# =========================================================================
class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=[9, 19, 39],
        bottleneck_channels=32,
    ):
        super(InceptionModule, self).__init__()

        # 1. 瓶颈层 (Bottleneck): 降低计算量
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, bias=False
            )
            input_channels = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            input_channels = in_channels

        # 2. 并行卷积层 (Inception)
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    input_channels,
                    out_channels,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )

        # 3. 最大池化分支 (Max Pooling)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        # 4. 批归一化 (BN)
        total_out_channels = out_channels * len(kernel_sizes) + out_channels
        self.bn = nn.BatchNorm1d(total_out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_in = self.bottleneck(x)

        conv_outs = [conv(x_in) for conv in self.conv_layers]
        pool_out = self.conv_pool(self.maxpool(x))

        out = torch.cat(conv_outs + [pool_out], dim=1)
        return self.relu(self.bn(out))


class InceptionEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=32, depth=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        for d in range(depth):
            layer_in = in_channels if d == 0 else out_channels * 4
            self.blocks.append(InceptionModule(layer_in, out_channels))

            if layer_in != out_channels * 4:
                self.shortcuts.append(
                    nn.Conv1d(layer_in, out_channels * 4, kernel_size=1, bias=False)
                )
            else:
                self.shortcuts.append(nn.Identity())

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = F.relu(block(x) + shortcut(x))
        return self.pool(x).squeeze(-1)


# =========================================================================
# 核心组件 2: LSTM 模块 
# 作用: 递归处理时序，擅长捕捉前后依赖关系。
# =========================================================================
class LSTMEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        output, (hn, cn) = self.lstm(x)
        return output[:, -1, :]


# =========================================================================
# 核心组件 3: 简单 CNN 模块 (基准线)
# 作用: 结构简单，跑得快，用来做 Baseline。
# =========================================================================
class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class TransformerEncoder(nn.Module):
    # 适合大数据
    def __init__(self, in_channels=2, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, d_model, 1)
        self.residual_proj = nn.Conv1d(in_channels, d_model, 1)  # 残差适配层
        
        # 混合位置编码（正弦余弦+可学习）
        self.fixed_pos = self._sinusoidal_pos_enc(1000, d_model)
        self.learnable_pos = nn.Parameter(torch.randn(1, 1000, d_model))
        self.pos_alpha = nn.Parameter(torch.tensor(0.5))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, 0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _sinusoidal_pos_enc(self, seq_len, d_model):
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pos_enc = torch.zeros(1, seq_len, d_model)
        pos_enc[0, :, 0::2] = torch.sin(pos * div)
        pos_enc[0, :, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pos_enc, requires_grad=False)

    def forward(self, x):
        x_res = self.residual_proj(x)  # 残差分支
        x = self.input_proj(x) + x_res  # 残差连接
        x = F.relu(x)
        
        x = x.permute(0,2,1)
        pos_enc = self.pos_alpha * self.fixed_pos + (1-self.pos_alpha)*self.learnable_pos
        x = x + pos_enc
        
        x = self.transformer(x)
        x = x.permute(0,2,1)
        return self.pool(x).squeeze(-1)






# =========================================================================
# Multi-Expert Fusion 模型 (多专家融合)
# =========================================================================

class StaticMLPEncoder(nn.Module):
    """静态特征编码器"""
    def __init__(self, in_dim, out_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(x)


class ConstitutionEmbedding(nn.Module):
    """体质类型 Embedding"""
    def __init__(self, num_constitutions=38, embed_dim=32, out_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_constitutions, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        embed = self.embedding(x)
        out = self.proj(embed)
        return out


class CrossAttentionGate(nn.Module):
    """改进的交叉注意力门控: 用静态特征调制动态特征
    - 先将动态特征降采样成多个 token
    - 使用 Multi-Head Attention
    - 静态特征生成 Query
    """
    def __init__(self, dim=128, static_dim=None, num_tokens=4, num_heads=4):
        super().__init__()
        static_dim = static_dim or dim
        self.num_tokens = num_tokens

        # 静态特征投影为 Query
        self.q_proj = nn.Linear(static_dim, dim)

        # 动态特征投影为 Key/Value
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Multi-Head Attention
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Gate 投影
        self.gate_proj = nn.Linear(dim, dim)

        # 注册位置编码为 buffer（自动移动到正确的设备）
        self.register_buffer('pos_encoding', torch.randn(1, num_tokens, dim) * 0.1)

    def forward(self, dynamic, static):
        """
        Args:
            dynamic: (B, D) - 动态特征编码
            static: (B, D) - 静态特征编码
        Returns:
            gated_dynamic: (B, D) - 门控后的动态特征
            attn_weights: (B, num_tokens) - 注意力权重
        """
        B, D = dynamic.shape

        # 1. 将动态特征降采样成多个 token
        # 方法：使用线性层 + reshape 创建多个 token
        dynamic_tokens = self.k_proj(dynamic).unsqueeze(1)  # (B, 1, D)
        # 复制多次并添加位置编码
        dynamic_tokens = dynamic_tokens.repeat(1, self.num_tokens, 1)  # (B, num_tokens, D)
        # 添加位置编码
        dynamic_tokens = dynamic_tokens + self.pos_encoding  # (B, num_tokens, D)

        # 2. 静态特征生成 Query
        q = self.q_proj(static).unsqueeze(1)  # (B, 1, D)

        # 3. 使用 MHA 计算注意力
        attn_output, attn_weights = self.mha(
            query=q,
            key=dynamic_tokens,
            value=dynamic_tokens,
            need_weights=True
        )
        # attn_output: (B, 1, D), attn_weights: (B, 1, num_tokens)

        gated = attn_output.squeeze(1)  # (B, D)
        attn_weights = attn_weights.squeeze(1)  # (B, num_tokens)

        # 4. 计算 Gate
        gate = torch.sigmoid(self.gate_proj(gated))

        # 5. 残差连接
        gated_dynamic = dynamic * gate + dynamic

        return gated_dynamic, attn_weights


class MultiExpertFusionModel(nn.Module):
    """多专家融合分类模型 - 输出 logits (CrossEntropyLoss 需要)
    
    改进：使用 nn.ModuleDict 组织子模块，便于：
    1. 不同模块使用不同学习率
    2. 模块级别的参数管理
    3. 验证 scheduler 是否正确更新所有参数组
    """
    def __init__(self, num_classes=3, num_constitutions=38, shared_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()

        # 使用 ModuleDict 组织编码器（便于参数分组）
        self.encoders = nn.ModuleDict({
            'dynamic': InceptionEncoder(in_channels=2, out_channels=shared_dim, depth=3),
            'static_basic': StaticMLPEncoder(in_dim=4, out_dim=shared_dim),
            'static_scores': StaticMLPEncoder(in_dim=2, out_dim=shared_dim),
            'constitution': ConstitutionEmbedding(num_constitutions=num_constitutions, embed_dim=32, out_dim=shared_dim),
        })

        # Cross-Attention Gating - dynamic 输出维度是 shared_dim * 4
        self.cross_attn = CrossAttentionGate(
            dim=shared_dim * 4,
            static_dim=shared_dim,
            num_tokens=4,  # 降采样成 4 个 token
            num_heads=4   # 4 个注意力头
        )

        # 使用 ModuleDict 组织投影层和融合层（便于参数分组）
        self.fusion_modules = nn.ModuleDict({
            'dynamic_proj': nn.Linear(shared_dim * 4, shared_dim),
            'static_basic_proj': nn.Linear(shared_dim, shared_dim),
            'static_scores_proj': nn.Linear(shared_dim, shared_dim),
            'constitution_proj': nn.Linear(shared_dim, shared_dim),
            'fusion': nn.Sequential(
                nn.Linear(shared_dim * 4, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Dropout(dropout),
            ),
        })

        # 分类头 (输出原始 logits，不使用 softmax)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
        # 兼容旧接口的属性访问
        self.dynamic_encoder = self.encoders['dynamic']
        self.static_basic_encoder = self.encoders['static_basic']
        self.static_scores_encoder = self.encoders['static_scores']
        self.constitution_encoder = self.encoders['constitution']
        self.fusion = self.fusion_modules['fusion']
        # 投影层兼容属性
        self.dynamic_proj = self.fusion_modules['dynamic_proj']
        self.static_basic_proj = self.fusion_modules['static_basic_proj']
        self.static_scores_proj = self.fusion_modules['static_scores_proj']
        self.constitution_proj = self.fusion_modules['constitution_proj']
        
        # 存储结构信息（用于实验记录）
        self.fusion_type = "cross_attention"
        self.model_name = "CrossAttentionFusion"

    def forward(self, dynamic, static_basic, static_scores, constitution, return_attention=False):
        """
        Args:
            dynamic: (B, 2, 1000)
            static_basic: (B, 4)
            static_scores: (B, 2)
            constitution: (B,) long
            return_attention: 是否返回注意力权重

        Returns:
            logits: (B, num_classes) 原始 logits
            attention_weights: (B, num_tokens) (optional)
        """
        # 1. 独立编码
        z_d = self.dynamic_encoder(dynamic)
        z_b = self.static_basic_encoder(static_basic)
        z_s = self.static_scores_encoder(static_scores)
        z_c = self.constitution_encoder(constitution)

        # 2. Cross-Attention Gating
        z_d_gated, attn_weights = self.cross_attn(z_d, z_b)

        # 3. 投影到统一维度
        z_d_proj = self.dynamic_proj(z_d_gated)
        z_b_proj = self.static_basic_proj(z_b)
        z_s_proj = self.static_scores_proj(z_s)
        z_c_proj = self.constitution_proj(z_c)

        # 4. 拼接融合
        fused = torch.cat([z_d_proj, z_b_proj, z_s_proj, z_c_proj], dim=-1)
        fusion_feat = self.fusion(fused)

        # 5. 分类 (输出 logits)
        logits = self.classifier(fusion_feat)

        if return_attention:
            return logits, attn_weights
        return logits


# =========================================================================
# 工厂函数: 根据配置返回模型
# =========================================================================
# =========================================================================
# Baseline A: Simple Concatenation
# 描述: 每个模态独立编码后直接拼接，经 MLP 输出
# 目的: 测试"独立编码 + 直接拼接"是否已能达到较好性能
# =========================================================================
class SimpleConcatModel(nn.Module):
    """
    Baseline A - 最简单的多模态融合基线
    
    架构:
    - Waveform: 2-3 层 1D Conv + GAP → D 维向量
    - Static Basic: 2 层 MLP → D 维向量
    - Static Scores: 2 层 MLP → D 维向量
    - Constitution: Embedding + Linear → D 维向量
    - Fusion: torch.cat() → MLP → logits
    """
    def __init__(
        self, 
        num_classes=3, 
        num_constitutions=38,
        shared_dim=64,
        hidden_dim=128,
        dropout=0.3
    ):
        super().__init__()
        
        # Waveform encoder: 简单 1D Conv + GAP
        self.waveform_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, shared_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            # GAP: (B, shared_dim, 250) → (B, shared_dim)
        )
        
        # Static basic encoder: 2 层 MLP
        self.static_basic_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, shared_dim),
            nn.ReLU(),
        )
        
        # Static scores encoder: 2 层 MLP
        self.static_scores_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, shared_dim),
            nn.ReLU(),
        )
        
        # Constitution embedding
        self.constitution_embedding = nn.Sequential(
            nn.Embedding(num_constitutions, 16),
            nn.Flatten(),
            nn.Linear(16, shared_dim),
            nn.ReLU(),
        )
        
        # Fusion MLP
        fusion_dim = shared_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # 存储结构信息（用于实验记录）
        self.fusion_type = "concat"
        self.model_name = "SimpleConcat"
        
    def forward(self, dynamic, static_basic, static_scores, constitution):
        """
        Args:
            dynamic: (B, 2, 1000) 波形数据
            static_basic: (B, 4) 基础静态特征
            static_scores: (B, 2) 评分静态特征
            constitution: (B,) 体质索引
            
        Returns:
            logits: (B, num_classes)
        """
        # 1. 独立编码
        z_wave = self.waveform_encoder(dynamic)  # (B, shared_dim, 250)
        z_wave = z_wave.mean(dim=-1)  # GAP → (B, shared_dim)
        
        z_basic = self.static_basic_encoder(static_basic)  # (B, shared_dim)
        z_scores = self.static_scores_encoder(static_scores)  # (B, shared_dim)
        z_const = self.constitution_embedding(constitution)  # (B, shared_dim)
        
        # 2. 直接拼接
        fused = torch.cat([z_wave, z_basic, z_scores, z_const], dim=-1)
        
        # 3. 分类
        logits = self.classifier(fused)
        
        return logits


# =========================================================================
# Baseline B: Late Fusion with Transformer
# 描述: 每个专家输出视为 token，用 Transformer Encoder 融合
# 目的: 评估自注意力融合是否优于简单拼接
# =========================================================================
class LateFusionTransformerModel(nn.Module):
    """
    Baseline B - Transformer 晚融合
    
    架构:
    - 各模态编码器输出投影到 D_shared 维
    - 加上 learnable modality positional embeddings
    - 1-2 层 TransformerEncoder 做融合
    - 全局池化或 class token → 分类头
    """
    def __init__(
        self,
        num_classes=3,
        num_constitutions=38,
        shared_dim=64,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()
        
        self.shared_dim = shared_dim
        self.num_modalities = 4
        
        # Waveform encoder (使用 Inception 风格但输出 pooled)
        self.waveform_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, shared_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
        )
        
        # Static encoders
        self.static_basic_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, shared_dim),
        )
        
        self.static_scores_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, shared_dim),
        )
        
        self.constitution_embedding = nn.Sequential(
            nn.Embedding(num_constitutions, 16),
            nn.Flatten(),
            nn.Linear(16, shared_dim),
        )
        
        # Learnable modality positional embeddings
        self.modality_pe = nn.Parameter(torch.randn(1, self.num_modalities, shared_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=shared_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # 存储结构信息
        self.fusion_type = "transformer"
        self.model_name = "LateFusionTransformer"
        
    def forward(self, dynamic, static_basic, static_scores, constitution):
        """
        Args:
            dynamic: (B, 2, 1000)
            static_basic: (B, 4)
            static_scores: (B, 2)
            constitution: (B,)
            
        Returns:
            logits: (B, num_classes)
        """
        B = dynamic.size(0)
        
        # 1. 独立编码
        z_wave = self.waveform_encoder(dynamic).mean(dim=-1)  # (B, shared_dim)
        z_basic = self.static_basic_encoder(static_basic)
        z_scores = self.static_scores_encoder(static_scores)
        z_const = self.constitution_embedding(constitution)
        
        # 2. 组装 tokens: (B, 4, shared_dim)
        tokens = torch.stack([z_wave, z_basic, z_scores, z_const], dim=1)
        
        # 3. 添加 modality positional embeddings
        tokens = tokens + self.modality_pe
        
        # 4. Transformer 融合
        fused_tokens = self.transformer(tokens)  # (B, 4, shared_dim)
        
        # 5. 全局平均池化
        pooled = fused_tokens.mean(dim=1)  # (B, shared_dim)
        
        # 6. 分类
        logits = self.classifier(pooled)
        
        return logits


# =========================================================================
# Baseline D: Simple Self-Attention Fusion
# 描述: 将所有模态特征拼接成序列，使用自注意力进行融合
# 目的: 验证注意力机制是否能更好地平衡各模态贡献
# =========================================================================
class SimpleAttentionFusion(nn.Module):
    """
    Baseline D - 简单自注意力融合
    
    架构:
    - 各模态编码器输出 → 拼接成 (B, 4, shared_dim)
    - 使用 Multi-Head Self-Attention 进行融合
    - 全局池化 → MLP → 分类
    
    优点: 
    - 简单直接，易于理解
    - 自注意力可以学习模态间的关联
    """
    def __init__(
        self,
        num_classes=3,
        num_constitutions=38,
        shared_dim=64,
        hidden_dim=128,
        num_heads=4,
        dropout=0.3
    ):
        super().__init__()
        
        self.shared_dim = shared_dim
        self.num_modalities = 4
        
        # 编码器（与 baseline_a 相同）
        self.waveform_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, shared_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
        )
        
        self.static_basic_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, shared_dim),
            nn.ReLU(),
        )
        
        self.static_scores_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, shared_dim),
            nn.ReLU(),
        )
        
        self.constitution_embedding = nn.Sequential(
            nn.Embedding(num_constitutions, 16),
            nn.Flatten(),
            nn.Linear(16, shared_dim),
            nn.ReLU(),
        )
        
        # Self-Attention 融合层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=shared_dim,
            nhead=num_heads,
            dim_feedforward=shared_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # 存储结构信息
        self.fusion_type = "self_attention"
        self.model_name = "SimpleAttentionFusion"
        
    def forward(self, dynamic, static_basic, static_scores, constitution):
        """
        Args:
            dynamic: (B, 2, 1000)
            static_basic: (B, 4)
            static_scores: (B, 2)
            constitution: (B,)
            
        Returns:
            logits: (B, num_classes)
        """
        # 1. 独立编码
        z_wave = self.waveform_encoder(dynamic).mean(dim=-1)  # (B, shared_dim)
        z_basic = self.static_basic_encoder(static_basic)
        z_scores = self.static_scores_encoder(static_scores)
        z_const = self.constitution_embedding(constitution)
        
        # 2. 组装 tokens: (B, 4, shared_dim)
        tokens = torch.stack([z_wave, z_basic, z_scores, z_const], dim=1)
        
        # 3. Self-Attention 融合
        fused_tokens = self.self_attention(tokens)
        
        # 4. 全局平均池化
        pooled = fused_tokens.mean(dim=1)  # (B, shared_dim)
        
        # 5. 分类
        logits = self.classifier(pooled)
        
        return logits


# =========================================================================
# Baseline E: Gated Fusion
# 描述: 使用门控机制动态控制各模态的贡献权重
# 目的: 防止模型"偷懒"，强制利用所有模态
# =========================================================================
class GatedFusion(nn.Module):
    """
    Baseline E - 门控融合
    
    架构:
    - 各模态编码器输出 → 独立投影
    - 门控网络学习每个模态的权重
    - 加权融合 → MLP → 分类
    
    优点:
    - 动态平衡各模态贡献
    - 可以防止某个模态主导
    """
    def __init__(
        self,
        num_classes=3,
        num_constitutions=38,
        shared_dim=64,
        hidden_dim=128,
        dropout=0.3
    ):
        super().__init__()
        
        self.shared_dim = shared_dim
        
        # 编码器（与 baseline_a 相同）
        self.waveform_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, shared_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
        )
        
        self.static_basic_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, shared_dim),
            nn.ReLU(),
        )
        
        self.static_scores_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, shared_dim),
            nn.ReLU(),
        )
        
        self.constitution_embedding = nn.Sequential(
            nn.Embedding(num_constitutions, 16),
            nn.Flatten(),
            nn.Linear(16, shared_dim),
            nn.ReLU(),
        )
        
        # 独立投影层
        self.dynamic_proj = nn.Linear(shared_dim, shared_dim)
        self.static_basic_proj = nn.Linear(shared_dim, shared_dim)
        self.static_scores_proj = nn.Linear(shared_dim, shared_dim)
        self.constitution_proj = nn.Linear(shared_dim, shared_dim)
        
        # 门控网络：输入所有模态特征，输出4个门控值
        self.gate_network = nn.Sequential(
            nn.Linear(shared_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)  # 归一化为概率分布
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # 存储结构信息
        self.fusion_type = "gated"
        self.model_name = "GatedFusion"
        
    def forward(self, dynamic, static_basic, static_scores, constitution):
        """
        Args:
            dynamic: (B, 2, 1000)
            static_basic: (B, 4)
            static_scores: (B, 2)
            constitution: (B,)
            
        Returns:
            logits: (B, num_classes)
        """
        # 1. 独立编码
        z_wave = self.waveform_encoder(dynamic).mean(dim=-1)  # (B, shared_dim)
        z_basic = self.static_basic_encoder(static_basic)
        z_scores = self.static_scores_encoder(static_scores)
        z_const = self.constitution_embedding(constitution)
        
        # 2. 独立投影
        z_wave_proj = self.dynamic_proj(z_wave)
        z_basic_proj = self.static_basic_proj(z_basic)
        z_scores_proj = self.static_scores_proj(z_scores)
        z_const_proj = self.constitution_proj(z_const)
        
        # 3. 计算门控权重
        all_features = torch.cat([z_wave, z_basic, z_scores, z_const], dim=-1)
        gates = self.gate_network(all_features)  # (B, 4)
        
        # 4. 加权融合
        gates = gates.unsqueeze(-1)  # (B, 4, 1)
        features = torch.stack([z_wave_proj, z_basic_proj, z_scores_proj, z_const_proj], dim=1)  # (B, 4, shared_dim)
        fused = (features * gates).sum(dim=1)  # (B, shared_dim)
        
        # 5. 分类
        logits = self.classifier(fused)
        
        return logits


# =========================================================================
# 工厂函数: 根据配置返回模型
# =========================================================================
def get_model(model_type="inception", num_classes=3, dyn_channels=2, static_dim=4, **kwarg):
    """工厂函数：根据配置返回模型"""
    # Baseline 模型
    if model_type == "baseline_a" or model_type == "simple_concat":
        return SimpleConcatModel(
            num_classes=num_classes,
            num_constitutions=kwarg.get('num_constitutions', 38),
            shared_dim=kwarg.get('shared_dim', 64),
            hidden_dim=kwarg.get('hidden_dim', 128),
            dropout=kwarg.get('dropout', 0.3),
        )
    
    if model_type == "baseline_b" or model_type == "late_fusion":
        return LateFusionTransformerModel(
            num_classes=num_classes,
            num_constitutions=kwarg.get('num_constitutions', 38),
            shared_dim=kwarg.get('shared_dim', 64),
            hidden_dim=kwarg.get('hidden_dim', 128),
            num_heads=kwarg.get('num_heads', 4),
            num_layers=kwarg.get('num_layers', 2),
            dropout=kwarg.get('dropout', 0.3),
        )
    
    # Baseline C (原有模型)
    if model_type == "baseline_c" or model_type == "multimodal":
        return MultiExpertFusionModel(
            num_classes=num_classes,
            num_constitutions=kwarg.get('num_constitutions', 38),
            shared_dim=kwarg.get('shared_dim', 128),
            hidden_dim=kwarg.get('hidden_dim', 256),
            dropout=kwarg.get('dropout', 0.3),
        )
    
    # 新增的融合策略
    if model_type == "attention_fusion" or model_type == "baseline_d":
        return SimpleAttentionFusion(
            num_classes=num_classes,
            num_constitutions=kwarg.get('num_constitutions', 38),
            shared_dim=kwarg.get('shared_dim', 64),
            hidden_dim=kwarg.get('hidden_dim', 128),
            num_heads=kwarg.get('num_heads', 4),
            dropout=kwarg.get('dropout', 0.3),
        )
    
    if model_type == "gated_fusion" or model_type == "baseline_e":
        return GatedFusion(
            num_classes=num_classes,
            num_constitutions=kwarg.get('num_constitutions', 38),
            shared_dim=kwarg.get('shared_dim', 64),
            hidden_dim=kwarg.get('hidden_dim', 128),
            dropout=kwarg.get('dropout', 0.3),
        )
    
    raise ValueError(f"Unknown model type: {model_type}")


# =========================================================================
# 自测代码
# =========================================================================
if __name__ == "__main__":
    dummy_dyn = torch.randn(8, 2, 1000)
    dummy_static_basic = torch.randn(8, 4)
    dummy_static_scores = torch.randn(8, 2)
    dummy_constitution = torch.randint(0, 38, (8,))

    print("Testing all 5 fusion models:")
    print("=" * 60)
    
    # 测试所有5个模型
    models = [
        ("baseline_a", "Simple Concat"),
        ("baseline_b", "Late Fusion Transformer"),
        ("baseline_c", "Multi-Expert Fusion"),
        ("baseline_d", "Simple Attention"),
        ("baseline_e", "Gated Fusion"),
    ]
    
    for model_type, model_name in models:
        print(f"\n--- Testing {model_name} ({model_type}) ---")
        try:
            model = get_model(model_type, num_classes=3)
            output = model(dummy_dyn, dummy_static_basic, dummy_static_scores, dummy_constitution)
            print(f"输出形状: {output.shape}")
            print("✅ 通过")
        except Exception as e:
            print(f"❌ 失败: {e}")
