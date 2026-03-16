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
# 主控中心: 按摩椅双流融合网络 (MassageFusionNet)
# =========================================================================
class MassageFusionNet(nn.Module):
    def __init__(
        self, model_type="inception", num_classes=3, dyn_channels=2, static_dim=4 ,**kwarg):
        super(MassageFusionNet, self).__init__()

        self.model_type = model_type
        print(f"🏗️ 初始化模型架构: [ {model_type.upper()} ]")

        # --- 1. 左路: 动态时序流 (根据配置切换) ---
        if model_type == "inception":
            self.dynamic_encoder = InceptionEncoder(in_channels=dyn_channels)
            dyn_out_dim = 128
        elif model_type == "lstm":
            self.dynamic_encoder = LSTMEncoder(in_channels=dyn_channels)
            dyn_out_dim = 64
        elif model_type == "cnn":
            self.dynamic_encoder = SimpleCNNEncoder(in_channels=dyn_channels)
            dyn_out_dim = 32
        elif model_type == 'transformer':
            self.dynamic_encoder = TransformerEncoder(
                in_channels=dyn_channels,
                d_model=kwarg.get("d_model",64),
                nhead=kwarg.get("nhead",4),
                num_layers=kwarg.get("num_layers",2),
            )
            dyn_out_dim = kwarg.get("d_model",64)
        else:
            raise ValueError(f"未知模型类型: {model_type}")

        # --- 2. 右路: 静态画像流 (MLP) ---
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        # --- 3. 汇合: 融合层 ---
        fusion_input_dim = dyn_out_dim + 16
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_dynamic, x_static):
        dyn_feat = self.dynamic_encoder(x_dynamic)
        stat_feat = self.static_encoder(x_static)
        combined = torch.cat((dyn_feat, stat_feat), dim=1)
        return self.classifier(combined)


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
    """多专家融合分类模型 - 输出 logits (CrossEntropyLoss 需要)"""
    def __init__(self, num_classes=3, num_constitutions=38, shared_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()

        # 独立编码器
        self.dynamic_encoder = InceptionEncoder(in_channels=2, out_channels=shared_dim, depth=3)
        self.static_basic_encoder = StaticMLPEncoder(in_dim=4, out_dim=shared_dim)
        self.static_scores_encoder = StaticMLPEncoder(in_dim=2, out_dim=shared_dim)
        self.constitution_encoder = ConstitutionEmbedding(num_constitutions=num_constitutions, embed_dim=32, out_dim=shared_dim)

        # Cross-Attention Gating - dynamic 输出维度是 shared_dim * 4
        self.cross_attn = CrossAttentionGate(
            dim=shared_dim * 4,
            static_dim=shared_dim,
            num_tokens=4,  # 降采样成 4 个 token
            num_heads=4   # 4 个注意力头
        )

        # 投影到统一维度
        self.dynamic_proj = nn.Linear(shared_dim * 4, shared_dim)
        self.static_basic_proj = nn.Linear(shared_dim, shared_dim)
        self.static_scores_proj = nn.Linear(shared_dim, shared_dim)
        self.constitution_proj = nn.Linear(shared_dim, shared_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(shared_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
        )

        # 分类头 (输出原始 logits，不使用 softmax)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

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
def get_model(model_type="inception", num_classes=3, dyn_channels=2, static_dim=4, **kwarg):
    """工厂函数：根据配置返回模型"""
    if model_type == "multimodal":
        return MultiExpertFusionModel(
            num_classes=num_classes,
            num_constitutions=kwarg.get('num_constitutions', 38),
            shared_dim=kwarg.get('shared_dim', 128),
            hidden_dim=kwarg.get('hidden_dim', 256),
            dropout=kwarg.get('dropout', 0.3),
        )
    return MassageFusionNet(
        model_type=model_type,
        num_classes=num_classes,
        dyn_channels=dyn_channels,
        static_dim=static_dim,
        **kwarg
    )


# =========================================================================
# 自测代码
# =========================================================================
if __name__ == "__main__":
    dummy_dyn = torch.randn(8, 2, 1000)
    dummy_stat = torch.randn(8, 4)

    for m_type in ["cnn", "lstm", "inception","transformer"]:
        print(f"\n--- Testing {m_type} ---")
        model = MassageFusionNet(model_type=m_type)
        output = model(dummy_dyn, dummy_stat)
        print(f"输出形状: {output.shape}")
        print("✅ 通过")
