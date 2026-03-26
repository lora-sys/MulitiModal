"""
公共编码器模块

提供可复用的编码器类，消除代码重复
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InceptionModule(nn.Module):
    """Inception模块：通过多尺度卷积核并行提取特征"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list = [9, 19, 39],
        bottleneck_channels: int = 32,
    ):
        super(InceptionModule, self).__init__()

        # 1. 瓶颈层：降低计算量
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, bias=False
            )
            input_channels = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            input_channels = in_channels

        # 2. 并行卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                input_channels,
                out_channels,
                kernel_size=k,
                padding=k // 2,
                bias=False,
            )
            for k in kernel_sizes
        ])

        # 3. 最大池化分支
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        # 4. 批归一化
        total_out_channels = out_channels * len(kernel_sizes) + out_channels
        self.bn = nn.BatchNorm1d(total_out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.bottleneck(x)

        conv_outs = [conv(x_in) for conv in self.conv_layers]
        pool_out = self.conv_pool(self.maxpool(x))

        out = torch.cat(conv_outs + [pool_out], dim=1)
        return self.relu(self.bn(out))


class InceptionEncoder(nn.Module):
    """Inception编码器：多尺度时序特征提取"""

    def __init__(self, in_channels: int, out_channels: int = 32, depth: int = 3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = F.relu(block(x) + shortcut(x))
        return self.pool(x).squeeze(-1)


class LSTMEncoder(nn.Module):
    """LSTM编码器：递归处理时序，捕捉前后依赖关系"""

    def __init__(self, in_channels: int, hidden_dim: int = 64, num_layers: int = 2, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        output, (hn, cn) = self.lstm(x)
        return output[:, -1, :]  # 返回最后一个时间步的输出


class SimpleCNNEncoder(nn.Module):
    """简单CNN编码器：基准线编码器"""

    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TransformerEncoder(nn.Module):
    """Transformer编码器：适合大数据量的时序建模"""

    def __init__(self, in_channels: int = 2, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, d_model, 1)
        self.residual_proj = nn.Conv1d(in_channels, d_model, 1)  # 残差适配层

        # 混合位置编码（正弦余弦+可学习）
        self.fixed_pos = self._sinusoidal_pos_enc(1000, d_model)
        self.learnable_pos = nn.Parameter(torch.randn(1, 1000, d_model))
        self.pos_alpha = nn.Parameter(torch.tensor(0.5))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, 0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _sinusoidal_pos_enc(self, seq_len: int, d_model: int) -> nn.Parameter:
        """生成正弦余弦位置编码"""
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_enc = torch.zeros(1, seq_len, d_model)
        pos_enc[0, :, 0::2] = torch.sin(pos * div)
        pos_enc[0, :, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pos_enc, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.residual_proj(x)  # 残差分支
        x = self.input_proj(x) + x_res  # 残差连接
        x = F.relu(x)

        x = x.permute(0, 2, 1)
        pos_enc = self.pos_alpha * self.fixed_pos + (1 - self.pos_alpha) * self.learnable_pos
        x = x + pos_enc

        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        return self.pool(x).squeeze(-1)


class StaticMLPEncoder(nn.Module):
    """静态特征编码器：处理静态特征"""

    def __init__(self, in_dim: int, out_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConstitutionEmbedding(nn.Module):
    """体质类型嵌入：将体质ID转换为向量"""

    def __init__(self, num_constitutions: int = 39, embed_dim: int = 32, out_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_constitutions, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(x)
        out = self.proj(embed)
        return out


class WaveformEncoder(nn.Module):
    """统一的波形编码器：封装多种编码器"""

    def __init__(
        self,
        encoder_type: str = 'inception',
        in_channels: int = 2,
        out_channels: int = 64,
        **kwargs
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == 'inception':
            self.encoder = InceptionEncoder(in_channels, out_channels, depth=kwargs.get('depth', 3))
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(
                in_channels,
                hidden_dim=kwargs.get('hidden_dim', 64),
                num_layers=kwargs.get('num_layers', 2),
                bidirectional=kwargs.get('bidirectional', True)
            )
        elif encoder_type == 'cnn':
            self.encoder = SimpleCNNEncoder(in_channels, out_channels)
        elif encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                in_channels,
                d_model=kwargs.get('d_model', 64),
                nhead=kwargs.get('nhead', 4),
                num_layers=kwargs.get('num_layers', 2)
            )
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)