import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pathlib import Path
from typing import Optional, List


class TCM_Encoder(nn.Module):
    def __init__(
        self,
        model_path='data/tcm_ft_transformer/checkpoints/best_model.pth',
        scaler_path='data/tcm_ft_transformer/data/scaler_params.npz',
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.scaler_params = self._load_scaler(scaler_path)
        self.register_buffer('mean', torch.tensor(self.scaler_params['mean'], dtype=torch.float32))
        self.register_buffer('std', torch.tensor(self.scaler_params['std'], dtype=torch.float32))
        self.model = self._load_ft_transformer(model_path)
        self.model.to(device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_projection = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(device)

        self.constitution_names = [
            "平和质", "气虚质", "阳虚质", "阴虚质",
            "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质"
        ]

    def _load_scaler(self, scaler_path):
        scaler_data = np.load(scaler_path)
        return {
            'mean': scaler_data['mean'],
            'std': scaler_data['std']
        }

    def _load_ft_transformer(self, model_path):
        import sys
        tcm_path = str(Path(model_path).parent.parent)
        if tcm_path not in sys.path:
            sys.path.insert(0, tcm_path)

        from ft_transformer import get_model

        model = get_model(n_features=8, n_classes=9)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def normalize(self, x):
        std_safe = torch.where(self.std < 1e-8, torch.ones_like(self.std), self.std)
        return (x - self.mean) / std_safe

    def extract_cls_token(self, x):
        batch_size = x.size(0)
        tokens = self.model.feature_tokenizer(x)
        cls_token = self.model.cls_token(batch_size)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = self.model.dropout_layer(tokens)
        encoded = self.model.transformer_encoder(tokens)
        cls_token = encoded[:, 0, :]
        cls_token = self.model.layer_norm(cls_token)
        return cls_token

    def forward(self, x):
        with torch.no_grad():
            x_normalized = self.normalize(x)
            cls_token = self.extract_cls_token(x_normalized)
            probs = self.model(x_normalized)
        features = self.feature_projection(cls_token)
        return features, probs

    def encode(self, x):
        return self.forward(x)

    def predict_constitution(self, x):
        _, probs = self.forward(x)
        constitution = torch.argmax(probs, dim=1)
        constitution_names = [self.constitution_names[idx] for idx in constitution.cpu().numpy()]
        return constitution, constitution_names, probs

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self


def create_tcm_encoder(
    model_path='data/tcm_ft_transformer/checkpoints/best_model.pth',
    scaler_path='data/tcm_ft_transformer/data/scaler_params.npz',
    device='cuda'
):
    encoder = TCM_Encoder(
        model_path=model_path,
        scaler_path=scaler_path,
        device=device
    )
    return encoder


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Optional[List[int]] = None,
        bottleneck_channels: int = 32,
    ):
        super(InceptionModule, self).__init__()
        kernel_sizes = kernel_sizes or [9, 19, 39]

        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, bias=False
            )
            input_channels = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            input_channels = in_channels

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

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

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


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, d_model, 1)
        self.residual_proj = nn.Conv1d(in_channels, d_model, 1)
        self.fixed_pos = self._sinusoidal_pos_enc(1000, d_model)
        self.learnable_pos = nn.Parameter(torch.randn(1, 1000, d_model))
        self.pos_alpha = nn.Parameter(torch.tensor(0.5))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, 0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _sinusoidal_pos_enc(self, seq_len: int, d_model: int) -> nn.Parameter:
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_enc = torch.zeros(1, seq_len, d_model)
        pos_enc[0, :, 0::2] = torch.sin(pos * div)
        pos_enc[0, :, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pos_enc, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.residual_proj(x)
        x = self.input_proj(x) + x_res
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        seq_len = x.size(1)

        if seq_len <= 1000:
            pos_enc = self.pos_alpha * self.fixed_pos[:, :seq_len, :] + (1 - self.pos_alpha) * self.learnable_pos[:, :seq_len, :]
        else:
            fixed_pos = self._sinusoidal_pos_enc(seq_len, self.input_proj.out_channels)
            pos_enc = self.pos_alpha * fixed_pos + (1 - self.pos_alpha) * self.learnable_pos.expand(1, seq_len, -1)

        x = x + pos_enc
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        return self.pool(x).squeeze(-1)


class StaticMLPEncoder(nn.Module):
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