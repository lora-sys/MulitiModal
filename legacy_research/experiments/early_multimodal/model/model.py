import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = None,
        bottleneck_channels: int = 32,
    ):
        super().__init__()
        kernel_sizes = kernel_sizes or [9, 19, 39]

        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if in_channels > 1
            else nn.Identity()
        )
        input_channels = bottleneck_channels if in_channels > 1 else in_channels

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_channels, out_channels, kernel_size=k, padding=k // 2, bias=False)
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
            self.shortcuts.append(
                nn.Conv1d(layer_in, out_channels * 4, kernel_size=1, bias=False)
                if layer_in != out_channels * 4
                else nn.Identity()
            )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = F.relu(block(x) + shortcut(x))
        return self.pool(x).squeeze(-1)


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
    def __init__(self, in_channels=2, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, d_model, 1)
        self.residual_proj = nn.Conv1d(in_channels, d_model, 1)
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
        x_res = self.residual_proj(x)
        x = self.input_proj(x) + x_res
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        pos_enc = self.pos_alpha * self.fixed_pos + (1-self.pos_alpha)*self.learnable_pos
        x = x + pos_enc
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        return self.pool(x).squeeze(-1)






class StaticMLPEncoder(nn.Module):
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
    def __init__(self, dim=128, static_dim=None, num_tokens=4, num_heads=4):
        super().__init__()
        static_dim = static_dim or dim
        self.num_tokens = num_tokens
        self.q_proj = nn.Linear(static_dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate_proj = nn.Linear(dim, dim)
        self.register_buffer('pos_encoding', torch.randn(1, num_tokens, dim) * 0.1)

    def forward(self, dynamic, static):
        B, D = dynamic.shape
        dynamic_tokens = self.k_proj(dynamic).unsqueeze(1)
        dynamic_tokens = dynamic_tokens.repeat(1, self.num_tokens, 1)
        dynamic_tokens = dynamic_tokens + self.pos_encoding
        q = self.q_proj(static).unsqueeze(1)
        attn_output, attn_weights = self.mha(query=q, key=dynamic_tokens, value=dynamic_tokens, need_weights=True)
        gated = attn_output.squeeze(1)
        attn_weights = attn_weights.squeeze(1)
        gate = torch.sigmoid(self.gate_proj(gated))
        gated_dynamic = dynamic * gate
        return gated_dynamic, attn_weights


class MultiExpertFusionModel(nn.Module):
    def __init__(self, num_classes=3, num_constitutions=38, shared_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.encoders = nn.ModuleDict({
            'dynamic': InceptionEncoder(in_channels=2, out_channels=shared_dim, depth=3),
            'static_basic': StaticMLPEncoder(in_dim=8, out_dim=shared_dim),
            'static_scores': StaticMLPEncoder(in_dim=2, out_dim=shared_dim),
            'constitution': ConstitutionEmbedding(num_constitutions=num_constitutions, embed_dim=32, out_dim=shared_dim),
        })

        self.cross_attn = CrossAttentionGate(
            dim=shared_dim * 4,
            static_dim=shared_dim,
            num_tokens=4,
            num_heads=4
        )

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

        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.fusion_type = "cross_attention"
        self.model_name = "CrossAttentionFusion"

    def forward(self, dynamic, static_basic, static_scores, constitution, return_attention=False):
        z_d = self.encoders['dynamic'](dynamic)
        z_b = self.encoders['static_basic'](static_basic)
        z_s = self.encoders['static_scores'](static_scores)
        z_c = self.encoders['constitution'](constitution)

        z_d_gated, attn_weights = self.cross_attn(z_d, z_b)

        z_d_proj = self.fusion_modules['dynamic_proj'](z_d_gated)
        z_b_proj = self.fusion_modules['static_basic_proj'](z_b)
        z_s_proj = self.fusion_modules['static_scores_proj'](z_s)
        z_c_proj = self.fusion_modules['constitution_proj'](z_c)

        fused = torch.cat([z_d_proj, z_b_proj, z_s_proj, z_c_proj], dim=-1)
        fusion_feat = self.fusion_modules['fusion'](fused)
        logits = self.classifier(fusion_feat)

        if return_attention:
            return logits, attn_weights
        return logits


class SimpleConcatModel(nn.Module):
    def __init__(self, num_classes=3, num_constitutions=38, shared_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()

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
            nn.Linear(8, 32),
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

        self.fusion_type = "concat"
        self.model_name = "SimpleConcat"

    def forward(self, dynamic, static_basic, static_scores, constitution):
        z_wave = self.waveform_encoder(dynamic).mean(dim=-1)
        z_basic = self.static_basic_encoder(static_basic)
        z_scores = self.static_scores_encoder(static_scores)
        z_const = self.constitution_embedding(constitution)
        fused = torch.cat([z_wave, z_basic, z_scores, z_const], dim=-1)
        logits = self.classifier(fused)
        return logits


class LateFusionTransformerModel(nn.Module):
    def __init__(self, num_classes=3, num_constitutions=38, shared_dim=64, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.shared_dim = shared_dim
        self.num_modalities = 4

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
            nn.Linear(8, 32),
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

        self.modality_pe = nn.Parameter(torch.randn(1, self.num_modalities, shared_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=shared_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.fusion_type = "transformer"
        self.model_name = "LateFusionTransformer"

    def forward(self, dynamic, static_basic, static_scores, constitution):
        B = dynamic.size(0)
        z_wave = self.waveform_encoder(dynamic).mean(dim=-1)
        z_basic = self.static_basic_encoder(static_basic)
        z_scores = self.static_scores_encoder(static_scores)
        z_const = self.constitution_embedding(constitution)
        tokens = torch.stack([z_wave, z_basic, z_scores, z_const], dim=1)
        tokens = tokens + self.modality_pe
        fused_tokens = self.transformer(tokens)
        pooled = fused_tokens.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


class SimpleAttentionFusion(nn.Module):
    def __init__(self, num_classes=3, num_constitutions=38, shared_dim=64, hidden_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        self.shared_dim = shared_dim
        self.num_modalities = 4

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
            nn.Linear(8, 32),
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

        encoder_layer = nn.TransformerEncoderLayer(d_model=shared_dim, nhead=num_heads, dim_feedforward=shared_dim * 2, dropout=dropout, batch_first=True)
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.fusion_type = "self_attention"
        self.model_name = "SimpleAttentionFusion"

    def forward(self, dynamic, static_basic, static_scores, constitution):
        z_wave = self.waveform_encoder(dynamic).mean(dim=-1)
        z_basic = self.static_basic_encoder(static_basic)
        z_scores = self.static_scores_encoder(static_scores)
        z_const = self.constitution_embedding(constitution)
        tokens = torch.stack([z_wave, z_basic, z_scores, z_const], dim=1)
        fused_tokens = self.self_attention(tokens)
        pooled = fused_tokens.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


class GatedFusion(nn.Module):
    def __init__(self, num_classes=3, num_constitutions=38, shared_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.shared_dim = shared_dim

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
            nn.Linear(8, 32),
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

        self.dynamic_proj = nn.Linear(shared_dim, shared_dim)
        self.static_basic_proj = nn.Linear(shared_dim, shared_dim)
        self.static_scores_proj = nn.Linear(shared_dim, shared_dim)
        self.constitution_proj = nn.Linear(shared_dim, shared_dim)

        self.gate_network = nn.Sequential(
            nn.Linear(shared_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.fusion_type = "gated"
        self.model_name = "GatedFusion"

    def forward(self, dynamic, static_basic, static_scores, constitution):
        z_wave = self.waveform_encoder(dynamic).mean(dim=-1)
        z_basic = self.static_basic_encoder(static_basic)
        z_scores = self.static_scores_encoder(static_scores)
        z_const = self.constitution_embedding(constitution)

        z_wave_proj = self.dynamic_proj(z_wave)
        z_basic_proj = self.static_basic_proj(z_basic)
        z_scores_proj = self.static_scores_proj(z_scores)
        z_const_proj = self.constitution_proj(z_const)

        all_features = torch.cat([z_wave, z_basic, z_scores, z_const], dim=-1)
        gates = self.gate_network(all_features)
        gates = gates.unsqueeze(-1)
        features = torch.stack([z_wave_proj, z_basic_proj, z_scores_proj, z_const_proj], dim=1)
        fused = (features * gates).sum(dim=1)
        logits = self.classifier(fused)
        return logits


# =========================================================================
class DualGatingFusionModel(nn.Module):
    def __init__(
        self,
        num_outputs=2,
        num_constitutions=9,
        shared_dim=128,
        projector_dim=128,
        gate_dim=128,
        hidden_dim=256,
        dropout=0.3,
        tcm_model_path='data/tcm_ft_transformer/checkpoints/best_model.pth',
        tcm_scaler_path='data/tcm_ft_transformer/data/scaler_params.npz',
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.num_constitutions = num_constitutions

        from .encoders import create_tcm_encoder
        self.tcm_encoder = create_tcm_encoder(
            model_path=tcm_model_path,
            scaler_path=tcm_scaler_path,
            device='cpu'
        )

        tcm_frozen = all(not p.requires_grad for p in self.tcm_encoder.parameters())
        print(f"[DualGating] TCM Encoder 参数冻结状态: {'✅ 已冻结' if tcm_frozen else '❌ 未冻结'}")

        self.pressure_encoder = InceptionEncoder(in_channels=2, out_channels=shared_dim, depth=3)
        self.pressure_proj = nn.Linear(shared_dim * 4, shared_dim)

        self.gate_a = nn.Sequential(
            nn.Linear(num_constitutions, gate_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gate_dim, shared_dim),
            nn.Sigmoid()
        )

        self.gate_b = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim, shared_dim),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(shared_dim * 2, projector_dim),
            nn.ReLU(),
            nn.LayerNorm(projector_dim),
            nn.Dropout(dropout)
        )

        self.regressor = nn.Sequential(
            nn.Linear(projector_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_outputs),
            nn.Sigmoid()
        )

        self.fusion_type = "dual_gating"
        self.model_name = "DualGatingFusion"

    def forward(self, dynamic, static_basic, return_intermediate=False):
        tcm_features, tcm_probs = self.tcm_encoder(static_basic)
        pressure_features = self.pressure_encoder(dynamic)
        pressure_features = self.pressure_proj(pressure_features)
        gate_a_weights = self.gate_a(tcm_probs)
        gated_pressure = pressure_features * gate_a_weights
        gate_b_weights = self.gate_b(pressure_features)
        gated_static = tcm_features * gate_b_weights
        fused = torch.cat([gated_static, gated_pressure], dim=1)
        fused_proj = self.fusion_proj(fused)
        outputs = self.regressor(fused_proj)

        if return_intermediate:
            return outputs, {
                'tcm_features': tcm_features,
                'tcm_probs': tcm_probs,
                'pressure_features': pressure_features,
                'gate_a_weights': gate_a_weights,
                'gate_b_weights': gate_b_weights,
                'gated_pressure': gated_pressure,
                'gated_static': gated_static,
                'fused_features': fused,
            }

        return outputs


def get_model(model_type="baseline_c", num_classes=3, dyn_channels=2, static_dim=4, **kwarg):
    if model_type in ["baseline_a", "simple_concat"]:
        return SimpleConcatModel(
            num_classes=num_classes,
            num_constitutions=kwarg.get('num_constitutions', 38),
            shared_dim=kwarg.get('shared_dim', 64),
            hidden_dim=kwarg.get('hidden_dim', 128),
            dropout=kwarg.get('dropout', 0.3),
        )

    if model_type in ["baseline_b", "late_fusion"]:
        return LateFusionTransformerModel(
            num_classes=num_classes,
            num_constitutions=kwarg.get('num_constitutions', 38),
            shared_dim=kwarg.get('shared_dim', 64),
            hidden_dim=kwarg.get('hidden_dim', 128),
            num_heads=kwarg.get('num_heads', 4),
            num_layers=kwarg.get('num_layers', 2),
            dropout=kwarg.get('dropout', 0.3),
        )

    if model_type in ["baseline_c", "multimodal"]:
        return MultiExpertFusionModel(
            num_classes=num_classes,
            num_constitutions=kwarg.get('num_constitutions', 38),
            shared_dim=kwarg.get('shared_dim', 128),
            hidden_dim=kwarg.get('hidden_dim', 256),
            dropout=kwarg.get('dropout', 0.3),
        )

    if model_type in ["attention_fusion", "baseline_d"]:
        return SimpleAttentionFusion(
            num_classes=num_classes,
            num_constitutions=kwarg.get('num_constitutions', 38),
            shared_dim=kwarg.get('shared_dim', 64),
            hidden_dim=kwarg.get('hidden_dim', 128),
            num_heads=kwarg.get('num_heads', 4),
            dropout=kwarg.get('dropout', 0.3),
        )

    if model_type in ["gated_fusion", "baseline_e"]:
        return GatedFusion(
            num_classes=num_classes,
            num_constitutions=kwarg.get('num_constitutions', 38),
            shared_dim=kwarg.get('shared_dim', 64),
            hidden_dim=kwarg.get('hidden_dim', 128),
            dropout=kwarg.get('dropout', 0.3),
        )

    if model_type == "dual_gating":
        return DualGatingFusionModel(
            num_outputs=2,
            num_constitutions=kwarg.get('num_constitutions', 9),
            shared_dim=kwarg.get('shared_dim', 128),
            projector_dim=kwarg.get('projector_dim', 128),
            gate_dim=kwarg.get('gate_dim', 128),
            hidden_dim=kwarg.get('hidden_dim', 256),
            dropout=kwarg.get('dropout', 0.3),
            tcm_model_path=kwarg.get('tcm_model_path', 'data/tcm_ft_transformer/checkpoints/best_model.pth'),
            tcm_scaler_path=kwarg.get('tcm_scaler_path', 'data/tcm_ft_transformer/data/scaler_params.npz'),
        )

    raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    dummy_dyn = torch.randn(8, 2, 1000)
    dummy_static_basic = torch.randn(8, 4)
    dummy_static_scores = torch.randn(8, 2)
    dummy_constitution = torch.randint(0, 38, (8,))

    print("Testing all 5 fusion models:")
    print("=" * 60)

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
