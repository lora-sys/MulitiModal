"""MulitiModal Demo — 多模态按摩决策演示系统
=============================================

三条检测链路:
  1. TCM 诊断 (舌诊/面诊/脉诊)    → FT-Transformer → 九型体质概率 + 128-D 特征
  2. 生理信号 (ECG + EDA)          → TCN 动态编码器  → 128-D 动态表征
  3. 脑电正念指数 (EEG Mindfulness) → 专用编码器      → 8-D 神经表征

统一融合 → 按摩方案推荐 + 力度等级
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
# 项目路径
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_ROOT = PROJECT_ROOT / "legacy_research"

sys.path.insert(0, str(LEGACY_ROOT / "source/tcm/tcm_ft_transformer"))
sys.path.insert(0, str(LEGACY_ROOT / "source/oplri/src"))
sys.path.insert(0, str(LEGACY_ROOT / "source/oplri/src/models"))

from ft_transformer import get_model               # noqa: E402
from models.encoders import get_dynamic_encoder    # noqa: E402

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────
CONSTITUTION_NAMES = [
    "平和质", "气虚质", "阳虚质", "阴虚质",
    "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质",
]

# 体质 → 推荐方案映射（待公司数据训练后替换为决策模型）
PROGRAM_CATALOG = [
    {
        "id": "stress_relief",
        "name": "舒缓解压",
        "name_en": "Stress Relief",
        "desc": "舒缓身心压力，释放日常疲劳与紧张",
        "icon": "🌿",
        "color": "#06b6d4",
        "techniques": ["瑞典式按摩", "芳香疗法", "轻柔拉伸", "热敷"],
    },
    {
        "id": "tcm_massage",
        "name": "中医推拿",
        "name_en": "TCM Massage",
        "desc": "传统中医经络推拿，调理气血运行",
        "icon": "🏮",
        "color": "#f59e0b",
        "techniques": ["经络推拿", "穴位按压", "拔罐", "刮痧"],
    },
    {
        "id": "thai_stretching",
        "name": "泰式拉筋",
        "name_en": "Thai Stretching",
        "desc": "深度拉伸与关节活动，改善身体柔韧性",
        "icon": "🤸",
        "color": "#10b981",
        "techniques": ["被动拉伸", "关节松动", "能量线按压", "足底反射"],
    },
    {
        "id": "brain_fitness",
        "name": "健脑强身",
        "name_en": "Brain & Body Fitness",
        "desc": "头部按摩与经络调理，增强精力与专注力",
        "icon": "🧠",
        "color": "#8b5cf6",
        "techniques": ["头部按摩", "肩颈放松", "足疗反射", "芳香呼吸"],
    },
]

INTENSITY_LEVELS = [
    {"key": "gentle",      "name": "轻柔",  "emoji": "🍃"},
    {"key": "comfortable", "name": "舒适",  "emoji": "🌿"},
    {"key": "strong",      "name": "强劲",  "emoji": "💪"},
]


# ──────────────────────────────────────────────────────────────
# 模型定义
# ──────────────────────────────────────────────────────────────

class OPLRIRegressor(nn.Module):
    """OPLRI 多模态骨干模型 — 生理信号编码器 + Gate A/B."""

    def __init__(self, encoder_name: str = "tcn",
                 use_gate_a: bool = True, use_gate_b: bool = True,
                 reg_head_input_dim: int = 137):
        super().__init__()
        self.dynamic_encoder = get_dynamic_encoder(encoder_name, in_channels=2)
        self.use_gate_a = bool(use_gate_a)
        self.use_gate_b = bool(use_gate_b)
        self.gate_a_linear = nn.Linear(9, 128)
        self.constitution_tokens = nn.Parameter(torch.randn(9, 128) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True
        )
        self.gate_b_linear = nn.Linear(128, 128)
        self.reg_head = nn.Sequential(
            nn.Linear(reg_head_input_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1),
        )

    def extract_dynamic(
        self,
        dynamic_x: torch.Tensor,
        tcm_probs_9d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        z_raw = self.dynamic_encoder(dynamic_x)
        attn_weights = None
        if self.use_gate_a:
            gate_a = torch.sigmoid(self.gate_a_linear(tcm_probs_9d))
            kv = self.constitution_tokens.unsqueeze(0) * tcm_probs_9d.unsqueeze(-1)
            query = z_raw.unsqueeze(1)
            attn_out, attn_weights = self.cross_attention(
                query, kv, kv, need_weights=True, average_attn_weights=False
            )
            weighted = torch.sigmoid(attn_out.squeeze(1))
            z_modulated = z_raw * ((1.0 - 1.0) + 1.0 * weighted)
        else:
            z_modulated = z_raw

        if self.use_gate_b:
            gate_b = torch.sigmoid(self.gate_b_linear(z_modulated))
            z_pure = z_modulated * (1.0 - 1.0 * gate_b)
        else:
            z_pure = z_modulated

        return z_pure.detach(), attn_weights

    def forward(
        self, dynamic_x: torch.Tensor, tcm_probs_9d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_pure, _ = self.extract_dynamic(dynamic_x, tcm_probs_9d)
        combined = torch.cat([z_pure, tcm_probs_9d], dim=-1)
        output = self.reg_head(combined)
        return output, z_pure


class TCMEncoder(nn.Module):
    """中医体质编码器 — FT-Transformer → 9-D 概率 + 128-D 特征."""

    def __init__(self, model_path: str, scaler_path: str, device: str = "cpu"):
        super().__init__()
        self.device = device

        scaler_data = np.load(scaler_path)
        raw_mean = scaler_data["mean"]
        raw_std = scaler_data["std"]
        n = min(4, len(raw_mean))
        self.mean = torch.tensor(raw_mean[:n], dtype=torch.float32).to(device)
        self.std = torch.tensor(raw_std[:n], dtype=torch.float32).to(device)

        self.model = get_model(n_features=4, n_classes=9)
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.feature_projection = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.1)
        ).to(device)

        self.best_params = ckpt.get("best_params", None)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        std_safe = torch.where(self.std < 1e-8, torch.ones_like(self.std), self.std)
        return (x - self.mean) / std_safe

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x_norm = self.normalize(x)
            tokens = self.model.feature_tokenizer(x_norm)
            bs = x.size(0)
            cls = self.model.cls_token(bs)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = self.model.dropout_layer(tokens)
            encoded = self.model.transformer_encoder(tokens)
            cls_out = self.model.layer_norm(encoded[:, 0, :])
            features = self.feature_projection(cls_out)
            probs = self.model(x_norm)
        return features, probs


class EEGEncoder(nn.Module):
    """脑电正念指数编码器 — 简化版 CNN → 8-D 神经表征.

    输入: (B, 1, L) 一维脑电波形
    输出: (B, 8) 神经状态向量
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveAvgPool1d(16),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.conv(x))


# ──────────────────────────────────────────────────────────────
# 模型管理器
# ──────────────────────────────────────────────────────────────

class ModelManager:
    """加载并管理所有预训练模型."""

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 路径
        self.oplri_path = LEGACY_ROOT / "checkpoints/oplri/server_f743da3/best_model.pth"
        self.tcm_path = LEGACY_ROOT / "checkpoints/tcm/server_f743da3/best_model.pth"
        self.tcm_scaler = LEGACY_ROOT / "checkpoints/tcm/server_f743da3/scaler_params_8d.npz"

        # 加载模型
        self._load_oplri()
        self._load_tcm()
        self._load_eeg()

        # 冻结所有参数
        for m in [self.oplri, self.tcm_encoder, self.eeg_encoder]:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

        print(f"[ModelManager] 所有模型已加载至 {self.device}")

    def _load_oplri(self):
        """加载 OPLRI 模型（从联合 checkpoint 中提取动态编码器和门控）."""
        from models.encoders import ResNet1DEncoder  # 导入正确的编码器

        # 加载联合 checkpoint 以检查维度
        ckpt = torch.load(str(self.oplri_path), map_location=self.device, weights_only=True)
        state_dict = ckpt["model_state_dict"]

        # 确定 reg_head 的输入维度
        reg_head_input_dim = state_dict["reg_head.0.weight"].shape[1]

        # 确定是否使用 resnet 编码器
        has_resnet = any("dynamic_encoder.stem" in k for k in state_dict.keys())

        # 创建模型
        encoder_name = "resnet" if has_resnet else "tcn"
        self.oplri = OPLRIRegressor(
            encoder_name=encoder_name,
            use_gate_a=True,
            use_gate_b=True,
            reg_head_input_dim=reg_head_input_dim,
        )

        # 提取动态编码器的 state_dict (去掉 'dynamic_encoder.' 前缀)
        dynamic_state = {
            k.replace("dynamic_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("dynamic_encoder.")
        }

        # 提取 Gate A
        gate_a_state = {
            k.replace("gate_a_linear.", ""): v
            for k, v in state_dict.items()
            if k.startswith("gate_a_linear.")
        }

        # 提取 Gate B
        gate_b_state = {
            k.replace("gate_b_linear.", ""): v
            for k, v in state_dict.items()
            if k.startswith("gate_b_linear.")
        }

        # 提取 reg_head
        reg_head_state = {
            k.replace("reg_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("reg_head.")
        }

        # 检查 constitution_tokens 是否在 checkpoint 中
        if "constitution_tokens" in state_dict:
            self.oplri.constitution_tokens.data = state_dict["constitution_tokens"].clone()

        # 加载各部分 (dynamic_encoder 使用 strict=False 以兼容不同编码器)
        self.oplri.dynamic_encoder.load_state_dict(dynamic_state, strict=False)
        self.oplri.gate_a_linear.load_state_dict(gate_a_state, strict=True)
        self.oplri.gate_b_linear.load_state_dict(gate_b_state, strict=True)
        self.oplri.reg_head.load_state_dict(reg_head_state, strict=True)
        self.oplri.to(self.device)

    def _load_tcm(self):
        self.tcm_encoder = TCMEncoder(
            model_path=str(self.tcm_path),
            scaler_path=str(self.tcm_scaler),
            device=str(self.device),
        )

    def _load_eeg(self):
        self.eeg_encoder = EEGEncoder(in_channels=1, embed_dim=8)
        self.eeg_encoder.to(self.device)

    def run_inference(self, sample: dict) -> dict:
        """对一条样本执行完整推理管线."""
        device = self.device

        # ── 准备输入张量 ──
        # ECG + EDA → [2, 1000]
        ecg = np.asarray(sample["ecg"], dtype=np.float32)
        eda = np.asarray(sample["eda"], dtype=np.float32)
        ecg_n = (ecg - ecg.mean()) / (ecg.std() + 1e-6)
        eda_n = (eda - eda.mean()) / (eda.std() + 1e-6)
        dyn_np = np.stack([ecg_n, eda_n]).astype(np.float32)
        dyn_t = torch.tensor(dyn_np, device=device).unsqueeze(0)

        # TCM 诊断 → [4]
        # 支持两种格式: 平铺 (sample["tongue"]) 或嵌套 (sample["tcm"]["tongue"])
        if "tcm" in sample and isinstance(sample["tcm"], dict):
            tcm_data = sample["tcm"]
        else:
            tcm_data = sample

        diag_t = torch.tensor(
            [tcm_data["tongue"], tcm_data["coating"], tcm_data["pulse"], tcm_data["face"]],
            dtype=torch.float32, device=device
        ).unsqueeze(0)

        # EEG → [1, 1000]
        eeg = np.asarray(sample["eeg"], dtype=np.float32)
        eeg_n = (eeg - eeg.mean()) / (eeg.std() + 1e-6)
        eeg_t = torch.tensor(eeg_n, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # ── 推理 ──
        with torch.no_grad():
            # TCM
            tcm_features, tcm_probs = self.tcm_encoder(diag_t)
            probs_np = tcm_probs.cpu().numpy()[0]
            features_np = tcm_features.cpu().numpy()[0]
            constitution_idx = int(np.argmax(probs_np))
            constitution_conf = float(probs_np[constitution_idx])

            # OPLRI 动态编码
            _, z_pure = self.oplri(dyn_t, tcm_probs)

            # EEG 编码
            eeg_repr = self.eeg_encoder(eeg_t)

        z_pure_np = z_pure.cpu().numpy()[0]
        eeg_np = eeg_repr.cpu().numpy()[0]

        # ── 方案推荐 ──
        recommendation = self._recommend(
            probs_np, constitution_idx, constitution_conf,
            sample["vitals"]["spo2"], sample["vitals"]["heart_rate"],
            sample["mindfulness"],
        )

        return {
            "constitution": {
                "name": CONSTITUTION_NAMES[constitution_idx],
                "index": constitution_idx,
                "confidence": round(constitution_conf, 4),
                "probabilities": {
                    name: round(float(p), 4)
                    for name, p in zip(CONSTITUTION_NAMES, probs_np)
                },
                "features_preview": features_np[:8].round(4).tolist(),
            },
            "dynamic_repr": {
                "preview": z_pure_np[:8].round(4).tolist(),
                "dim": 128,
                "gate_a": True,
                "gate_b": True,
            },
            "neuro_repr": {
                "preview": eeg_np.round(4).tolist(),
                "dim": 8,
                "mindfulness_score": round(float(sample["mindfulness"]), 3),
            },
            "recommendation": recommendation,
        }

    @staticmethod
    def _recommend(probs, c_idx, c_conf, spo2, hr, mindfulness) -> dict:
        constitution = CONSTITUTION_NAMES[c_idx]

        # 体质 → 按摩模式映射
        program_map = {
            0: "brain_fitness",     # 平和质 → 健脑强身
            1: "stress_relief",     # 气虚质 → 舒缓解压
            2: "tcm_massage",       # 阳虚质 → 中医推拿
            3: "brain_fitness",     # 阴虚质 → 健脑强身
            4: "thai_stretching",   # 痰湿质 → 泰式拉筋
            5: "thai_stretching",   # 湿热质 → 泰式拉筋
            6: "tcm_massage",       # 血瘀质 → 中医推拿
            7: "stress_relief",     # 气郁质 → 舒缓解压
            8: "brain_fitness",     # 特禀质 → 健脑强身
        }
        program = next(p for p in PROGRAM_CATALOG if p["id"] == program_map[c_idx])

        # 正念指数影响力度
        if mindfulness < 0.3:
            intensity_ix = 0   # 轻柔
        elif mindfulness < 0.6:
            intensity_ix = 1   # 舒适
        else:
            intensity_ix = 2   # 强劲

        # 血氧异常 → 降力度
        if spo2 < 94:
            intensity_ix = min(intensity_ix, 1)

        intensity = INTENSITY_LEVELS[intensity_ix]

        return {
            "program": program,
            "intensity": intensity,
            "constitution": constitution,
            "confidence": round(c_conf, 4),
        }


# ──────────────────────────────────────────────────────────────
# 全局模型单例
# ──────────────────────────────────────────────────────────────
_manager: ModelManager | None = None


def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
