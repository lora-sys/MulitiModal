from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from src.config import TCM_CHECKPOINT_PATH
from src.models.encoders import get_dynamic_encoder


class TCMEncoderAdapter(nn.Module):
    """Adapter over pre-trained FT-Transformer.

    Outputs:
      - static_embedding: [B, 128]
      - tcm_probs: [B, 9]
    """

    def __init__(self, checkpoint_path: Path | str = TCM_CHECKPOINT_PATH, freeze: bool = True):
        super().__init__()
        # Import existing project model directly.
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "tcm_ft_transformer"))
        from ft_transformer import get_model  # pylint: disable=import-error

        self.model = get_model(n_features=8, n_classes=9)
        self.static_proj = nn.Linear(9, 128)
        self._load_checkpoint(Path(checkpoint_path))

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def _xavier_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _load_checkpoint(self, ckpt: Path) -> None:
        if not ckpt.exists():
            warnings.warn(f"TCM checkpoint not found: {ckpt}. Use Xavier init fallback.")
            self._xavier_init()
            return
        try:
            loaded = torch.load(ckpt, map_location="cpu", weights_only=True)
            state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
            self.model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            warnings.warn(f"Failed to load TCM checkpoint ({exc}); fallback to Xavier init.")
            self._xavier_init()

    def forward(self, static_8d: torch.Tensor):
        probs = self.model(static_8d)  # [B, 9], softmax probs
        static_embedding = self.static_proj(probs)  # [B, 128]
        return static_embedding, probs


class BaselineSignalRegressor(nn.Module):
    """Exp1 baseline: no TCM, no gating."""

    def __init__(self, encoder_name: str = "inceptiontime"):
        super().__init__()
        self.dynamic_encoder = get_dynamic_encoder(encoder_name, in_channels=2)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, dynamic: torch.Tensor, static_8d: torch.Tensor | None = None):
        z = self.dynamic_encoder(dynamic)
        return self.head(z)


class DualGatingModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "inceptiontime",
        tcm_checkpoint_path: Path | str = TCM_CHECKPOINT_PATH,
        freeze_tcm: bool = True,
        use_gate_a: bool = True,
        use_gate_b: bool = True,
        use_tcm_encoder: bool = True,
    ):
        super().__init__()
        self.dynamic_encoder = get_dynamic_encoder(encoder_name, in_channels=2)
        self.use_gate_a = use_gate_a
        self.use_gate_b = use_gate_b
        self.use_tcm_encoder = use_tcm_encoder

        if use_tcm_encoder:
            self.tcm_encoder = TCMEncoderAdapter(tcm_checkpoint_path, freeze=freeze_tcm)
        else:
            self.tcm_encoder = None
            self.static_encoder = nn.Linear(8, 128)

        # strict dimensions required by spec
        self.gate_a_linear = nn.Linear(9, 128)
        self.gate_b_linear = nn.Linear(128, 128)

        self.reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def extract_fused_features(self, dynamic: torch.Tensor, static_8d: torch.Tensor) -> torch.Tensor:
        pressure_embedding = self.dynamic_encoder(dynamic)  # [B, 128]

        if self.use_tcm_encoder:
            static_embedding, tcm_probs = self.tcm_encoder(static_8d)
        else:
            static_embedding = self.static_encoder(static_8d)
            # dummy probabilities keep shape compatibility
            tcm_probs = torch.softmax(torch.zeros(static_embedding.shape[0], 9, device=static_embedding.device), dim=1)

        if self.use_gate_a:
            gate_a = torch.sigmoid(self.gate_a_linear(tcm_probs))  # [B, 128]
            modulated_pressure = pressure_embedding * gate_a
        else:
            modulated_pressure = pressure_embedding

        if self.use_gate_b:
            gate_b = torch.sigmoid(self.gate_b_linear(pressure_embedding))  # [B, 128]
            modulated_static = static_embedding * gate_b
        else:
            modulated_static = static_embedding

        return torch.cat([modulated_pressure, modulated_static], dim=1)  # [B, 256]

    def forward(self, dynamic: torch.Tensor, static_8d: torch.Tensor):
        fused = self.extract_fused_features(dynamic, static_8d)
        return self.reg_head(fused)
