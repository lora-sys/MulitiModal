from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import TCM_CHECKPOINT_PATH, TCM_SCALER_PATH
from src.models.encoders import get_dynamic_encoder


class TCMEncoderAdapter(nn.Module):
    """Adapter over pre-trained FT-Transformer internal features."""

    def __init__(
        self,
        checkpoint_path: Path | str = TCM_CHECKPOINT_PATH,
        scaler_path: Path | str = TCM_SCALER_PATH,
        freeze: bool = True,
    ):
        super().__init__()
        # Import existing project model directly.
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "tcm_ft_transformer"))
        from ft_transformer import get_model  # pylint: disable=import-error

        self.model = get_model(n_features=4, n_classes=9)
        self.static_proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
        )
        self.scaler = self._load_scaler(Path(scaler_path), Path(checkpoint_path))
        self._load_checkpoint(Path(checkpoint_path))

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        self.model.eval()

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

    def _load_scaler(self, scaler_path: Path, checkpoint_path: Path):
        candidate_paths = [scaler_path]
        sibling_default = checkpoint_path.parent / "tcm_scaler.pkl"
        if sibling_default not in candidate_paths:
            candidate_paths.append(sibling_default)
        for path in candidate_paths:
            if not path.exists():
                continue
            # Preferred: sklearn StandardScaler pickled by joblib/pickle.
            try:
                import joblib  # type: ignore

                return joblib.load(path)
            except Exception:
                pass
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
            # Compatibility: npz mean/std fallback.
            try:
                data = np.load(path)
                if "mean" in data and "std" in data:
                    mean = data["mean"].astype(np.float32)
                    std = data["std"].astype(np.float32)
                    std = np.where(std == 0, 1.0, std).astype(np.float32)
                    return {"mean": mean, "std": std}
            except Exception:
                pass
        warnings.warn(f"TCM scaler not found or invalid ({scaler_path}); use identity scaling fallback.")
        return None

    def _scale_static(self, static_4d: torch.Tensor) -> torch.Tensor:
        # 铁律：特征顺序必须严格为 [Age, Gender, BMI, Heart Rate]
        x = static_4d.detach().to("cpu").numpy().astype(np.float32)
        if self.scaler is None:
            scaled = x
        elif isinstance(self.scaler, dict) and "mean" in self.scaler and "std" in self.scaler:
            mean = np.asarray(self.scaler["mean"], dtype=np.float32)
            std = np.asarray(self.scaler["std"], dtype=np.float32)
            std = np.where(std == 0, 1.0, std)
            scaled = (x - mean) / std
        else:
            scaled = self.scaler.transform(x)
        return torch.from_numpy(np.asarray(scaled, dtype=np.float32)).to(static_4d.device)

    def _extract_cls(self, scaled_static_4d: torch.Tensor) -> torch.Tensor:
        batch_size = scaled_static_4d.size(0)
        tokens = self.model.feature_tokenizer(scaled_static_4d)  # [B, 4, d_token]
        cls_token = self.model.cls_token(batch_size)  # [B, 1, d_token]
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = self.model.dropout_layer(tokens)
        encoded = self.model.transformer_encoder(tokens)
        cls_encoded = encoded[:, 0, :]
        return self.model.layer_norm(cls_encoded)

    def extract_features_and_probs(self, static_4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Accept wider static vectors for backward compatibility (e.g. BUT/MIMIC 8-d),
        # while enforcing the TCM 4-d input contract.
        if static_4d.dim() != 2:
            raise ValueError(f"static features must be 2D [B, F], got {tuple(static_4d.shape)}")
        if static_4d.size(1) < 4:
            raise ValueError(f"static features must contain at least 4 dims, got {static_4d.size(1)}")
        x4 = static_4d[:, :4]
        x_scaled = self._scale_static(x4)

        # 铁律：TCM 前向全程 no_grad，禁止主任务反向更新 TCM 分支
        with torch.no_grad():
            probs = self.model(x_scaled)  # [B, 9]
            cls_encoded = self._extract_cls(x_scaled)  # [B, 64]
        internal = self.static_proj(cls_encoded)  # [B, 128] (trainable projection head)
        return internal, probs

    def get_tcm_internal_features(self, static_4d: torch.Tensor) -> torch.Tensor:
        internal, _ = self.extract_features_and_probs(static_4d)
        return internal

    def forward(self, static_4d: torch.Tensor):
        return self.get_tcm_internal_features(static_4d)

    def train(self, mode: bool = True):
        super().train(mode)
        # 铁律：TCM 先验始终 eval
        self.model.eval()
        return self


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

    def forward(self, dynamic: torch.Tensor, static_4d: torch.Tensor | None = None):
        z = self.dynamic_encoder(dynamic)
        return self.head(z)


class DualGatingModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "inceptiontime",
        tcm_checkpoint_path: Path | str = TCM_CHECKPOINT_PATH,
        tcm_scaler_path: Path | str = TCM_SCALER_PATH,
        freeze_tcm: bool = True,
        use_tcm: bool = True,
        use_gate_a: bool = True,
        use_gate_b: bool = True,
        use_tcm_encoder: bool | None = None,
    ):
        super().__init__()
        self.dynamic_encoder = get_dynamic_encoder(encoder_name, in_channels=2)
        # Backward compatibility for existing callsites.
        if use_tcm_encoder is not None:
            use_tcm = bool(use_tcm_encoder)
        self.use_tcm = use_tcm
        # Gate A is undefined without TCM internal features.
        self.use_gate_a = bool(use_gate_a) if self.use_tcm else False
        self.use_gate_b = bool(use_gate_b)

        if self.use_tcm:
            self.tcm_encoder = TCMEncoderAdapter(
                checkpoint_path=tcm_checkpoint_path,
                scaler_path=tcm_scaler_path,
                freeze=freeze_tcm,
            )
            self.static_proj = None
        else:
            self.tcm_encoder = None
            self.static_proj = nn.Linear(4, 128)

        # Gate A consumes 9-dim TCM prior probabilities.
        self.gate_a_linear = nn.Linear(9, 128)
        self.gate_b_linear = nn.Linear(128, 128)
        self.fusion_norm = nn.LayerNorm(256)

        self.reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def extract_fused_features(self, dynamic: torch.Tensor, static_4d: torch.Tensor) -> torch.Tensor:
        pressure_embedding = self.dynamic_encoder(dynamic)  # [B, 128]

        if self.use_tcm:
            tcm_internal, tcm_probs = self.tcm_encoder.extract_features_and_probs(static_4d)
            static_embedding = tcm_internal
        else:
            static_embedding = self.static_proj(static_4d[:, :4] if static_4d.size(1) > 4 else static_4d)
            tcm_internal = None
            tcm_probs = None

        if self.use_gate_a:
            if tcm_probs is None:
                raise RuntimeError("Gate A requires TCM probability prior, but TCM is disabled.")
            gate_a = torch.sigmoid(self.gate_a_linear(tcm_probs))  # [B, 128]
            modulated_pressure = pressure_embedding * gate_a
        else:
            modulated_pressure = pressure_embedding

        if self.use_gate_b:
            gate_b = torch.sigmoid(self.gate_b_linear(pressure_embedding))  # [B, 128]
            modulated_static = static_embedding * gate_b
        else:
            modulated_static = static_embedding

        fused = torch.cat([modulated_pressure, modulated_static], dim=1)  # [B, 256]
        return self.fusion_norm(fused)

    def get_tcm_internal_features(self, static_4d: torch.Tensor) -> torch.Tensor:
        if not self.use_tcm or self.tcm_encoder is None:
            raise RuntimeError("TCM branch disabled; internal features unavailable.")
        return self.tcm_encoder.get_tcm_internal_features(static_4d)

    def forward(self, dynamic: torch.Tensor, static_4d: torch.Tensor):
        fused = self.extract_fused_features(dynamic, static_4d)
        return self.reg_head(fused)
