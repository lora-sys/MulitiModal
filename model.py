from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import get_dynamic_encoder


class OPLRIRegressor(nn.Module):
    """
    OP-LRI style model core (dynamic-only in model body).

    Iron rules implemented here:
    - Model body only handles dynamic stream and gates.
    - Static 4D -> scaler -> frozen FT-Transformer -> 9D probs is done in script layer.
    - Late reinjection is also done in script layer.
    """

    def __init__(
        self,
        encoder_name: str = "tcn",
        use_gate_a: bool = True,
        use_gate_b: bool = True,
    ) -> None:
        super().__init__()
        self.dynamic_encoder = get_dynamic_encoder(encoder_name, in_channels=2)  # z_raw: [B, 128]
        self.use_gate_a = bool(use_gate_a)
        self.use_gate_b = bool(use_gate_b)

        # Gate A: 9D TCM probability -> 128D channel gate
        self.gate_a_linear = nn.Linear(9, 128)
        # Gate B: dynamic disentanglement gate in dynamic feature space
        self.gate_b_linear = nn.Linear(128, 128)

        # Late reinjection input dim = 128 (pure dynamic) + 9 (TCM probs)
        self.reg_head = nn.Sequential(
            nn.Linear(137, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def extract_pure_dynamic(self, dynamic_x: torch.Tensor, tcm_probs_9d: torch.Tensor) -> torch.Tensor:
        """
        Dynamic flow:
          z_raw -> Gate A modulated -> Gate B disentangled -> detach
        """
        z_raw = self.dynamic_encoder(dynamic_x)  # [B, 128]

        if self.use_gate_a:
            gate_a = torch.sigmoid(self.gate_a_linear(tcm_probs_9d))  # [B, 128]
            z_modulated = z_raw * gate_a
        else:
            z_modulated = z_raw

        if self.use_gate_b:
            gate_b = torch.sigmoid(self.gate_b_linear(z_modulated))  # [B, 128]
            # Suppress static-correlated shortcut channels
            z_pure_dynamic = z_modulated * (1.0 - gate_b)
        else:
            z_pure_dynamic = z_modulated

        # Core red line: stop gradients to dynamic trunk / gates to avoid pollution.
        return z_pure_dynamic.detach()

    def forward_from_final_input(self, final_input: torch.Tensor) -> torch.Tensor:
        return self.reg_head(final_input)

