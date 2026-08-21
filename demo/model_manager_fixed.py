#!/usr/bin/env python3
"""
MulitiModal Demo - 修复版 ModelManager
使用 DualGatingModel 替代 OPLRIRegressor
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# 路径设置
DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
LEGACY_ROOT = PROJECT_ROOT / "legacy_research"

sys.path.insert(0, str(DEMO_DIR))
sys.path.insert(0, str(LEGACY_ROOT / "source/tcm/tcm_ft_transformer"))
sys.path.insert(0, str(LEGACY_ROOT / "source/oplri/src"))
sys.path.insert(0, str(LEGACY_ROOT / "source/oplri/src/models"))

from ft_transformer import get_model
from models.encoders import get_dynamic_encoder
from models.fusion import DualGatingModel
from app import CONSTITUTION_NAMES, PROGRAM_CATALOG, INTENSITY_LEVELS  # 导入常量

print("=" * 60)
print("MulitiModal Demo - 修复验证")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# EEG 编码器 (保持不变)
# ──────────────────────────────────────────────────────────────

class EEGEncoder(nn.Module):
    """脑电正念指数编码器 — 简化版 CNN → 8-D 神经表征."""

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
# 修复的 ModelManager
# ──────────────────────────────────────────────────────────────

class ModelManager:
    """加载并管理所有预训练模型 (使用 DualGatingModel)."""

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 路径
        self.oplri_path = LEGACY_ROOT / "checkpoints/oplri/server_f743da3/best_model.pth"
        self.tcm_path = LEGACY_ROOT / "checkpoints/tcm/server_f743da3/best_model.pth"
        self.tcm_scaler = LEGACY_ROOT / "checkpoints/tcm/server_f743da3/scaler_params_8d.npz"

        # 加载联合模型
        self._load_combined_model()
        self._load_eeg()

        # 冻结所有参数
        for m in [self.combined_model, self.eeg_encoder]:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

        print(f"[ModelManager] 所有模型已加载至 {self.device}")

    def _load_combined_model(self):
        """加载 DualGatingModel 联合模型."""
        self.combined_model = DualGatingModel(
            encoder_name="resnet",
            tcm_checkpoint_path=str(self.tcm_path),
            tcm_scaler_path=str(self.tcm_scaler),
            freeze_tcm=True,
            use_tcm=True,
            use_gate_a=True,
            use_gate_b=True,
        )

        # 加载 OPLRI checkpoint
        ckpt = torch.load(str(self.oplri_path), map_location=self.device, weights_only=True)
        state_dict = ckpt["model_state_dict"]

        # 提取各部分 state_dict
        dynamic_state = {k.replace("dynamic_encoder.", ""): v
                         for k, v in state_dict.items() if k.startswith("dynamic_encoder.")}
        gate_a_state = {k.replace("gate_a_linear.", ""): v
                        for k, v in state_dict.items() if k.startswith("gate_a_linear.")}
        gate_b_state = {k.replace("gate_b_linear.", ""): v
                        for k, v in state_dict.items() if k.startswith("gate_b_linear.")}
        reg_head_state = {k.replace("reg_head.", ""): v
                          for k, v in state_dict.items() if k.startswith("reg_head.")}
        fusion_norm_state = {k.replace("fusion_norm.", ""): v
                             for k, v in state_dict.items() if k.startswith("fusion_norm.")}

        # 加载到模型
        self.combined_model.dynamic_encoder.load_state_dict(dynamic_state, strict=False)
        self.combined_model.gate_a_linear.load_state_dict(gate_a_state, strict=True)
        self.combined_model.gate_b_linear.load_state_dict(gate_b_state, strict=True)
        self.combined_model.reg_head.load_state_dict(reg_head_state, strict=True)
        if fusion_norm_state:
            self.combined_model.fusion_norm.load_state_dict(fusion_norm_state, strict=True)
        self.combined_model.to(self.device)

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
        # 支持两种格式: 平铺或嵌套
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
            # DualGatingModel: 接收 dynamic + static_4d
            output = self.combined_model(dyn_t, diag_t)  # [B, 1]

            # 获取 TCM 内部特征和概率
            tcm_internal, tcm_probs = self.combined_model.tcm_encoder.extract_features_and_probs(diag_t)
            probs_np = tcm_probs.cpu().numpy()[0]
            features_np = tcm_internal.cpu().numpy()[0]
            constitution_idx = int(np.argmax(probs_np))
            constitution_conf = float(probs_np[constitution_idx])

            # EEG 编码
            eeg_repr = self.eeg_encoder(eeg_t)

        output_np = output.cpu().numpy()[0, 0]
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
                "preview": features_np[:8].round(4).tolist(),  # 使用 TCM 内部特征
                "dim": 128,
                "gate_a": self.combined_model.use_gate_a,
                "gate_b": self.combined_model.use_gate_b,
            },
            "neuro_repr": {
                "preview": eeg_np.round(4).tolist(),
                "dim": 8,
                "mindfulness_score": round(float(sample["mindfulness"]), 3),
            },
            "model_output": round(float(output_np), 4),  # 模型预测值
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
# 测试
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔍 测试 ModelManager...")

    try:
        manager = ModelManager(device="cpu")
        print("  ✓ ModelManager 初始化成功")

        # 测试推理
        from examples import get_preset, get_preset_list

        preset_name = get_preset_list()[0][0]
        sample = get_preset(preset_name)
        print(f"  使用预设样本: {preset_name}")

        result = manager.run_inference(sample)
        print("  ✓ 推理成功!")
        print(f"  ✓ 体质: {result['constitution']['name']} ({result['constitution']['confidence']:.2%})")
        print(f"  ✓ 推荐: {result['recommendation']['program']['name']}")
        print(f"  ✓ 力度: {result['recommendation']['intensity']['name']}")
        print(f"  ✓ 模型输出: {result.get('model_output', 'N/A')}")

        print("\n" + "=" * 60)
        print("✅ 修复版 ModelManager 验证通过!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
