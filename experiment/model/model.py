import torch
import torch.nn as nn
import torch.nn.functional as F


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


# =========================================================================
# 主控中心: 按摩椅双流融合网络 (MassageFusionNet)
# =========================================================================
class MassageFusionNet(nn.Module):
    def __init__(
        self, model_type="inception", num_classes=4, dyn_channels=2, static_dim=4
    ):
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
# 工厂函数: 根据配置返回模型
# =========================================================================
def get_model(model_type="inception", num_classes=4, dyn_channels=2, static_dim=4):
    """工厂函数：根据配置返回模型"""
    return MassageFusionNet(
        model_type=model_type,
        num_classes=num_classes,
        dyn_channels=dyn_channels,
        static_dim=static_dim,
    )


# =========================================================================
# 自测代码
# =========================================================================
if __name__ == "__main__":
    dummy_dyn = torch.randn(8, 2, 1000)
    dummy_stat = torch.randn(8, 4)

    for m_type in ["cnn", "lstm", "inception"]:
        print(f"\n--- Testing {m_type} ---")
        model = MassageFusionNet(model_type=m_type)
        output = model(dummy_dyn, dummy_stat)
        print(f"输出形状: {output.shape}")
        print("✅ 通过")
