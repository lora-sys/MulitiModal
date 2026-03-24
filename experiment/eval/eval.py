from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import sys

import torch

sys.path.append("experiment/dataset")
sys.path.append("experiment/model")
from csv_source import NPZDataSource
from nk2_processor import NK2Preprocessor
from massage_dataset import MassageDataset
from model import get_model
import yaml

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "experiment/model/best_model_lstm.pth"  # 改为实际路径
MODEL_TYPE = "lstm"  # 根据选择的模型修改


def evaluate_in_detail():
    # 加载配置
    with open("experiment/dataset/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 创建数据集
    npz_path = "experiment/model/processed_data.npz"
    source = NPZDataSource(npz_path)
    source.initialize()
    preprocessor = NK2Preprocessor(config)
    dataset = MassageDataset(source, preprocessor)

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # 加载模型
    model = get_model(model_type=MODEL_TYPE, num_classes=3, dyn_channels=2, static_dim=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    all_preds = []
    all_labels = []

    print("🚀 开始评估...")
    with torch.no_grad():
        for batch in loader:
            dyn = batch["dynamic"].to(DEVICE)
            stat = batch["static"].to(DEVICE)
            outputs = model(dyn, stat)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    # 混淆矩阵 - 英文标签
    cm = confusion_matrix(all_labels, all_preds)
    classes_en = ["Very Bad(0)", "Bad(1)", "Normal(2)", "Good(3)"]
    classes_cn = ["很差(0)", "一般(1)", "正常(2)", "良好(3)"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes_en,
        yticklabels=classes_en,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{MODEL_TYPE.upper()} Confusion Matrix")
    plt.savefig("experiment/test/result/confusion_matrix_lstm.png")
    plt.close()

    print("\n📝 分类报告 (Classification Report):")
    print(classification_report(all_labels, all_preds, target_names=classes_cn))


if __name__ == "__main__":
    evaluate_in_detail()
