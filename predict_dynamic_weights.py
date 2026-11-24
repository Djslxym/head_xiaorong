import torch
import torch.nn as nn
from pathlib import Path

class HeadImportancePredictor(nn.Module):
    def __init__(self, input_dim=7168, hidden1=1024,
                 hidden2=512, output_dim=28, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path("outputs/head_predictor/best_model.pt")
    feature_path = Path("outputs/vlm_features/test_features.pt")
    output_path = Path("outputs/head_predictor/test_predictions.pt")

    print("Loading checkpoint...")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    print("Loading test features...")
    data = torch.load(
        feature_path, map_location="cpu", weights_only=True
    )

    features = data["fused_features"].float()
    sample_ids = data["sample_ids"]

    # 使用训练时保存的标准化参数
    mean = checkpoint["feature_mean"].float()
    std = checkpoint["feature_std"].float()

    features = (features - mean) / std
    features = torch.nan_to_num(features)

    # 初始化网络
    model = HeadImportancePredictor(
        input_dim=checkpoint["input_dim"],
        hidden1=checkpoint["hidden1"],
        hidden2=checkpoint["hidden2"],
        output_dim=checkpoint["output_dim"],
        dropout=checkpoint["dropout"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 逐样本预测动态权重
    print("Predicting dynamic Head weights...")

    with torch.no_grad():
        logits = model(features.to(device))
        predictions = torch.softmax(logits, dim=-1).cpu()

    assert predictions.shape == (265, 28)
    assert torch.isfinite(predictions).all()

    # 保存预测结果
    package = {
        "sample_ids": sample_ids,
        "predicted_weights": predictions,
        "mean_baseline_weights": checkpoint["mean_target_weights"],
        "top1_heads": predictions.argmax(dim=1),
        "best_epoch": checkpoint["best_epoch"]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, output_path)

    print("\n========== Prediction Completed ==========")
    print("Samples:", predictions.shape[0])
    print("Heads:", predictions.shape[1])
    print("Prediction shape:", tuple(predictions.shape))
    print("First sample weights:", predictions[0])
    print("First 5 row sums:", predictions.sum(dim=1)[:5])
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
