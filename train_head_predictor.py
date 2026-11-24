import os
import json
import random
import argparse
from pathlib import Path
from copy import deepcopy

# ============================================================
# 1.  参数配置
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MLP to predict 28 Query Head importance weights."
    )

    parser.add_argument("--gpus", type=str, default="2")
    parser.add_argument(
        "--train_features",
        type=str,
        default="outputs/vlm_features/train_features.pt"
    )
    parser.add_argument(
        "--test_features",
        type=str,
        default="outputs/vlm_features/test_features.pt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/head_predictor"
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


args = parse_args()

# 必须在导入 torch 之前设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 2. 固定随机种子
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 3. MLP 网络
# ============================================================

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
        logits = self.network(x)
        return logits


# ============================================================
# 4. 数据读取与检查
# ============================================================

def load_data(device):

    train_path = Path(args.train_features)
    test_path = Path(args.test_features)

    if not train_path.exists():
        raise FileNotFoundError(f"训练特征不存在：{train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"测试特征不存在：{test_path}")

    print("\n========== Loading Feature Files ==========")

    train_data = torch.load(
        train_path,
        map_location="cpu",
        weights_only=True
    )

    test_data = torch.load(
        test_path,
        map_location="cpu",
        weights_only=True
    )

    train_x = train_data["fused_features"].float()
    train_y = train_data["target_weights"].float()
    test_x = test_data["fused_features"].float()

    train_ids = train_data["sample_ids"]
    test_ids = test_data["sample_ids"]

    print("Train features:", tuple(train_x.shape))
    print("Train labels:", tuple(train_y.shape))
    print("Test features:", tuple(test_x.shape))

    # 检查数据形状
    assert train_x.shape == (500, 7168)
    assert train_y.shape == (500, 28)
    assert test_x.shape == (265, 7168)

    # 检查训练集和测试集不重叠
    assert set(train_ids.tolist()).isdisjoint(
        set(test_ids.tolist())
    )

    # 检查数值有效性
    assert torch.isfinite(train_x).all()
    assert torch.isfinite(train_y).all()
    assert torch.isfinite(test_x).all()

    # 标签必须是非负分布
    assert (train_y >= 0).all()

    # 每条标签归一化为概率分布
    target_sums = train_y.sum(dim=1, keepdim=True)

    assert (target_sums > 0).all()

    train_y = train_y / target_sums

    print("\nData verification PASSED.")

    # ========================================================
    # 只使用训练集计算特征标准化参数
    # 不使用测试集统计量，避免数据泄漏
    # ========================================================

    feature_mean = train_x.mean(dim=0, keepdim=True)
    feature_std = train_x.std(
        dim=0,
        unbiased=False,
        keepdim=True
    )

    feature_std = feature_std.clamp(min=1e-6)

    train_x = (train_x - feature_mean) / feature_std
    test_x = (test_x - feature_mean) / feature_std

    train_x = torch.nan_to_num(train_x)
    test_x = torch.nan_to_num(test_x)

    print("Feature normalization completed.")

    return (
        train_x,
        train_y,
        test_x,
        train_ids,
        test_ids,
        feature_mean,
        feature_std
    )


# ============================================================
# 5. 训练一个 Epoch
# ============================================================

def train_one_epoch(model, loader, optimizer, device):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(batch_x)

        log_probs = F.log_softmax(logits, dim=-1)

        # KL(P_target || P_prediction)
        loss = F.kl_div(
            log_probs,
            batch_y,
            reduction="batchmean"
        )

        loss.backward()

        optimizer.step()

        batch_size = batch_x.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# 6. 预测函数
# ============================================================

@torch.no_grad()
def predict(model, features, device):

    model.eval()

    features = features.to(device)

    logits = model(features)

    probabilities = torch.softmax(logits, dim=-1)

    return probabilities.cpu()


# ============================================================
# 7. 主训练流程
# ============================================================

def main():

    set_seed(args.seed)

    print("========== GPU Configuration ==========")
    print("Physical GPU:", args.gpus)
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 GPU 环境。")

    device = torch.device("cuda:0")

    print("Training device:", torch.cuda.get_device_name(0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    (
        train_x,
        train_y,
        test_x,
        train_ids,
        test_ids,
        feature_mean,
        feature_std
    ) = load_data(device)

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataset = TensorDataset(train_x, train_y)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    # --------------------------------------------------------
    # 初始化网络
    # --------------------------------------------------------

    model = HeadImportancePredictor(
        input_dim=7168,
        hidden1=1024,
        hidden2=512,
        output_dim=28,
        dropout=args.dropout
    ).to(device)

    total_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("\n========== Model ==========")
    print(model)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # 全局平均标签：作为后续实验的固定权重基线
    mean_target_weights = train_y.mean(dim=0)
    mean_target_weights = (
        mean_target_weights / mean_target_weights.sum()
    )

    # --------------------------------------------------------
    # 正式训练
    # --------------------------------------------------------

    print("\n========== Start Training ==========")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Dropout: {args.dropout}")
    print("Training samples: 500")
    print("Validation split: None")
    print("Test samples are NOT used during training.")

    best_train_loss = float("inf")
    best_epoch = -1
    best_state = None

    training_log = []

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            device
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_kl_loss": float(train_loss),
            "learning_rate": float(current_lr)
        }

        training_log.append(record)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch [{epoch:03d}/{args.epochs}] "
                f"Train KL Loss: {train_loss:.8f} "
                f"Best: {best_train_loss:.8f} "
                f"LR: {current_lr:.8e}"
            )

    if best_state is None:
        raise RuntimeError("训练失败，没有得到有效模型参数。")

    # --------------------------------------------------------
    # 保存训练过程中训练 Loss 最低的模型
    # 注意：这是训练 Loss 最低的 checkpoint，不是验证集最优
    # --------------------------------------------------------

    model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "mean_target_weights": mean_target_weights,
        "input_dim": 7168,
        "hidden1": 1024,
        "hidden2": 512,
        "output_dim": 28,
        "dropout": args.dropout,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "seed": args.seed
    }

    model_path = output_dir / "best_model.pt"
    torch.save(checkpoint, model_path)

    print("\n========== Training Completed ==========")
    print("Best epoch:", best_epoch)
    print("Best training KL loss:", best_train_loss)
    print("Model saved:", model_path)

    # --------------------------------------------------------
    # 保存训练日志
    # --------------------------------------------------------

    log_path = output_dir / "training_log.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "seed": args.seed
                },
                "best_epoch": best_epoch,
                "best_train_loss": best_train_loss,
                "history": training_log
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Training log saved:", log_path)

    # --------------------------------------------------------
    # 对265条独立测试数据进行预测
    # --------------------------------------------------------

    print("\n========== Testing on 265 Samples ==========")

    predictions = predict(
        model,
        test_x,
        device
    )

    assert predictions.shape == (265, 28)
    assert torch.isfinite(predictions).all()

    assert torch.allclose(
        predictions.sum(dim=1),
        torch.ones(265),
        atol=1e-5
    )

    # 每条测试样本的 Top-1 Head
    top1_heads = predictions.argmax(dim=1)

    # 预测分布的平均熵
    entropy = -(
        predictions * predictions.clamp(min=1e-12).log()
    ).sum(dim=1)

    # 生成全局平均权重基线
    mean_baseline = mean_target_weights.unsqueeze(0).repeat(
        predictions.shape[0], 1
    )

    prediction_package = {
        "sample_ids": test_ids,
        "predicted_weights": predictions,
        "mean_baseline_weights": mean_baseline,
        "top1_heads": top1_heads,
        "entropy": entropy,
        "best_epoch": best_epoch
    }

    prediction_path = output_dir / "test_predictions.pt"

    torch.save(
        prediction_package,
        prediction_path
    )

    print("Prediction shape:", tuple(predictions.shape))
    print("Prediction row sum:", predictions.sum(dim=1)[:5])
    print("Top-1 Head examples:", top1_heads[:20].tolist())
    print("Mean prediction entropy:", entropy.mean().item())

    print("Predictions saved:", prediction_path)

    # --------------------------------------------------------
    # 输出各 Head 的平均预测权重
    # --------------------------------------------------------

    mean_predicted_weights = predictions.mean(dim=0)

    print("\n========== Mean Predicted Head Weights ==========")

    sorted_heads = torch.argsort(
        mean_predicted_weights,
        descending=True
    )

    for head_id in sorted_heads.tolist():
        print(
            f"Head {head_id:02d}: "
            f"{mean_predicted_weights[head_id].item():.6f}"
        )

    print("\n========== ALL TASKS COMPLETED ==========")
    print("Model:", model_path)
    print("Log:", log_path)
    print("Predictions:", prediction_path)


if __name__ == "__main__":
    main()
