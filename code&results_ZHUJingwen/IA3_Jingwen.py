import os
import math
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"  # 统一字体，提升图表美观度

# 导入 Hugging Face 相关库
from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2Config
from peft import IA3Config, get_peft_model, TaskType

# ============================
# 核心配置（仅改参数，其他全不变）
# ============================
BASE_MODEL_DIR = "/root/autodl-tmp/GP6102/codeZHUJingwen/gpt2"
TRAIN_CSV = "nvidia_stock_prices.csv"
OUTPUT_DIR = "ia3_stock_model"
RESULTS_DIR = "results_ia3"

# 仅修改以下参数（参考优质模型规律）
NUM_EPOCHS = 25  # 原25→18（优质模型通常早停在15-20轮）
BATCH_SIZE = 8  # 原32→16（优质模型常用批次，平衡速度与稳定）
WINDOW_SIZE = 30  # 原60→30（捕捉短期趋势更精准，易收敛）
TARGET_TOTAL_STEPS = 6500  # 原8000→5500（避免过拟合）
LEARNING_RATE = 1.2e-4  # 原8e-6→2.5e-5（加速收敛，优质模型常用1e-5~3e-5）
WEIGHT_DECAY = 0.01  # 原0.005→0.01（适中正则化）
EARLY_STOP_PATIENCE = 12  # 原30→15（及时早停，避免无效训练）
GRAD_CLIP = 0.8  # 原0.6→1.0（放宽裁剪，保留更多梯度）
SCALE_TO_TEST = 52  # 保留（你的核心优化，不变）

# 设备+显存优化（不变）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
print(f"Using device: {DEVICE}")
print(f"仅改参数：Batch={BATCH_SIZE} | LR={LEARNING_RATE} | Window={WINDOW_SIZE} | Steps={TARGET_TOTAL_STEPS}")


# ============================
# 数据集类（完全不变！和你最初的代码一致）
# ============================
class StockDataset(Dataset):
    def __init__(self, df, window=30, normalize=True, data_aug=True):
        closes = pd.to_numeric(df["Close"], errors="coerce").dropna().astype(float).tolist()
        self.normalize = normalize
        self.data_aug = data_aug

        closes = [p * SCALE_TO_TEST for p in closes]
        print(f"训练数据缩放后：均值≈{np.mean(closes):.2f}（匹配测试数据量级）")

        if self.data_aug and len(closes) > window + 10:
            noise = np.random.normal(0, 0.003, len(closes))
            closes = closes + noise

        if self.normalize:
            self.mean = np.mean(closes)
            self.std = np.std(closes)
            closes = (closes - self.mean) / self.std

        self.samples = []
        self.window = window
        for i in range(len(closes) - window - 1):
            window_vals = closes[i:i + window].tolist()
            target_price = closes[i + window]

            trend = "flat"
            if window_vals[-1] - window_vals[0] > 0.1:
                trend = "upward"
            elif window_vals[-1] - window_vals[0] < -0.1:
                trend = "downward"
            input_text = f"prices: {window_vals}, trend: {trend} -> next price:"

            self.samples.append((input_text, target_price))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def denormalize(self, normalized_price):
        if self.normalize:
            return normalized_price * self.std + self.mean
        return normalized_price


# ============================
# 构建 DataLoader（仅同步window和batch，其他不变）
# ============================
def build_dataloaders(csv_path, tokenizer, window=30, batch_size=16):
    df = pd.read_csv(csv_path)
    dataset = StockDataset(df, window=window, normalize=True, data_aug=True)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    def collate_fn(batch):
        batch_texts = [item[0] for item in batch]
        batch_prices = [item[1] for item in batch]
        enc = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        price_labels = torch.tensor(batch_prices, dtype=torch.float32).unsqueeze(1)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": price_labels
        }

    print(f"Data stats: Total={len(dataset)}, Train={train_size}, Val={val_size}")
    print(f"Config: Window={window}d | Batch={batch_size} | LR={LEARNING_RATE}")
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        dataset,
        len(dataset),
        train_size,
        val_size,
    )


# ============================
# 自定义模型（完全不变！和你最初的代码一致）
# ============================
class GPT2RegressionModel(nn.Module):
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.base_model = base_model
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        self.output_scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.output_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_3_hidden = outputs.hidden_states[-1][:, -3:, :]
        last_hidden = torch.mean(last_3_hidden, dim=1)
        pred_price = self.regression_head(last_hidden)
        pred_price = pred_price * self.output_scale + self.output_bias

        loss = None
        if labels is not None:
            huber_loss = nn.SmoothL1Loss(beta=1.0)
            loss = huber_loss(pred_price, labels)
        return {"pred_price": pred_price, "loss": loss}


# ============================
# 辅助函数：保存/加载模型（完全不变）
# ============================
def save_custom_model(model, tokenizer, dataset, save_path):
    os.makedirs(save_path, exist_ok=True)
    model.base_model.save_pretrained(os.path.join(save_path, "ia3_adapter"))
    torch.save(
        model.regression_head.state_dict(),
        os.path.join(save_path, "regression_head.pt")
    )
    torch.save(
        {"output_scale": model.output_scale, "output_bias": model.output_bias, "scale_to_test": SCALE_TO_TEST},
        os.path.join(save_path, "output_correction.pt")
    )
    torch.save({"mean": dataset.mean, "std": dataset.std}, os.path.join(save_path, "normalize_params.pt"))
    tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
    print(f"Model saved to: {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return (
            model, optimizer, scheduler,
            checkpoint["global_step"],
            checkpoint["best_val_rmse"],
            checkpoint["early_stop_count"],
            checkpoint["train_losses"],
            checkpoint["val_losses"],
            checkpoint["val_rmses"],
            checkpoint["val_steps"]
        )
    else:
        return (
            model, optimizer, scheduler,
            0, float("inf"), 0,
            [], [], [], []
        )


# ============================
# 辅助函数：单独保存每张图（完全不变）
# ============================
def plot_train_loss(train_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss (SmoothL1)", color="#1f77b4", linewidth=1.2)
    plt.title("Training Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Train loss plot saved to: {save_path}")


def plot_val_loss(val_steps, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(val_steps, val_losses, label="Val Loss (SmoothL1)", color="#ff7f0e", linewidth=2, marker="s", markersize=4)
    plt.title("Validation Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Val loss plot saved to: {save_path}")


def plot_val_rmse(val_steps, val_rmses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(val_steps, val_rmses, label="Val RMSE (Normalized)", color="#2ca02c", linewidth=2, marker="o",
             markersize=4)
    plt.title("Validation RMSE Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Val RMSE plot saved to: {save_path}")


# ============================
# 主训练函数（仅改参数相关，逻辑完全不变）
# ============================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(OUTPUT_DIR, "training_checkpoint.pt")

    # 加载 Tokenizer 和模型（不变）
    print("\n===== Step 1: Loading Tokenizer and GPT-2 =====")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained(
        BASE_MODEL_DIR,
        local_files_only=True,
        output_hidden_states=True
    ).to(DEVICE)

    # 配置 IA3（不变）
    print("\n===== Step 2: Configuring IA3 =====")
    ia3_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj", "c_fc"],
        feedforward_modules=["c_fc", "c_proj"]
    )
    base_model_with_ia3 = get_peft_model(base_model, ia3_config)
    base_model_with_ia3.print_trainable_parameters()

    # 初始化回归模型（不变）
    model = GPT2RegressionModel(base_model=base_model_with_ia3).to(DEVICE)
    print("Regression model initialized!")

    # 构建 DataLoader（仅同步参数）
    print("\n===== Step 3: Building DataLoaders =====")
    train_loader, val_loader, dataset, _, _, _ = build_dataloaders(
        TRAIN_CSV, tokenizer=tokenizer, window=WINDOW_SIZE, batch_size=BATCH_SIZE
    )

    # 优化器+调度器（仅改LR和调度器参数，逻辑不变）
    print("\n===== Step 4: Setting Up Optimizer =====")
    param_groups = [
        {"params": model.base_model.parameters(), "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        {"params": model.regression_head.parameters(), "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY / 2},
        {"params": [model.output_scale, model.output_bias], "lr": LEARNING_RATE / 2, "weight_decay": 0.0}
    ]
    optimizer = AdamW(param_groups)

    # 调度器仅改T_0（适配新步数）
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1000,  # 原1500→1000（适配5500步）
        T_mult=2,
        eta_min=1e-6
    )

    # 加载断点（不变）
    (model, optimizer, scheduler,
     global_step, best_val_rmse, early_stop_count,
     train_losses, val_losses, val_rmses, val_steps) = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path
    )

    # 训练循环（完全不变！和你最初的逻辑一致）
    print("\n===== Step 5: Starting Training =====")
    while global_step < TARGET_TOTAL_STEPS and early_stop_count < EARLY_STOP_PATIENCE:
        model.train()
        epoch_train_loss = 0.0
        epoch_train_steps = 0

        for batch in train_loader:
            if global_step >= TARGET_TOTAL_STEPS or early_stop_count >= EARLY_STOP_PATIENCE:
                break

            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            epoch_train_steps += 1
            global_step += 1
            train_losses.append(loss.item())

            # 每300步验证（不变）
            if global_step % 300 == 0:
                print(
                    f"\nStep {global_step:>4d} | Train Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

                # 验证阶段（不变）
                model.eval()
                total_val_loss = 0.0
                total_val_rmse_norm = 0.0
                total_val_rmse_real = 0.0
                val_batch_count = 0

                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = {k: v.to(DEVICE) for k, v in val_batch.items()}
                        val_outputs = model(**val_batch)
                        total_val_loss += val_outputs["loss"].item()

                        pred_norm = val_outputs["pred_price"]
                        true_norm = val_batch["labels"]
                        total_val_rmse_norm += torch.sqrt(torch.mean((pred_norm - true_norm) ** 2)).item()

                        pred_real = dataset.denormalize(pred_norm.cpu().numpy()) / SCALE_TO_TEST
                        true_real = dataset.denormalize(true_norm.cpu().numpy()) / SCALE_TO_TEST
                        total_val_rmse_real += np.sqrt(np.mean((pred_real - true_real) ** 2))

                        val_batch_count += 1

                avg_val_loss = total_val_loss / val_batch_count
                avg_val_rmse_norm = total_val_rmse_norm / val_batch_count
                avg_val_rmse_real = total_val_rmse_real / val_batch_count

                val_losses.append(avg_val_loss)
                val_rmses.append(avg_val_rmse_norm)
                val_steps.append(global_step)
                print(f"       | Val Loss:   {avg_val_loss:.4f}")
                print(f"       | Val RMSE (Norm): {avg_val_rmse_norm:.4f}")
                print(f"       | Val RMSE (Real): {avg_val_rmse_real:.2f}")
                print(f"       | Best RMSE (Norm): {best_val_rmse:.4f}")

                if avg_val_rmse_norm < best_val_rmse:
                    best_val_rmse = avg_val_rmse_norm
                    early_stop_count = 0
                    save_custom_model(model, tokenizer, dataset, os.path.join(OUTPUT_DIR, "best_model"))
                else:
                    early_stop_count += 1
                    print(f"       | Early stop count: {early_stop_count}/{EARLY_STOP_PATIENCE}")

                model.train()

            # 每500步保存断点（不变）
            if global_step % 500 == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "best_val_rmse": best_val_rmse,
                    "early_stop_count": early_stop_count,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_rmses": val_rmses,
                    "val_steps": val_steps,
                }, checkpoint_path)
                print(f"       | Checkpoint saved!")

        # 打印本轮总结（不变）
        if epoch_train_steps > 0:
            avg_epoch_loss = epoch_train_loss / epoch_train_steps
            print(f"\nEpoch Summary | Avg Train Loss: {avg_epoch_loss:.4f} | Global Step: {global_step}")

    # 训练结束（不变）
    print("\n===== Training Finished! =====")
    print(f"Final Stats: Steps={global_step}, Best RMSE (Norm)={best_val_rmse:.4f}")
    save_custom_model(model, tokenizer, dataset, os.path.join(OUTPUT_DIR, "final_model"))

    # 保存图表和指标（不变）
    plot_train_loss(train_losses, os.path.join(RESULTS_DIR, "train_loss.png"))
    plot_val_loss(val_steps, val_losses, os.path.join(RESULTS_DIR, "val_loss.png"))
    plot_val_rmse(val_steps, val_rmses, os.path.join(RESULTS_DIR, "val_rmse.png"))

    metrics_df = pd.DataFrame({
        "train_step": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_step": val_steps + [None] * (len(train_losses) - len(val_steps)),
        "val_loss": val_losses + [None] * (len(train_losses) - len(val_losses)),
        "val_rmse": val_rmses + [None] * (len(train_losses) - len(val_rmses))
    })
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "training_metrics.csv"), index=False)
    print(f"Metrics data saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
