"""
mistralai/Mistral-7B-v0.1 + LoRA 支持Loss/RMSE曲线可视化
"""
import os
import time
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_rmse(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()  # 修复点1：回归任务，直接squeeze
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    return {"rmse": rmse}

def preprocess_stock_data(csv_path, window_size=30):
    df = pd.read_csv(csv_path).sort_values("Date").reset_index(drop=True)
    assert df["Date"].is_monotonic_increasing
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[features].copy()

    data_list = []
    for i in range(len(df) - window_size):
        window = df.iloc[i:i+window_size][features].values          # 形状 (window_size, 5)
        next_close = df.iloc[i+window_size]["Close"]
        last_close = window[-1, 3]                                  # 窗口最后一天收盘价

        # 归一化：预测收益率
        target = (next_close - last_close) / last_close

        # 输入仍用自然语言，可后续再压缩
        text = ", ".join([
            f"Day {d+1}: O={r[0]:.2f}, H={r[1]:.2f}, L={r[2]:.2f}, C={r[3]:.2f}, V={int(r[4])}"
            for d, r in enumerate(window)
        ])
        prompt = f"Based on this stock data: [{text}], predict next day's return:"

        data_list.append({
            "input": prompt,
            "target": target,           # 现在是收益率，[-0.2, 0.2] 左右
            "last_close": last_close,   # 保留，用于反归一化
        })
    return Dataset.from_list(data_list)

class RegressionMistral(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.regressor = torch.nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None, last_close=None, return_pred_close=False):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = h.last_hidden_state[:, -1, :]
        logits = self.regressor(pooled).squeeze(-1)      # 预测收益率

        if return_pred_close and last_close is not None:
            # 反归一化
            pred_close = logits * last_close + last_close
        else:
            pred_close = logits

        loss = None
        if labels is not None:
            loss = torch.nn.functional.smooth_l1_loss(logits, labels.float())
        return {"loss": loss, "logits": pred_close}

def plot_metrics(training_logs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 解析日志
    train_steps, train_losses = [], []
    val_steps, val_losses, val_rmses = [], [], []

    for log in training_logs:
        if "loss" in log and "step" in log:          # 训练 loss
            train_losses.append(log["loss"])
            train_steps.append(log["step"])
        if "eval_loss" in log:                       # 验证 loss
            val_losses.append(log["eval_loss"])
        if "eval_rmse" in log:                       # 验证 RMSE
            val_rmses.append(log["eval_rmse"])

    # 验证步骤用 epoch 序数简单对齐（HuggingFace 默认按 epoch 评估）
    val_steps = np.linspace(0, max(train_steps) if train_steps else 1, len(val_losses))

    # 2. 画 Train Loss
    if train_losses:
        plt.figure()
        plt.plot(train_steps, train_losses, color="blue")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "train_loss.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # 3. 画 Val Loss
    if val_losses:
        plt.figure()
        plt.plot(val_steps, val_losses, color="red", marker="o")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "val_loss.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # 4. 画 Val RMSE
    if val_rmses:
        plt.figure()
        plt.plot(val_steps, val_rmses, color="green", marker="o")
        plt.xlabel("Steps")
        plt.ylabel("RMSE")
        plt.title("Validation RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "val_rmse.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Three separate plots saved to {output_dir}")

def main(args):
    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = preprocess_stock_data(args.data_path)
    train_size = int(0.8 * len(dataset))
    train_ds = dataset.select(range(train_size))
    eval_ds = dataset.select(range(train_size, len(dataset)))

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_func(examples):
        tok = tokenizer(examples["input"], 
                        truncation=True,
                        max_length=256, 
                        padding="max_length")
        tok["labels"] = examples["target"]  
        return tok

    train_ds = train_ds.map(tokenize_func, batched=True, num_proc=4)
    eval_ds = eval_ds.map(tokenize_func, batched=True, num_proc=4)
    train_ds = train_ds.with_format('torch')
    eval_ds = eval_ds.with_format('torch')


    model = RegressionMistral("mistralai/Mistral-7B-v0.1")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="FEATURE_EXTRACTION"
    )
    model.backbone = get_peft_model(model.backbone, peft_config)

    class RegTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            inputs["last_close"] = None
            out = model(**inputs, labels=labels)
            return (out["loss"], out) if return_outputs else out["loss"]
        
        def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
            last_close = inputs.pop("last_close", None)          # 拿出来
            with torch.no_grad():
                out = model(return_pred_close=True,              # 评估要反归一化
                            last_close=last_close,
                            **inputs)
            loss = out["loss"]
            logits = out["logits"]                               # 已经是收盘价
            labels = inputs["labels"] * last_close + last_close  # 也把标签反归一化
            return (loss, logits, labels)

    def collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.float32).to(input_ids.device)
        last_close  = torch.tensor([b["last_close"] for b in batch], dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": mask,
            "labels": labels,
            "last_close": last_close          # 新增
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rmse",
        greater_is_better=False,
        fp16=True,
        report_to="none",
        disable_tqdm=False
    )

    trainer = RegTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_rmse,
        data_collator=collate
    )

    trainer.train()
    plot_metrics(trainer.state.log_history, args.output_dir)

    os.makedirs(os.path.join(args.output_dir, "final"), exist_ok=True)
    #torch.save(model.state_dict(), os.path.join(args.output_dir, "final", "full_model.pt"))
    # 保存LoRA权重（仅适配器，几MB）
    model.backbone.save_pretrained(os.path.join(args.output_dir, "final", "lora"))

    # 单独保存regressor层参数（极小）
    torch.save(model.regressor.state_dict(), os.path.join(args.output_dir, "final", "regressor.pt"))

    # 不再保存full_model.pt（避免13G大文件）
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final", "tokenizer"))
    with open(os.path.join(args.output_dir, "final", "training_logs.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    final = trainer.evaluate()
    print("\nFinal validation — MSE:", round(final["eval_loss"], 4), "RMSE:", round(final["eval_rmse"], 4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='dataset/nvidia_stock_prices.csv')
    parser.add_argument("--output_dir", type=str, default="ckpt_normalize")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()
    main(args)
