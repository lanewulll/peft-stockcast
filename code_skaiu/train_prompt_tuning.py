# train_prompt_tuning.py
# Prompt Tuning 训练脚本：生成 Training Loss / Validation Loss / Validation RMSE 三张图

import os
import re
import math
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, AutoTokenizer

# ============================
# 配置
# ============================

BASE_MODEL_DIR = r"C:\Users\admin\models\gpt2"    # 你的 GPT-2 本地目录
TRAIN_CSV = "nvidia_stock_prices.csv"
OUTPUT_DIR = "prompt_tuned_model"
RESULTS_DIR = "results_prompt"

WINDOW_SIZE = 30
MAX_LEN = 256
NUM_VIRTUAL_TOKENS = 20       # prompt tuning: 虚拟 token 数量

BATCH_SIZE = 8
NUM_EPOCHS = 6
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# 数据集
# ============================

class StockDataset(Dataset):
    """
    把过去 WINDOW_SIZE 天 closing price 拼成一句话：
    "prices: [....] -> next: 123.45"
    用 LM 方式训练，让模型学会在这种句尾输出数字。
    """
    def __init__(self, df, window=30):
        self.window = window
        self.samples = []

        closes = df["Close"].astype(float).tolist()
        for i in range(len(closes) - window - 1):
            window_vals = closes[i:i + window]
            target = closes[i + window]
            text = f"prices: {window_vals} -> next: {target:.4f}"
            self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================
# Prompt Encoder
# ============================

class PromptEncoder(nn.Module):
    """
    学习一段长度为 NUM_VIRTUAL_TOKENS 的连续 prompt 向量
    形状: [V, H]
    """
    def __init__(self, hidden_size, num_virtual_tokens):
        super().__init__()
        self.embeds = nn.Parameter(torch.randn(num_virtual_tokens, hidden_size))

    def forward(self):
        return self.embeds   # [V, H]


# ============================
# 主训练函数
# ============================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Step 1: loading tokenizer and GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_DIR, local_files_only=True).to(DEVICE)
    hidden = model.config.hidden_size

    # 冻结 GPT-2 参数
    for p in model.parameters():
        p.requires_grad = False

    print("Step 1 done.")

    print("Step 2: building prompt encoder...")
    prompt_encoder = PromptEncoder(hidden, NUM_VIRTUAL_TOKENS).to(DEVICE)

    # optimizer 只优化 Prompt Embedding
    optimizer = AdamW(prompt_encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print("Step 2 done.")

    print("Step 3: loading CSV...")
    df = pd.read_csv(TRAIN_CSV)

    dataset = StockDataset(df, window=WINDOW_SIZE)
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    print(f"Train = {train_size}, Val = {val_size}")
    print("Step 3 done.")

    # ============================
    # 训练
    # ============================

    global_step = 0
    train_losses = []
    val_losses = []
    val_rmses = []
    val_steps_plot = []

    print("Step 4: training...")

    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            texts = list(batch)

            # -------- 编码文本 --------
            encoded = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(DEVICE)

            input_ids = encoded["input_ids"]           # [B, L]
            labels = input_ids.clone()                 # [B, L]

            # -------- Prompt Embeddings --------
            prompt_embeds = prompt_encoder()           # [V, H]
            prompt_embeds = prompt_embeds.unsqueeze(0).repeat(input_ids.size(0), 1, 1)  # [B, V, H]

            # -------- GPT-2 原始 embedding --------
            gpt_embeds = model.transformer.wte(input_ids)  # [B, L, H]

            # 拼接：prompt + 原 embedding
            full_embeds = torch.cat([prompt_embeds, gpt_embeds], dim=1)  # [B, V+L, H]

            # ⚠️ labels 也要在前面补 V 个 -100，对齐序列长度
            ignore = torch.full(
                (labels.size(0), NUM_VIRTUAL_TOKENS),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            full_labels = torch.cat([ignore, labels], dim=1)  # [B, V+L]

            # -------- 前向计算 --------
            outputs = model(inputs_embeds=full_embeds, labels=full_labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            global_step += 1
            train_losses.append(loss.item())

            if global_step % 300 == 0:
                print(f"Step {global_step}: train_loss = {loss.item():.4f}")

                # ===== 验证 =====
                model.eval()
                total_loss = 0.0
                count = 0
                total_rmse = 0.0

                with torch.no_grad():
                    for val_batch in val_loader:
                        texts_val = list(val_batch)

                        encoded_val = tokenizer(
                            texts_val,
                            padding="max_length",
                            truncation=True,
                            max_length=MAX_LEN,
                            return_tensors="pt"
                        ).to(DEVICE)

                        ids = encoded_val["input_ids"]          # [B, L]
                        labels_val = ids.clone()                # [B, L]

                        prompt_val = prompt_encoder().unsqueeze(0).repeat(ids.size(0), 1, 1)   # [B, V, H]
                        embeds_val = model.transformer.wte(ids)                                # [B, L, H]
                        full_val = torch.cat([prompt_val, embeds_val], dim=1)                  # [B, V+L, H]

                        ignore_val = torch.full(
                            (labels_val.size(0), NUM_VIRTUAL_TOKENS),
                            -100,
                            dtype=labels_val.dtype,
                            device=labels_val.device
                        )
                        full_labels_val = torch.cat([ignore_val, labels_val], dim=1)           # [B, V+L]

                        out = model(inputs_embeds=full_val, labels=full_labels_val)
                        total_loss += out.loss.item()
                        count += 1

                        # 用 sqrt(loss) 粗略当成 RMSE 指标，主要是看趋势
                        total_rmse += math.sqrt(out.loss.item())

                avg_loss = total_loss / count
                avg_rmse = total_rmse / count

                val_losses.append(avg_loss)
                val_rmses.append(avg_rmse)
                val_steps_plot.append(global_step)

                print(f"   Eval loss = {avg_loss:.4f}, RMSE ≈ {avg_rmse:.4f}")

    print("Training finished.")

    # ============================
    # 画图
    # ============================

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Training Loss
    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss (Prompt Tuning)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "training_loss.png"))
    plt.close()

    # 2. Validation Loss
    if val_steps_plot:
        plt.figure()
        plt.plot(val_steps_plot, val_losses, marker="o", color="red")
        plt.title("Validation Loss (Prompt Tuning)")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "validation_loss.png"))
        plt.close()

    # 3. Validation RMSE
    if val_steps_plot:
        plt.figure()
        plt.plot(val_steps_plot, val_rmses, marker="o", color="green")
        plt.title("Validation RMSE (Prompt Tuning, approx by sqrt(loss))")
        plt.xlabel("Steps")
        plt.ylabel("RMSE (approx)")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "validation_rmse.png"))
        plt.close()

    # ============================
    # 保存模型
    # ============================

    torch.save(prompt_encoder.state_dict(), os.path.join(OUTPUT_DIR, "prompt_encoder.pt"))
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("All done. Model + plots saved.")


if __name__ == "__main__":
    main()