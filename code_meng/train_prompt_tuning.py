# train_prompt_tuning.py
# Prompt Tuning + Huber loss 训练脚本
# 会生成三张图：train_loss_prompt_huber.png / val_loss_prompt_huber.png / val_rmse_prompt_huber.png

import os
import math
import random
import json
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt

from transformers import GPT2LMHeadModel, AutoTokenizer

# ==================== 配置 ====================

# 这里填你本地 gpt2 的目录（里面有 config.json / merges.txt / tokenizer.json / tokenizer_config.json / vocab.json / model.safetensors）
BASE_MODEL_DIR = r"C:\Users\admin\models\gpt2"

TRAIN_CSV = os.path.join("dataset", "nvidia_stock_prices.csv")
OUTPUT_DIR = "prompt_tuned_model"
RESULTS_DIR = "results_prompt"

WINDOW_SIZE = 30          # 用过去 30 天预测下一天
MAX_LEN = 256
NUM_VIRTUAL_TOKENS = 20   # prompt 长度

BATCH_SIZE = 8
NUM_EPOCHS = 6
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100

LOG_EVERY = 50            # 每多少 step 打一次 log / 记一次训练 loss

SEED = 42

# ==================== 工具函数 ====================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class StockPromptDataset(Dataset):
    """
    把收盘价序列切成 (过去 WINDOW_SIZE 天, 目标第 WINDOW_SIZE+1 天)。
    """
    def __init__(self, csv_path: str, window_size: int = 30):
        df = pd.read_csv(csv_path)
        closes = df["Close"].astype(float).tolist()

        self.samples = []
        for i in range(window_size, len(closes)):
            hist = closes[i - window_size:i]
            target = closes[i]
            self.samples.append((hist, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PromptEncoder(nn.Module):
    """
    最简单的 Prompt Encoder：直接把 NUM_VIRTUAL_TOKENS 个虚拟 token 的 embedding 当成可训练参数。
    """
    def __init__(self, hidden_size: int, num_virtual_tokens: int = 20):
        super().__init__()
        self.emb = nn.Embedding(num_virtual_tokens, hidden_size)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, batch_size: int):
        # (batch, num_virtual, hidden)
        virtual_ids = torch.arange(self.emb.num_embeddings, device=self.emb.weight.device)
        virtual_ids = virtual_ids.unsqueeze(0).expand(batch_size, -1)
        return self.emb(virtual_ids)


def build_dataloaders(csv_path: str, window_size: int, batch_size: int, tokenizer):
    dataset = StockPromptDataset(csv_path, window_size)
    n_total = len(dataset)
    n_train = int(n_total * 0.9)
    n_val = n_total - n_train

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    def collate(batch):
        """
        batch: list of (hist_list, target_price)
        """
        texts = []
        targets = []
        for hist, tgt in batch:
            hist_str = ", ".join([f"{x:.2f}" for x in hist])
            text = f"Past {WINDOW_SIZE} closes: {hist_str}. Next close:"
            texts.append(text)
            targets.append(float(tgt))

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        targets = torch.tensor(targets, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
        }

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    return train_loader, val_loader, n_train, n_val


# ==================== 主训练逻辑 ====================

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----- Step 1: 加载 tokenizer 和 GPT-2 -----
    print("Step 1: loading tokenizer and GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    # GPT-2 没有 padding token，用 EOS 代替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_DIR)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.train()
    print("Step 1 done.")

    hidden_size = model.config.hidden_size

    # ----- Step 2: 构造 Prompt Encoder -----
    print("Step 2: building prompt encoder...")
    prompt_encoder = PromptEncoder(hidden_size, NUM_VIRTUAL_TOKENS).to(device)
    print("Step 2 done.")

    # ----- Step 3: 构造 DataLoader -----
    print("Step 3: loading CSV...")
    train_loader, val_loader, n_train, n_val = build_dataloaders(
        TRAIN_CSV, WINDOW_SIZE, BATCH_SIZE, tokenizer
    )
    print(f"Train = {n_train}, Val = {n_val}")
    print("Step 3 done.")

    # ----- 优化器 -----
    optimizer = AdamW(
        list(model.parameters()) + list(prompt_encoder.parameters()),
        lr=LEARNING_RATE,
    )

    # Huber 用 smooth_l1_loss；我们把 base_loss (cross-entropy) 当作 x，目标为 0
    def huber_from_ce(ce_loss: torch.Tensor, delta: float = 1.0):
        # ce_loss: 标量
        diff = ce_loss
        abs_diff = diff.abs()
        quadratic = 0.5 * diff * diff
        linear = delta * (abs_diff - 0.5 * delta)
        return torch.where(abs_diff <= delta, quadratic, linear)

    global_step = 0
    train_losses = []
    eval_steps = []
    val_losses = []
    val_rmses = []

    print("Step 4: training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")
        model.train()
        prompt_encoder.train()

        for batch in train_loader:
            global_step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            batch_size = input_ids.size(0)

            # 构建带 prompt 的 embedding
            with torch.no_grad():
                token_embeds = model.transformer.wte(input_ids)  # (B, L, H)

            prompt_embeds = prompt_encoder(batch_size)          # (B, V, H)
            full_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)

            # labels: 前 V 个虚拟 token 不算 loss => -100
            labels = input_ids.clone()
            prefix = torch.full(
                (batch_size, NUM_VIRTUAL_TOKENS),
                -100,
                dtype=torch.long,
                device=device,
            )
            labels = torch.cat([prefix, labels], dim=1)

            optimizer.zero_grad()
            outputs = model(inputs_embeds=full_embeds, labels=labels)
            ce_loss = outputs.loss

            # 用 Huber 包一下 CE，作为真正的训练 loss
            loss = huber_from_ce(ce_loss)

            loss.backward()
            optimizer.step()

            if global_step % LOG_EVERY == 0:
                train_losses.append(loss.item())
                print(f"Step {global_step}: train_loss (Huber(CE)) = {loss.item():.4f}")

        # ---------- 每个 epoch 结束做一次验证 ----------
        model.eval()
        prompt_encoder.eval()

        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_size = input_ids.size(0)

                token_embeds = model.transformer.wte(input_ids)
                prompt_embeds = prompt_encoder(batch_size)
                full_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)

                labels = input_ids.clone()
                prefix = torch.full(
                    (batch_size, NUM_VIRTUAL_TOKENS),
                    -100,
                    dtype=torch.long,
                    device=device,
                )
                labels = torch.cat([prefix, labels], dim=1)

                outputs = model(inputs_embeds=full_embeds, labels=labels)
                ce_loss = outputs.loss
                huber_loss = huber_from_ce(ce_loss)

                val_loss_sum += huber_loss.item() * batch_size
                val_count += batch_size

        val_loss_epoch = val_loss_sum / val_count
        val_rmse_epoch = math.sqrt(max(val_loss_epoch, 1e-8))

        eval_steps.append(global_step)
        val_losses.append(val_loss_epoch)
        val_rmses.append(val_rmse_epoch)

        print(
            f"Epoch {epoch} finished. "
            f"Val HuberLoss = {val_loss_epoch:.4f}, Val RMSE≈{val_rmse_epoch:.4f}"
        )

    # ==================== 画图 ====================

    # 1) Training Loss
    plt.figure()
    plt.plot(range(LOG_EVERY, LOG_EVERY * len(train_losses) + 1, LOG_EVERY), train_losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss (Prompt Tuning + Huber)")
    plt.grid(True)
    train_fig_path = os.path.join(RESULTS_DIR, "train_loss_prompt_huber.png")
    plt.savefig(train_fig_path, bbox_inches="tight")
    plt.close()

    # 2) Validation Loss
    plt.figure()
    plt.plot(eval_steps, val_losses, marker="o")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Validation Loss (Prompt Tuning + Huber)")
    plt.grid(True)
    val_loss_fig_path = os.path.join(RESULTS_DIR, "val_loss_prompt_huber.png")
    plt.savefig(val_loss_fig_path, bbox_inches="tight")
    plt.close()

    # 3) Validation RMSE
    plt.figure()
    plt.plot(eval_steps, val_rmses, marker="o")
    plt.xlabel("Steps")
    plt.ylabel("RMSE (approx)")
    plt.title("Validation RMSE (Prompt Tuning + Huber)")
    plt.grid(True)
    val_rmse_fig_path = os.path.join(RESULTS_DIR, "val_rmse_prompt_huber.png")
    plt.savefig(val_rmse_fig_path, bbox_inches="tight")
    plt.close()

    # ==================== 保存模型和 Prompt Encoder ====================

    torch.save(
        {
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "num_virtual_tokens": NUM_VIRTUAL_TOKENS,
                "hidden_size": hidden_size,
                "window_size": WINDOW_SIZE,
                "max_len": MAX_LEN,
            },
        },
        os.path.join(OUTPUT_DIR, "prompt_encoder.pt"),
    )
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nAll done. Model + plots saved.")
    print("  Model dir :", OUTPUT_DIR)
    print("  Plots dir :", RESULTS_DIR)


if __name__ == "__main__":
    main()