"""
GPT-2 + LoRA 支持Loss/RMSE曲线可视化
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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_rmse(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    return {"rmse": rmse}

def preprocess_stock_data(csv_path, window_size=30):
    df = pd.read_csv(csv_path).sort_values("Date").reset_index(drop=True)
    assert df["Date"].is_monotonic_increasing
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[features].copy()

    data_list = []
    for i in range(len(df) - window_size):
        window = df.iloc[i:i+window_size][features].values
        next_close = df.iloc[i+window_size]["Close"]
        last_close = window[-1, 3]

        # 预测收益率（归一化目标）
        target = (next_close - last_close) / last_close

        # 构建输入文本
        text = ", ".join([
            f"Day {d+1}: O={r[0]:.2f}, H={r[1]:.2f}, L={r[2]:.2f}, C={r[3]:.2f}, V={int(r[4])}"
            for d, r in enumerate(window)
        ])
        prompt = f"Based on this stock data: [{text}], predict next day's return:"

        data_list.append({
            "input": prompt,
            "target": target,
            "last_close": last_close,
        })
    return Dataset.from_list(data_list)

class RegressionGPT2(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # GPT-2使用CausalLM模型
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True  # 需要获取隐藏状态
        )
        self.regressor = torch.nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None, last_close=None, return_pred_close=False):
        # GPT-2的forward返回包含hidden_states的字典
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # 取最后一层隐藏状态的最后一个token
        pooled = outputs.hidden_states[-1][:, -1, :]
        logits = self.regressor(pooled).squeeze(-1)  # 预测收益率

        if return_pred_close and last_close is not None:
            # 反归一化到收盘价
            pred_close = logits * last_close + last_close
        else:
            pred_close = logits

        loss = None
        if labels is not None:
            loss = torch.nn.functional.smooth_l1_loss(logits, labels.float())
        return {"loss": loss, "logits": pred_close}

def plot_metrics(training_logs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_steps, train_losses = [], []
    val_steps, val_losses, val_rmses = [], [], []

    for log in training_logs:
        if "loss" in log and "step" in log:
            train_losses.append(log["loss"])
            train_steps.append(log["step"])
        if "eval_loss" in log:
            val_losses.append(log["eval_loss"])
        if "eval_rmse" in log:
            val_rmses.append(log["eval_rmse"])

    val_steps = np.linspace(0, max(train_steps) if train_steps else 1, len(val_losses))

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

    # 使用GPT-2分词器
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # GPT-2使用eos_token作为pad_token
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_func(examples):
        tok = tokenizer(
            examples["input"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )
        tok["labels"] = examples["target"]
        tok["last_close"] = examples["last_close"]  # 保留last_close用于反归一化
        return tok

    train_ds = train_ds.map(tokenize_func, batched=True, num_proc=4)
    eval_ds = eval_ds.map(tokenize_func, batched=True, num_proc=4)
    train_ds = train_ds.with_format('torch')
    eval_ds = eval_ds.with_format('torch')

    # 初始化GPT-2回归模型
    model = RegressionGPT2("gpt2")

    # 配置LoRA（适配GPT-2的注意力层）
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        # GPT-2的注意力层名称与Mistral不同
        target_modules=["c_attn"],  # GPT-2中查询/键/值合并在c_attn层
        task_type="CAUSAL_LM"
    )
    model.backbone = get_peft_model(model.backbone, peft_config)

    class RegTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            inputs["last_close"] = None
            out = model(** inputs, labels=labels)
            return (out["loss"], out) if return_outputs else out["loss"]
        
        def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
            last_close = inputs.pop("last_close", None)
            with torch.no_grad():
                out = model(
                    return_pred_close=True,
                    last_close=last_close,
                    **inputs
                )
            loss = out["loss"]
            logits = out["logits"]  # 反归一化后的收盘价
            # 标签也反归一化
            labels = inputs["labels"] * last_close + last_close
            return (loss, logits, labels)
            

        # 重写保存模型方法，修正tokenizer引用
        def _save(self, output_dir: str):
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存LoRA权重（禁用safe_serialization）
            self.model.backbone.save_pretrained(
                os.path.join(output_dir, "lora"),
                safe_serialization=False  # 避免共享张量错误
            )
            
            # 保存回归头
            torch.save(
                self.model.regressor.state_dict(),
                os.path.join(output_dir, "regressor.pt")
            )
            
            # 保存分词器（使用初始化时传入的tokenizer）
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(
                    os.path.join(output_dir, "tokenizer"),
                    safe_serialization=False
            )

    def collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.float32).to(input_ids.device)
        last_close = torch.tensor([b["last_close"] for b in batch], dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": mask,
            "labels": labels,
            "last_close": last_close
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
        data_collator=collate,
        tokenizer=tokenizer  # 传递tokenizer到自定义Trainer
    )

    trainer.train()
    plot_metrics(trainer.state.log_history, args.output_dir)

    os.makedirs(os.path.join(args.output_dir, "final"), exist_ok=True)
    #torch.save(model.state_dict(), os.path.join(args.output_dir, "final", "full_model.pt"))
    # 保存LoRA权重
    model.backbone.save_pretrained(
        os.path.join(args.output_dir, "final", "lora"),
        safe_serialization=False  # 保持一致
    )
    
    # 单独保存regressor层参数
    torch.save(model.regressor.state_dict(), os.path.join(args.output_dir, "final", "regressor.pt"))
    
    # 保存分词器
    tokenizer.save_pretrained(
        os.path.join(args.output_dir, "final", "tokenizer"),
        safe_serialization=False  # 保持一致
    )
    with open(os.path.join(args.output_dir, "final", "training_logs.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    final = trainer.evaluate()
    print("\nFinal validation — MSE:", round(final["eval_loss"], 4), "RMSE:", round(final["eval_rmse"], 4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='dataset/nvidia_stock_prices.csv')
    parser.add_argument("--output_dir", type=str, default="ckpt_gpt2")
    parser.add_argument("--batch_size", type=int, default=4)  # GPT-2参数量较小，可适当增大batch_size
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)  # GPT-2可能需要稍高学习率
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()
    main(args)