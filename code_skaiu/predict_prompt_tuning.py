# predict_prompt_tuning.py
# 使用 Prompt Tuning 训练好的 prompt_encoder.pt，对 sample_test.csv 预测未来 3 天收盘价

import os
import re
import pandas as pd
import torch
from torch import nn
from transformers import GPT2LMHeadModel, AutoTokenizer

# ========= 配置 =========

BASE_MODEL_DIR = r"C:\Users\admin\models\gpt2"   # GPT-2 本地目录
PROMPT_MODEL_DIR = "prompt_tuned_model"          # 里面有 prompt_encoder.pt
TEST_CSV = "sample_test.csv"                     # 测试数据
OUTPUT_CSV = "predictions_prompt.csv"            # 输出预测

WINDOW_SIZE = 30
MAX_LEN = 256
NUM_VIRTUAL_TOKENS = 20                          # 与训练一致

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 工具 =========

NUM_PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")


def extract_number(text: str):
    nums = NUM_PATTERN.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def fallback_baseline(window):
    """
    回退策略：最近 5 天均值 MA5。
    """
    if not window:
        return 0.0
    k = min(5, len(window))
    recent = window[-k:]
    return float(sum(recent) / len(recent))


class PromptEncoder(nn.Module):
    """
    和训练脚本保持完全一致
    """
    def __init__(self, hidden_size, num_virtual_tokens):
        super().__init__()
        self.embeds = nn.Parameter(torch.randn(num_virtual_tokens, hidden_size))

    def forward(self):
        return self.embeds


# ========= 主逻辑 =========

def main():
    print("Using device:", DEVICE)

    # 1. 加载 tokenizer & base GPT-2
    print("Step 1: loading tokenizer and GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_DIR, local_files_only=True).to(DEVICE)
    base_model.eval()
    print("Step 1 done.")

    # 2. 加载 prompt encoder
    print("Step 2: loading prompt encoder...")
    hidden_size = base_model.config.hidden_size
    prompt_encoder = PromptEncoder(hidden_size, NUM_VIRTUAL_TOKENS).to(DEVICE)

    state_path = os.path.join(PROMPT_MODEL_DIR, "prompt_encoder.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Cannot find prompt_encoder.pt under {PROMPT_MODEL_DIR}")

    # 这里的 FutureWarning 可以忽略，不是错误
    state = torch.load(state_path, map_location=DEVICE)
    prompt_encoder.load_state_dict(state)
    prompt_encoder.eval()
    print("Step 2 done.")

    # 3. 读取测试 CSV（这里是修复的关键）
    print("Step 3: loading test CSV...")
    df = pd.read_csv(TEST_CSV)

    if "Close" not in df.columns:
        raise ValueError(f"CSV file {TEST_CSV} has no 'Close' column, columns = {list(df.columns)}")

    # 把 Close 列强制转成数值，无法转的（比如 'AAPL'）变成 NaN，然后丢掉
    closes_series = pd.to_numeric(df["Close"], errors="coerce")
    closes = closes_series.dropna().astype(float).tolist()

    if len(closes) < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} numeric closes in {TEST_CSV}, got {len(closes)}")

    window = closes[-WINDOW_SIZE:]
    print(f"Initial window (last {WINDOW_SIZE} closes):")
    print(window)

    predictions = []

    # 4. 预测 Day1, Day2, Day3
    print("Step 4: predicting next 3 days...")
    for day in range(1, 4):
        prompt_text = f"prices: {window} -> next:"

        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
        ).to(DEVICE)

        input_ids = enc["input_ids"]              # [1, L]
        attention_mask = enc["attention_mask"]    # [1, L]

        # 原 token embedding
        gpt_embeds = base_model.transformer.wte(input_ids)   # [1, L, H]

        # prompt embedding
        p_embeds = prompt_encoder()                         # [V, H]
        p_embeds = p_embeds.unsqueeze(0)                    # [1, V, H]

        # 拼接
        full_embeds = torch.cat([p_embeds, gpt_embeds], dim=1)  # [1, V+L, H]

        # attention mask 前面补 V 个 1
        p_mask = torch.ones(1, NUM_VIRTUAL_TOKENS, dtype=attention_mask.dtype, device=DEVICE)
        full_attn = torch.cat([p_mask, attention_mask], dim=1)  # [1, V+L]

        gen_ids = base_model.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_attn,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

        full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        suffix = full_text[len(prompt_text):]
        y_pred = extract_number(suffix)

        if y_pred is None:
            y_pred = fallback_baseline(window)
            print(f"Day {day}: no number parsed, fallback to MA5 {y_pred:.4f}")
        else:
            print(f"Day {day}: parsed from model output -> {y_pred:.4f}")

        predictions.append(float(y_pred))
        print(f"Predicted Day {day}: {y_pred:.4f}")

        # 滑动窗口：丢掉最早一天，加上新预测
        window = window[1:] + [float(y_pred)]

    # 5. 保存预测结果
    print("Step 5: saving predictions...")
    out_df = pd.DataFrame(
        [[predictions[0], predictions[1], predictions[2]]],
        columns=["Predicted_Close_Day1", "Predicted_Close_Day2", "Predicted_Close_Day3"],
    )
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions to {OUTPUT_CSV}")
    print(out_df)


if __name__ == "__main__":
    main()