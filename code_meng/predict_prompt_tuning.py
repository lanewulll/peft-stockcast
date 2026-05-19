# predict_prompt_tuning.py
# 使用 Prompt Tuning + GPT-2 做 3 日股价预测（带 debug 打印 + 更合理的 fallback）

import os
import re
import pandas as pd
import torch
from torch import nn
from transformers import GPT2LMHeadModel, AutoTokenizer

# ================== 配置区域 ==================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_DIR = r"C:\Users\admin\models\gpt2"
PROMPT_ENCODER_PATH = os.path.join("prompt_tuned_model", "prompt_encoder.pt")

TEST_CSV = os.path.join("dataset", "sample_test.csv")
WINDOW_SIZE = 30


# ================== 与训练脚本一致的 PromptEncoder ==================

class PromptEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_virtual_tokens: int = 20):
        super().__init__()
        self.emb = nn.Embedding(num_virtual_tokens, hidden_size)

    def forward(self, batch_size: int):
        virtual_ids = torch.arange(self.emb.num_embeddings, device=self.emb.weight.device)
        virtual_ids = virtual_ids.unsqueeze(0).expand(batch_size, -1)
        return self.emb(virtual_ids)


# ================== 工具函数 ==================

def build_prompt_from_window(window_prices):
    lines = [f"Day {i+1}: {p:.2f}" for i, p in enumerate(window_prices)]
    history_str = "\n".join(lines)
    prompt = (
        "The following are the last 30 daily closing prices of NVIDIA stock (in USD):\n"
        f"{history_str}\n"
        "Please predict the next day's closing price in USD.\n"
        "Answer with a single number only (no extra words):"
    )
    return prompt


def parse_price_from_text(text):
    """
    找到第一个浮点数。找不到就返回 None，由外面决定 fallback。
    """
    m = re.search(r"\d+(\.\d+)?", text)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


# ================== 主函数 ==================

def main():
    print(f"Using device: {DEVICE}")

    # ----- Step 1: 加载 tokenizer + GPT-2 -----
    print("Step 1: loading tokenizer and GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_DIR)
    base_model.to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    print("Step 1 done.")

    # ----- Step 2: 加载 prompt encoder -----
    print("Step 2: loading prompt encoder...")
    if not os.path.exists(PROMPT_ENCODER_PATH):
        raise FileNotFoundError(f"Prompt encoder not found: {PROMPT_ENCODER_PATH}")

    state = torch.load(PROMPT_ENCODER_PATH, map_location=DEVICE)
    if "prompt_encoder_state_dict" in state and "config" in state:
        pe_state_dict = state["prompt_encoder_state_dict"]
        cfg = state["config"]
        num_virtual_tokens = cfg.get("num_virtual_tokens", 20)
        hidden_size = cfg.get("hidden_size", base_model.config.hidden_size)
    else:
        pe_state_dict = state
        num_virtual_tokens = 20
        hidden_size = base_model.config.hidden_size

    prompt_encoder = PromptEncoder(hidden_size, num_virtual_tokens).to(DEVICE)
    prompt_encoder.load_state_dict(pe_state_dict)
    prompt_encoder.eval()
    print(f"Loaded prompt encoder: num_virtual_tokens={num_virtual_tokens}, hidden_size={hidden_size}")
    print("Step 2 done.")

    # ----- Step 3: 加载测试 CSV -----
    print("Step 3: loading test CSV...")
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    df = pd.read_csv(TEST_CSV)
    if "Close" not in df.columns:
        raise ValueError(f"'Close' column not found in {TEST_CSV}. Columns = {df.columns.tolist()}")

    closes_series = pd.to_numeric(df["Close"], errors="coerce")
    closes = closes_series.dropna().astype(float).tolist()

    if len(closes) < WINDOW_SIZE:
        raise ValueError(f"Not enough numeric closes: {len(closes)} < {WINDOW_SIZE}")

    window = closes[-WINDOW_SIZE:]
    print(f"Initial window (last {WINDOW_SIZE} closes):")
    print(window)
    print("Step 3 done.")

    # ----- Step 4: 预测未来 3 天 -----
    print("Step 4: predicting next 3 days...")
    preds = []

    with torch.no_grad():
        for day in range(1, 4):
            prompt_text = build_prompt_from_window(window)
            enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=512,
            ).to(DEVICE)

            input_embeds = base_model.transformer.wte(enc["input_ids"])
            prefix_embeds = prompt_encoder(batch_size=1)
            full_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)

            outputs = base_model.generate(
                inputs_embeds=full_embeds,
                max_new_tokens=32,      # 多给一点长度
                do_sample=True,         # 采样，更容易生成数字
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_ids = outputs[0, full_embeds.size(1):]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # 打印原始生成文本，方便你看
            print(f"\n[Day {day}] raw generated text:")
            print(repr(gen_text))

            parsed = parse_price_from_text(gen_text)

            if parsed is None:
                # 如果还是没解析到数字，用最近 3 天的平均作为 fallback
                fallback = sum(window[-3:]) / 3.0
                print(f"[Day {day}] No number parsed, fallback to 3-day mean {fallback:.4f}")
                pred_price = fallback
            else:
                pred_price = parsed
                print(f"[Day {day}] Parsed price: {pred_price:.4f}")

            preds.append(pred_price)
            window = window[1:] + [pred_price]
            print(f"Predicted Day {day}: {pred_price:.4f}")

    # ----- Step 5: 保存结果 -----
    print("\nStep 5: saving predictions...")
    out_df = pd.DataFrame(
        {
            "Predicted_Close_Day1": [preds[0]],
            "Predicted_Close_Day2": [preds[1]],
            "Predicted_Close_Day3": [preds[2]],
        }
    )
    out_path = "predictions_prompt.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    print(out_df)


if __name__ == "__main__":
    main()