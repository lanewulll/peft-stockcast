import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import GPT2LMHeadModel, AutoTokenizer
from peft import PeftModel, IA3Config
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')

# ============================
# 核心配置（不变）
# ============================
MODEL_ROOT_DIR = "ia3_stock_model"
SELECTED_MODEL = "best_model"
TEST_CSV = "sample_test.csv"  # 包含完整历史数据（最后3天作为待预测的真实值）
OUTPUT_CSV = "stock_prediction_last3days.csv"  # 输出：预测值+真实值对比
WINDOW_SIZE = 30 #与训练一致
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IA3_TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]
BASE_MODEL_DIR = "/root/autodl-tmp/GP6102/codeZHUJingwen/gpt2"

torch.cuda.empty_cache()
print(f"Using device: {DEVICE}")
print(f"核心逻辑：用除最后3天外的历史数据 → 预测最后3天的真实值（回测验证）")


# ============================
# 第一步：数据拆分（关键！分离“输入数据”和“待预测的真实值”）
# ============================
def split_data_for_backtest(csv_path: str, n_predict_days: int = 3) -> Tuple[List[float], List[float], List[str]]:
    """
    拆分数据：
    - 输入数据：所有数据除了最后3天（用于滚动预测）
    - 真实值：最后3天的收盘价（用于对比预测准确性）
    - 日期：对应真实值的日期
    """
    print("\n===== 数据拆分（回测模式） =====")
    df = pd.read_csv(csv_path)

    # 提取收盘价和日期（处理异常值）
    closes = pd.to_numeric(df["Close"], errors="coerce").dropna().astype(float).tolist()
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna().astype(str).tolist()

    # 确保数据量足够（输入数据至少需要1个窗口 + 预测3天）
    min_required = WINDOW_SIZE + n_predict_days
    if len(closes) < min_required:
        raise ValueError(f"数据量不足！需要至少 {min_required} 个有效价格，当前只有 {len(closes)} 个")

    # 拆分：输入数据 = 所有数据[:-3]，真实值 = 最后3天数据[-3:]
    input_closes = closes[:-n_predict_days]  # 用于预测的输入历史数据
    true_values = closes[-n_predict_days:]  # 待预测的最后3天真实值
    true_dates = dates[-n_predict_days:]  # 最后3天的日期

    print(f"✅ 输入数据量：{len(input_closes)} 个价格（除最后3天）")
    print(f"✅ 待预测真实值：{len(true_values)} 个价格（最后3天）")
    print(f"✅ 真实值日期：{true_dates}")
    print(f"✅ 真实值：{[round(v, 2) for v in true_values]}")

    return input_closes, true_values, true_dates


# ============================
# 第二步：加载模型（不变）
# ============================
def load_model_components() -> Tuple[nn.Module, AutoTokenizer]:
    print("\n===== 加载模型组件 =====")
    model_dir = os.path.join(MODEL_ROOT_DIR, SELECTED_MODEL)
    adapter_dir = os.path.join(model_dir, "ia3_adapter")
    tokenizer_dir = os.path.join(model_dir, "tokenizer")
    regression_head_path = os.path.join(model_dir, "regression_head.pt")

    # 检查模型文件
    required_files = [
        (adapter_dir, "IA3 adapter folder"),
        (os.path.join(adapter_dir, "adapter_model.safetensors"), "Adapter weights"),
        (tokenizer_dir, "Tokenizer folder"),
        (regression_head_path, "Regression head weights"),
        (BASE_MODEL_DIR, "Base GPT2 model")
    ]
    for path, name in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺失 {name}：{path}")
    print(f"✅ 所有模型文件存在")

    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token if not tokenizer.pad_token else tokenizer.pad_token
    print(f"✅ Tokenizer加载完成")

    # 加载基础模型+IA3适配器
    base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_DIR, local_files_only=True, output_hidden_states=True).to(
        DEVICE)
    ia3_config = IA3Config(task_type="CAUSAL_LM", target_modules=IA3_TARGET_MODULES,
                           feedforward_modules=["c_fc", "c_proj"], inference_mode=True)
    model_with_ia3 = PeftModel.from_pretrained(base_model, adapter_dir, config=ia3_config,
                                               ignore_mismatched_sizes=True).to(DEVICE)

    # 加载回归头
    class GPT2RegressionModel(nn.Module):
        def __init__(self, base_model, hidden_size=768):
            super().__init__()
            self.base_model = base_model
            self.regression_head = nn.Sequential(
                nn.Linear(hidden_size, 128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(64, 1)
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            return self.regression_head(outputs.hidden_states[-1][:, -1, :])

    model = GPT2RegressionModel(model_with_ia3).to(DEVICE)
    model.regression_head.load_state_dict(torch.load(regression_head_path, map_location=DEVICE, weights_only=False))
    model.eval()
    print(f"✅ 模型加载完成（评估模式）")

    return model, tokenizer


# ============================
# 第三步：滚动预测最后3天（核心！用输入数据逐天预测）
# ============================
def predict_last_3days(model, tokenizer, input_closes) -> List[float]:
    """
    滚动预测逻辑（和训练时的多步预测一致）：
    1. 第一天预测：用输入数据的最后45天窗口 → 预测第1天
    2. 第二天预测：将第1天预测值加入窗口 → 预测第2天
    3. 第三天预测：将第2天预测值加入窗口 → 预测第3天
    """
    print("\n===== 滚动预测最后3天 =====")
    # 用输入数据的统计量归一化（避免训练/测试分布差异）
    input_mean = np.mean(input_closes)
    input_std = np.std(input_closes)
    input_norm = (np.array(input_closes) - input_mean) / input_std
    print(f"✅ 输入数据归一化：mean={input_mean:.2f}, std={input_std:.2f}")

    # 初始窗口：输入数据的最后45天
    current_window = input_norm[-WINDOW_SIZE:].tolist()
    predictions = []  # 存储3天的预测值（反归一化后）

    with torch.no_grad():
        for day in range(1, 4):
            # 1. 构建输入文本（和训练格式完全一致）
            input_text = f"prices: {current_window} -> next:"

            # 2. Tokenize
            enc = tokenizer(
                input_text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)

            # 3. 模型预测（归一化后的预测值）
            pred_norm = model(enc["input_ids"], enc["attention_mask"]).cpu().numpy()[0][0]

            # 4. 反归一化 → 得到真实价格
            pred_real = pred_norm * input_std + input_mean
            pred_real = round(pred_real, 2)
            predictions.append(pred_real)

            # 5. 滚动更新窗口：将预测值（归一化后）加入窗口，移除最旧数据
            current_window = current_window[1:] + [pred_norm]

            print(f"📈 第{day}天预测值：{pred_real}")

    return predictions


# ============================
# 第四步：生成对比CSV（预测值+真实值+误差）
# ============================
def generate_comparison_csv(true_dates, true_values, predictions, output_path):
    """
    生成包含“日期、真实值、预测值、误差”的CSV，方便对比准确性
    """
    print(f"\n===== 生成预测对比CSV =====")
    # 计算误差（绝对误差和相对误差）
    absolute_errors = [round(abs(t - p), 2) for t, p in zip(true_values, predictions)]
    relative_errors = [round(abs(t - p) / t * 100, 2) for t, p in zip(true_values, predictions)]

    # 构建DataFrame（清晰对比）
    df = pd.DataFrame({
        "日期": true_dates,
        "最后3天真实收盘价": [round(v, 2) for v in true_values],
        "模型预测收盘价": predictions,
        "绝对误差（真实-预测）": absolute_errors,
        "相对误差（%）": relative_errors
    })

    # 保存CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ CSV保存路径：{output_path}")

    # 打印完整对比结果
    print("\n📊 最后3天预测 vs 真实值 对比表：")
    print("-" * 80)
    print(df.to_string(index=False))
    print("-" * 80)

    # 计算平均误差（评估模型准确性）
    avg_abs_error = round(np.mean(absolute_errors), 2)
    avg_rel_error = round(np.mean(relative_errors), 2)
    print(f"📊 平均绝对误差：{avg_abs_error}")
    print(f"📊 平均相对误差：{avg_rel_error}%")


# ============================
# 主函数（串联回测流程）
# ============================
def main():
    try:
        # 1. 拆分数据：输入数据（除最后3天）+ 真实值（最后3天）
        input_closes, true_values, true_dates = split_data_for_backtest(TEST_CSV)

        # 2. 加载模型
        model, tokenizer = load_model_components()

        # 3. 滚动预测最后3天
        predictions = predict_last_3days(model, tokenizer, input_closes)

        # 4. 生成对比CSV
        generate_comparison_csv(true_dates, true_values, predictions, OUTPUT_CSV)

        print("\n🎉 回测预测完成！")
        print(f"核心结论：用除最后3天外的历史数据，成功预测出最后3天的价格，可通过CSV查看详细对比")

    except Exception as e:
        print(f"\n❌ 错误：{str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
