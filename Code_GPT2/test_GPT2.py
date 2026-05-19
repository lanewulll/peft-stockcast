import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # 改用GPT-2相关类
from peft import LoraConfig, get_peft_model, PeftModel


class RegressionGPT2(torch.nn.Module):  # 类名修改为GPT2相关
    def __init__(self, model_name):
        super().__init__()
        # 加载GPT-2模型（替换Mistral）
        self.backbone = GPT2LMHeadModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # 冻结语言建模头，仅使用其特征提取能力
        for param in self.backbone.lm_head.parameters():
            param.requires_grad = False
        # GPT-2的隐藏层维度通过config获取
        self.regressor = torch.nn.Linear(self.backbone.config.hidden_size, 1, dtype=torch.float16)

    def forward(self, input_ids, attention_mask, labels=None, last_close=None, return_pred_close=False):
        # 需要显式指定output_hidden_states=True才能获取隐藏状态
        h = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True  # 开启隐藏状态输出
        )
        # 从hidden_states中获取最后一层的隐藏状态（倒数第一个元素是最后一层）
        pooled = h.hidden_states[-1][:, -1, :]  # 取最后一个token的隐藏状态
        logits = self.regressor(pooled).squeeze(-1)  # 预测收益率
        print("predict return (not final result):", logits)

        if return_pred_close and last_close is not None:
            # 反归一化
            pred_close = logits * last_close + last_close
        else:
            pred_close = logits

        loss = None
        if labels is not None:
            loss = torch.nn.functional.smooth_l1_loss(logits, labels.float())
        return {"loss": loss, "logits": pred_close}


def preprocess_window_data(window_data, tokenizer, window_size=30):
    """保持数据预处理逻辑不变，确保与训练时格式一致"""
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    window_data = np.array(window_data) if not isinstance(window_data, np.ndarray) else window_data
    
    # 提示文本格式不变，GPT-2同样支持自然语言输入
    text = ", ".join([
        f"Day {d+1}: O={r[0]:.2f}, H={r[1]:.2f}, L={r[2]:.2f}, C={r[3]:.2f}, V={int(r[4])}"
        for d, r in enumerate(window_data)
    ])
    prompt = f"Based on this stock data: [{text}], predict next day's return:"
    print("model prompt:", prompt)
    
    # 编码逻辑不变，GPT-2分词器兼容相同参数
    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )
    last_close = float(window_data[-1][3])
    print("last_close:", last_close)
    
    return inputs, last_close              


def main(test_csv='sample_test.csv', model_dir="ckpt_gpt2", window_size=30):  # 模型目录名调整
    print("test_csv:", test_csv)
    # 数据加载逻辑不变
    df = pd.read_csv(test_csv)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    last_window = df.iloc[-window_size:][features].values.astype(np.float32)
    
    # 设备检查不变
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载GPT-2分词器
    tokenizer_path = os.path.join(model_dir, "final", "tokenizer")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "gpt2"  # 改用GPT-2基础模型
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    # GPT-2默认没有pad_token，使用eos_token作为pad_token
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 初始化GPT-2模型
    print("Initializing GPT-2 model...")
    
    # 加载GPT-2主干（可根据需要选择不同规模：gpt2-small, gpt2-medium, gpt2-large）
    model = RegressionGPT2("gpt2")
    # 加载LoRA权重（需确保训练时使用相同的GPT-2模型）
    model.backbone = PeftModel.from_pretrained(model.backbone, os.path.join(model_dir, "final", "lora"))
    
    model.backbone.requires_grad_(False)
    model.backbone.enable_adapter_layers()  # 启用LoRA适配器
    # 加载回归头权重
    model.regressor.load_state_dict(torch.load(os.path.join(model_dir, "final", "regressor.pt")))
    model.regressor.to(dtype=torch.float16)
    
    model = model.to(device)
    model.eval()
    
    
    # GPU内存统计不变
    if torch.cuda.is_available():
        reset_peak_memory_stats(device)
    
    # 预测逻辑完全不变
    predictions = []
    current_window = last_window.copy()

    for i in range(3):
        print(f"\nPredicting day {i+1}...")
        inputs, last_close = preprocess_window_data(current_window, tokenizer, window_size)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        last_close_t = torch.tensor([last_close], dtype=torch.float16, device=device)

        with torch.no_grad():
            outputs = model(** inputs, return_pred_close=True, last_close=last_close_t)
            pred_close = outputs["logits"].item()        
            print(f"Predicted close price: {pred_close:.2f}")
        
        predictions.append(pred_close)
        
        # 滑动窗口更新逻辑不变
        last_row = current_window[-1].copy()
        new_row = last_row.copy()
        new_row[3] = pred_close  # 更新Close
        new_row = np.array(new_row, dtype=np.float32)
        
        current_window = np.vstack((current_window[1:], new_row))

    # 输出统计信息
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peak_gpu_mem_mb = max_memory_allocated(device) / (1024 **2) if torch.cuda.is_available() else 0
    
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Peak GPU memory usage (MB): {peak_gpu_mem_mb:.2f}")
    print(f"Final predictions: {[f'{p:.2f}' for p in predictions]}")
    
    # 保存预测结果
    pred_df = pd.DataFrame({
        'Predicted_Close_Day1': [predictions[0]],
        'Predicted_Close_Day2': [predictions[1]],
        'Predicted_Close_Day3': [predictions[2]]
    })
    pred_df.to_csv('predictions_gpt2.csv', index=False)  # 输出文件名区分
    print("Predictions saved to 'predictions_gpt2.csv'")


if __name__ == "__main__":
    # 使用GPT-2的模型目录和测试数据
    main(test_csv='sample_test.csv', model_dir="ckpt_gpt2")