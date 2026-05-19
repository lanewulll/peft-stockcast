import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
import os
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


class RegressionMistral(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"  # 自动分配设备
        )
        self.regressor = torch.nn.Linear(self.backbone.config.hidden_size, 1, dtype=torch.float16)

    def forward(self, input_ids, attention_mask, labels=None, last_close=None, return_pred_close=False):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = h.last_hidden_state[:, -1, :]
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
    """将窗口数据转换为模型输入格式（与训练时对齐）"""
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 确保数据是numpy数组或普通Python类型
    window_data = np.array(window_data) if not isinstance(window_data, np.ndarray) else window_data
    
    # 构建提示文本（与训练时完全一致的格式）
    text = ", ".join([
        f"Day {d+1}: O={r[0]:.2f}, H={r[1]:.2f}, L={r[2]:.2f}, C={r[3]:.2f}, V={int(r[4])}"
        for d, r in enumerate(window_data)
    ])
    prompt = f"Based on this stock data: [{text}], predict next day's return:"
    print("model prompt:", prompt)
    
    # 编码
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


def main(test_csv='sample_test.csv', model_dir="ckpt_normalize", window_size=30):
    print("test_csv:", test_csv)
    # Load data
    df = pd.read_csv(test_csv)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    last_window = df.iloc[-window_size:][features].values.astype(np.float32)
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载tokenizer
    tokenizer_path = os.path.join(model_dir, "final", "tokenizer")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "mistralai/Mistral-7B-v0.1"  # 如果本地没有，从huggingface加载
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 初始化模型（与训练时一致）
    print("Initializing model...")
    
    # 加载Mistral-7B主干
    model = RegressionMistral("mistralai/Mistral-7B-v0.1")
    ################ lora部分模型，不再使用完整权重，会OOM
    # 加载LoRA权重
    model.backbone = PeftModel.from_pretrained(model.backbone, os.path.join(model_dir, "final", "lora"))
    
    model.backbone.requires_grad_(False)
    
    # 只开 LoRA（PEFT 提供的快捷接口，不会误伤主干）
    model.backbone.enable_adapter_layers() 
    # 加载regressor
    model.regressor.load_state_dict(torch.load(os.path.join(model_dir, "final", "regressor.pt")))
    model.regressor.to(dtype=torch.float16)
    
    model = model.to(device)
    model.eval()
    
    
    # Reset GPU stats
    if torch.cuda.is_available():
        reset_peak_memory_stats(device)
    
    # 迭代预测（滑动窗口，每次预测下一天）
    predictions = []
    current_window = last_window.copy()

    for i in range(3):
        print(f"\nPredicting day {i+1}...")
        # 预处理当前窗口数据
        inputs, last_close = preprocess_window_data(current_window, tokenizer, window_size)
        
        # 将输入移到设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        last_close_t = torch.tensor([last_close], dtype=torch.float16, device=device)

        with torch.no_grad():
            outputs = model(**inputs, return_pred_close=True, last_close=last_close_t)
            pred_close = outputs["logits"].item()        
            print(f"Predicted close price: {pred_close:.2f}")
        
        predictions.append(pred_close)
        
        # 滑动窗口：创建新行，用预测的收盘价更新，其他值用最后一行的（占位）
        last_row = current_window[-1].copy()
        new_row = last_row.copy()
        new_row[3] = pred_close  # 更新Close（索引3）
        new_row = np.array(new_row, dtype=np.float32)
        
        # 移除第一行，添加新行
        current_window = np.vstack((current_window[1:], new_row))

    # 计算可训练参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # GPU内存使用
    peak_gpu_mem_mb = max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0
    
    # 输出结果
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Peak GPU memory usage (MB): {peak_gpu_mem_mb:.2f}")
    print(f"Final predictions: {[f'{p:.2f}' for p in predictions]}")
    
    # 保存预测结果
    pred_df = pd.DataFrame({
        'Predicted_Close_Day1': [predictions[0]],
        'Predicted_Close_Day2': [predictions[1]],
        'Predicted_Close_Day3': [predictions[2]]
    })
    pred_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")


if __name__ == "__main__":
    # 使用训练时的数据集路径
#    main(test_csv='dataset/nvidia_stock_prices.csv', model_dir="ckpt_normalize")
    main(test_csv='sample_test.csv', model_dir="ckpt_normalize")