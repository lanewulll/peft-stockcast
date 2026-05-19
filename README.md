# PEFT-StockCast

PEFT-StockCast 是一个实验项目，主题是使用大语言模型的参数高效微调方法（PEFT）进行股票收盘价预测。项目把历史 OHLCV 股价序列转换成自然语言提示或价格序列文本，再用 Mistral-7B、GPT-2 作为特征提取/生成骨干，比较 LoRA、IA3 和 Prompt Tuning 在短期股价预测任务上的表现。

> 注意：本项目是机器学习课程实验和方法比较，不构成任何投资建议。当前代码更适合作为 PEFT 方法实验样例，而不是可直接用于真实交易的预测系统。

## 项目内容

- 训练数据：`nvidia_stock_prices.csv`，来自 NVIDIA 历史股价，时间跨度为 2004-01-02 到 2023-12-29。
- 测试样例：`sample_test.csv`，结构与训练数据一致，但样例内容为另一支股票时间序列。
- 任务形式：使用过去 30 个交易日窗口预测下一日收益率或收盘价，并滚动预测未来 3 天。
- 对比方法：
  - Mistral-7B + LoRA + 回归头
  - GPT-2 + LoRA + 回归头
  - GPT-2 + Prompt Tuning
  - GPT-2 + IA3 + 回归头

## 目录结构

```text
.
├── code_likeyu_final_20251123/    # Mistral-7B + LoRA 训练、测试、检查点、预测结果
├── Code_GPT2/                     # GPT-2 + LoRA 训练、测试、检查点、预测结果
├── code_meng/                     # GPT-2 Prompt Tuning 版本
├── code_skaiu/                    # 另一个 GPT-2 Prompt Tuning 版本
├── code&results_ZHUJingwen/       # GPT-2 IA3 训练、测试与结果
├── Group12 Stock Price Prediction with LLM Fine tuning final.pptx
└── Li_Keyu_Report for Stock Price Prediction using Mistral-7B with LoRA Finetuning.pdf
```

## 实验结果摘要

最终展示文件中记录了四种模型的对比结果：

| Model | Val MSE | Test MSE | GPU Peak |
| --- | ---: | ---: | ---: |
| Prompt Tuning (GPT-2) | 0.10 | 5.05 | 约 5.5 GB |
| IA3 (GPT-2) | 0.07 | 16.86 | 约 5.7 GB |
| LoRA (Mistral-7B) | 0.85 | 3.09 | 约 13.7 GB |
| LoRA (GPT-2) | 0.74 | 4.95 | 约 6.0 GB |

从现有结果看，Mistral-7B + LoRA 的测试误差最低，但显存成本最高；GPT-2 系列更轻量，适合做 PEFT 方法对照实验。

## 环境准备

建议为每个子项目单独创建虚拟环境，或先使用根目录下目标子项目的 `requirements.txt` 安装依赖。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code_likeyu_final_20251123/requirements.txt
```

如果运行 GPT-2 Prompt Tuning 或 IA3 脚本，请先检查脚本顶部的 `BASE_MODEL_DIR`。部分脚本写死了 Windows 或 AutoDL 环境中的本地 GPT-2 路径，需要改为你机器上的模型目录，或改成 Hugging Face 模型名 `gpt2`。

## 运行示例

### Mistral-7B + LoRA

```bash
cd code_likeyu_final_20251123
python train_normalize.py --data_path dataset/nvidia_stock_prices.csv --output_dir ckpt_normalize
python test.py
```

预测结果会写入 `predictions.csv`。

### GPT-2 + LoRA

```bash
cd Code_GPT2
python train_GPT2.py --data_path dataset/nvidia_stock_prices.csv --output_dir ckpt_gpt2
python test_GPT2.py
```

预测结果会写入 `predictions_gpt2.csv`。

### GPT-2 + Prompt Tuning

```bash
cd code_meng
python train_prompt_tuning.py
python predict_prompt_tuning.py
```

或：

```bash
cd code_skaiu
python train_prompt_tuning.py
python predict_prompt_tuning.py
```

预测结果会写入 `predictions_prompt.csv`。

### GPT-2 + IA3

```bash
cd 'code&results_ZHUJingwen'
python IA3_Jingwen.py
python IA3_test.py
```

预测对比结果会写入 `stock_prediction_last3days.csv`。

## 当前局限

- 训练集是 NVIDIA 历史股价，但测试样例文件第二行标记为 AAPL，训练和测试分布并不一致。
- 多个脚本包含硬编码的本地模型路径，跨机器复现需要手动修改。
- Prompt Tuning 脚本主要优化语言模型生成损失，验证指标中的 RMSE 是近似指标，不等价于真实价格 RMSE。
- 多步预测使用滚动窗口，把上一日预测值作为下一日输入，误差会逐日累积。
- 子目录中有重复数据集、重复 tokenizer 和重复训练逻辑，项目还没有统一配置入口。

## 后续发展方向

1. 统一工程结构：抽出共享的数据加载、滑动窗口、评估指标、模型保存/加载模块。
2. 增加可复现实验配置：用 YAML/JSON 管理模型名、数据路径、窗口长度、学习率、训练轮数和输出目录。
3. 修正数据协议：明确训练股票、测试股票、时间范围和预测目标，避免混用 NVDA/AAPL 时误读结果。
4. 加入真实评估指标：MAE、RMSE、MAPE、方向准确率、分股票/分时间段回测指标。
5. 建立强基线：与 naive last-close、移动平均、ARIMA/LSTM/Transformer 等基线对比。
6. 改善多步预测：训练直接输出 3 日预测向量，或使用 seq2seq/多头回归方式减少滚动误差扩散。
7. 扩展输入特征：加入技术指标、收益率、波动率、成交量变化、市场指数或新闻情绪特征。
8. 增加自动化检查：添加最小单元测试、数据 schema 校验和轻量 CI，先保证脚本在干净环境中可运行。
