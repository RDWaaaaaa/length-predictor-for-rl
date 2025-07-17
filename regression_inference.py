import json
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
from tqdm import tqdm
import os
import numpy as np
import argparse

# ================== 配置项 ==================
MODEL_PATH = "./regression-lora-standardscaler/best_model"
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 16
MAX_LENGTH = 512
# ==========================================

class InferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 格式化输入文本
        text = f"Prompt: {item['prompt']}\nRL Step: {item['rl_step']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

def main():
    # --- 新增：设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="使用训练好的模型对JSON文件进行推理。")
    parser.add_argument("input_filename", type=str, help="需要进行推理的输入JSON文件的路径。")
    args = parser.parse_args()
    input_filename = args.input_filename # 从命令行获取输入文件名

    # --- 路径和设备检查 ---
    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型路径 '{MODEL_PATH}' 不存在。")
        return
    
    scaler_path = os.path.join(MODEL_PATH, "label_scaler.pkl")
    if not os.path.exists(scaler_path):
        print(f"错误：在 '{MODEL_PATH}' 中未找到 'label_scaler.pkl'。")
        return
        
    if not os.path.exists(input_filename):
        print(f"错误：输入数据文件 '{input_filename}' 不存在。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的设备: {device}")

    # --- 加载模型和分词器 ---
    print(f"正在从 '{MODEL_PATH}' 加载组件...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    if tokenizer.pad_token is None:
        print("警告: 分词器未设置 pad_token。正在将其设置为 eos_token。")
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=1,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
        trust_remote_code=True
    )
    
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model = model.to(device)
    model.eval()
    
    scaler = joblib.load(scaler_path)
    print("模型、分词器和Scaler加载成功！")

    # --- 准备数据 ---
    print(f"正在加载推理数据: {input_filename}")
    with open(input_filename, 'r', encoding='utf-8') as f:
        inference_data = json.load(f)
    
    inference_dataset = InferenceDataset(inference_data, tokenizer, max_length=MAX_LENGTH)
    
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_predictions = []
    print(f"\n开始对 {len(inference_dataset)} 条数据进行推理...")
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="推理中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits.detach().cpu().to(torch.float32).numpy()
            all_predictions.append(logits)

    scaled_predictions = np.vstack(all_predictions)

    print("正在将预测结果转换回原始数值...")
    original_scale_predictions = scaler.inverse_transform(scaled_predictions)

    results_with_predictions = []
    for i, item in enumerate(tqdm(inference_data, desc="整合结果")):
        new_item = item.copy()
        new_item['predicted_output'] = float(original_scale_predictions[i][0])
        results_with_predictions.append(new_item)

    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_inference_results{ext}"
    
    print(f"\n推理完成！正在将结果保存到: {output_filename}")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results_with_predictions, f, ensure_ascii=False, indent=2)
        print("结果保存成功！")
    except IOError as e:
        print(f"文件保存失败。错误: {e}")

if __name__ == "__main__":
    main()
