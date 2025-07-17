import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, classification_report
from peft import get_peft_model, LoraConfig, TaskType

# 测试模式标志，控制使用测试数据还是完整数据
isTest = False

class AlpacaDataset(Dataset):
    """
    用于Alpaca格式数据的自定义数据集类，处理文本输入和分类标签
    """
    def __init__(self, data, tokenizer, num_classes, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        self.max_length = max_length
        self.encoded_data = self._preprocess_data()
    
    def _preprocess_data(self):
        encoded_data = []
        print("Preprocessing data...")
        for i, item in enumerate(self.data):
            text = f"Prompt: {item['prompt']}\nRL Step: {item['rl_step']}"
            encoding = self.tokenizer(text, 
                                     truncation=True, 
                                     max_length=self.max_length,
                                     padding="max_length",
                                     return_tensors="pt")
            output = item['output']
            label = output // 50
            assert 0 <= label < self.num_classes, f"Error at data index {i}: Generated label {label} is out of bounds for num_classes={self.num_classes}."
            encoded_data.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            })
        print("Data preprocessing complete.")
        return encoded_data
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        return self.encoded_data[idx]

# 加载数据函数
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Data loaded successfully.")
    return data

# 创建模型函数
def create_model(model_name, num_classes, tokenizer):
    """
    加载用于序列分类的预训练模型。
    """
    print("Loading model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        num_labels=num_classes, # 直接为分类任务配置模型
    )
    # 将模型的 pad_token_id 设置为 tokenizer 的 pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model for sequence classification loaded successfully.")
    return model

# 数据划分函数
def split_data(data, train_ratio=0.8, val_ratio=0.1):
    np.random.seed(42)
    np.random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    # 确保验证集和测试集至少有一个样本
    val_size = max(1, val_size)
    test_size = max(1, len(data) - train_size - val_size)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # 如果测试集为空，则从验证集中划分一个
    if not test_data and len(val_data) > 1:
        test_data = val_data[-1:]
        val_data = val_data[:-1]

    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

def preprocess_logits_for_metrics(logits, labels):
    """
    预处理logits，直接返回预测的类别ID。
    """
    # 对于 SequenceClassification 模型，logits 的形状是 (batch_size, num_classes)
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

# 计算评估指标函数
def compute_metrics(eval_pred):
    """
    计算模型评估指标。
    `eval_pred` 是一个 EvalPrediction 对象，其中包含 predictions 和 label_ids。
    """
    # predictions 可能是 logits，也可能是经过 preprocess_logits_for_metrics 处理后的结果
    predictions, labels = eval_pred
    
    # 如果 predictions 是一个元组（例如，当模型返回多个输出时），我们只取第一个
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # 如果 preprocess_logits_for_metrics 没有被使用，predictions 仍然是 logits
    # 在这种情况下，我们需要从 logits 中获取预测的类别
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    # 确保数据类型正确
    predictions = predictions.flatten().astype(int)
    labels = labels.flatten().astype(int)

    # 现在，它们的长度应该是匹配的
    accuracy = accuracy_score(labels, predictions)
    print("\nClassification Report (Validation Set):")
    # 设置 target_names 可以让报告更具可读性
    # num_classes = len(set(labels)) # 动态获取类别数
    # target_names = [f'class_{i}' for i in range(num_classes)]
    report = classification_report(labels, predictions, zero_division=0, output_dict=False) #, target_names=target_names)
    print(report)
    return {"accuracy": accuracy}

# 主函数
def main():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    if isTest:
        data_path = "alpaca_gpt4_data_processed_test.json"
    else:
        data_path = "alpaca_gpt4_data_processed.json"
    output_dir = "./qwen_sft_output"
    max_length = 512
    
    data = load_data(data_path)
    train_data, val_data, test_data = split_data(data)
    
    max_output = max(item['output'] for item in data)
    num_classes = (max_output // 50) + 1
    print(f"Number of classes determined from data: {num_classes}")
    
    # 先创建 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用 tokenizer 创建模型
    model = create_model(model_name, num_classes, tokenizer)
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    print("\n" + "="*50)
    print("Final trainable parameters after applying LoRA:")
    model.print_trainable_parameters()
    print("="*50 + "\n")

    train_dataset = AlpacaDataset(train_data, tokenizer, num_classes, max_length)
    val_dataset = AlpacaDataset(val_data, tokenizer, num_classes, max_length)
    test_dataset = AlpacaDataset(test_data, tokenizer, num_classes, max_length)
    
    print("Setting TrainingArguments for BF16 (mixed precision) training.")
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir="./logs",
        logging_steps=100,
        fp16=False,
        bf16=True,
        bf16_full_eval=True,
        max_grad_norm=1.0,
        save_total_limit=3,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    print("Starting model training...")
    trainer.train()
    
    print("Training finished. Saving the best model...")
    trainer.save_model(os.path.join(output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
    
    print("\nEvaluating on the test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Final test results: {test_results}")

if __name__ == "__main__":
    main()
