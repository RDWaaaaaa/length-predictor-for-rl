import json
import torch
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error
from peft import get_peft_model, LoraConfig, TaskType
from swanlab.integration.transformers import SwanLabCallback
# ================== 配置项 ==================
# --- 基本配置 ---
IS_TEST = False
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH_PROCESSED = "examples/alpaca_gpt4_data/alpaca_gpt4_data_processed.json"
DATA_PATH_TEST = "examples/alpaca_gpt4_data/alpaca_gpt4_data_processed_test.json"
OUTPUT_DIR = "./regression"
MAX_LENGTH = 512

# --- 数据集划分比例 ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42

# --- LoRA 配置 ---
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_BIAS = "none"

# --- 训练参数 ---
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 10
LOGGING_DIR = "./regression-lora-standardscaler"
# ==========================================

class AlpacaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Prompt: {item['prompt']}\nRL Step: {item['rl_step']}"
        
        encoding = self.tokenizer(text,
                                 truncation=True,
                                 max_length=self.max_length,
                                 padding="max_length",
                                 return_tensors="pt")
        
        label = float(item['output'])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Data loaded successfully.")
    return data

def create_model_for_regression(model_name, tokenizer):
    print("Loading model for sequence regression...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        num_labels=1,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model for sequence regression loaded successfully.")
    return model

def split_data(data, train_ratio, val_ratio, seed):
    np.random.seed(seed)
    np.random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    val_size = max(1, val_size)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    if not test_data and len(val_data) > 1:
        test_data = val_data[-1:]
        val_data = val_data[:-1]
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

def compute_metrics_for_regression(eval_pred, scaler):
    predictions, labels = eval_pred
    
    predictions_reshaped = predictions.reshape(-1, 1)
    labels_reshaped = labels.reshape(-1, 1)
    
    original_predictions = scaler.inverse_transform(predictions_reshaped)
    original_labels = scaler.inverse_transform(labels_reshaped)
    
    mse = mean_squared_error(original_labels, original_predictions)
    mae = mean_absolute_error(original_labels, original_predictions)
    
    print(f"\nValidation Set Evaluation (Original Scale): MSE={mse:.4f}, MAE={mae:.4f}")
    
    return {
        "mse": mse,
        "mae": mae,
    }

def main():
    data_path = DATA_PATH_TEST if IS_TEST else DATA_PATH_PROCESSED
    
    data = load_data(data_path)
    train_data, val_data, test_data = split_data(data, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    
    print("\nStandardizing labels using StandardScaler...")
    
    train_labels = np.array([float(item['output']) for item in train_data]).reshape(-1, 1)
    
    scaler = StandardScaler()
    scaler.fit(train_labels)
    print(f"Scaler fitted on training data. Mean: {scaler.mean_[0]:.2f}, Scale: {scaler.scale_[0]:.2f}")
    
    for dataset in [train_data, val_data, test_data]:
        if not dataset: continue
        original_labels = np.array([float(item['output']) for item in dataset]).reshape(-1, 1)
        scaled_labels = scaler.transform(original_labels)
        for i, item in enumerate(dataset):
            item['output'] = scaled_labels[i][0]
            
    print("Labels for all datasets have been standardized.")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    model = create_model_for_regression(MODEL_NAME, tokenizer)
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
    )
    model = get_peft_model(model, peft_config)

    print("\n" + "="*50)
    print("Final trainable parameters after applying LoRA:")
    model.print_trainable_parameters()
    print("="*50 + "\n")

    train_dataset = AlpacaDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = AlpacaDataset(val_data, tokenizer, MAX_LENGTH)
    test_dataset = AlpacaDataset(test_data, tokenizer, MAX_LENGTH)
    
    print("Setting TrainingArguments for Regression Task.")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        push_to_hub=False,
        logging_dir=LOGGING_DIR,
        logging_steps=1,
        fp16=False,
        bf16=True,
        bf16_full_eval=True,
        max_grad_norm=1.0,
        save_total_limit=3,
        report_to="none",
    )

    
    compute_metrics_with_scaler = lambda eval_pred: compute_metrics_for_regression(eval_pred, scaler)
    
    swanlab_callback = SwanLabCallback(
        project="Qwen2.5-SFT-Regression", 
        experiment_name=f"{MODEL_NAME.replace('/', '_')}-lora"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_with_scaler,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, 
                early_stopping_threshold=0.01
            ),
            swanlab_callback
        ],
    )
    
    print("Starting model training for regression task...")
    trainer.train()
    
    print("Training finished. Saving the best model and the scaler...")
    import os
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    
    joblib.dump(scaler, os.path.join(best_model_dir, "label_scaler.pkl"))
    
    print("\nEvaluating on the test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Final test results (on original scale): {test_results}")

if __name__ == "__main__":
    main()