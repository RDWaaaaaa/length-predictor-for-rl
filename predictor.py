import torch
import torch.nn as nn
from datasets import load_dataset
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 加载数据集
# print(dataset.keys())会输出dict_keys(['train'])
# load_dataset 函数会根据 JSON 文件路径加载一个 DatasetDict 对象
# 这个对象中默认包含一个名为 'train' 的数据分割（split），即使原始 JSON 文件中没有显式地定义 "train" 这个键。
dataset = load_dataset("json", data_files="alpaca_gpt4_data_processed.json")

# 提取原始的 train 分割
raw_train = dataset['train']

train_test_split = raw_train.train_test_split(test_size=0.4, seed=42)
train_data = train_test_split['train']
test_val = train_test_split['test']

test_val_split = test_val.train_test_split(test_size=0.5, seed=42)
test_data = test_val_split['test']
val_data = test_val_split['train']
print(f"数据集划分：训练集{len(train_data)}条，验证集{len(val_data)}条，测试集{len(test_data)}条")

# 加载分词器
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 参考 https://huggingface.co/Qwen/Qwen2.5-3B-Instruct 
# tokenizer_config.json中有`"pad_token": "<|endoftext|>"`和`"eos_token": "<|im_end|>"`
# 所以不需要在加载分词器时候写`tokenizer.pad_token = tokenizer.eos_token`  



# 数据预处理，将原始数据转换为模型可接收的输入格式
# 函数最终返回 text_inputs，这是一个字典，其中包含以下内容：
# input_ids：经过分词器转换得到的输入文本的 token ID。
# attention_mask：用于标识哪些是有效 token 的注意力掩码。
# rl_step：强化学习的步数信息。
# labels：模型训练时需要用到的目标文本。
def preprocess_function(examples):
    text_inputs = tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )
    text_inputs["rl_step"] = examples["rl_step"]
    text_inputs["labels"] = examples["output"]
    return text_inputs

# 应用预处理
# remove_columns=train_data.column_names 会移除所有原始列，只保留预处理函数返回的列（如 input_ids, attention_mask）
tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
tokenized_val = val_data.map(preprocess_function, batched=True, remove_columns=val_data.column_names)
tokenized_test = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)

# 定义预测模型（融合prompt和rl_step特征）
class LengthPredictor(nn.Module):
    def __init__(self, base_model_name):
        # 调用父类nn.Module的init方法，因为当子类定义了自己的__init__方法时，Python 不会自动调用父类的__init__，因此需要显式调用。
        super().__init__()
        # 基础模型：处理prompt文本特征
        # 关键参数：
        # input_ids（必需）：形状为 (batch_size, sequence_length) 的整数张量，表示文本的 token ID。
        # attention_mask（可选）：形状与 input_ids 相同的二进制张量，用于指示哪些 token 需要被关注（1 表示有效，0 表示填充）。
        # labels（仅训练时使用）：与 input_ids 形状相同的张量，用于计算损失（通常与 input_ids 相同或偏移一个位置）。
        # 输出是一个包含多个字段的字典，核心输出包括：
        # logits语言模型预测（未经过 softmax）：(batch_size, sequence_length, vocab_size)
        # hidden_states隐藏层表示，tuple of tensors包含了模型各层的隐藏状态, 每个张量形状为 (batch_size, sequence_length, hidden_size)
        # loss（仅当提供 labels 时返回）
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=None
        )
        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # rl_step特征处理层（将整数转换为特征向量）
        # 输入形状(batch_size,)，输出形状(batch_size, 64)
        self.rl_step_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=100, embedding_dim=32),  # 假设rl_step最大不超过100，该层实际上维护着一个大小为 100×32 的查找表。
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Value Head：融合文本特征和rl_step特征
        # 输入维度 = 基础模型隐藏层维度 + rl_step编码维度
        self.fusion_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU()
        )
        
        # 移动子模块到基础模型设备
        # 当使用device_map="auto"参数加载基础模型时
        # Hugging Face Transformers 库会自动将模型分布到可用的 GPU 上（如果有多个 GPU），或默认使用 CPU。
        # 但此时，新创建的子模块（如rl_step_encoder和fusion_head）仍位于 CPU 上，因此需要手动将它们移动到与基础模型相同的设备。
        # self.rl_step_encoder.to(self.base_model.device)
        # self.fusion_head.to(self.base_model.device)
        # 用accelerate则无需手动指定

    def forward(self, input_ids, attention_mask=None, rl_step=None, labels=None):
        # 提取prompt文本特征
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # 文本特征，hidden_states[-1]即最后一层的隐藏状态，形状为 (batch_size, sequence_length, hidden_size)
        # [:, 0, :] 压缩为(batch_size, hidden_size)，还有平均池化、最大池化、注意力池化等
        text_feature = outputs.hidden_states[-1][:, 0, :]  
        
        # 编码rl_step特征
        rl_step_feature = self.rl_step_encoder(rl_step)  # rl_step特征，形状为 (batch_size, 64)
        
        # 融合两种特征
        fused_feature = torch.cat([text_feature, rl_step_feature], dim=1)
        
        # 预测输出长度，形状(batch_size, 1) -> (batch_size)
        predicted_length = self.fusion_head(fused_feature).squeeze(-1)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(predicted_length, labels.float())
        
        return {"loss": loss, "predictions": predicted_length}


# 自定义数据收集器（处理rl_step特征的批量化）
class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # 分离rl_step和其他文本特征
        rl_steps = [{"rl_step": f["rl_step"]} for f in features]
        text_features = [{k: v for k, v in f.items() if k != "rl_step"} for f in features]
        
        # 文本特征用默认方式批量化
        # 确定最大长度：找出最长的 input_ids 长度
        # 填充短序列：对较短的序列补 0
        # 转换为张量：将所有特征转换为 PyTorch 张量
        batch = super().__call__(text_features)
        
        # rl_step特征单独处理
        batch["rl_step"] = torch.tensor([f["rl_step"] for f in rl_steps], device=batch["input_ids"].device)
        return batch

# 评估指标
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    return {
        "mae": mean_absolute_error(labels, predictions), # 平均绝对误差
        "rmse": np.sqrt(mean_squared_error(labels, predictions)) # 均方根误差
    }

# 训练配置
training_args = TrainingArguments(
    output_dir="./qwen_predictor",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    logging_steps=200,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rmse",
    fp16=True,
    report_to="none"
)

model = LengthPredictor(model_name)
data_collator = CustomDataCollator(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()

test_results = trainer.evaluate(tokenized_test)
print("测试集评估结果：", test_results)

trainer.save_model("./qwen_predictor_final")
tokenizer.save_pretrained("./qwen_predictor_final")