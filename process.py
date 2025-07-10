import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

def process_data(input_data):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    processed_data = []
    for item in input_data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        # 合并instruction和input作为prompt
        prompt = instruction
        if input_text:
            prompt += " " + input_text
        
        # 计算分词数量
        # token_count = len(tokenizer.encode(output_text))
        token_count = len(tokenizer.encode(output_text, truncation=True, max_length=1024))
        
        # 构建新的数据结构
        processed_item = {
            "prompt": prompt,
            "output_sequence": output_text,
            "rl_step": 0,
            "output": token_count
        }
        processed_data.append(processed_item)
    return processed_data

def main():
    if len(sys.argv) != 2:
        print("使用方法: python json_processor.py <输入文件名>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    input_path = Path(input_file)
    
    # 确保输入文件存在且为JSON文件
    if not input_path.exists():
        print(f"错误: 文件 '{input_file}' 不存在")
        sys.exit(1)
    if input_path.suffix.lower() != '.json':
        print(f"错误: 文件 '{input_file}' 不是JSON文件")
        sys.exit(1)
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"错误: 读取文件 '{input_file}' 失败: {e}")
        sys.exit(1)
    
    # 处理数据
    processed_data = process_data(input_data)
    
    # 生成输出文件名
    output_file = input_path.stem + "_processed" + input_path.suffix
    output_path = input_path.parent / output_file
    
    # 写入输出文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"成功处理并保存到 '{output_path}'")
    except Exception as e:
        print(f"错误: 写入文件 '{output_path}' 失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()