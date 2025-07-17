import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

# ================== 配置项 ==================
TOKENIZER_NAME = 'gpt2'
DEFAULT_RL_STEP = 0
TEST_SET_SIZE = 10
# ==========================================

def process_data(input_data, tokenizer):
    processed_data = []
    for item in input_data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        prompt = instruction
        if input_text:
            prompt += " " + input_text
        
        token_count = len(tokenizer.encode(output_text, truncation=False, max_length=None, return_tensors=None))
        
        processed_item = {
            "prompt": prompt,
            "rl_step": DEFAULT_RL_STEP,
            "output": token_count
        }
        processed_data.append(processed_item)
    return processed_data

def main():
    if len(sys.argv) != 2:
        print("使用方法: python processor.py <输入文件名>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"错误: 文件 '{input_file}' 不存在")
        sys.exit(1)
    if input_path.suffix.lower() != '.json':
        print(f"错误: 文件 '{input_file}' 不是JSON文件")
        sys.exit(1)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"错误: 读取文件 '{input_file}' 失败: {e}")
        sys.exit(1)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, model_max_length=10**12)
    processed_data = process_data(input_data, tokenizer)
    
    output_file = input_path.stem + "_processed" + input_path.suffix
    output_path = input_path.parent / output_file
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"成功处理并保存到 '{output_path}'")
    except Exception as e:
        print(f"错误: 写入文件 '{output_path}' 失败: {e}")
        sys.exit(1)

    if len(processed_data) >= TEST_SET_SIZE:
        test_data = processed_data[:TEST_SET_SIZE]
        test_file = input_path.stem + "_processed_test" + input_path.suffix
        test_path = input_path.parent / test_file
        
        try:
            with open(test_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            print(f"成功生成测试数据集并保存到 '{test_path}'")
        except Exception as e:
            print(f"错误: 写入测试文件 '{test_path}' 失败: {e}")
            sys.exit(1)
        
if __name__ == "__main__":
    main()
