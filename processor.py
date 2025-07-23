import json
import sys
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ================== 配置项 ==================
TOKENIZER_NAME = 'gpt2'
DEFAULT_RL_STEP = 0
TEST_SET_SIZE = 10
# ==========================================

def download_from_hf(repo_id, filepath, save_dir="."):
    """
    从Hugging Face Hub下载指定的数据集文件。
    (使用 shutil.copy 替代 rename 来避免跨设备链接错误)

    :param repo_id: Hugging Face上的仓库ID (例如 'databricks/databricks-dolly-15k')
    :param filepath: 仓库中的文件路径 (例如 'databricks-dolly-15k.jsonl')
    :param save_dir: 文件保存的本地目录
    :return: 下载文件的本地路径对象 (Path object)
    """
    print(f"正在从 Hugging Face Hub 下载文件...")
    print(f"仓库: {repo_id}")
    print(f"文件: {filepath}")
    try:
        downloaded_path_str = hf_hub_download(repo_id=repo_id, filename=filepath, repo_type='dataset')
        source_path = Path(downloaded_path_str)
        local_path = Path(save_dir) / Path(filepath).name
        shutil.copy(source_path, local_path)
        print(f"文件成功下载并保存到: '{local_path}'")
        return local_path
    except Exception as e:
        print(f"错误: 从 Hugging Face Hub 下载或复制文件失败: {e}")
        sys.exit(1)

def load_data_from_file(file_path):
    """
    从本地文件加载数据，支持 .json 和 .jsonl 格式。

    :param file_path: Path对象，指向输入文件
    :return: 包含数据的列表
    """
    if not file_path.exists():
        print(f"错误: 文件 '{file_path}' 不存在")
        sys.exit(1)

    suffix = file_path.suffix.lower()
    print(f"正在读取文件: '{file_path}' (格式: {suffix})")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if suffix == '.json':
                return json.load(f)
            elif suffix == '.jsonl':
                return [json.loads(line) for line in f]
            else:
                print(f"错误: 不支持的文件格式 '{suffix}'。仅支持 '.json' 和 '.jsonl'。")
                sys.exit(1)
    except Exception as e:
        print(f"错误: 读取或解析文件 '{file_path}' 失败: {e}")
        sys.exit(1)

def process_data(input_data, tokenizer):
    """
    处理加载的数据，转换成目标格式。

    :param input_data: 从文件加载的数据列表
    :param tokenizer: Hugging Face Tokenizer实例
    :return: 处理后的数据列表
    """
    processed_data = []
    print("正在处理数据:")
    for item in tqdm(input_data, desc="处理进度", unit="条"):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        prompt = instruction
        if input_text:
            prompt += " " + input_text
        
        token_count = len(tokenizer.encode(output_text, truncation=False, max_length=None, return_tensors=None))
        
        processed_item = {
            "prompt": prompt.strip(),
            "rl_step": DEFAULT_RL_STEP,
            "output": token_count
        }
        processed_data.append(processed_item)
    
    return processed_data

def save_data_to_file(data, output_path):
    """
    将处理后的数据保存到JSON文件。

    :param data: 要保存的数据列表
    :param output_path: Path对象，指向输出文件
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"成功处理并保存到 '{output_path}'")
    except Exception as e:
        print(f"错误: 写入文件 '{output_path}' 失败: {e}")
        sys.exit(1)

def main():
    """
    主执行函数
    """
    if len(sys.argv) == 2:
        input_file_path = Path(sys.argv[1])
    elif len(sys.argv) == 3:
        repo_id = sys.argv[1]
        file_in_repo = sys.argv[2]
        input_file_path = download_from_hf(repo_id, file_in_repo)
    else:
        print("使用方法:")
        print("  1. 处理本地文件: python processor.py <本地JSON或JSONL文件名>")
        print("  2. 下载并处理HF数据集: python processor.py <HF仓库ID> <仓库中的文件名>")
        print("\n示例:")
        print("  python processor.py my_data.jsonl")
        print("  python processor.py databricks/databricks-dolly-15k databricks-dolly-15k.jsonl")
        sys.exit(1)

    input_data = load_data_from_file(input_file_path)
    
    print(f"正在加载分词器: '{TOKENIZER_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, model_max_length=10**12)
    
    processed_data = process_data(input_data, tokenizer)
    
    # --- 保存处理后的训练集文件 ---
    print("\n--- 开始保存处理后的文件 ---")
    output_file = input_file_path.stem + "_processed" + ".json"
    output_path = input_file_path.parent / output_file
    save_data_to_file(processed_data, output_path)

    # --- 生成并保存测试集 ---
    if len(processed_data) >= TEST_SET_SIZE:
        print(f"\n--- 数据量充足，开始生成 {TEST_SET_SIZE} 条数据的测试集 ---")
        test_data = processed_data[:TEST_SET_SIZE]
        test_file = input_file_path.stem + "_processed_test" + ".json"
        test_path = input_file_path.parent / test_file
        save_data_to_file(test_data, test_path)
    else:
        print(f"\n数据量不足 {TEST_SET_SIZE} 条，不生成测试集。")

if __name__ == "__main__":
    main()
