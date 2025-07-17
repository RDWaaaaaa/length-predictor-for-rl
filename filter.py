import json
import argparse
import os
from tqdm import tqdm

# 只保留token > OUTPUT_VALUE_THRESHOLD的数据 
OUTPUT_VALUE_THRESHOLD = 200

def filter_data(input_filename, output_filename):
    """
    根据预设阈值筛选JSON文件中的数据。

    该函数会读取一个包含JSON对象的列表，并筛选出其中'output'字段值
    大于OUTPUT_VALUE_THRESHOLD的条目，然后将结果写入一个新的JSON文件。

    Args:
        input_filename (str): 输入的JSON文件路径。
        output_filename (str): 输出的JSON文件路径。
    """
    # 验证输入文件是否存在，以避免后续操作出错
    if not os.path.exists(input_filename):
        print(f"错误：输入文件 '{input_filename}' 不存在。")
        return

    # 读取并解析JSON数据，同时处理可能发生的异常
    try:
        print(f"正在读取文件: {input_filename}...")
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("文件读取成功。")
    except json.JSONDecodeError:
        print(f"错误：文件 '{input_filename}' 的JSON格式无效。")
        return
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
        return

    # 筛选数据，并使用tqdm显示处理进度
    print(f"\n开始筛选 'output' 值 > {OUTPUT_VALUE_THRESHOLD} 的条目...")
    filtered_results = [
        item for item in tqdm(data, desc="数据处理进度")
        if 'output' in item and isinstance(item['output'], (int, float)) and item['output'] > OUTPUT_VALUE_THRESHOLD
    ]

    # 根据筛选结果决定后续操作
    if not filtered_results:
        print(f"\n筛选完成，未找到任何符合条件的条目。")
        return

    # 将筛选后的数据写入新文件
    print(f"\n筛选完成！共找到 {len(filtered_results)} 条符合条件的条目。")
    print(f"正在将结果保存到文件: {output_filename}...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # 使用indent参数美化输出的JSON，方便人工阅读
            json.dump(filtered_results, f, ensure_ascii=False, indent=2)
        print(f"结果已成功保存到 {output_filename}。")
    except IOError as e:
        print(f"文件保存失败。错误: {e}")

def main():
    """
    程序主入口，负责解析命令行参数并启动数据筛选流程。
    """
    # 初始化命令行参数解析器，并提供清晰的程序说明
    parser = argparse.ArgumentParser(
        description=f"从JSON文件中筛选出 'output' 值大于 {OUTPUT_VALUE_THRESHOLD} 的条目，并存入新文件。"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="需要处理的输入JSON文件路径。"
    )

    # 解析用户从命令行传入的参数
    args = parser.parse_args()
    input_filename = args.input_file

    # 自动生成输出文件名，例如将 'data.json' 转换为 'data_filtered.json'
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_filtered{ext}"

    # 调用核心函数执行数据筛选
    filter_data(input_filename, output_filename)

if __name__ == '__main__':
    main()
