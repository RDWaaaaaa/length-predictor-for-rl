# 导入所需的库
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 设置matplotlib使用'Agg'后端，适用于无图形界面的服务器环境
plt.switch_backend('Agg')

# --- 1. 处理输入参数 ---
if len(sys.argv) < 2:
    print("Usage: python output_analyzer.py <input_json_file>")
    sys.exit(1)

input_filename = sys.argv[1]

# --- 2. 读取和解析JSON文件 ---
output_sizes = []
try:
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            if "output" in item:
                output_sizes.append(item["output"])
except FileNotFoundError:
    print(f"Error: File '{input_filename}' not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{input_filename}'.")
    sys.exit(1)

if not output_sizes:
    print("No 'output' values found in the JSON file.")
    sys.exit(0)

# --- 3. 计算并打印基本统计信息 ---
print(f"总样本数：{len(output_sizes)}")
print(f"output最小值：{min(output_sizes)}")
print(f"output最大值：{max(output_sizes)}")
print(f"output平均值：{np.mean(output_sizes):.2f}")

# --- 4. 生成并保存统计图表 ---
plt.figure(figsize=(12, 5))

# 绘制直方图
plt.subplot(1, 2, 1)
plt.hist(output_sizes, bins=54, range=(0, 2700), color='skyblue', edgecolor='black')
plt.title('Output Size Distribution (Histogram)')
plt.xlabel('Output Size')
plt.ylabel('Frequency')

# 绘制箱线图
plt.subplot(1, 2, 2)
plt.boxplot(output_sizes, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Output Size Distribution (Boxplot)')
plt.ylabel('Output Size')

plt.tight_layout()

# --- 5. 保存图像到文件 ---

base_path_without_ext = os.path.splitext(input_filename)[0]
output_image_filename = f"{base_path_without_ext}_distribution.png"

try:
    plt.savefig(output_image_filename, dpi=300, bbox_inches='tight')
    print(f"图像已保存为 '{output_image_filename}'")
except Exception as e:
    print(f"错误：保存图像失败。原因: {e}")

