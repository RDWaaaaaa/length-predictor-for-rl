# 导入所需的库
import json  # 用于处理JSON数据
import matplotlib.pyplot as plt  # 用于数据可视化
import numpy as np  # 用于数值计算，如此处的平均值
import sys  # 用于访问命令行参数
import os  # 用于处理文件路径

# 设置matplotlib使用'Agg'后端。这是一个非交互式后端，它将图像渲染到文件而不是屏幕上，
# 适用于在没有图形用户界面的服务器环境中运行脚本。
plt.switch_backend('Agg')

# --- 1. 处理输入参数 ---
# 检查脚本运行时是否提供了足够的命令行参数（至少需要一个文件名）
if len(sys.argv) < 2:
    # 如果没有提供文件名，则打印用法说明并退出脚本
    print("Usage: python analyze_dolly.py <input_json_file>")
    sys.exit(1)

# 从命令行参数中获取输入的JSON文件名
input_filename = sys.argv[1]

# --- 2. 读取和解析JSON文件 ---
# 初始化一个空列表，用于存储从文件中提取的'output'字段的长度或值
output_sizes = []
try:
    # 使用'with'语句安全地打开文件，确保文件在使用后能被正确关闭
    with open(input_filename, "r", encoding="utf-8") as f:
        # 加载整个JSON文件内容到'data'变量中
        data = json.load(f)
        # 遍历JSON数据中的每一个项目（通常是一个字典）
        for item in data:
            # 检查当前项目中是否存在'output'这个键
            if "output" in item:
                # 如果存在，则将其对应的值添加到'output_sizes'列表中
                # 假设这里的'output'值是数值类型，如长度
                output_sizes.append(item["output"])
except FileNotFoundError:
    # 如果文件不存在，则捕获异常，打印错误信息并退出
    print(f"Error: File '{input_filename}' not found.")
    sys.exit(1)
except json.JSONDecodeError:
    # 如果文件不是有效的JSON格式，则捕获异常，打印错误信息并退出
    print(f"Error: Could not decode JSON from '{input_filename}'.")
    sys.exit(1)

# 检查是否成功提取到任何数据
if not output_sizes:
    print("No 'output' values found in the JSON file.")
    sys.exit(0)

# --- 3. 计算并打印基本统计信息 ---
print(f"总样本数：{len(output_sizes)}")
print(f"output最小值：{min(output_sizes)}")
print(f"output最大值：{max(output_sizes)}")
print(f"output平均值：{np.mean(output_sizes):.2f}") # 使用numpy计算平均值并格式化为两位小数

# --- 4. 生成并保存统计图表 ---
# 创建一个新的图形，设置其尺寸为12x5英寸
plt.figure(figsize=(12, 5))

# 创建一个1行2列的子图网格，并激活第一个子图（左侧）
plt.subplot(1, 2, 1)
# 绘制直方图来展示数据的分布
plt.hist(output_sizes, bins=54, range=(0, 2700), color='skyblue', edgecolor='black')
plt.title('Output Size Distribution (Histogram)') # 设置子图标题
plt.xlabel('Output Size') # 设置X轴标签
plt.ylabel('Frequency') # 设置Y轴标签

# 激活第二个子图（右侧）
plt.subplot(1, 2, 2)
# 绘制箱线图，用于展示数据的中位数、四分位数和异常值
plt.boxplot(output_sizes, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Output Size Distribution (Boxplot)') # 设置子图标题
plt.ylabel('Output Size') # 设置Y轴标签

# 自动调整子图参数，使其填充整个图像区域，防止标签重叠
plt.tight_layout()

# --- 5. 保存图像到文件 ---
# 从输入文件名中提取基本名称（不含扩展名和路径）
base_name = os.path.splitext(os.path.basename(input_filename))[0]
# 构建输出图像的文件名
output_image_filename = f"{base_name}_distribution.png"

# 将生成的图形保存到文件
# dpi=300 设置图像分辨率为300点每英寸，以获得高质量图像
# bbox_inches='tight' 自动裁剪图像边缘的空白
plt.savefig(output_image_filename, dpi=300, bbox_inches='tight')
print(f"图像已保存为 '{output_image_filename}'")
