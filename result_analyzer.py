import json
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 尝试从 scikit-learn 导入评估指标
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn 库未安装。将无法计算 MAE, RMSE, 和 R² 指标。")
    print("请运行 'pip install scikit-learn' 进行安装。")


def analyze_differences(file_path):
    """
    读取JSON数据文件，全面分析'output'和'predicted_output'的准确率，
    并生成和保存多张统计图表。
    """
    # --- 1. JSON文件解析 ---
    print(f"正在分析文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return
    except json.JSONDecodeError as e:
        print(f"错误：文件内容不是一个有效的JSON。请检查文件格式。")
        print(f"JSON解析错误信息: {e}")
        return
    except Exception as e:
        print(f"读取或解析文件时发生未知错误: {e}")
        return

    # 确保数据是列表格式，以统一处理
    if not isinstance(data, list):
        if isinstance(data, dict):
            data = [data]  # 如果文件是单个JSON对象，将其放入列表中
        else:
            print(f"错误：期望从文件中解析出JSON数组（列表），但得到了类型 {type(data)}。")
            return

    if not data:
        print("错误：JSON文件中不包含任何数据。")
        return
        
    if data and isinstance(data[0], list):
        data = data[0]

    # --- 2. 数据提取和计算 ---
    try:
        outputs = [item['output'] for item in data]
        predicted_outputs = [item['predicted_output'] for item in data]
        differences = [p - o for o, p in zip(outputs, predicted_outputs)]
        residuals = [o - p for o, p in zip(outputs, predicted_outputs)] # 残差 = 真实 - 预测
    except KeyError as e:
        print(f"错误：数据中缺少键 {e}。请确保每个JSON对象都包含 'output' 和 'predicted_output'。")
        return
    except TypeError:
        print(f"错误：数据结构不正确。期望一个字典列表，但列表中的元素不是字典。第一个元素的类型是: {type(data[0])}")
        return

    # --- 3. 打印统计摘要和性能指标 ---
    df = pd.DataFrame({'difference': differences})
    print("\n--- 差值统计摘要 (predicted - actual) ---")
    print(df.describe())

    if SKLEARN_AVAILABLE:
        mae = mean_absolute_error(outputs, predicted_outputs)
        mse = mean_squared_error(outputs, predicted_outputs)
        rmse = mse**0.5
        r2 = r2_score(outputs, predicted_outputs)

        print("\n--- 性能评估指标 ---")
        print(f"平均绝对误差 (MAE):   {mae:.4f}")
        print(f"均方根误差 (RMSE):   {rmse:.4f}")
        print(f"R² 分数 (R-squared): {r2:.4f}")
        print("------------------------")

    # --- 4. 绘图并保存 ---
    
    # 获取不带扩展名的基本文件名，用于命名输出文件
    base_name = os.path.splitext(file_path)[0]

    # 图表一：误差分布直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(differences, kde=True, bins=30)
    plt.title('Prediction Difference (predicted_output - output) Distribution')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.grid(True)
    dist_filename = f"{base_name}_distribution.png"
    plt.savefig(dist_filename)
    plt.close()
    print(f"\n[1/3] 误差分布图已保存到: {dist_filename}")

    # 图表二：真实值 vs. 预测值 散点图
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=outputs, y=predicted_outputs, alpha=0.6)
    max_val = max(max(outputs), max(predicted_outputs))
    min_val = min(min(outputs), min(predicted_outputs))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x (Perfect Prediction)')
    plt.title('True Values vs. Predicted Values')
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    scatter_filename = f"{base_name}_scatter_plot.png"
    plt.savefig(scatter_filename)
    plt.close()
    print(f"[2/3] 散点图已保存到: {scatter_filename}")

    # 图表三：残差图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=predicted_outputs, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs. Predicted Values')
    plt.xlabel('Predicted Output')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    residual_filename = f"{base_name}_residuals_plot.png"
    plt.savefig(residual_filename)
    plt.close()
    print(f"[3/3] 残差图已保存到: {residual_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="全面分析模型预测的准确率，并生成摘要报告和多张可视化图表。"
    )
    parser.add_argument(
        "filename", 
        type=str, 
        help="包含JSON数据的文件的路径 (例如: my_results.json)。"
    )
    
    args = parser.parse_args()
    analyze_differences(args.filename)
