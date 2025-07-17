#!/bin/bash

# 脚本出错时立即退出
set -e

# 定义要处理的数据集名称
DATASETS=(
  "dolly"
)

# 获取数据集总数
total_datasets=${#DATASETS[@]}
current_dataset_num=1

# 循环处理每个数据集
for dataset_name in "${DATASETS[@]}"; do
  echo "=================================================="
  echo "开始处理数据集 ${current_dataset_num}/${total_datasets}: ${dataset_name}"
  echo "开始时间: $(date)"
  echo "=================================================="
  echo ""

  # --- 步骤 1: 数据预处理 ---
  echo "[步骤 1/4] 正在进行数据预处理..."
  input_file="examples/${dataset_name}/${dataset_name}.json"
  processed_file="examples/${dataset_name}/${dataset_name}_processed.json"
  python processor.py "${input_file}"
  echo "数据预处理完成，输出文件: ${processed_file}"
  echo ""

  # --- 步骤 2: 初始推理与分析 ---
  echo "[步骤 2/4] 正在对预处理后的数据进行推理与分析..."
  inference_results_file="${processed_file}_inference_results.json"
  
  python regression_inference.py "${processed_file}"
  echo "推理完成。"
  
  # 分析初始推理结果
  python result_analyzer.py "${inference_results_file}"
  echo "初始结果分析完成。"
  echo ""

  # --- 步骤 3: 数据过滤 ---
  echo "[步骤 3/4] 正在过滤数据..."
  filtered_file="examples/${dataset_name}/${dataset_name}_processed_filtered.json"
  python filter.py "${processed_file}"
  echo "数据过滤完成，输出文件: ${filtered_file}"
  echo ""

  # --- 步骤 4: 对过滤后的数据进行推理与分析 ---
  echo "[步骤 4/4] 正在对过滤后的数据进行推理与分析..."
  filtered_inference_results_file="${filtered_file}_inference_results.json"
  
  python regression_inference.py "${filtered_file}"
  echo "对过滤后数据的推理完成。"

  # 分析过滤后的推理结果
  python result_analyzer.py "${filtered_inference_results_file}"
  echo "过滤后结果分析完成。"
  echo ""


  echo "=================================================="
  echo "数据集 ${dataset_name} 已处理完毕。"
  echo "结束时间: $(date)"
  echo "=================================================="
  echo ""

  ((current_dataset_num++))
done

echo "所有数据集均已成功处理完毕。"
