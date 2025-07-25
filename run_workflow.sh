#!/bin/bash

# 启动一个tmux会话：tmux new -s my_training
# 在会话中运行：./run_workflow.sh
# 此时即使服务器断连也能正常运行
# 列出所有正在运行的tmux会话：tmux ls
# 附加到会话可以看到脚本运行的输出：tmux attach -t my_training

# 脚本出错时立即退出
set -e


PYTHON_SCRIPTS=(
  "regression-lora-standardscaler.py"
)

# 获取脚本总数
total_scripts=${#PYTHON_SCRIPTS[@]}
current_script_num=1

# 循环遍历并执行数组中的每个脚本
for script in "${PYTHON_SCRIPTS[@]}"; do
  # 检查文件是否存在
  if [ ! -f "$script" ]; then
    echo "=================================================="
    echo "错误：脚本 '$script' 未找到，跳过执行。"
    echo "=================================================="
    # 更新计数器并继续下一个循环
    ((current_script_num++))
    continue
  fi

  # 从 Python 文件名生成日志文件名 (例如: script.py -> script.log)
  log_file="${script%.py}.log"

  echo "=================================================="
  echo "开始任务 ${current_script_num}/${total_scripts}: 正在运行 $script..."
  echo "开始时间: $(date)"
  echo ""

  # 使用 accelerate launch 运行脚本，并将标准输出和错误都重定向到日志文件
  accelerate launch "$script" > "$log_file" 2>&1

  echo ""
  echo "任务 ${current_script_num}/${total_scripts} ($script) 已成功完成。"
  echo "日志已保存到: $log_file"
  echo "结束时间: $(date)"
  echo "=================================================="
  echo ""

  ((current_script_num++))
done

echo "所有任务已执行完毕。"
