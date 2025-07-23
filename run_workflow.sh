#!/bin/bash

# ==============================================================================
# 使用说明:
#
# 启动一个tmux会话:
#   tmux new -s my_training
#
# 在会话中根据需要运行脚本:
#   ./run_workflow.sh
#
# 即使服务器断连，任务也能在 tmux 中继续运行。
#
# 其他 tmux 命令:
#   tmux ls                      # 列出所有正在运行的tmux会话
#   tmux attach -t my_training   # 附加到会话，查看实时输出
#   tmux kill-session -t my_training # 结束会话
# ==============================================================================

# 脚本出错时立即退出
set -e

# 使用 accelerate 运行回归脚本
run_task_accelerate() {
  echo "##################################################"
  echo "#               开始执行 Accelerate 任务                 #"
  echo "##################################################"
  
  # 定义需要运行的 Python 脚本列表
  local PYTHON_SCRIPTS=("regression.py")
  local total_scripts=${#PYTHON_SCRIPTS[@]}
  local current_script_num=1

  for script in "${PYTHON_SCRIPTS[@]}"; do
    # 检查脚本文件是否存在
    if [ ! -f "$script" ]; then
      echo "=================================================="
      echo "错误：脚本 '$script' 未找到，跳过执行。"
      echo "=================================================="
      ((current_script_num++))
      continue
    fi

    # 为日志文件添加后缀以作区分
    local log_file="${script%.py}_accelerate.log"

    echo "=================================================="
    echo "开始子任务 ${current_script_num}/${total_scripts}: 正在运行 $script..."
    echo "开始时间: $(date)"
    echo ""

    # 使用 accelerate launch 运行脚本，并将标准输出和错误重定向到日志文件
    accelerate launch "$script" > "$log_file" 2>&1

    echo ""
    echo "子任务 ${current_script_num}/${total_scripts} ($script) 已成功完成。"
    echo "日志已保存到: $log_file"
    echo "结束时间: $(date)"
    echo "=================================================="
    echo ""

    ((current_script_num++))
  done
  
  echo "##################################################"
  echo "#               Accelerate 任务执行完毕                #"
  echo "##################################################"
  echo ""
}

# --- 主程序入口 ---

# 直接调用任务函数
run_task_accelerate

echo "所有任务已执行完毕。"
