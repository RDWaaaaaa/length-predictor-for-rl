#!/bin/bash

# ==============================================================================
# 使用说明:
#
# 启动一个tmux会话:
#   tmux new -s my_training
#
# 在会话中运行脚本:
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

# --- 任务函数定义 ---

# 任务: 使用 llamafactory-cli 运行回归任务
run_task_llamafactory() {
  echo "##################################################"
  echo "#             开始执行 LlamaFactory 任务             #"
  echo "##################################################"
  
  local config_file="train_regression.yaml"
  local log_file="regression_llamafactory.log"

  if [ ! -f "$config_file" ]; then
    echo "=================================================="
    echo "错误：配置文件 '$config_file' 未找到，跳过执行。"
    echo "=================================================="
    return
  fi

  echo "=================================================="
  echo "正在使用 LlamaFactory 运行 $config_file..."
  echo "开始时间: $(date)"
  echo ""

  # 运行 llamafactory-cli 命令
  rm -rf outputs
  llamafactory-cli train "$config_file" > "$log_file" 2>&1

  echo ""
  echo "LlamaFactory 任务已成功完成。"
  echo "日志已保存到: $log_file"
  echo "结束时间: $(date)"
  echo "=================================================="
  echo ""
  
  echo "##################################################"
  echo "#             LlamaFactory 任务执行完毕             #"
  echo "##################################################"
  echo ""
}

# --- 主逻辑 ---

run_task_llamafactory

echo "所有任务已执行完毕。"
