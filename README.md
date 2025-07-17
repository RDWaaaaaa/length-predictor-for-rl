# 基于SFT的推理输出长度预测器

## 文件说明

### 数据集处理
1. processor.py
转换格式
```sh
python processor.py <input_json_file>
```
输入文件名为`xxx.json`，输出文件名为`xxx_processed.json`，以及`xxx_processed_test.json`。`xxx_processed_test.json`保留了前10条数据以便简单测试代码能否正常运行。

2. output_analyzer.py
显示长度分布
```sh
python output_analyzer.py <input_json_file>
```
输入文件名为`xxx.json`，输出文件名为`xxx_processed_distribution.png`。

3. filter.py
筛选出token数大于特定值的数据
```sh
python filter.py <input_json_file>
```

### 微调
后台进行训练，不受服务器断连影响。  
```sh
tmux new -s my_training # 启动一个tmux会话
./run_workflow.sh # 在会话中运行
# 此时即使服务器断连也能正常运行
tmux ls # 列出所有正在运行的tmux会话
tmux attach -t my_training # 附加到会话，可以看到脚本运行的输出
```

单独运行时，nohup可避免关闭终端导致训练中断。  

1. 分类classification.py
```sh
nohup accelerate launch classification.py > classification.log 2>&1 &
```
以 50 tokens 为粒度，LoRA，正确率56%

2. 回归regression.py
```sh
nohup accelerate launch regression.py > regression.log 2>&1 &
```
直接预测长度，LoRA，长度用Z-Score标准化
mse: 2914.61：原始尺度上的均方误差
mae: 34.9187: 原始尺度上的平均绝对误差


### 推理
1. 推理
```sh
python regression_inference.py
```
要修改代码，在其中设置模型路径以及数据集
输入文件名为`xxx.json`，输出文件名为`xxx_inference_results.json`，每条数据只比输入文件多一项predicted_output。  
暂不支持accelerate launch。  

2. 数据分析
```sh
python result_analyzer.py alpaca_gpt4_data_processed_inference_results.json
```

3. 集成
可以直接处理原始数据集、生成过滤前后的数据集，分别推理、分析输出。
```
inference_and_analyze.sh
```
### 示例
模型在Qwen/Qwen2.5-3B-Instruct用alpaca_gpt4_data数据集微调。  
alpaca_gpt4_data的完整测试结果在examples/alpaca_gpt4_data。  
dolly的完整测试结果在examples/dolly。  
测试结果的图像包括：  
1. distribution输入数据集的分布。
2. inference_results_distribution推理结果与真实结果差的分布
3. inference_results_scatter_plot散点图
4. inference_results_residuals_plot残差图
其中2、3、4分别有完整数据集的版本，和只显示长数据（真实token>500）的版本。

## 问题
目前微调和测试用到的数据集均为英文，且数据量不大。  
真实token>500的数据，它们对应的预测整体偏小。  
TODO：LLaMA-Factory