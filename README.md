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
参考`run_workflow.sh`文件，后台进行训练，不受服务器断连影响。  
单独运行时，nohup可避免关闭终端导致训练中断。  

两种方式：
1. 分类classification.py
```sh
nohup accelerate launch classification.py > classification.log 2>&1 &
```
以50 tokens为粒度，LoRA，正确率56%

2. 回归
```sh
nohup accelerate launch classification.py > classification.log 2>&1 &
```
regression.py，LoRA，长度Z-Score标准化
mse: 2914.61：原始尺度上的均方误差
mae: 34.9187: 原始尺度上的平均绝对误差


### 推理
```sh
python regression_inference.py <input_json_file>
```
输入文件名为`xxx.json`，输出文件名为`xxx_inference_results.json`，每条数据只比输入文件多一项predicted_output。  
暂不支持accelerate launch。


## 问题
目前微调和测试用到的数据集均为英文，且数据量不大。
