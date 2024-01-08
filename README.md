# 1216-battery_temperature_prediction
Predicting cell temperature in a data-driven manner
## 可视化程序
for /l %x in (1, 0, 80) do (type result.txt) & (TIMEOUT /T 1) & CLS

# 01 单模型方案
## 数据处理方面
输入为初始温度，输出为999个时序的温度，每组数据集内包含了1000个时序。  
Encoder输入固定长度的10个序列，当预测的点数加上初始序列的长度小于10时，在后端补0。  
Decoder输入与Encoder类似，但会多一个下一时刻的电流与SOC值。
## 模型结构方面
在本方案中，构建了Encoder和Decoder。  
Encoder通过LSTM和Self-Attention对充电电流与SOC进行编码，生成q。  
Decoder先计算当前时刻测温点温度的均值与标准差，再通过Self-Attention对历史的各测点温度经过LayNorm之后进行编码，生成k和v。
通过全连接层对下一时刻的均值与标准差进行更新。使用Cross-Attention对qkv计算，得到分布，再与更新之后的均值与标准差进行计算，生成下一时刻的各测点温度。
## 当前结果
使用逐点累加的方式进行训练，但当点数增加到一定程度时，梯度爆炸，无法收敛，_**放弃了**_。
