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

# 02 多模型方案
## 数据处理方面
load_data之后，不做任何处理，在各自的dataloader内进行处理。
## 模型结构方面
多模型包含了Pre_Encoder模型和All_Prediction模型。
Pre_Encoder用于对前半部分的温度分布进行预测，目前是预测七个测温点各自的温度。  
先计算初始温度的均值、方差和分布，然后通过Cross_Attention预测均值与方差随时间的变化，以及分布随时间的变化，最后进行反算。

# 03 & 04 
# 单模型预测单电芯
## 数据处理方面
load_data之后，不做任何处理，在各自的dataloader内进行处理。  
在dataloader内，用滑动窗口的方式提取数据，并每隔1个点提取一个数据点。模型输入数据包含六个变量，分别是位置编码，NTC最高温，NTC最低温，电压，电流，SOC。
## 模型结构方面
包含一个lstm和attention，每隔50个数据点进行一次attention，并重新lstm。  
在100 * 2个序列上进行训练，在650 * 2个序列上有稳定的表现。  
预测的是最大值的均值和方差，使用高斯负对数似然进行误差评估。  
效果不错！！

# 预测全时序
# 在仿真数据中
## 对25组电芯求取其均值，最大值和最小值，可得，平均均值误差为0.4051K，平均最大值为0.8167K，平均最小值为0.0032K
![0.png](best_version%2Fsingle%2Fsim%2F0.png)
![1.png](best_version%2Fsingle%2Fsim%2F1.png)
![2.png](best_version%2Fsingle%2Fsim%2F2.png)
![3.png](best_version%2Fsingle%2Fsim%2F3.png)
![4.png](best_version%2Fsingle%2Fsim%2F4.png)
![5.png](best_version%2Fsingle%2Fsim%2F5.png)
![6.png](best_version%2Fsingle%2Fsim%2F6.png)
![7.png](best_version%2Fsingle%2Fsim%2F7.png)
![8.png](best_version%2Fsingle%2Fsim%2F8.png)
![9.png](best_version%2Fsingle%2Fsim%2F9.png)
![10.png](best_version%2Fsingle%2Fsim%2F10.png)
![11.png](best_version%2Fsingle%2Fsim%2F11.png)
![12.png](best_version%2Fsingle%2Fsim%2F12.png)
![13.png](best_version%2Fsingle%2Fsim%2F13.png)
![14.png](best_version%2Fsingle%2Fsim%2F14.png)
![15.png](best_version%2Fsingle%2Fsim%2F15.png)
![16.png](best_version%2Fsingle%2Fsim%2F16.png)
![17.png](best_version%2Fsingle%2Fsim%2F17.png)
![18.png](best_version%2Fsingle%2Fsim%2F18.png)
![19.png](best_version%2Fsingle%2Fsim%2F19.png)
![20.png](best_version%2Fsingle%2Fsim%2F20.png)
![21.png](best_version%2Fsingle%2Fsim%2F21.png)
![22.png](best_version%2Fsingle%2Fsim%2F22.png)
![23.png](best_version%2Fsingle%2Fsim%2F23.png)
![24.png](best_version%2Fsingle%2Fsim%2F24.png)
# 实际数据中
![0.png](best_version%2Fsingle%2Freal%2F0.png)
![1.png](best_version%2Fsingle%2Freal%2F1.png)
![2.png](best_version%2Fsingle%2Freal%2F2.png)
![3.png](best_version%2Fsingle%2Freal%2F3.png)