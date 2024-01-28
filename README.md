<!-- TOC -->
* [1216-battery_temperature_prediction](#1216-battery_temperature_prediction)
  * [可视化程序](#可视化程序)
* [01 单模型方案](#01-单模型方案)
  * [数据处理方面](#数据处理方面)
  * [模型结构方面](#模型结构方面)
  * [当前结果](#当前结果)
* [02 多模型方案](#02-多模型方案)
  * [数据处理方面](#数据处理方面-1)
  * [模型结构方面](#模型结构方面-1)
* [03 & 04](#03--04-)
* [多模型预测单电芯](#多模型预测单电芯)
  * [数据处理方面](#数据处理方面-2)
  * [模型结构方面](#模型结构方面-2)
    * [模型1](#模型1)
    * [模型2](#模型2)
* [基于电流和SOC预测电压，NTC_max，NTC_min变化](#基于电流和soc预测电压ntc_maxntc_min变化)
  * [在仿真数据中](#在仿真数据中)
    * [对1组电芯求取其均值，最大值和最小值，可得：](#对1组电芯求取其均值最大值和最小值可得)
  * [在真实数据中](#在真实数据中)
    * [对1组电芯求取其均值，最大值和最小值，可得：](#对1组电芯求取其均值最大值和最小值可得-1)
* [基于理想数据预测温度](#基于理想数据预测温度)
  * [在仿真数据中](#在仿真数据中-1)
    * [对25组电芯求取其均值，最大值和最小值，可得：](#对25组电芯求取其均值最大值和最小值可得)
  * [实际数据中](#实际数据中)
    * [对4组电芯求取其均值，最大值和最小值，可得：](#对4组电芯求取其均值最大值和最小值可得)
* [模型结合进行温度预测](#模型结合进行温度预测)
  * [在仿真数据中](#在仿真数据中-2)
    * [对25组电芯求取其均值，最大值和最小值，可得:](#对25组电芯求取其均值最大值和最小值可得-1)
  * [在真实数据中](#在真实数据中-1)
    * [对4组电芯求取其均值，最大值和最小值，可得:](#对4组电芯求取其均值最大值和最小值可得-1)
<!-- TOC -->

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
# 多模型预测单电芯
## 数据处理方面
load_data之后，不做任何处理，在各自的dataloader内进行处理。  
在dataloader内，用滑动窗口的方式提取数据，并每隔1个点提取一个数据点。模型输入数据包含六个变量，分别是位置编码，NTC最高温，NTC最低温，电压，电流，SOC。
## 模型结构方面
首先，模型1根据电流策略对SOC、电压、NTC最高温和NTC最低温进行预测  
其次，模型2基于预测出的数据进行温度预测  
### 模型1

### 模型2
包含一个lstm和attention，每隔50个数据点进行一次attention，并重新lstm。  
在100 * 2个序列上进行训练，在650 * 2个序列上有稳定的表现。  
预测的是最大值的均值和方差，使用高斯负对数似然进行误差评估。  
效果不错！！

# 基于电流和SOC预测电压，NTC_max，NTC_min变化
## 在仿真数据中
### 对1组电芯求取其均值，最大值和最小值，可得：
    Voltage:
        平均均值误差：0.1390V
        平均最大误差：1.5855V
        平均最小误差：0.0001V
    NTC_max:
        平均均值误差：0.2247℃
        平均最大误差：1.4775℃
        平均最小误差：0.0002℃
    NTC_min:
        平均均值误差：0.4879℃
        平均最大误差：1.4821℃
        平均最小误差：0.0006℃
|                                        |
|----------------------------------------|
| ![0.png](best_version/state/sim/0.png) |

## 在真实数据中
### 对1组电芯求取其均值，最大值和最小值，可得：
    Voltage:
        平均均值误差：0.2472V
        平均最大误差：1.3492V
        平均最小误差：0.0001V
    NTC_max:
        平均均值误差：1.5150℃
        平均最大误差：3.6811℃
        平均最小误差：0.0015℃
    NTC_min:
        平均均值误差：1.0841℃
        平均最大误差：2.5481℃
        平均最小误差：0.0042℃
|                                         |
|-----------------------------------------|
| ![0.png](best_version/state/real/0.png) |

# 基于理想数据预测温度
## 在仿真数据中
### 对25组电芯求取其均值，最大值和最小值，可得：
    平均均值误差：0.4504K
    平均最大误差：0.8132K
    平均最小误差：0.0140K
|                                                |                                                |
|------------------------------------------------|------------------------------------------------|
| ![0.png](best_version/temperature/sim/0.png)   | ![13.png](best_version/temperature/sim/13.png) |
| ![1.png](best_version/temperature/sim/1.png)   | ![14.png](best_version/temperature/sim/14.png) |
| ![2.png](best_version/temperature/sim/2.png)   | ![15.png](best_version/temperature/sim/15.png) |
| ![3.png](best_version/temperature/sim/3.png)   | ![16.png](best_version/temperature/sim/16.png) |
| ![4.png](best_version/temperature/sim/4.png)   | ![17.png](best_version/temperature/sim/17.png) |
| ![5.png](best_version/temperature/sim/5.png)   | ![18.png](best_version/temperature/sim/18.png) |
| ![6.png](best_version/temperature/sim/6.png)   | ![19.png](best_version/temperature/sim/19.png) |
| ![7.png](best_version/temperature/sim/7.png)   | ![20.png](best_version/temperature/sim/20.png) |
| ![8.png](best_version/temperature/sim/8.png)   | ![21.png](best_version/temperature/sim/21.png) |
| ![9.png](best_version/temperature/sim/9.png)   | ![22.png](best_version/temperature/sim/22.png) |
| ![10.png](best_version/temperature/sim/10.png) | ![23.png](best_version/temperature/sim/23.png) |
| ![11.png](best_version/temperature/sim/11.png) | ![24.png](best_version/temperature/sim/24.png) |
| ![12.png](best_version/temperature/sim/12.png) |                                                |
## 实际数据中
### 对4组电芯求取其均值，最大值和最小值，可得：
    平均均值误差：0.9195K
    平均最大误差：2.1509K
    平均最小误差：0.0011K
|                                               |                                               |
|-----------------------------------------------|-----------------------------------------------|
| ![0.png](best_version/temperature/real/0.png) | ![2.png](best_version/temperature/real/2.png) |                                       |
| ![1.png](best_version/temperature/real/1.png) | ![3.png](best_version/temperature/real/3.png) |

# 模型结合进行温度预测
## 在仿真数据中
### 对25组电芯求取其均值，最大值和最小值，可得:
    Voltage:
        平均均值误差：0.1390V
        平均最大误差：1.5855V
        平均最小误差：0.0001V
    NTC_max:
        平均均值误差：0.2247℃
        平均最大误差：1.4775℃
        平均最小误差：0.0002℃
    NTC_min:
        平均均值误差：0.4879℃
        平均最大误差：1.4821℃
        平均最小误差：0.0006℃
    Temperature:
        平均均值误差：0.4503℃
        平均最大误差：0.8023℃
        平均最小误差：0.0113℃
|                                        |                                        |
|----------------------------------------|----------------------------------------|
| ![0.png](best_version/all/sim/0.png)   | ![13.png](best_version/all/sim/13.png) |
| ![1.png](best_version/all/sim/1.png)   | ![14.png](best_version/all/sim/14.png) |
| ![2.png](best_version/all/sim/2.png)   | ![15.png](best_version/all/sim/15.png) |
| ![3.png](best_version/all/sim/3.png)   | ![16.png](best_version/all/sim/16.png) |
| ![4.png](best_version/all/sim/4.png)   | ![17.png](best_version/all/sim/17.png) |
| ![5.png](best_version/all/sim/5.png)   | ![18.png](best_version/all/sim/18.png) |
| ![6.png](best_version/all/sim/6.png)   | ![19.png](best_version/all/sim/19.png) |
| ![7.png](best_version/all/sim/7.png)   | ![20.png](best_version/all/sim/20.png) |
| ![8.png](best_version/all/sim/8.png)   | ![21.png](best_version/all/sim/21.png) |
| ![9.png](best_version/all/sim/9.png)   | ![22.png](best_version/all/sim/22.png) |
| ![10.png](best_version/all/sim/10.png) | ![23.png](best_version/all/sim/23.png) |
| ![11.png](best_version/all/sim/11.png) | ![24.png](best_version/all/sim/24.png) |
| ![12.png](best_version/all/sim/12.png) |                                        |

## 在真实数据中
### 对4组电芯求取其均值，最大值和最小值，可得:
    Voltage:
        平均均值误差：0.2472V
        平均最大误差：1.3492V
        平均最小误差：0.0001V
    NTC_max:
        平均均值误差：1.5150℃
        平均最大误差：3.6811℃
        平均最小误差：0.0014℃
    NTC_min:
        平均均值误差：1.0841℃
        平均最大误差：2.5481℃
        平均最小误差：0.0042℃
    Temperature:
        平均均值误差：0.9212℃
        平均最大误差：2.3700℃
        平均最小误差：0.0050℃
|                                       |                                       |
|---------------------------------------|---------------------------------------|
| ![0.png](best_version/all/real/0.png) | ![2.png](best_version/all/real/2.png) |                                       |
| ![1.png](best_version/all/real/1.png) | ![3.png](best_version/all/real/3.png) |
