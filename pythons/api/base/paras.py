import torch

# 开式温度计算常数
K = 273.16

# 单个电芯测点数量
num_measure_point = 7

# 时序设定
sequence_init = 1
sequence_predict = 9

# 模型参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
