import torch

# 开式温度计算常数
K = 273.16

# 单个电芯测点数量
num_measure_point = 7

# 时序设定
sequence_init = 1
sequence_predict = 999

# 模型参数
device = torch.device('cuda:0')

# PreEncoder
paras_PreEncoder = {
    'num_measure_point': num_measure_point,
    'num_pre_encoder': 1000,
    'max_epochs': 200,
    'lr_init': 1e-3,
    'size_middle': 16
}

# Prediction_Seq2Seq
paras_Prediction_Seq2Seq = {
    'num_measure_point': num_measure_point,
    'seq_history': 50,
    'seq_predict': 500,
    'max_epochs': 200,
    'lr_init': 1e-3,
    'size_middle': 16,
    'device': device
}

base_size = 32
batch_size = 32
lr_init = 1e-4
epoch_max = 100
seq_concern = 10
paras_encoder = {
    'size_inp': 2,
    'size_middle': base_size,
    'size_oup': num_measure_point,
    'size_state': num_measure_point
}
paras_decoder = {
    'size_multi_head': 1,
    'size_state': num_measure_point,
    'size_middle': base_size,
    'seq_inp': seq_concern,
    'seq_oup': 1
}
paras = {
    'encoder': paras_encoder,
    'decoder': paras_decoder,
    'seq_concern': seq_concern,
    'seq_predict': sequence_predict,
    'num_measure_point': num_measure_point
}
