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
base_size = 32
paras_encoder = {
    'size_inp': 3,
    'size_middle': base_size,
    'size_oup': num_measure_point
}
paras_decoder = {
    'size_multi_head': 2,
    'size_state': num_measure_point,
    'size_middle': base_size,
    'seq_inp': sequence_init + sequence_predict,
    'seq_oup': 1
}
paras = {
    'encoder': paras_encoder,
    'decoder': paras_decoder,
    'seq_all': sequence_init + sequence_predict,
    'seq_predict': sequence_predict,
    'num_measure_point': num_measure_point
}
