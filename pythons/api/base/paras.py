import torch
import numpy as np
# 开式温度计算常数
K = 273.16

# 单个电芯测点数量
num_measure_point = 7

# 模型参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prediction_State
paras_Prediction_State = {
    'current_to_soc': 2.561927693277231e+03,
    'info_len': 3,
    'state_len': 3,
    'seq_history': 1,
    'seq_predict': 200,
    'seq_attention_once': 50,
    'split_length': 2,  # 取点间隔，间隔为n个数时，split_length=n+1
    'max_epochs': 100,
    'lr_init': 1e-3,
    'size_middle': 16,
    'num_layers': 1,
    'device': device,
    'scale': 100,
    'delta_limit_mean': np.array([50.0, 50.0, 50.0]),
    'delta_limit_var': np.array([10.0, 10.0, 10.0])
}

# Prediction_Temperature
paras_Prediction_Temperature = {
    'num_measure_point': num_measure_point,
    'info_len': 6,
    'seq_history': 1,
    'seq_predict': 200,
    'seq_attention_once': 50,
    'split_length': 2,  # 取点间隔，间隔为n个数时，split_length=n+1
    'max_epochs': 100,
    'lr_init': 1e-3,
    'size_middle': 32,
    'num_layers': 1,
    'device': device,
    'scale': 100
}

# paras_Prediction_All
paras_Prediction_All = {
    'current_to_soc': 2.561927693277231e+03,
    'predict_state_inp': paras_Prediction_State['info_len'],
    'predict_state_oup': paras_Prediction_State['state_len'],
    'predict_temperature_inp': paras_Prediction_Temperature['info_len'],
    'predict_temperature_oup': 1,
    'seq_history': 1,
    'seq_predict': 200,
    'seq_attention_once': 50,
    'split_length': 2,  # 取点间隔，间隔为n个数时，split_length=n+1
    'device': device
}
