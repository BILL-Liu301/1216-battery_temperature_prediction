import torch
import pytorch_lightning as pl

pl.seed_everything(2024)

# 开式温度计算常数
# K = 273.16
K = 0

# 单个电芯测点数量
num_measure_point = 7

# 模型参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prediction_Seq2Seq
paras_Prediction_Seq2Seq = {
    'num_measure_point': num_measure_point,
    'info_len': 6,
    'seq_history': 1,
    'seq_predict': 200,  # 100和600
    'seq_attention_once': 100,
    'split_length': 2,  # 取点间隔，间隔为n个数时，split_length=n+1
    'max_epochs': 100,
    'lr_init': 1e-3,
    'size_middle': 16,
    'num_layers': 4,
    'device': device,
    'scale': 100
}

# Prediction_Seq2Seq_All
paras_Prediction_Seq2Seq_All = {
    'info_len': 6,
    'seq_history': 5,
    'seq_predict': 650,
    'seq_attention_once': 50,
    'split_length': 2,  # 取点间隔，间隔为n个数时，split_length=n+1
    'max_epochs': 100,
    'lr_init': 1e-3,
    'size_middle': 16,
    'num_layers': 4,
    'device': device,
    'scale': 100
}

