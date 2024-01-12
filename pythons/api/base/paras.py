import torch
from torch.utils.data import DataLoader, random_split

from pythons.api.base.paths import path_data_origin_pkl
from pythons.api.datasets.prediction_seq2seq import Prediction_Seq2Seq_Dataset

# 开式温度计算常数
K = 273.16

# 单个电芯测点数量
num_measure_point = 7

# 时序设定
sequence_init = 1
sequence_predict = 999

# 模型参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PreEncoder
paras_PreEncoder = {
    'num_measure_point': num_measure_point,
    'num_pre_encoder': 20,
    'max_epochs': 200,
    'lr_init': 1e-3,
    'size_middle': 16
}

# Prediction_Seq2Seq
paras_Prediction_Seq2Seq = {
    'num_measure_point': num_measure_point,
    'seq_history': 50,
    'seq_predict': 200,
    'max_epochs': 200,
    'lr_init': 1e-3,
    'size_middle': 16,
    'device': device
}
# 加载数据集
dataset_base = Prediction_Seq2Seq_Dataset(path_data_origin_pkl=path_data_origin_pkl,
                                          paras=paras_Prediction_Seq2Seq,
                                          device=device)

# 分割train, test, val，并进行数据加载
train_valid_set_size = int(len(dataset_base) * 0.9)
test_set_size = len(dataset_base) - train_valid_set_size
train_valid_set, test_set = random_split(dataset_base, [train_valid_set_size, test_set_size])

train_set_size = int(len(train_valid_set) * 0.9)
valid_set_size = len(train_valid_set) - train_set_size
train_set, valid_set = random_split(train_valid_set, [train_set_size, valid_set_size])

dataset_loader_train = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=True, num_workers=2)
dataset_loader_val = DataLoader(valid_set, batch_size=8, pin_memory=True, num_workers=2)
dataset_loader_test = DataLoader(test_set, batch_size=8, pin_memory=True, num_workers=2)
paras_Prediction_Seq2Seq_dataset = {
    'dataset_loader_train': dataset_loader_train,
    'dataset_loader_val': dataset_loader_val,
    'dataset_loader_test': dataset_loader_test
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
