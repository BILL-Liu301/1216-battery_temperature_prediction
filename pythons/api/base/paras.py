import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from pythons.api.base.paths import path_data_origin_pkl
from pythons.api.datasets.prediction_seq2seq import Prediction_Seq2Seq_Dataset

pl.seed_everything(2024)

# 开式温度计算常数
K = 273.16

# 单个电芯测点数量
num_measure_point = 7

# 模型参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prediction_Seq2Seq
paras_Prediction_Seq2Seq = {
    'num_measure_point': num_measure_point,
    'seq_history': 50,
    'seq_predict': 500,
    'max_epochs': 100,
    'lr_init': 1e-3,
    'size_middle': 32,
    'device': device,
    'scale': 50
}
if os.path.exists(path_data_origin_pkl):
    # 加载数据集
    dataset_base = Prediction_Seq2Seq_Dataset(path_data_origin_pkl=path_data_origin_pkl,
                                              paras=paras_Prediction_Seq2Seq)

    # 分割train, test, val，并进行数据加载
    train_valid_set_size = int(len(dataset_base) * 0.9)
    test_set_size = len(dataset_base) - train_valid_set_size
    train_valid_set, test_set = random_split(dataset_base, [train_valid_set_size, test_set_size])

    train_set_size = int(len(train_valid_set) * 0.9)
    valid_set_size = len(train_valid_set) - train_set_size
    train_set, valid_set = random_split(train_valid_set, [train_set_size, valid_set_size])

    dataset_loader_train = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=True, num_workers=0)
    dataset_loader_val = DataLoader(valid_set, batch_size=8, pin_memory=True, num_workers=0)
    dataset_loader_test = DataLoader(test_set, batch_size=1, pin_memory=True, num_workers=0)
    paras_Prediction_Seq2Seq_dataset = {
        'dataset_loader_train': dataset_loader_train,
        'dataset_loader_val': dataset_loader_val,
        'dataset_loader_test': dataset_loader_test
    }
