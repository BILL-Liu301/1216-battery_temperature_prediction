import torch
import pickle
import numpy as np
import librosa.util as librosa_util
from torch.utils.data import Dataset, random_split, DataLoader

from pythons.api.base.paras import paras_Prediction_Temperature
from pythons.api.base.paths import path_data_origin_pkl_real, path_data_origin_pkl_sim


class Prediction_Temperature_Dataset(Dataset):
    def __init__(self, path_data, paras, modules=None, flag_slide=True):
        self.hop_length = 1  # 滑窗间隔
        self.flag_slide = flag_slide
        self.split_length = paras['split_length']  # 取点间隔，间隔为n个数时，split_length=n+1
        self.modules = [0] if (modules is None) else modules
        self.seq_history, self.seq_predict, self.scale = paras['seq_history'], paras['seq_predict'], paras['scale']
        self.data = self.load_from_pkl(path_data, self.modules)

    def load_from_pkl(self, path_data, modules):
        with open(path_data, 'rb') as pkl:
            dataset = pickle.load(pkl)
            pkl.close()
        data = list()
        for module in modules:
            dataset_module = dataset[f'module-{module}']
            for dataset_group in dataset_module:
                data_group = np.concatenate([dataset_group['stamp'],
                                             dataset_group['location'], dataset_group['NTC_max'], dataset_group['NTC_min'],
                                             dataset_group['Voltage'], dataset_group['Current'], dataset_group['SOC'],
                                             np.max(dataset_group['Temperature_max'], axis=1, keepdims=True)], axis=1)
                if self.flag_slide:
                    data_group_slide = librosa_util.frame(x=data_group.transpose(), frame_length=(self.seq_history + self.seq_predict) * self.split_length, hop_length=self.hop_length)
                    for slide in range(data_group_slide.shape[-1]):
                        data.append(torch.from_numpy(data_group_slide[:, 0:-1:self.split_length, slide].transpose().copy()).to(torch.float32))
                else:
                    data.append(torch.from_numpy(data_group[0:-1:self.split_length, :].copy()).to(torch.float32))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# 加载数据集
path_data_origin_pkl = path_data_origin_pkl_sim
# path_data_origin_pkl = path_data_origin_pkl_real
dataset_train_val = Prediction_Temperature_Dataset(path_data=path_data_origin_pkl, paras=paras_Prediction_Temperature, modules=[0], flag_slide=True)
dataset_test = Prediction_Temperature_Dataset(path_data=path_data_origin_pkl, paras=paras_Prediction_Temperature, modules=[1], flag_slide=False)

# 分割train, test, val，并进行数据加载
train_set_size = int(len(dataset_train_val) * 0.8)
valid_set_size = len(dataset_train_val) - train_set_size
dataset_train, dataset_val = random_split(dataset_train_val, [train_set_size, valid_set_size])

dataset_loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, pin_memory=True, num_workers=0)
dataset_loader_val = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=0)
dataset_loader_test = DataLoader(dataset_test, batch_size=5, pin_memory=True, num_workers=0)
paras_Prediction_Temperature_dataset = {
    'dataset_loader_train': dataset_loader_train,
    'dataset_loader_val': dataset_loader_val,
    'dataset_loader_test': dataset_loader_test
}
