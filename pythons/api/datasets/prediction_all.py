import torch
import pickle
import numpy as np
import librosa.util as librosa_util
from torch.utils.data import Dataset, random_split, DataLoader

from pythons.api.base.paras import paras_Prediction_All
from pythons.api.base.paths import path_data_origin_pkl_real, path_data_origin_pkl_sim


class Prediction_All_Dataset(Dataset):
    def __init__(self, path_data, paras, modules=None):
        self.hop_length = 1  # 滑窗间隔
        self.split_length = paras['split_length']  # 取点间隔，间隔为n个数时，split_length=n+1
        self.modules = [0] if (modules is None) else modules
        self.seq_history, self.seq_predict, self.current_to_soc = paras['seq_history'], paras['seq_predict'], paras['current_to_soc']
        self.data = self.load_from_pkl(path_data, self.modules)

    def integral_i(self, stamp, current, soc_0):
        soc = np.zeros(current.shape)
        for t in range(1, stamp.shape[0]):
            soc[t] = (current[t] + current[t - 1]) * (stamp[t] - stamp[t - 1]) / 2 + soc[t - 1]
        soc = soc / self.current_to_soc + soc_0
        return soc

    def load_from_pkl(self, path_data, modules):
        with open(path_data, 'rb') as pkl:
            dataset = pickle.load(pkl)
            pkl.close()
        data = list()

        # 按工况遍历
        for condition, dataset_condition in dataset.items():
            # if condition == '低温充电':
            #     continue
            # 按模组遍历
            for module, dataset_module in dataset_condition.items():
                if int(module.split('-')[1]) in modules:
                    data_module = list()
                    # 遍历电芯组
                    for group, dataset_group in dataset_module.items():

                        stamp = dataset_group['stamp']
                        location = dataset_group['location']
                        current = dataset_group['Current']
                        soc = self.integral_i(stamp, current, dataset_group['SOC'][0, 0])
                        condition = np.ones(soc.shape) * dataset_group['NTC'].mean(axis=1)[0]
                        voltage = dataset_group['Voltage']
                        ntc_max = dataset_group['NTC'].max(axis=1, keepdims=True)
                        ntc_min = dataset_group['NTC'].min(axis=1, keepdims=True)
                        temperature_max = dataset_group['Temperature_max'].min(axis=1, keepdims=True)

                        data_temp = np.concatenate([stamp, dataset_group['SOC'], location, current, soc, condition, voltage, ntc_max, ntc_min, temperature_max],
                                                   axis=1)
                        data_module.append(np.expand_dims(data_temp[0:-1:self.split_length, :], axis=0))
                    data.append(torch.from_numpy(np.concatenate(data_module, axis=0)).to(torch.float32))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# 加载数据集
path_data_origin_pkl = path_data_origin_pkl_sim
# path_data_origin_pkl = path_data_origin_pkl_real
dataset_test = Prediction_All_Dataset(path_data=path_data_origin_pkl, paras=paras_Prediction_All, modules=[1])
