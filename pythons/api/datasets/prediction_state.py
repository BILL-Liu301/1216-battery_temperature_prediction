import torch
import pickle
import numpy as np
import librosa.util as librosa_util
from torch.utils.data import Dataset, random_split, DataLoader

from pythons.api.base.paras import paras_Prediction_State
from pythons.api.base.paths import path_data_origin_pkl_real, path_data_origin_pkl_sim


class Prediction_State_Dataset(Dataset):
    def __init__(self, path_data, paras, modules=None, flag_slide=True):
        self.hop_length = 1  # 滑窗间隔
        self.flag_slide = flag_slide
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
                    # 遍历电芯组，由于每个组的state都是一样的，所以只提取一次即可break
                    for group, dataset_group in dataset_module.items():
                        stamp = dataset_group['stamp']
                        current = dataset_group['Current']
                        soc = self.integral_i(stamp, current, dataset_group['SOC'][0, 0])
                        condition = np.ones(soc.shape) * dataset_group['NTC'].mean(axis=1)[0]
                        voltage = dataset_group['Voltage']
                        ntc = dataset_group['NTC']
                        data_origin = np.concatenate([stamp, current, soc, condition,
                                                      voltage, ntc.max(axis=1, keepdims=True), ntc.min(axis=1, keepdims=True), ntc.min(axis=1, keepdims=True), ntc.min(axis=1, keepdims=True)],
                                                     axis=1)
                        # 数据分割与否
                        if self.flag_slide:
                            data_slide = librosa_util.frame(x=data_origin.transpose(), frame_length=(self.seq_history + self.seq_predict) * self.split_length, hop_length=self.hop_length)
                            for slide in range(data_slide.shape[-1]):
                                data.append(torch.from_numpy(data_slide[:, 0:-1:self.split_length, slide].transpose().copy()).to(torch.float32))
                        else:
                            data.append(torch.from_numpy(data_origin[0:-1:self.split_length, :].copy()).to(torch.float32))
                        break
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# 加载数据集
path_data_origin_pkl = path_data_origin_pkl_sim
# path_data_origin_pkl = path_data_origin_pkl_real

dataset_train_val = Prediction_State_Dataset(path_data=path_data_origin_pkl, paras=paras_Prediction_State, modules=[0], flag_slide=True)
train_set_size = int(len(dataset_train_val) * 0.8)
valid_set_size = len(dataset_train_val) - train_set_size
dataset_train, dataset_val = random_split(dataset_train_val, [train_set_size, valid_set_size])
dataset_test = Prediction_State_Dataset(path_data=path_data_origin_pkl, paras=paras_Prediction_State, modules=[1], flag_slide=False)

# 分割train, test, val，并进行数据加载
dataset_loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, pin_memory=True, num_workers=0)
dataset_loader_val = DataLoader(dataset_val, batch_size=8, pin_memory=True, num_workers=0)
dataset_loader_test = DataLoader(dataset_test, batch_size=1, pin_memory=True, num_workers=0)
paras_Prediction_State_dataset = {
    'dataset_loader_train': dataset_loader_train,
    'dataset_loader_val': dataset_loader_val,
    'dataset_loader_test': dataset_loader_test
}
