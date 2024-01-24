import torch
import pickle
import numpy as np
import librosa.util as librosa_util
from torch.utils.data import Dataset, random_split, DataLoader

from pythons.api.base.paras import paras_Prediction_Seq2Seq
from pythons.api.base.paths import path_data_origin_pkl


class Prediction_Seq2Seq_Dataset(Dataset):
    def __init__(self, path_data_origin_pkl, paras):
        self.seq_history, self.seq_predict, self.scale = paras['seq_history'], paras['seq_predict'], paras['scale']
        self.hop_length = 1  # 滑窗间隔
        self.split_length = paras['split_length']  # 取点间隔，间隔为n个数时，split_length=n+1
        self.data = self.load_from_pkl(path_data_origin_pkl)

    def load_from_pkl(self, path_data_origin_pkl):
        with open(path_data_origin_pkl, 'rb') as pkl:
            data_pkl = pickle.load(pkl)
            pkl.close()
        data_load = list()
        for data_name, data_origin in data_pkl.items():
            data_slide = librosa_util.frame(x=data_origin, frame_length=(self.seq_history + self.seq_predict) * self.split_length, hop_length=self.hop_length)
            for group in range(data_slide.shape[-1]):
                data = data_slide[:, :, group]
                data = np.append(np.append(data[0:7], np.expand_dims(np.max(data[7:], axis=0), axis=0), axis=0), data[7:], axis=0)
                # data[7] = gaussian_filter(data[7], sigma=3.0)
                data[7] = self.lpf(data[7], alpha=0.15)
                data = data[:, 0:-1:self.split_length]
                data_load.append(torch.from_numpy(data.transpose()).to(torch.float32))
        return data_load

    def lpf(self, data_origin, alpha=0.7):
        data_oup = np.zeros(data_origin.shape)
        data_oup[0] = data_origin[0]
        for i in range(1, data_oup.shape[0]):
            data_oup[i] = (1 - alpha) * data_oup[i - 1] + alpha * data_origin[i]
        return data_oup

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# 加载数据集
dataset_base = Prediction_Seq2Seq_Dataset(path_data_origin_pkl=path_data_origin_pkl,
                                          paras=paras_Prediction_Seq2Seq)

# 分割train, test, val，并进行数据加载
train_valid_set_size = int(len(dataset_base) * 0.8)
test_set_size = len(dataset_base) - train_valid_set_size
train_valid_set, test_set = random_split(dataset_base, [train_valid_set_size, test_set_size])

train_set_size = int(len(train_valid_set) * 0.8)
valid_set_size = len(train_valid_set) - train_set_size
train_set, valid_set = random_split(train_valid_set, [train_set_size, valid_set_size])

dataset_loader_train = DataLoader(train_set, batch_size=16, shuffle=True, pin_memory=True, num_workers=0)
dataset_loader_val = DataLoader(valid_set, batch_size=16, pin_memory=True, num_workers=0)
dataset_loader_test = DataLoader(test_set, batch_size=5, pin_memory=True, num_workers=0)
paras_Prediction_Seq2Seq_dataset = {
    'dataset_loader_train': dataset_loader_train,
    'dataset_loader_val': dataset_loader_val,
    'dataset_loader_test': dataset_loader_test
}
