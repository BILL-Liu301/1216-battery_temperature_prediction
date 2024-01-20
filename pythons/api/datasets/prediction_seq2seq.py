import torch
import pickle
import numpy as np
import librosa.util as librosa_util
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter


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
