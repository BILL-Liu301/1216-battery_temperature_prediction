import torch
import pickle
import numpy as np
import librosa.util as librosa_util
from torch.utils.data import Dataset


class Prediction_Seq2Seq_Dataset(Dataset):
    def __init__(self, path_data_origin_pkl, paras, device):
        self.seq_history, self.seq_predict, device = paras['seq_history'], paras['seq_predict'], device
        self.hop_length = 10
        self.data = self.load_from_pkl(path_data_origin_pkl)

    def load_from_pkl(self, path_data_origin_pkl):
        with open(path_data_origin_pkl, 'rb') as pkl:
            data_pkl = pickle.load(pkl)
            pkl.close()
        data_load = list()
        for data_name, data_origin in data_pkl.items():
            data_slide = librosa_util.frame(x=data_origin, frame_length=(self.seq_history + self.seq_predict), hop_length=self.hop_length)
            for group in range(data_slide.shape[-1]):
                data = data_slide[:, :, group]
                data = np.append(np.append(data[0:4], np.expand_dims(np.max(data[4:], axis=0), axis=0), axis=0), data[4:], axis=0)
                data_load.append(torch.from_numpy(data.transpose()).to(torch.float32))
        return data_load

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
