import torch
import pickle
import numpy as np
from torch.utils.data import Dataset


class PreEncoder_Dataset(Dataset):
    def __init__(self, path_data_origin_pkl, num_pre_encoder, device):
        self.num_pre_encoder, self.device = num_pre_encoder, device
        self.num_expand = 20
        self.data = self.load_from_pkl(path_data_origin_pkl)

    def load_from_pkl(self, path_data_origin_pkl):
        with open(path_data_origin_pkl, 'rb') as pkl:
            data_origin = pickle.load(pkl)
            pkl.close()
        data_load = list()
        for data_name, data in data_origin.items():
            data = data[:, 0:self.num_pre_encoder]
            data = np.append(np.linspace(0, 1, self.num_pre_encoder).reshape(1, self.num_pre_encoder), data[1:], axis=0)
            for _ in range(self.num_expand):
                random_expand = np.random.normal(0, 0.08, (data.shape[0]-1, data.shape[1]))
                data_expand = data.copy()
                data_expand[1:] += random_expand
                data_load.append(torch.from_numpy(data_expand).to(torch.float32))
            data_load.append(torch.from_numpy(data).to(torch.float32))
        return data_load

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

