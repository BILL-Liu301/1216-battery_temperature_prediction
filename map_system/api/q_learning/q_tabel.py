import numpy as np
import torch
import pandas as pd


class MAP_Tabel:
    def __init__(self, path_tabel):
        self.temperature_labels, self.soc_labels, self.tabel = self.load_tabel(path_tabel)

    def load_tabel(self, path_tabel):
        data_xlsx = pd.read_excel(path_tabel)
        temperature_labels = np.asarray(data_xlsx['temperature'])
        soc_labels = np.asarray(data_xlsx.columns[1:])
        tabel = data_xlsx[data_xlsx.columns[1:]]
        return temperature_labels, soc_labels, tabel

    def change_tabel(self, temperature_1, soc_1, new_data):
        self.tabel.loc[temperature_1, self.soc_labels[soc_1]] = new_data

    def search_tabel(self, value, labels):
        value_id = 0
        if value < labels[0]:
            value_id = 0
        elif value >= labels[-1]:
            value_id = len(labels) - 1
        else:
            # 遍历所有有限区间
            for i in range(len(labels) - 1):
                if labels[i] <= value < labels[i + 1]:
                    value_id = i
                    break
        return value_id

    def get_current(self, policy):
        # 乘上电池容量
        return policy * 68.6

    def get_policy_from_current(self, current):
        return current / 68.6

    def get_policy(self, temperature, soc, return_id=True):
        # MAP表查表

        # 获取temperature和soc的区间
        temperature_id = self.search_tabel(temperature, self.temperature_labels)
        soc_id = self.search_tabel(soc, self.soc_labels)

        # 查表
        policy = self.tabel[self.soc_labels[soc_id]][temperature_id]

        if return_id:
            return policy, (temperature_id, soc_id)
        else:
            return policy
