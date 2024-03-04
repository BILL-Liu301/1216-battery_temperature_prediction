import numpy as np
import matplotlib.pyplot as plt

from .q_tabel import MAP_Tabel
from .reward import Reward
from ..system.charging import Charging_System


class Learning_Core:
    def __init__(self, path_tabel, path_system, path_new_map_tabel):
        # 初始化
        self.map_tabel = MAP_Tabel(path_tabel)
        self.reward_func = Reward()
        self.charging_system = Charging_System(path_system)
        self.path_new_map_tabel = path_new_map_tabel

        # 参数变量
        self.alpha = 0.001
        self.gamma = 0.001
        self.episode_max = 200
        self.charging_time = 50
        num_condition = 20
        self.condition = self.init_condition(num_condition)

        # 每个模组内由25个电芯组，每隔2s为一个时间戳
        self.split_time = 2
        self.num_group = 25

    def init_condition(self, num_condition):
        temperature = np.linspace(start=20, stop=40, num=num_condition)
        soc = np.random.uniform(-1, 1, num_condition) * 0.0 + 10.0
        voltage = np.random.uniform(-1, 1, num_condition) * 10.0 + 350.0
        return np.stack([temperature, soc, voltage], axis=1)

    def charging(self, temperature_0, soc_0, voltage_0, policy):
        # 使用代理模型进行预测
        charging_data = self.charging_system.charging(self.num_group, self.split_time, self.charging_time,
                                                      self.map_tabel.get_current(policy), [temperature_0, soc_0, voltage_0])

        # 获取所有模组中最大的温度数据
        temperature_max = charging_data[:, :, -1].max(axis=0)
        soc = charging_data[0, :, 3]
        voltage = charging_data[0, :, 5]

        return temperature_max, soc, voltage

    def plot_map(self, pause, temperature_id, soc_id, flag_end=False):
        plt.clf()

        for soc in self.map_tabel.soc_labels:
            for temperature in self.map_tabel.temperature_labels:
                policy, _ = self.map_tabel.get_policy(temperature, soc)
                plt.text(soc, temperature, f'{policy:.2f}', ha='center', va='center')
        plt.fill_between([self.map_tabel.soc_labels[soc_id] - 0.03, self.map_tabel.soc_labels[soc_id] + 0.03],
                         [self.map_tabel.temperature_labels[temperature_id] + 2, self.map_tabel.temperature_labels[temperature_id] + 2],
                         [self.map_tabel.temperature_labels[temperature_id] - 2, self.map_tabel.temperature_labels[temperature_id] - 2],
                         color='r' if flag_end else 'g', alpha=0.5)

        plt.xticks(self.map_tabel.soc_labels.tolist())
        plt.yticks(self.map_tabel.temperature_labels.tolist())
        plt.xlim([self.map_tabel.soc_labels[0] - 0.1, self.map_tabel.soc_labels[-1] + 0.1])
        plt.ylim([self.map_tabel.temperature_labels[0] - 5, self.map_tabel.temperature_labels[-1] + 5])
        plt.grid(True, lw=0.1)

        plt.pause(pause)

    def save_new_map(self):
        tabel = self.map_tabel.tabel.copy()
        tabel['temperature'] = self.map_tabel.temperature_labels
        tabel = tabel.reindex(columns=(['temperature'] + self.map_tabel.soc_labels.tolist()))
        tabel.to_excel(self.path_new_map_tabel, index=False)
