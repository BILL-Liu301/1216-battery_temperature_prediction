import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
        self.charging_time = 51
        num_condition = 40
        self.condition = self.init_condition(num_condition)

        # 每个模组内由25个电芯组，每隔2s为一个时间戳
        self.split_time = 2
        self.num_group = 25

    def init_condition(self, num_condition):
        temperature = np.linspace(start=20, stop=40, num=num_condition)
        soc = np.random.uniform(-1, 1, num_condition) * 2 + 10.0
        voltage = np.random.uniform(-1, 1, num_condition) * 20.0 + 350.0
        ntc_max = temperature.copy()
        ntc_min = temperature.copy()
        return np.stack([temperature, soc, voltage, ntc_max, ntc_min], axis=1)

    def charging(self, temperature_0, soc_0, voltage_0, policy, condition_temperature, ntc_max, ntc_min):
        # 使用代理模型进行预测
        charging_data = self.charging_system.charging(self.num_group, self.split_time, self.charging_time,
                                                      self.map_tabel.get_current(policy), [temperature_0, soc_0, voltage_0],
                                                      condition_temperature, ntc_max, ntc_min)

        # 获取所有模组中最大的温度数据
        temperature_max = charging_data[:, :, 8].max(axis=0)
        soc = charging_data[0, :, 3]
        voltage = charging_data[0, :, 5]
        ntc_max = charging_data[0, :, 6]
        ntc_min = charging_data[0, :, 7]

        return temperature_max, soc, voltage, ntc_max, ntc_min

    def plot_map(self, fig, pause, temperature_id, soc_id, soc_now, condition_id, flag_finish=False):
        # 设置字体
        fontsize = 5  # 坐标轴的字体大小
        pad = 0.55  # 坐标与坐标轴之间的距离

        plt.clf()

        plt.subplot(1, 7, (1, 6))
        for soc in self.map_tabel.soc_labels:
            for temperature in self.map_tabel.temperature_labels:
                policy, _ = self.map_tabel.get_policy(temperature, soc)
                plt.text(soc, temperature, f'{policy:.2f}', fontsize=fontsize, ha='center', va='center')
        plt.fill_between([self.map_tabel.soc_labels[soc_id] - 0.03, self.map_tabel.soc_labels[soc_id] + 0.03],
                         [self.map_tabel.temperature_labels[temperature_id] + 2, self.map_tabel.temperature_labels[temperature_id] + 2],
                         [self.map_tabel.temperature_labels[temperature_id] - 2, self.map_tabel.temperature_labels[temperature_id] - 2],
                         color='r' if flag_finish else 'g', alpha=0.5)

        plt.xticks(self.map_tabel.soc_labels.tolist(), fontsize=fontsize)
        plt.yticks(self.map_tabel.temperature_labels.tolist(), fontsize=fontsize)
        plt.tick_params(pad=pad)
        plt.xlim([self.map_tabel.soc_labels[0] - 0.1, self.map_tabel.soc_labels[-1] + 0.1])
        plt.ylim([self.map_tabel.temperature_labels[0] - 5, self.map_tabel.temperature_labels[-1] + 5])
        plt.grid(True, lw=0.1)

        ax = plt.subplot(1, 7, 7)
        # 进度条外壳
        rect = Rectangle((0, 0), 10, 100, edgecolor='k', facecolor='None')
        ax.add_patch(rect)
        # 进度线
        plt.plot([0, 10], [soc_now, soc_now], 'k-')
        plt.fill_between([0, 10],
                         [0, 0],
                         [soc_now, soc_now],
                         color='r' if flag_finish else 'g', alpha=0.7)
        # 进度值
        plt.text(5, soc_now, f'SOC:{soc_now:.2f}%', ha='center', va='bottom', fontsize=fontsize)
        plt.text(5, 100, f'condition ID:{condition_id}', ha='center', va='bottom', fontsize=fontsize)
        # 设置外型
        plt.xlim((-1, 11))
        plt.ylim((-5, 105))
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tick_params(pad=pad)

        plt.pause(pause)

    def save_new_map(self):
        tabel = self.map_tabel.tabel.copy()
        tabel['temperature'] = self.map_tabel.temperature_labels
        tabel = tabel.reindex(columns=(['temperature'] + self.map_tabel.soc_labels.tolist()))
        tabel.to_excel(self.path_new_map_tabel, index=False)
