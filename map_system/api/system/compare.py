import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

from ..q_learning.q_tabel import MAP_Tabel
from .charging import Charging_System


class Compare_System:
    def __init__(self, path_tabel_origin, path_tabel_new, path_ckpt):
        # 加载MAP表
        self.tabel_origin = MAP_Tabel(path_tabel_origin)
        self.tabel_new = MAP_Tabel(path_tabel_new)

        # 加载充电系统（代理模型）
        self.charging_system = Charging_System(path_ckpt)

        # 状态记录表
        self.condition_record_origin, self.condition_record_new = None, None

        # 每个模组内由25个电芯组，每隔2s为一个时间戳，每轮充电50s
        self.split_time = 2
        self.num_group = 25
        self.charging_time = 50

    def init_condition(self, temperature=None, soc=None, voltage=None):
        stamp = 1
        temperature = 25.0 if (temperature is None) else temperature
        soc = 10 if (soc is None) else soc
        voltage = 350 if (voltage is None) else voltage
        policy = 0.0
        current = 0.0

        condition = np.array([[stamp, temperature, soc, voltage, policy, current]])

        # 赋值并计算policy
        self.condition_record_origin, self.condition_record_new = condition.copy(), condition.copy()
        self.condition_record_origin[-1, 4], _ = self.tabel_origin.get_policy(temperature, soc / 100)
        self.condition_record_origin[-1, 5] = self.tabel_origin.get_current(self.condition_record_origin[-1, 4])
        self.condition_record_new[-1, 4], _ = self.tabel_new.get_policy(temperature, soc / 100)
        self.condition_record_new[-1, 5] = self.tabel_new.get_current(self.condition_record_origin[-1, 4])

    def charging_single(self, condition_record, tabel, soc_end, map_name):
        # 设置进度条
        pbar = tqdm(range(soc_end), desc=map_name, leave=False, ncols=100, disable=False)
        pbar.update(round(condition_record[-1, 2] - 0, 1))

        # 开启循环充电
        while condition_record[-1, 2] <= soc_end:
            # 提取数据
            stamp_0, temperature_0, soc_0, voltage_0, policy_0, current_0 = condition_record[-1]
            current = tabel.get_current(policy_0)

            # 充电
            charging_data = self.charging_system.charging(self.num_group, self.split_time, self.charging_time,
                                                          current, [temperature_0, soc_0, voltage_0])

            # 进行插值，并提取
            stamp_1 = stamp_0 + self.split_time
            temperature_1 = charging_data[:, 1, -1].mean(axis=0)
            soc_1 = charging_data[0, 1, 3]
            voltage_1 = charging_data[0, 1, 5]
            policy_1, _ = tabel.get_policy(temperature_1, soc_1 / 100)
            current_1 = tabel.get_current(policy_1)

            # 数据拼接
            condition_record = np.append(condition_record, np.array([[stamp_1, temperature_1, soc_1, voltage_1, policy_1, current_1]]), axis=0)

            # 更新进度条
            pbar.update(round(soc_1 - soc_0, 4))
        pbar.close()
        return condition_record

    def charging_both(self, soc_end=90):
        # 基于原始的MAP进行充电
        self.condition_record_origin = self.charging_single(self.condition_record_origin, self.tabel_origin, soc_end, 'MAP_origin')

        # 基于优化后的MAP进行充电
        self.condition_record_new = self.charging_single(self.condition_record_new, self.tabel_new, soc_end, 'MAP_new')

    def plot_single(self, flag_finish, plot_times, plot_side, PolicyorCurrent, condition_record, map_name):
        # 判断是否充电完毕
        plot_times = condition_record.shape[0] if flag_finish else plot_times
        stamp, temperature, soc, voltage, policy, current = np.hsplit(condition_record[0:plot_times], condition_record.shape[1])

        plt.subplot(5, 2, 1 + plot_side)
        plt.title(map_name)
        plt.plot(stamp, temperature, 'k-', label='temperature')
        xlim = plt.xlim()
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(5, 2, 3 + plot_side)
        plt.plot(stamp, soc, 'k-', label='soc')
        plt.xlim(xlim)
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(5, 2, 5 + plot_side)
        plt.plot(stamp, voltage, 'k-', label='voltage')
        plt.xlim(xlim)
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(5, 2, 7 + plot_side)
        if PolicyorCurrent == 'policy':
            plt.plot(stamp, policy, 'k-', label='policy')
        else:
            plt.plot(stamp, current, 'k-', label='current')
        plt.xlim(xlim)
        plt.grid(True)
        plt.legend(loc='upper right')

        ax = plt.subplot(5, 2, 9 + plot_side)
        # 进度条外壳
        rect = Rectangle((0, 0), 100, 10, edgecolor='k', facecolor='None')
        ax.add_patch(rect)
        # 进度线
        plt.plot([soc[-1], soc[-1]], [0, 10], 'k-')
        plt.fill_between([0.0, soc[-1, 0]],
                         [0, 0],
                         [10, 10],
                         color='r' if flag_finish else 'g', alpha=0.7)
        # 进度值
        plt.text(soc[-1], 5, f'{soc[-1, 0]:.2f}%', ha='left', va='center')
        plt.xlim((-5, 105))
        plt.ylim((-1, 11))

    def plot_both(self, plot_speed=2):
        # 初始化参数
        flag_finish_origin, flag_finish_new = False, False
        plot_times = 1

        # 循环绘图，直至全部充电完毕
        while not (flag_finish_origin and flag_finish_new):
            plt.clf()

            # 基于原始的MAP进行充电的结果
            self.plot_single(flag_finish_origin, plot_times, 0, 'policy', self.condition_record_origin, 'MAP_origin')

            # 基于优化后的MAP进行充电的结果
            self.plot_single(flag_finish_new, plot_times, 1, 'policy', self.condition_record_new, 'MAP_new')

            # 判断到达充电终点
            flag_finish_origin = True if plot_times >= self.condition_record_origin.shape[0] else False
            flag_finish_new = True if plot_times >= self.condition_record_new.shape[0] else False
            plot_times += plot_speed

            # 展示
            plt.pause(0.01)
