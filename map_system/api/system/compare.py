import numpy as np
import matplotlib.pyplot as plt
import torch
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
        self.charging_time = 51

    def init_condition(self, temperature=None, soc=None, voltage=None):
        stamp = 1
        temperature = 25.0 if (temperature is None) else temperature
        soc = 10 if (soc is None) else soc
        voltage = 350 if (voltage is None) else voltage
        policy = 0.0
        current = 0.0
        ntc_max = temperature
        ntc_min = temperature

        condition = np.array([[stamp, temperature, soc, voltage, policy, current, ntc_max, ntc_min]])

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
            stamp_0, temperature_0, soc_0, voltage_0, policy_0, current_0, ntc_max_0, ntc_min_0 = condition_record[-1]
            current = tabel.get_current(policy_0)

            # 充电
            charging_data = self.charging_system.charging(self.num_group, self.split_time, self.charging_time * self.split_time,
                                                          current, [temperature_0, soc_0, voltage_0],
                                                          condition_temperature=condition_record[0, 1], ntc_max=ntc_max_0, ntc_min=ntc_min_0)

            # 提取数据
            stamp_1 = stamp_0 + self.split_time
            temperature_1 = charging_data[:, 1, 8].mean(axis=0)
            soc_1 = charging_data[0, 1, 3]
            voltage_1 = charging_data[0, 1, 5]
            policy_1, _ = tabel.get_policy(temperature_1, soc_1 / 100)
            current_1 = tabel.get_current(policy_1)
            ntc_max_1 = charging_data[0, 1, 6]
            ntc_min_1 = charging_data[0, 1, 7]
            # ntc_max_1 = temperature_1
            # ntc_min_1 = temperature_1

            # 数据拼接
            condition_record = np.append(condition_record, np.array([[stamp_1, temperature_1, soc_1, voltage_1, policy_1, current_1, ntc_max_1, ntc_min_1]]), axis=0)

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
        fontsize = 5

        # 判断是否充电完毕
        plot_times = condition_record.shape[0] if flag_finish else plot_times
        stamp, temperature, soc, voltage, policy, current, ntc_max, ntc_min = np.hsplit(condition_record[0:plot_times], condition_record.shape[1])

        plt.subplot(6, 1, 1)
        # 绘制数据
        plt.plot(stamp, temperature, 'k-' if (plot_side == 0) else 'k--', label=map_name + '_temperature')
        # 设置外型
        xlim = plt.xlim()
        plt.xticks(np.linspace(1, stamp[-1, 0], 10), fontsize=fontsize)
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=fontsize)

        plt.subplot(6, 1, 2)
        # 绘制数据
        plt.plot(stamp, soc, 'k-' if (plot_side == 0) else 'k--', label=map_name + '_soc')
        # 设置外型
        plt.xlim(xlim)
        plt.xticks(np.linspace(1, stamp[-1, 0], 10), fontsize=fontsize)
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=fontsize)

        plt.subplot(6, 1, 3)
        # 绘制数据
        plt.plot(stamp, voltage, 'k-' if (plot_side == 0) else 'k--', label=map_name + '_voltage')
        # 设置外型
        plt.xlim(xlim)
        plt.xticks(np.linspace(1, stamp[-1, 0], 10), fontsize=fontsize)
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=fontsize)

        plt.subplot(6, 1, 4)
        # 绘制数据
        if PolicyorCurrent == 'policy':
            plt.plot(stamp, policy, 'k-' if (plot_side == 0) else 'k--', label=map_name + '_policy')
        else:
            plt.plot(stamp, current, 'k-' if (plot_side == 0) else 'k--', label=map_name + '_current')
        # 设置外型
        plt.xlim(xlim)
        plt.xticks(np.linspace(1, stamp[-1, 0], 10), fontsize=fontsize)
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=fontsize)

        plt.subplot(6, 1, 5)
        # 绘制数据
        plt.plot(stamp, ntc_max, 'r-' if (plot_side == 0) else 'r--', label=map_name + '_ntc_max')
        plt.plot(stamp, ntc_min, 'g-' if (plot_side == 0) else 'g--', label=map_name + '_ntc_min')
        # 设置外型
        plt.xlim(xlim)
        plt.xticks(np.linspace(1, stamp[-1, 0], 10), fontsize=fontsize)
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=fontsize)

        ax = plt.subplot(6, 2, 11 + plot_side)
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
        plt.text(soc[-1], 5, f'SOC:{soc[-1, 0]:.2f}%, \nStamp:{stamp[-1, 0]}s ', ha='right', va='center', fontsize=fontsize)
        # 设置外型
        plt.xlim((-5, 105))
        plt.ylim((-1, 11))
        plt.xticks(fontsize=fontsize)

    def plot_both(self, plot_speed=2, condition_record_origin=None, condition_record_new=None, name_origin=None, name_new=None):
        # 赋值数据
        condition_record_origin = self.condition_record_origin if (condition_record_origin is None) else condition_record_origin
        condition_record_new = self.condition_record_new if (condition_record_new is None) else condition_record_new
        name_origin = 'MAP_origin' if (name_origin is None) else name_origin
        name_new = 'MAP_new' if (name_new is None) else name_new

        # 初始化参数
        flag_finish_origin, flag_finish_new = False, False
        plot_times = 0

        # 循环绘图，直至全部充电完毕
        while not (flag_finish_origin and flag_finish_new):
            # 判断是否到达充电终点
            flag_finish_origin = True if plot_times > condition_record_origin.shape[0] else False
            flag_finish_new = True if plot_times > condition_record_new.shape[0] else False
            plot_times += plot_speed

            plt.clf()

            # 基于原始的MAP进行充电的结果
            self.plot_single(flag_finish_origin, plot_times, 0, 'current', condition_record_origin, name_origin)

            # 基于优化后的MAP进行充电的结果
            self.plot_single(flag_finish_new, plot_times, 1, 'current', condition_record_new, name_new)

            # 展示
            plt.pause(0.01)

    def re_charging(self, condition_record):
        # 基于已生成的电流策略进行重新充电过程
        stamp, temperature, soc, voltage, policy, current, ntc_max, ntc_min = np.hsplit(condition_record, condition_record.shape[1])
        # charging_data: [25, charging_stamp, [stamp, location, current, soc, condition_temperature, voltage, ntc_max, ntc_min, temperature_max]]

        charging_data = list()
        # 计算通用数据
        # stamp = stamp.reshape(-1, 1)
        # current = current.reshape(-1, 1)
        # soc = soc.reshape(-1, 1)
        # condition_temperature = np.ones([stamp.shape[0], 1]) * temperature[0, 0]
        # voltage = voltage.reshape(-1, 1)
        # ntc_max = ntc_max.reshape(-1, 1)
        # ntc_min = ntc_min.reshape(-1, 1)
        # temperature_max = temperature

        # 计算通用数据
        stamp = stamp.reshape(-1, 1)
        current = np.ones([stamp.shape[0], 1]) * current[0]
        soc = soc.reshape(-1, 1)
        condition_temperature = np.ones([stamp.shape[0], 1]) * temperature[0, 0]
        voltage = np.ones([stamp.shape[0], 1]) * voltage[0, 0]
        ntc_max = np.ones([stamp.shape[0], 1]) * temperature[0, 0]
        ntc_min = np.ones([stamp.shape[0], 1]) * temperature[0, 0]
        temperature_max = np.ones([stamp.shape[0], 1]) * temperature[0, 0]

        for group in range(self.num_group):
            # 计算各组独有数据
            location = np.ones([stamp.shape[0], 1]) * group / (self.num_group - 1) * 100

            # 组合数据
            charging_data.append(np.concatenate([stamp, location, current, soc, condition_temperature, voltage, ntc_max, ntc_min, temperature_max], axis=1)[None, :])

        charging_data = np.concatenate(charging_data, axis=0)

        # 呼叫模型
        charging_data_re = self.charging_system.call_model(charging_data)

        return charging_data_re

    def charging_data_2_condition_record(self, charging_data, tabel):
        stamp, location, current, soc, condition_temperature, voltage, ntc_max, ntc_min, _ = np.hsplit(charging_data[0], charging_data[0].shape[1])
        policy = tabel.get_policy_from_current(current)
        temperature_max = charging_data[:, :, -1].max(axis=0).reshape(-1, 1)

        # 数据拼接
        condition_record = np.concatenate([stamp, temperature_max, soc, voltage, policy, current, ntc_max, ntc_min], axis=1)

        return condition_record

