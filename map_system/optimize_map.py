import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from api.base.path import path_map_tabel, path_ckpt_best_version, path_map_tabel_new
from api.q_learning.core import Learning_Core

if __name__ == '__main__':
    # 创建强化学习核心
    core = Learning_Core(path_map_tabel, path_ckpt_best_version, path_map_tabel_new)
    # 创建可视化窗口
    fig = plt.figure(figsize=(20, 11.25))

    num_flag_finish = 0
    while num_flag_finish < core.condition.shape[0]:
        num_flag_finish = 0
        for condition_id in tqdm(range(core.condition.shape[0]), desc='condition', leave=False, ncols=100, disable=False):

            # 初始值
            temperature_0, soc_0, voltage_0, ntc_max_0, ntc_min_0 = core.condition[condition_id]

            # 获取当前状态下的电流策略
            policy, (temperature_id, soc_id) = core.map_tabel.get_policy(temperature_0, soc_0 / 100)

            # 不考虑soc>90%的工况
            if soc_0 >= 90:
                num_flag_finish += 1
                # 可视化
                core.plot_map(fig, 0.1, temperature_id, soc_id, core.condition[condition_id, 1], condition_id, core.num_condition, flag_finish=True)
            else:

                # 获取充电最后的温度
                temperature, soc, voltage, ntc_max, ntc_min = core.charging(temperature_0, soc_0, voltage_0, policy,
                                                                            core.condition[condition_id, 0], ntc_max_0, ntc_min_0)
                temperature_1, soc_1, voltage_1, ntc_max_1, ntc_min_1 = temperature[-1], soc[-1], voltage[-1], ntc_max[-1], ntc_min[-1]

                # 计算reward
                reward = core.reward_func.calculate(temperature_1, soc_1)

                # 计算下一时刻，一定范围内的最佳policy
                policy_next, _ = core.map_tabel.get_policy(temperature_1, soc_1 / 100)
                policies_next = np.random.uniform(-1, 1, 10) * 1 + policy_next
                policy_best, reward_best = 0, 0
                for policy_next_sample in policies_next:
                    temperature_sample, soc_sample, _, _, _ = core.charging(temperature_1, soc_1, voltage_1, policy_next_sample,
                                                                            core.condition[condition_id, 0], ntc_max_1, ntc_min_1)
                    temperature_sample, soc_sample = temperature_sample[-1], soc_sample[-1]

                    # 计算reward
                    reward_sample = core.reward_func.calculate(temperature_sample, soc_sample)

                    policy_best = policy_next_sample if reward_sample > reward_best else policy_best
                    reward_best = max(reward_best, reward_sample)

                # 更新策略
                new_policy = policy + core.alpha * (reward + core.gamma * policy_best - policy)

                core.map_tabel.change_tabel(temperature_id, soc_id, new_policy)

                # 更新condition
                core.condition[condition_id] = temperature_1, soc_1, voltage_1, ntc_max_1, ntc_min_1

                # 可视化
                core.plot_map(fig, 0.1, temperature_id, soc_id, core.condition[condition_id, 1], condition_id, core.num_condition)

                # 保存数据
                core.save_new_map()
