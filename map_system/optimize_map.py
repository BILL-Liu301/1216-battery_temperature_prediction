import matplotlib.pyplot as plt
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
                core.plot_map(fig, 0.1, temperature_id, soc_id, core.condition[condition_id, 1], condition_id, flag_finish=True)
            else:

                # 获取充电最后的温度
                temperature, soc, voltage, ntc_max, ntc_min = core.charging(temperature_0, soc_0, voltage_0, policy,
                                                                            core.condition[condition_id, 0], ntc_max_0, ntc_min_0)
                temperature_1, soc_1, voltage_1, ntc_max_1, ntc_min_1 = temperature[-1], soc[-1], voltage[-1], ntc_max[-1], ntc_min[-1]

                # 计算reward
                reward = core.reward_func.calculate(temperature_1, soc_1)

                # 更新策略
                policy_next, _ = core.map_tabel.get_policy(temperature_1, soc_1 / 100)
                new_policy = policy + core.alpha * (reward + core.gamma * policy_next - policy)
                core.map_tabel.change_tabel(temperature_id, soc_id, new_policy)

                # 更新condition
                core.condition[condition_id] = temperature_1, soc_1, voltage_1, ntc_max_1, ntc_min_1

                # 可视化
                core.plot_map(fig, 0.1, temperature_id, soc_id, core.condition[condition_id, 1], condition_id)

                # 保存数据
                core.save_new_map()
