from tqdm import tqdm

from api.base.path import path_map_tabel, path_ckpt_best_version
from api.q_learning.core import Learning_Core

if __name__ == '__main__':
    core = Learning_Core(path_map_tabel, path_ckpt_best_version)

    for episode in tqdm(range(core.episode_max), desc='episode', leave=False, ncols=100, disable=False):
        for condition_id in tqdm(range(core.condition.shape[0]), desc='condition', leave=False, ncols=100, disable=False):
            # 初始temperature和soc
            temperature_0, soc_0 = core.condition[condition_id]

            # 获取当前状态下的电流策略
            policy, (temperature_id, soc_id) = core.map_tabel.get_policy(temperature_0, soc_0)

            # 获取充电最后的温度
            temperature, soc = core.charging(temperature_0, soc_0, policy)
            temperature_1, soc_1 = temperature[-1], soc[-1]

            # 计算reward
            reward = core.reward_func.calculate(temperature_1, soc_1)

            # 更新策略
            policy_next, _ = core.map_tabel.get_policy(temperature_0, soc_0)
            new_policy = policy + core.alpha * (reward + core.gamma * policy_next - policy)
            core.map_tabel.change_tabel(temperature_1, soc_1, new_policy)


