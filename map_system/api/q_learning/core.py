import numpy as np

from .q_tabel import MAP_Tabel
from .reward import Reward
from ..system.charging import Charging_System


class Learning_Core:
    def __init__(self, path_tabel, path_system):
        # 初始化
        self.map_tabel = MAP_Tabel(path_tabel)
        self.reward_func = Reward()
        self.environment = Charging_System(path_system)

        # 参数变量
        self.alpha = 0.1
        self.gamma = 0.1
        self.episode_max = 30
        self.charging_stamp = 30
        num_condition = 20
        self.condition = self.init_condition(num_condition)

    def init_condition(self, num_condition):
        temperature = np.linspace(start=-25, stop=60, num=num_condition)
        soc = np.random.rand(num_condition) * 0.1
        return np.stack([temperature, soc], axis=1)

    def integral_current(self, current, soc_0):
        soc = np.zeros(current.shape)
        for t in range(1, current.shape[0]):
            soc[t] = (current[t] + current[t - 1]) * 2 / 2 + soc[t - 1]
        soc = soc / 2.561927693277231e+03 + soc_0
        return soc

    def charging(self, temperature_0, soc_0, policy):
        # 每个模组内由25个电芯组，每隔2s为一个时间戳
        num_group = 25
        current = np.ones([self.charging_stamp, 1]) * self.map_tabel.get_current(policy)
        soc = self.integral_current(current, soc_0)

        temperature, soc = None, None
        return temperature, soc
