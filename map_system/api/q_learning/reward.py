import math


class Reward:
    def __init__(self):
        self.weight = [1.0, 0.05]
        self.scale = 5

    def calculate(self, temperature, soc):
        if temperature <= 0:
            reward = 1 / (1 + math.exp(temperature)) + 1 / (1 + math.exp(soc / 100 - 1))
        else:
            reward = 25 / (temperature + 50) * self.weight[0] + soc * self.weight[1]
        return reward * self.scale
