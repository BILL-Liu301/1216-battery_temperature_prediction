
class Reward:
    def __init__(self):
        self.weight = [0.7, 0.3]

    def calculate(self, temperature, soc):
        if temperature <= 0:
            return temperature
        else:
            return 20 / temperature * self.weight[0] + soc * self.weight[1]
