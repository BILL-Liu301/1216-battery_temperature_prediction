import torch
import pickle

from api.module.model_Temperature_Predict import TemperaturePrediction
from api.base.paras import device, paras
from api.base.paths import path_dataset_pkl

# 设定随机种子
torch.manual_seed(2024)

# 加载基础模型
TP = TemperaturePrediction(device=device, paras=paras)

# 加载数据集
with open(path_dataset_pkl, 'rb') as pkl:
    dataset = pickle.load(pkl)
    pkl.close()

# 模式选择器
mode_switch = {
    'Train': 1,
    'Test': 0
}

if __name__ == '__main__':
    if mode_switch['Train'] == 1:
        print('正在进行模型训练')
    elif mode_switch['Test'] == 0:
        print('正在进行模型测试')
