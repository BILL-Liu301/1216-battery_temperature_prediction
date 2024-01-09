import os
import pickle
import psutil
import pynvml
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm

from api.module.model_Temperature_Predict import TemperaturePrediction
from api.base.ids import temperature_measure_points
from api.base.paras import device, paras, batch_size, lr_init, epoch_max, num_measure_point, sequence_predict
from api.base.paths import path_data_origin_pkl, path_result, path_pts, path_pt_best, path_figs, path_figs_test
from api.util.Mode_Choose import ModeTrain, ModeTest
from api.util.Criterion_Choose import CriterionTrain, CriterionTest
from api.util.plots import plot_during_train, plot_for_test_loss, plot_for_predicted_temperature
from api.util.data_integration import integration_predicted_temperature

# 设定随机种子
torch.manual_seed(2024)

# 加载基础模型
TP = TemperaturePrediction(device=device, paras=paras)

# 加载数据集
with open(path_data_origin_pkl, 'rb') as pkl:
    dataset = pickle.load(pkl)
    pkl.close()

# 模式选择器
mode_switch = {
    'Paras': 1,
    'Train': 1,
    'Test': 0
}

plt.figure(figsize=[20, 11.25])
if __name__ == '__main__':
    if mode_switch['Paras'] == 1:
        paras_name, paras_sum = list(), list()
        for name, param in TP.named_parameters():
            if param.requires_grad:
                paras_name.append(name)
                paras_sum.append(param.numel())
        print(f'当前模型内涉及了{len(paras_name)}个单元，总计{np.array(paras_sum).sum()}个参数。')
        for i in range(len(paras_name)):
            print(f'name: {paras_name[i]}, sum: {paras_sum[i]}')
    if mode_switch['Train'] == 1:
        print('正在进行模型训练')

        criterion_train = CriterionTrain()
        optimizer = optim.Adam(TP.parameters(), lr=lr_init)
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=3, eta_min=1e-5)
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr_init, total_steps=epoch_max, pct_start=0.1)
        # scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.8, last_epoch=-1, verbose=False)
        loss_all = np.zeros([epoch_max, 2])
        lr_all = np.zeros([epoch_max, 1])

        mode_train = ModeTrain(model=TP, batch_size=batch_size, dataset=dataset['train'], criterion=criterion_train, optimizer=optimizer, scheduler=scheduler,
                               epoch_max=epoch_max, sequence_predict=sequence_predict, num_measure_point=num_measure_point,
                               path_pts=path_pts, path_pt_best=path_pt_best, path_figs=path_figs)
        mode_train.start()

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        while True:
            time.sleep(0.1)
            with open(path_result, 'w') as result:
                result.write(f'Now: {time.asctime(time.localtime())}')
                result.write(f'\nMode: {TP.mode_for_train}')
                result.write(f'\nPoint: {mode_train.point_now + 1}/{sequence_predict}')
                result.write(f'\nEpoch: {mode_train.epoch_now + 1}/{epoch_max}, Lr:{mode_train.scheduler.get_last_lr()[0]}')
                result.write(f'\nLoss_min: {mode_train.loss_min:.6f}, Loss_max: {mode_train.loss_max:.6f}')
                result.write(f'\nLoss_min_now: {mode_train.loss.min():.6f}, Loss_max_now: {mode_train.loss.max():.6f}')
                result.write(f'\nJudge: {mode_train.loss.max()} ?< {mode_train.judge}')
                result.write(f'\nGrad_Origin: {mode_train.grad_max_origin}, Name: {mode_train.grad_max_name_origin}')
                result.write(f'\nGrad_Limit: {mode_train.grad_max_limit}, Name: {mode_train.grad_max_name_limit}')
                result.write(f'\nCPU Percent: {psutil.cpu_percent()}%')
                for d in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(d)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    result.write(f'\nGPU: {d}, Used: {info.used / 1048576:.2f}MB, Free:{info.free / 1048576:.2f}MB, Temperature: {temperature}℃')
                result.flush()
            if mode_train.flag_finish:
                with open(path_result, 'w') as result:
                    result.write(f'Now: {time.asctime(time.localtime())}')
                    result.write(f'\nEpoch: {mode_train.epoch_now + 1}/{epoch_max}, Lr:{mode_train.scheduler.get_last_lr()[0]}')
                    result.write(f'\nLoss_min: {mode_train.loss_min:.6f}, Loss_max: {mode_train.loss_max:.6f}')
                    result.write(f'\nCPU Percent: {psutil.cpu_percent()}%')
                    for d in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(d)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        result.write(f'\nGPU: {d}, Used: {info.used / 1048576}MB, Free:{info.free / 1048576}MB, Temperature: {temperature}℃')
                    result.flush()
                pynvml.nvmlShutdown()
                break

    if mode_switch['Test'] == 1:
        print('正在进行模型测试')

        TP = torch.load(path_pt_best)

        criterion_test = CriterionTest()
        mode_test = ModeTest(model=TP, dataset=dataset['test'], criterion=criterion_test)
        with torch.no_grad():
            mode_test.run()
        plot_for_test_loss(mode_test.loss)
        plt.savefig(f'{path_figs_test}/test_loss.png')
        print(f'已保存test_loss.png')
        for cell_name in mode_test.temperature_prediction:
            plot_for_predicted_temperature(cell_name, mode_test.temperature_prediction[cell_name], dataset['origin'][cell_name], num_measure_point, temperature_measure_points[cell_name])
            plt.savefig(f'{path_figs_test}/predicted_temperature_{cell_name}.png')
            print(f'已保存predicted_temperature_{cell_name}.png')

