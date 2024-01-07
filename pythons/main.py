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
from api.base.paras import device, paras, batch_size, lr_init, epoch_max, num_measure_point
from api.base.paths import path_dataset_pkl, path_result, path_pts, path_pt_best, path_figs, path_figs_test
from api.util.Mode_Choose import ModeTrain, ModeTest
from api.util.Criterion_Choose import CriterionTrain, CriterionTest
from api.util.plots import plot_during_train, plot_for_test_loss, plot_for_predicted_temperature
from api.util.data_integration import integration_predicted_temperature

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
    'Train': 0,
    'Test': 1
}

plt.figure(figsize=[20, 11.25])
if __name__ == '__main__':
    if mode_switch['Train'] == 1:
        print('正在进行模型训练')

        criterion_train = CriterionTrain()
        optimizer = optim.Adam(TP.parameters(), lr=lr_init)
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=3, eta_min=1e-5)
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr_init, total_steps=epoch_max, pct_start=0.1)
        # scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.8, last_epoch=-1, verbose=False)
        loss_all = np.zeros([epoch_max, 2])
        lr_all = np.zeros([epoch_max, 1])

        for epoch in tqdm(range(epoch_max), desc='Epoch', leave=False, ncols=80, disable=False):
            mode_train = ModeTrain(model=TP, batch_size=batch_size, dataset=dataset['train'], criterion=criterion_train, optimizer=optimizer)
            mode_train.start()

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            while True:
                time.sleep(0.1)
                with open(path_result, 'w') as result:
                    result.write(f"Now: {time.asctime(time.localtime())}")
                    result.write(f"\nEpoch: {epoch + 1}/{epoch_max}, Lr:{scheduler.get_last_lr()[0]}")
                    result.write(f"\nLoss_min: {mode_train.loss_min:.6f}, Loss_max: {mode_train.loss_max:.6f}")
                    result.write(f"\nLoss_min_now: {mode_train.loss.min():.6f}, Loss_max_now: {mode_train.loss.max():.6f}")
                    result.write(f"\nGrad: {mode_train.grad_max}, Name: {mode_train.grad_max_name}")
                    result.write(f"\nCPU Percent: {psutil.cpu_percent()}%")
                    for d in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(d)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        result.write(f"\nGPU: {d}, Used: {info.used / 1048576:.2f}MB, Free:{info.free / 1048576:.2f}MB, Temperature: {temperature}℃")
                    result.flush()
                if mode_train.flag_finish:
                    with open(path_result, 'w') as result:
                        result.write(f"Now: {time.asctime(time.localtime())}")
                        result.write(f"\nEpoch: {epoch + 1}/{epoch_max}, Lr:{scheduler.get_last_lr()[0]}")
                        result.write(f"\nLoss_min: {mode_train.loss_min:.6f}, Loss_max: {mode_train.loss_max:.6f}")
                        result.write(f"\nGrad: {mode_train.grad_max}, Name: {mode_train.grad_max_name}")
                        result.write(f"\nCPU Percent: {psutil.cpu_percent()}%")
                        for d in range(device_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(d)
                            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            result.write(f"\nGPU: {d}, Used: {info.used / 1048576}MB, Free:{info.free / 1048576}MB, Temperature: {temperature}℃")
                        result.flush()
                    pynvml.nvmlShutdown()
                    break

            for cell_id in dataset['train']:
                random.shuffle(dataset['train'][cell_id])

            loss_all[epoch, :] = np.array([mode_train.loss.min(), mode_train.loss.max()])
            lr_all[epoch, 0] = scheduler.get_last_lr()[0]
            scheduler.step()
            torch.save(TP, os.path.join(path_pts, f"{epoch}_{loss_all[epoch, 1]:.2f}.pt"))
            if loss_all[epoch, 1] == loss_all[0:(epoch + 1), 1].min():
                torch.save(TP, path_pt_best)
            plot_during_train(epoch, loss_all, lr_all)
            plt.savefig(f"{path_figs}/train.png")
    elif mode_switch['Test'] == 1:
        print('正在进行模型测试')

        TP = torch.load(path_pt_best)

        criterion_test = CriterionTest()
        mode_test = ModeTest(model=TP, dataset=dataset['test'], criterion=criterion_test)
        with torch.no_grad():
            mode_test.run()
            temperature_prediction = integration_predicted_temperature(mode_test.temperature_prediction, dataset['origin'], 1)
        plot_for_test_loss(mode_test.loss)
        plt.savefig(f"{path_figs_test}/test_loss.png")
        print(f'已保存test_loss.png')
        for cell_name in temperature_prediction:
            plot_for_predicted_temperature(temperature_prediction[cell_name], dataset['origin'][cell_name], num_measure_point, temperature_measure_points[cell_name])
            plt.savefig(f"{path_figs_test}/predicted_temperature_{cell_name}.png")
            print(f'已保存predicted_temperature_{cell_name}.png')

