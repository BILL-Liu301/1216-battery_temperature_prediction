import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from scipy.stats import norm
from scipy.io import savemat
from tqdm import tqdm

from api.base.paras import paras_Prediction_All
from api.datasets.prediction_all import dataset_test
from api.models.prediction_state import Prediction_State_Module
from api.models.prediction_temperature import Prediction_Temperature_Module
from api.models.prediction_all import Prediction_All_Module
from api.base.paths import path_ckpt_best_version, path_ckpts, path_figs_test, path_mat, path_result_store_all
from api.util.plots import plot_for_prediction_all_val_test

if __name__ == '__main__':
    pl.seed_everything(2024)
    fig = plt.figure(figsize=(20, 11.25))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置显示中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号

    # 找到ckpt
    path_model_state = path_ckpt_best_version + 'state/version_0/checkpoints/'
    path_model_temperature = path_ckpt_best_version + 'temperature/version_0/checkpoints/'
    ckpt_model_state = path_model_state + os.listdir(path_model_state)[0]
    ckpt_model_temperature = path_model_temperature + os.listdir(path_model_temperature)[0]

    # 加载模型
    ckpt = torch.load(ckpt_model_state)
    model_state = Prediction_State_Module(ckpt['hyper_parameters']['paras'])
    for key, value in ckpt['state_dict'].copy().items():
        key_new = key[(len(key.split('.')[0]) + 1):]
        ckpt['state_dict'].update({key_new: ckpt['state_dict'].pop(key)})
    model_state.load_state_dict(state_dict=ckpt['state_dict'])

    ckpt = torch.load(ckpt_model_temperature)
    model_temperature = Prediction_Temperature_Module(ckpt['hyper_parameters']['paras'])
    for key, value in ckpt['state_dict'].copy().items():
        key_new = key[(len(key.split('.')[0]) + 1):]
        ckpt['state_dict'].update({key_new: ckpt['state_dict'].pop(key)})
    model_temperature.load_state_dict(state_dict=ckpt['state_dict'])

    model_all = Prediction_All_Module(model_state.to(paras_Prediction_All['device']).eval(), model_temperature.to(paras_Prediction_All['device']).eval())

    # 设定存储
    test_results = list()
    test_losses = {
        'Voltage': {
            'mean': list(),
            'max': list(),
            'min': list()
        },
        'NTC_max': {
            'mean': list(),
            'max': list(),
            'min': list()
        },
        'NTC_min': {
            'mean': list(),
            'max': list(),
            'min': list()
        },
        'Temperature': {
            'mean': list(),
            'max': list(),
            'min': list()
        }
    }
    criterion_test = nn.L1Loss(reduction='none')

    # 温度预测
    data_mat, condition_id = dict(), 0
    for condition, condition_data in tqdm(dataset_test.data.items(), desc='Test', leave=False, ncols=100, disable=False):
        condition_data = condition_data.to(paras_Prediction_All['device'])
        inp_his, inp_info = condition_data[:, 0:1, 2:], condition_data[:, 1:, 2:6]
        with torch.no_grad():
            pre_mean, pre_var = model_all(inp_his, inp_info)
            ref_mean = condition_data[:, 1:, 6:]
        losses = criterion_test(pre_mean, ref_mean)

        # 将数据保存成.mat文件
        savemat(path_mat + condition + '.mat', {
            'origin': condition_data.cpu().numpy(),
            'pre_mean': pre_mean.cpu().numpy(),
            'pre_var': pre_var.cpu().numpy(),
            'ref_mean': ref_mean.cpu().numpy(),
            'losses': losses.cpu().numpy()
        })

        # 在python内绘制结果
        prob = np.abs(norm.cdf(ref_mean.cpu().numpy(), pre_mean.cpu().numpy(), torch.sqrt(pre_var).cpu().numpy()) - 0.5) * 2 * 100
        for b in range(len(condition_data)):
            test_results.append({
                'origin': condition_data[b].T.cpu().numpy(),
                'pre_mean': pre_mean[b].T.cpu().numpy(),
                'pre_std': np.sqrt(pre_var[b].T.cpu().numpy()),
                'ref_mean': ref_mean[b].T.cpu().numpy(),
                'loss': losses[b].T.cpu().numpy(),
                'prob': prob[b].transpose()
            })
            loss = losses[b]
            test_losses['Voltage']['mean'].append(loss[:, 0].mean().unsqueeze(0))
            test_losses['Voltage']['max'].append(loss[:, 0].max().unsqueeze(0))
            test_losses['Voltage']['min'].append(loss[:, 0].min().unsqueeze(0))

            test_losses['NTC_max']['mean'].append(loss[:, 1].mean().unsqueeze(0))
            test_losses['NTC_max']['max'].append(loss[:, 1].max().unsqueeze(0))
            test_losses['NTC_max']['min'].append(loss[:, 1].min().unsqueeze(0))

            test_losses['NTC_min']['mean'].append(loss[:, 2].mean().unsqueeze(0))
            test_losses['NTC_min']['max'].append(loss[:, 2].max().unsqueeze(0))
            test_losses['NTC_min']['min'].append(loss[:, 2].min().unsqueeze(0))

            test_losses['Temperature']['mean'].append(loss[:, 3].mean().unsqueeze(0))
            test_losses['Temperature']['max'].append(loss[:, 3].max().unsqueeze(0))
            test_losses['Temperature']['min'].append(loss[:, 3].min().unsqueeze(0))

    # 结果展示
    print(f'总计有{len(dataset_test)}组数据')
    states = ['Voltage', 'NTC_max', 'NTC_min', 'Temperature']
    for state in states:
        test_loss = test_losses[state]
        loss_mean, loss_max, loss_min = torch.cat(test_loss['mean'], dim=0), torch.cat(test_loss['max'], dim=0), torch.cat(test_loss['min'], dim=0)
        print(f'{state}:')
        print(f'\t平均均值误差：{loss_mean.mean():.4f}V?℃')
        print(f'\t平均最大误差：{loss_max.mean():.4f}V?℃')
        print(f'\t平均最小误差：{loss_min.mean():.4f}V?℃')
    for i in tqdm(range(0, len(test_results), 1), desc='Test', leave=False, ncols=100, disable=False):
        test_result = test_results[i]

        plot_for_prediction_all_val_test(fig, f'{dataset_test.data_condition[i]}_{i % 25}', test_result, paras_Prediction_All)
        # plt.savefig(path_figs_test + f'{i}.png')  # 用于日常debug
        plt.savefig(path_result_store_all + f'{i}.png')  # 用于保存最终结果
