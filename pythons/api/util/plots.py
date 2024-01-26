import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def plot_for_prediction_temperature_val_test(_result, paras):
    plt.clf()
    xx = _result['origin'][0]
    info = _result['info']

    # 结果展示
    ax = plt.subplot(6, 1, 1)
    # for j in range(paras['num_measure_point']):
    #     plt.plot(xx, _result['origin'][1 + j], 'k-.')

    plt.plot(_result['ref'][0], _result['ref'][1], 'b-', label='ref_mean')

    plt.plot(xx, _result['pre'][0], 'r-', label='pre_mean')
    # plt.plot(x, result[0] + np.sqrt(result[1]) * 3, 'r-', label='pre_var_max')
    # plt.plot(x, result[0] - np.sqrt(result[1]) * 3, 'g-', label='pre_var_min')
    x = np.append(xx, np.flip(xx), axis=0)
    colors = ['r', 'b', 'k']
    for j in range(3):
        y = np.append(_result['pre'][0] + np.sqrt(_result['pre'][1]) * (3 - j), np.flip(_result['pre'][0] - np.sqrt(_result['pre'][1]) * (3 - j)), axis=0)
        polygon = Polygon(np.column_stack((x, y)), color=colors[j], alpha=0.3)
        ax.add_patch(polygon)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.yticks(np.arange(np.floor(ylim[0]), np.floor(ylim[-1]), 2))
    plt.grid(True)
    plt.legend(loc='lower right')

    # 概率
    plt.subplot(6, 1, 2)
    plt.plot([xx[0], xx[-1]], [99.73, 99.73], 'r--', label='3σ')
    plt.plot([xx[0], xx[-1]], [95.45, 95.45], 'b--', label='2σ')
    plt.plot([xx[0], xx[-1]], [68.27, 68.27], 'g--', label='1σ')
    plt.plot(xx, _result['prob'], 'k', label='Prob')
    for p in np.where(_result['prob'] > 99.73)[0]:
        plt.bar(xx[p], _result['prob'][p], width=1, color='r')
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=8)

    # Loss
    plt.subplot(6, 1, 3)
    plt.plot(xx, _result['loss'], 'k', label='Loss')
    for p in np.where(_result['loss'] <= 1)[0]:
        plt.bar(xx[p], _result['loss'][p], width=1, color='g')
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

    # 输入：NTC温度
    plt.subplot(6, 1, 4)
    plt.plot(xx, info[1], 'r', label='NTC-HEIGHT')
    plt.plot(xx, info[2], 'b', label='NTC-LOW')
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

    # 输入：电压，电流
    plt.subplot(6, 1, 5)
    plt.plot(xx, info[3], 'r', label='Voltage')
    plt.plot(xx, info[4], 'b', label='Current')
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

    # 输入：SOC
    plt.subplot(6, 1, 6)
    plt.plot(xx, info[5], 'k', label='SOC/%')
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')


def plot_for_prediction_state_val_test(_result, paras):
    plt.clf()
    xx = _result['origin'][0]

    # Voltage
    pre_mean, pre_std, ref_mean, prob, loss = (
        _result['pre_mean'][0], _result['pre_std'][0], _result['ref_mean'][0], _result['prob'][0], _result['loss'][0])

    # Voltage: 数值
    ax = plt.subplot(6, 1, 1)
    plt.plot(xx, ref_mean, 'b-', label='ref_mean')
    plt.plot(xx, pre_mean, 'b--', label='pre_mean')
    x = np.append(xx, np.flip(xx), axis=0)
    colors = ['r', 'b', 'k']
    for j in range(3):
        y = np.append(pre_mean + pre_std * (3 - j), np.flip(pre_mean - pre_std * (3 - j)), axis=0)
        polygon = Polygon(np.column_stack((x, y)), color=colors[j], alpha=0.3)
        ax.add_patch(polygon)
    plt.ylabel('Voltage')
    xlim = plt.xlim()
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

    # Voltage: 概率
    plt.subplot(6, 1, 2)
    plt.ylabel('Voltage_Prob')
    plt.plot([xx[0], xx[-1]], [99.73, 99.73], 'r--', label='3σ')
    plt.plot([xx[0], xx[-1]], [95.45, 95.45], 'b--', label='2σ')
    plt.plot([xx[0], xx[-1]], [68.27, 68.27], 'g--', label='1σ')
    plt.plot(xx, prob, 'k', label='Prob')
    for p in np.where(prob > 99.73)[0]:
        plt.bar(xx[p], prob[p], width=1, color='r', alpha=1.0)
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=8)

    # Voltage: 偏差
    plt.subplot(6, 1, 3)
    plt.ylabel('Voltage_Loss')
    plt.plot(xx, loss, 'k', label='Loss')
    for p in np.where(loss <= 0.1)[0]:
        plt.bar(xx[p], loss[p], width=1, color='g', alpha=1.0)
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

    # NTC
    pre_mean, pre_std, ref_mean, prob, loss = (
        _result['pre_mean'][1:3], _result['pre_std'][1:3], _result['ref_mean'][1:3], _result['prob'][1:3], _result['loss'][1:3])

    # NTC: 数值
    ax = plt.subplot(6, 1, 4)
    plt.plot(xx, ref_mean[0], 'r-', label='ref_mean_Max')
    plt.plot(xx, pre_mean[0], 'r--', label='pre_mean_Max')
    plt.plot(xx, ref_mean[1], 'b-', label='ref_mean_Max')
    plt.plot(xx, pre_mean[1], 'b--', label='pre_mean_Max')
    x = np.append(xx, np.flip(xx), axis=0)
    colors = ['r', 'b', 'k']
    for j in range(3):
        y = np.append(pre_mean[0] + pre_std[0] * (3 - j), np.flip(pre_mean[0] - pre_std[0] * (3 - j)), axis=0)
        polygon = Polygon(np.column_stack((x, y)), color=colors[j], alpha=0.3)
        ax.add_patch(polygon)
    for j in range(3):
        y = np.append(pre_mean[1] + pre_std[1] * (3 - j), np.flip(pre_mean[1] - pre_std[1] * (3 - j)), axis=0)
        polygon = Polygon(np.column_stack((x, y)), color=colors[j], alpha=0.3)
        ax.add_patch(polygon)
    plt.ylabel('NTC')
    xlim = plt.xlim()
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

    # NTC: 概率
    plt.subplot(6, 1, 5)
    plt.ylabel('NTC_Prob')
    plt.plot([xx[0], xx[-1]], [99.73, 99.73], 'r--', label='3σ')
    plt.plot([xx[0], xx[-1]], [95.45, 95.45], 'b--', label='2σ')
    plt.plot([xx[0], xx[-1]], [68.27, 68.27], 'g--', label='1σ')
    plt.plot(xx, prob[0], 'r-', label='Prob_Max')
    plt.plot(xx, prob[1], 'b-', label='Prob_Min')
    for p in np.where(prob[0] > 99.73)[0]:
        plt.bar(xx[p], prob[0, p], width=1, color='r', alpha=0.5)
    for p in np.where(prob[1] > 99.73)[0]:
        plt.bar(xx[p], prob[1, p], width=1, color='b', alpha=0.5)
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=8)

    # NTC: 偏差
    plt.subplot(6, 1, 6)
    plt.ylabel('NTC_Loss')
    plt.plot(xx, loss[0], 'r', label='Loss_Max')
    plt.plot(xx, loss[1], 'b', label='Loss_Min')
    for p in np.where(loss[0] <= 0.1):
        plt.bar(xx[p], loss[0, p], width=1, color='r', alpha=0.5)
    for p in np.where(loss[1] <= 0.1):
        plt.bar(xx[p], loss[1, p], width=1, color='b', alpha=0.5)
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

