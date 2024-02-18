import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def plot_for_prediction_state_val_test(fig, title, _result, paras):
    plt.clf()
    xx = _result['origin'][0]

    fig.suptitle(f'Prediction_State_{title}', y=0.91, fontsize=15)

    # Voltage
    pre_mean, pre_std, ref_mean, prob, loss, origin = (
        _result['pre_mean'][0], _result['pre_std'][0], _result['ref_mean'][0], _result['prob'][0], _result['loss'][0], _result['origin'])

    # 电流
    plt.subplot(8, 1, 1)
    plt.plot(xx, origin[1], 'k', label='current')
    xlim = plt.xlim()
    plt.ylabel('Current')
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

    # SOC
    plt.subplot(8, 1, 2)
    plt.plot(xx, origin[2], 'k', label='soc')
    plt.ylabel('SOC')
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

    # Voltage: 数值
    ax = plt.subplot(8, 1, 3)
    plt.plot(xx, ref_mean, 'b-', label='ref_mean')
    plt.plot(xx, pre_mean, 'b--', label='pre_mean')
    x = np.append(xx, np.flip(xx), axis=0)
    colors = ['r', 'b', 'k']
    for j in range(3):
        y = np.append(pre_mean + pre_std * (3 - j), np.flip(pre_mean - pre_std * (3 - j)), axis=0)
        polygon = Polygon(np.column_stack((x, y)), color=colors[j], alpha=0.3)
        ax.add_patch(polygon)
    plt.ylabel('Voltage')
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

    # Voltage: 概率
    plt.subplot(8, 1, 4)
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
    plt.subplot(8, 1, 5)
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
    ax = plt.subplot(8, 1, 6)
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
    plt.subplot(8, 1, 7)
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
    plt.subplot(8, 1, 8)
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


def plot_for_prediction_temperature_val_test(fig, title, _result, paras):
    plt.clf()
    xx = _result['origin'][0]
    info = _result['info']

    fig.suptitle(f'Prediction_Temperature_{title}', y=0.91, fontsize=15)

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
    plt.plot(xx, info[4], 'r', label='NTC-HEIGHT')
    plt.plot(xx, info[5], 'b', label='NTC-LOW')
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

    # 输入：电压，电流
    plt.subplot(6, 1, 5)
    plt.plot(xx, info[3], 'r', label='Voltage')
    plt.plot(xx, info[1], 'b', label='Current')
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')

    # 输入：SOC
    plt.subplot(6, 1, 6)
    plt.plot(xx, info[2], 'k', label='SOC/%')
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.grid(True)
    plt.legend(loc='lower right')


def plot_for_prediction_all_val_test(fig, title, _result, paras):
    plt.clf()
    xx = _result['origin'][0]
    fontsize = 7

    fig.suptitle(f'Prediction_Temperature_{title}', y=0.91, fontsize=15)

    # location, Current, SOC
    ## location
    ax = plt.subplot(8, 2, 1)
    location = round(_result['origin'][2, 0] * 24)
    for loc in range(25):
        if loc == location:
            plt.bar(loc + 1, 1, 0.5, color='r', alpha=0.7)
        else:
            plt.bar(loc + 1, 1, 0.5, color='k', alpha=0.1)
        plt.text(loc + 1, 0.5, f'{loc + 1}', color='k', fontsize=fontsize, ha='center', va='center')
    plt.xticks(np.linspace(1, 25, 25))
    ax.set_xticklabels([])
    plt.xlim([0.5, 25.5])
    plt.ylim([0.0, 1.0])

    ## Current
    ax = plt.subplot(8, 2, 3)
    ## 提取数据
    current = _result['origin'][3]
    ## 绘制数据
    plt.plot(xx, current, 'k')
    ## 设定样式
    xlim = plt.xlim()
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    ax.set_xticklabels([])
    plt.ylabel('Current', fontsize=fontsize)
    plt.grid(True)

    ## SOC
    ax = plt.subplot(4, 2, 2)
    ## 提取数据
    soc_ref = _result['origin'][1]
    soc_pre = _result['origin'][4]
    ## 绘制数据
    plt.plot(xx, soc_ref, 'k--', label='ref')
    plt.plot(xx, soc_pre, 'k', label='pre')
    ## 设定样式
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    ax.set_xticklabels([])
    plt.ylabel('SOC', fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize)
    plt.grid(True)

    # --------------------------------------------------------------- #

    # 电压
    ax = plt.subplot(4, 2, 3)
    ## 提取数据
    ref = _result['origin'][6]
    pre_mean = np.append(_result['origin'][6, 0:1], _result['pre_mean'][0], axis=0)
    pre_std = np.append(np.zeros(1), _result['pre_std'][0], axis=0)
    prob = np.append(np.zeros(1), _result['prob'][0], axis=0)
    ## 绘制数据
    plt.plot(xx, ref, 'k--', label='ref')
    plt.plot(xx, pre_mean, 'k', label='pre')
    ## 绘制分布
    x = np.append(xx, np.flip(xx), axis=0)
    colors = ['r', 'b', 'k']
    for j in range(3):
        y = np.append(pre_mean + pre_std * (3 - j), np.flip(pre_mean - pre_std * (3 - j)), axis=0)
        polygon = Polygon(np.column_stack((x, y)), color=colors[j], alpha=0.3)
        ax.add_patch(polygon)
    for p in np.where(prob > 99.73)[0]:
        plt.bar(xx[p], pre_mean[p], width=1, color='r', alpha=1.0)
    ## 设定样式
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    ax.set_xticklabels([])
    plt.ylabel('Voltage(V)', fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize)
    plt.grid(True)

    ax = plt.subplot(4, 2, 4)
    ## 提取数据
    loss = np.append(np.zeros([1]), _result['loss'][0], axis=0)
    ## 绘制数据
    plt.plot(xx, loss, 'k')
    for p in np.where(loss <= 0.5):
        plt.bar(xx[p], loss[p], width=1, color='g', alpha=1.0)
    ## 设定样式
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    ax.set_xticklabels([])
    plt.ylabel('Loss(V)', fontsize=fontsize)
    plt.grid(True)

    # --------------------------------------------------------------- #

    # NTC
    ax = plt.subplot(4, 2, 5)
    ## 提取数据
    ref = _result['origin'][7:9]
    pre_mean = np.append(_result['origin'][7:9, 0:1], _result['pre_mean'][1:3], axis=1)
    pre_std = np.append(np.zeros([2, 1]), _result['pre_std'][1:3], axis=1)
    prob = np.append(np.zeros([2, 1]), _result['prob'][1:3], axis=1)
    ## 绘制数据
    plt.plot(xx, ref[0], 'r--', label='ref_max')
    plt.plot(xx, pre_mean[0], 'r', label='pre_max')
    plt.plot(xx, ref[1], 'b--', label='ref_min')
    plt.plot(xx, pre_mean[1], 'b', label='pre_min')
    ## 绘制分布
    x = np.append(xx, np.flip(xx), axis=0)
    colors = ['r', 'b', 'k']
    for j in range(3):
        y = np.append(pre_mean[0] + pre_std[0] * (3 - j), np.flip(pre_mean[0] - pre_std[0] * (3 - j)), axis=0)
        polygon = Polygon(np.column_stack((x, y)), color=colors[j], alpha=0.3)
        ax.add_patch(polygon)

        y = np.append(pre_mean[1] + pre_std[1] * (3 - j), np.flip(pre_mean[1] - pre_std[1] * (3 - j)), axis=0)
        polygon = Polygon(np.column_stack((x, y)), color=colors[j], alpha=0.3)
        ax.add_patch(polygon)
    for p in np.where(prob[0] > 99.73)[0]:
        plt.bar(xx[p], pre_mean[0, p], width=1, color='r', alpha=0.3)
    for p in np.where(prob[1] > 99.73)[0]:
        plt.bar(xx[p], pre_mean[1, p], width=1, color='b', alpha=0.3)
    ## 设定样式
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    ax.set_xticklabels([])
    plt.ylabel('NTC(℃)', fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize)
    plt.grid(True)

    ax = plt.subplot(4, 2, 6)
    ## 提取数据
    loss = np.append(np.zeros([2, 1]), _result['loss'][1:3], axis=1)
    ## 绘制数据
    plt.plot(xx, loss[0], 'r', label='NTC_Max')
    plt.plot(xx, loss[1], 'b', label='NTC_Min')
    for p in np.where(loss[0] <= 0.5):
        plt.bar(xx[p], loss[0, p], width=1, color='r', alpha=0.3)
    for p in np.where(loss[1] <= 0.5):
        plt.bar(xx[p], loss[1, p], width=1, color='b', alpha=0.3)
    ## 设定样式
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    ax.set_xticklabels([])
    plt.ylabel('Loss(℃)', fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize)
    plt.grid(True)

    # --------------------------------------------------------------- #

    # 温度
    ax = plt.subplot(4, 2, 7)
    ## 提取数据
    ref = _result['origin'][9]
    pre_mean = np.append(_result['origin'][9, 0:1], _result['pre_mean'][3], axis=0)
    pre_std = np.append(np.zeros(1), _result['pre_std'][3], axis=0)
    prob = np.append(np.zeros(1), _result['prob'][3], axis=0)
    ## 绘制数据
    plt.plot(xx, ref, 'k--', label='ref')
    plt.plot(xx, pre_mean, 'k', label='pre')
    ## 绘制分布
    x = np.append(xx, np.flip(xx), axis=0)
    colors = ['r', 'b', 'k']
    for j in range(3):
        y = np.append(pre_mean + pre_std * (3 - j), np.flip(pre_mean - pre_std * (3 - j)), axis=0)
        polygon = Polygon(np.column_stack((x, y)), color=colors[j], alpha=0.3)
        ax.add_patch(polygon)
    for p in np.where(prob > 99.73)[0]:
        plt.bar(xx[p], pre_mean[p], width=1, color='r', alpha=1.0)
    ## 设定样式
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    ax.set_xticklabels([])
    plt.ylabel('Temperature(℃)', fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize)
    plt.grid(True)

    ax = plt.subplot(4, 2, 8)
    ## 提取数据
    loss = np.append(np.zeros([1]), _result['loss'][3], axis=0)
    ## 绘制数据
    plt.plot(xx, loss, 'k')
    for p in np.where(loss <= 1.0):
        plt.bar(xx[p], loss[p], width=1, color='g', alpha=1.0)
    ## 设定样式
    plt.xlim(xlim)
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    ax.set_xticklabels([])
    plt.ylabel('Loss(℃)', fontsize=fontsize)
    plt.grid(True)
