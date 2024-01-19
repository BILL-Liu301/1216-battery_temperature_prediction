import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_for_prediction_seq2seq_val_test(_result, paras):
    plt.clf()
    xx = _result['origin'][0]
    info = _result['info']

    # 结果展示
    ax = plt.subplot(6, 1, 1)
    for j in range(paras['num_measure_point']):
        plt.plot(xx, _result['origin'][1 + j], 'k-.')

    plt.plot(_result['ref'][0], _result['ref'][1], 'b-', label='ref_mean')

    plt.plot(xx, _result['pre'][0], 'r-', label='pre_mean')
    # plt.plot(x, result[0] + np.sqrt(result[1]) * 3, 'r-', label='pre_var_max')
    # plt.plot(x, result[0] - np.sqrt(result[1]) * 3, 'g-', label='pre_var_min')
    x = np.append(xx, np.flip(xx), axis=0)
    colors = ['r', 'b', 'k']
    for j in range(3):
        y = np.append(_result['pre'][0] + np.sqrt(_result['pre'][1]) * (3 - j), np.flip(_result['pre'][0] - np.sqrt(_result['pre'][1]) * (3 - j)), axis=0)
        polygon = patches.Polygon(np.column_stack((x, y)), color=colors[j], alpha=0.3)
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
    for p in np.where(_result['prob'] > 95.45)[0]:
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



