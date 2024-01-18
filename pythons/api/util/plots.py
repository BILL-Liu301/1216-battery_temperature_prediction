import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_for_prediction_seq2seq_val_test(_result, paras):
    plt.clf()
    xx = _result['origin'][0]
    info = _result['info']

    # 结果展示
    ax = plt.subplot(3, 1, 1)
    for result_name, result in _result.items():
        if result_name == 'pre':
            plt.plot(xx, result[0], 'r-', label='pre_mean')
            # plt.plot(x, result[0] + np.sqrt(result[1]) * 3, 'r-', label='pre_var_max')
            # plt.plot(x, result[0] - np.sqrt(result[1]) * 3, 'g-', label='pre_var_min')
            x = np.append(xx, np.flip(xx), axis=0)
            colors = ['r', 'b', 'k']
            for j in range(3):
                y = np.append(result[0] + np.sqrt(result[1]) * (3 - j), np.flip(result[0] - np.sqrt(result[1]) * (3 - j)), axis=0)
                polygon = patches.Polygon(np.column_stack((x, y)), color=colors[j], alpha=0.3)
                ax.add_patch(polygon)
        elif result_name == 'ref':
            plt.plot(result[0], result[1], 'b-', label='ref_mean')
        elif result_name == 'origin':
            for j in range(paras['num_measure_point']):
                plt.plot(xx, result[1 + j], 'k-.')
        else:
            pass
    ylim = plt.ylim()
    plt.xticks(np.arange(np.floor(xx[0]), np.floor(xx[-1]), 50))
    plt.yticks(np.arange(np.floor(ylim[0]), np.floor(ylim[-1]), 2))
    plt.grid(True)
    plt.legend(loc='lower right')

    # 输入：NTC温度
    plt.subplot(3, 1, 2)
    plt.plot(xx, info[1], 'r', label='NTC-HEIGHT')
    plt.plot(xx, info[2], 'b', label='NTC-LOW')
    plt.grid(True)
    plt.legend(loc='lower right')

    # 输入：电压，电流
    plt.subplot(6, 1, 5)
    plt.plot(xx, info[3], 'r', label='Voltage')
    plt.plot(xx, info[4], 'b', label='Current')
    plt.grid(True)
    plt.legend(loc='lower right')

    # 输入：SOC
    plt.subplot(6, 1, 6)
    plt.plot(xx, info[5], 'r', label='SOC')
    plt.grid(True)
    plt.legend(loc='lower right')

