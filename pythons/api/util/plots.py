import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_for_prediction_seq2seq_val_test(_results, paras):
    plt.clf()

    num_results = len(_results)
    for i in range(num_results):
        _result = _results[i]
        ax = plt.subplot(num_results, 1, i + 1)
        xx = _result['origin'][0]
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
            else:
                for j in range(paras['num_measure_point']):
                    plt.plot(xx, result[1 + j], 'k-.')
        ylim = plt.ylim()
        plt.xticks(np.arange(xx[0], xx[-1] + 10, 10))
        plt.yticks(np.arange(ylim[0], ylim[-1], 1.0))
        plt.grid(True)
        plt.legend(loc='upper right')
