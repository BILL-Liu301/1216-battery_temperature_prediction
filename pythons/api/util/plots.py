import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm


def plot_during_train(epoch, loss_all, lr_all):
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, epoch + 1, 1), loss_all[0:(epoch + 1), 0], "g", label="min")
    plt.plot(np.arange(0, epoch + 1, 1), loss_all[0:(epoch + 1), 1], "r", label="max")
    plt.plot([0, epoch], [0.0, 0.0], "k--")
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, epoch + 1, 1), lr_all[0:(epoch + 1), 0], "k", label="lr_init")
    plt.legend(loc='upper right')

    # plt.pause(0.01)


def plot_for_test_loss(loss):
    plt.clf()
    for cell_name, cell_loss in loss.items():
        x = np.linspace(1, cell_loss.shape[0], cell_loss.shape[0])
        plt.bar(x, cell_loss, width=1, label=cell_name, alpha=0.5)
    plt.grid(True)
    plt.ylabel('Temperature/℃')
    plt.legend(loc='upper right')


def plot_for_predicted_temperature(cell_name, temperature_prediction, temperature_reference, num_measure_point, measure_point_ids):
    plt.clf()
    for point in tqdm(range(num_measure_point), desc='Point', leave=False, ncols=100, disable=False):
        plt.subplot(num_measure_point, 1, point+1)
        plt.plot(temperature_reference[0, :], temperature_reference[3 + point, :], 'k--', label='ref')
        for group in tqdm(temperature_prediction, desc=f'Plot_{cell_name}', leave=False, ncols=100, disable=False):
            group_np = group.cpu().numpy()
            plt.plot(group_np[0, :], group_np[3+point], 'k-')
            plt.grid(True)
            plt.tick_params(axis='both', labelsize=10)
            plt.ylabel(measure_point_ids[point] + '/℃', fontsize=10)
            plt.legend(loc='lower right')


def plot_for_pre_encoder_val_test(_results, num_measure_point):
    plt.clf()

    num_results = len(_results)
    for i in range(num_results):
        _result = _results[i]
        plt.subplot(num_results, 1, i + 1)
        for result_name, result in _result.items():
            if result_name == 'pre':
                plt.plot(np.arange(0, result.shape[1], 1), result[4], 'k-')
                plt.plot(np.arange(0, result.shape[1], 1), result[4] + result[5] * 3, 'r-')
                plt.plot(np.arange(0, result.shape[1], 1), result[4] - result[5] * 3, 'b-')
            else:
                for j in range(num_measure_point):
                    plt.plot(np.arange(0, result.shape[1], 1), result[4 + j], 'k-.')
        plt.grid(True)


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
                plt.plot(xx, result[0], 'b-', label='ref_mean')
            else:
                for j in range(paras['num_measure_point']):
                    plt.plot(xx, result[1 + j], 'k-.')
        ylim = plt.ylim()
        plt.xticks(np.arange(xx[0], xx[-1] + 10, 10))
        plt.yticks(np.arange(ylim[0], ylim[-1], 1.0))
        plt.grid(True)
        plt.legend(loc='upper right')
