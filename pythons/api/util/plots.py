import matplotlib.pyplot as plt
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
            for j in range(num_measure_point):
                if result_name == 'pre':
                    plt.plot(np.arange(0, result.shape[1], 1), result[4 + j], 'k-')
                else:
                    plt.plot(np.arange(0, result.shape[1], 1), result[4 + j], 'k-.')
        plt.grid(True)

