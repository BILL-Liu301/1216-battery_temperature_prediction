import matplotlib.pyplot as plt
import numpy as np


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


def plot_for_predicted_temperature(temperature_prediction, temperature_reference, num_measure_point, measure_point_ids):
    plt.clf()
    for point in range(num_measure_point):
        plt.subplot(num_measure_point, 1, point+1)
        plt.plot(temperature_reference[0, :], temperature_reference[3+point, :], 'k-', label='ref')
        plt.plot(temperature_prediction[0, :], temperature_prediction[3+point, :], 'k--', label='pre')
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=10)
        plt.ylabel(measure_point_ids[point] + '/℃', fontsize=10)
        plt.legend(loc='lower right')
