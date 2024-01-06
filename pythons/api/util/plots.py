import matplotlib.pyplot as plt


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

