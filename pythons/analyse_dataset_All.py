import numpy as np
import pickle
import matplotlib.pyplot as plt

from api.base.paths import path_data_origin_pkl_single, path_data_origin_pkl_all, path_figs

if __name__ == '__main__':
    with open(path_data_origin_pkl_single, 'rb') as pkl:
        dataset_single = pickle.load(pkl)
        pkl.close()
    with open(path_data_origin_pkl_all, 'rb') as pkl:
        dataset_all = pickle.load(pkl)
        pkl.close()

    plt.figure(figsize=(20, 11.25))
    colors = [
        '#FF0000',  # Red
        '#0000FF',  # Blue
        '#008000',  # Green
        '#FFFF00',  # Yellow
        '#800080',  # Purple
        '#00FFFF',  # Cyan
        '#FFA500',  # Orange
        '#FFC0CB',  # Pink
        '#A52A2A',  # Brown
        '#808080',  # Gray
        '#000000',  # Black
        '#FFFFFF',  # White
        '#ADD8E6',  # Light Blue
        '#90EE90',  # Light Green
        '#FFFFE0',  # Light Yellow
        '#D8BFD8',  # Light Purple
        '#E0FFFF',  # Light Cyan
        '#FFD700',  # Light Orange
        '#FFB6C1',  # Light Pink
        '#D2B48C',  # Light Brown
        '#8B0000',  # Dark Red
        '#00008B',  # Dark Blue
        '#006400',  # Dark Green
        '#800080',  # Dark Purple
        '#008B8B'  # Dark Cyan
    ]

    # 模组数据对比分析
    plt.clf()
    plt.title('Module-0: -, Module-1: --')
    for group in range(25):
        plt.plot(dataset_all[f'module-0'][group]['stamp'], np.max(dataset_all[f'module-0'][group]['Temperature_max'], axis=1, keepdims=True),
                 '-', color=colors[group], label=f'{group}')
        plt.plot(dataset_all[f'module-1'][group]['stamp'], np.max(dataset_all[f'module-1'][group]['Temperature_max'], axis=1, keepdims=True),
                 '--', color=colors[group], label=f'{group}')
    plt.grid(True)
    plt.savefig(f'{path_figs}/analyse_simulation.png')

    # 仿真数据和原始数据相比
    for cell_name, data_single in dataset_single.items():
        if cell_name.split('-')[0] == 'M1':
            module = 0
        elif cell_name.split('-')[0] == 'M2':
            module = 1
        else:
            module = -1
        location = data_single[1, 0]
        data_all = None
        for data_all in dataset_all[f'module-{module}']:
            if data_all['location'][0] == location:
                break

        plt.clf()
        plt.subplot(5, 1, 1)
        plt.plot(data_single[0], data_single[2], 'r-', label='max_single')
        plt.plot(data_all['stamp'], data_all['NTC_max'], 'r--', label='max_all')
        plt.plot(data_single[0], data_single[3], 'b-', label='min_single')
        plt.plot(data_all['stamp'], data_all['NTC_min'], 'b--', label='min_all')
        plt.ylabel('NTC')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.subplot(5, 1, 2)
        plt.plot(data_single[0], data_single[4], 'r-', label='single')
        plt.plot(data_all['stamp'], data_all['Voltage'], 'r--', label='all')
        plt.ylabel('Voltage')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.subplot(5, 1, 3)
        plt.plot(data_single[0], data_single[5], 'r-', label='single')
        plt.plot(data_all['stamp'], data_all['Current'], 'r--', label='all')
        plt.ylabel('Current')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.subplot(5, 1, 4)
        plt.plot(data_single[0], data_single[6], 'r-', label='single')
        plt.plot(data_all['stamp'], data_all['SOC'], 'r--', label='all')
        plt.ylabel('SOC')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.subplot(5, 1, 5)
        plt.plot(data_single[0], np.max(data_single[7:], axis=0), 'r-', label='single')
        plt.plot(data_all['stamp'], np.max(data_all['Temperature_max'], axis=1), 'r--', label='all')
        plt.ylabel('Temperature_max')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.savefig(f'{path_figs}/analyse_{cell_name}.png')
