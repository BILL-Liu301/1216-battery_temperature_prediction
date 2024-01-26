import numpy as np
import pickle
import matplotlib.pyplot as plt

from api.base.paths import path_data_origin_pkl_real, path_data_origin_pkl_sim, path_figs

if __name__ == '__main__':
    with open(path_data_origin_pkl_real, 'rb') as pkl:
        dataset_real = pickle.load(pkl)
        pkl.close()
    with open(path_data_origin_pkl_sim, 'rb') as pkl:
        dataset_sim = pickle.load(pkl)
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
        plt.plot(dataset_sim[f'module-0'][group]['stamp'], np.max(dataset_sim[f'module-0'][group]['Temperature_max'], axis=1, keepdims=True),
                 '-', color=colors[group], label=f'{group}')
        plt.plot(dataset_sim[f'module-1'][group]['stamp'], np.max(dataset_sim[f'module-1'][group]['Temperature_max'], axis=1, keepdims=True),
                 '--', color=colors[group], label=f'{group}')
    plt.grid(True)
    plt.savefig(f'{path_figs}/analyse_simulation.png')

    # 仿真数据和原始数据相比
    for cell_name, datas_real in dataset_real.items():
        module = cell_name
        for data_real in datas_real:
            location = data_real[f'location'][0, 0]
            data_sim = None
            for data_sim in dataset_sim[module]:
                if data_sim['location'][0, 0] == location:
                    break

            plt.clf()
            plt.subplot(5, 1, 1)
            plt.plot(data_real['stamp'], data_real['NTC_max'], 'r-', label='max_real')
            plt.plot(data_sim['stamp'], data_sim['NTC_max'], 'r--', label='max_sim')
            plt.plot(data_real['stamp'], data_real['NTC_min'], 'b-', label='min_real')
            plt.plot(data_sim['stamp'], data_sim['NTC_min'], 'b--', label='min_sim')
            plt.ylabel('NTC')
            plt.legend(loc='upper right')
            plt.grid(True)

            plt.subplot(5, 1, 2)
            plt.plot(data_real['stamp'], data_real['Voltage'], 'r-', label='real')
            plt.plot(data_sim['stamp'], data_sim['Voltage'], 'r--', label='sim')
            plt.ylabel('Voltage')
            plt.legend(loc='upper right')
            plt.grid(True)

            plt.subplot(5, 1, 3)
            plt.plot(data_real['stamp'], data_real['Current'], 'r-', label='real')
            plt.plot(data_sim['stamp'], data_sim['Current'], 'r--', label='sim')
            plt.ylabel('Current')
            plt.legend(loc='upper right')
            plt.grid(True)

            plt.subplot(5, 1, 4)
            plt.plot(data_real['stamp'], data_real['SOC'], 'r-', label='real')
            plt.plot(data_sim['stamp'], data_sim['SOC'], 'r--', label='sim')
            plt.ylabel('SOC')
            plt.legend(loc='upper right')
            plt.grid(True)

            plt.subplot(5, 1, 5)
            plt.plot(data_real['stamp'], np.max(data_real['Temperature_max'], axis=1), 'r-', label='real')
            plt.plot(data_sim['stamp'], np.max(data_sim['Temperature_max'], axis=1), 'r--', label='sim')
            plt.ylabel('Temperature_max')
            plt.legend(loc='upper right')
            plt.grid(True)

            plt.savefig(f'{path_figs}/analyse_{cell_name}_{location}.png')
