import torch
import pickle
import numpy as np
import pandas as pd

from api.base.paths import path_data_t9m, path_dataset_pkl
from api.base.ids import temperature_measure_points, temperature_match
from api.base.paras import K, num_measure_point, sequence_init, sequence_predict, device


def slice_origin(data_origin, data_aim, batches, seq_all):
    for group in range(data_origin.shape[0]):
        for batch in range(batches):
            data_aim[group, batch] = data_origin[group, batch:(batch + seq_all)]


if __name__ == '__main__':
    # 从xlsx中提取数据
    print(f'从【{path_data_t9m}】提取数据，正在提取...')
    data_xlsx = pd.read_excel(path_data_t9m, sheet_name='常温充电')
    duration = np.asarray(data_xlsx['时间[s]'])[np.isfinite(np.asarray(data_xlsx['时间[s]']))]
    batch_all = duration.shape[0] - (sequence_init + sequence_predict) + 1

    # 基础存储单元，同时存储了训练集和测试集
    # 时间[s]，电流，SOC，测点1，测点2，测点3，测点4，测点5，测点6，测点7
    #         上
    #       1    2
    # 内  3    4   5  外
    #       6    7
    #         下
    data_np_origin = np.zeros([len(temperature_match['train']), 2, duration.shape[0], num_measure_point + 3])
    print(f'测试集和训练集分别有{data_np_origin.shape[0]}个电芯，正在提取各电芯数据...')
    for group in range(data_np_origin.shape[0]):
        # 提取train和test对应的电芯id
        cell_train = temperature_match['train'][group]
        cell_test = temperature_match['test'][group]

        # 提取训练集
        data_np_origin[group, 0, :, 0:3] = np.asarray(data_xlsx[['时间[s]', '电流', 'SOC']])[0:duration.shape[0]]
        data_np_origin[group, 0, :, 3:] = np.asarray(data_xlsx[temperature_measure_points[cell_train]])[0:duration.shape[0]] + K

        # 提取测试集
        data_np_origin[group, 1, :, 0:3] = np.asarray(data_xlsx[['时间[s]', '电流', 'SOC']])[0:duration.shape[0]]
        data_np_origin[group, 1, :, 3:] = np.asarray(data_xlsx[temperature_measure_points[cell_test]])[0:duration.shape[0]] + K

    # 从基础存储单元中分割出数据集
    print(f'每个电芯分别可分割出{batch_all}组数据，正在分割...')
    data_np_train = np.zeros([len(temperature_match['train']), batch_all, sequence_init + sequence_predict, data_np_origin.shape[-1]])
    data_np_test = np.zeros([len(temperature_match['test']), batch_all, sequence_init + sequence_predict, data_np_origin.shape[-1]])
    slice_origin(data_np_origin[:, 0], data_np_train, batch_all, sequence_init + sequence_predict)
    slice_origin(data_np_origin[:, 0], data_np_test, batch_all, sequence_init + sequence_predict)

    # 将数据集重新排列，移到device内，并保存于dict中
    print(f'进行数据重排列，并将数据移至{device}，正在处理...')
    data_pkl = dict()
    data_pkl_train, data_pkl_test = dict(), dict()
    for i in range(data_np_train.shape[0]):
        data_pkl_temp = list()
        for j in range(data_np_train.shape[1]):
            data_pkl_temp.append(torch.from_numpy(data_np_train[i, j].transpose()).to(torch.float32).to(device))
        data_pkl_train[temperature_match['train'][i]] = data_pkl_temp
    for i in range(data_np_test.shape[0]):
        data_pkl_temp = list()
        for j in range(data_np_test.shape[1]):
            data_pkl_temp.append(torch.from_numpy(data_np_test[i, j].transpose()).to(torch.float32).to(device))
        data_pkl_test[temperature_match['test'][i]] = data_pkl_temp
    data_pkl['train'] = data_pkl_train
    data_pkl['test'] = data_pkl_test

    # 保存数据
    with open(path_dataset_pkl, 'wb') as pkl:
        pickle.dump(data_pkl, pkl)
        pkl.close()
    print(f"pkl文件已保存至【{path_dataset_pkl}】")
