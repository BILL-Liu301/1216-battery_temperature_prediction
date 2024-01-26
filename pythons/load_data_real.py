import pickle
import numpy as np
import pandas as pd

from api.base.paths import path_data_t9m, path_data_origin_pkl_real
from api.base.ids import temperature_measure_points, temperature_location
from api.base.paras import K, num_measure_point


if __name__ == '__main__':
    # 从xlsx中提取数据
    print(f'从【{path_data_t9m}】提取数据，正在提取...')
    data_xlsx = pd.read_excel(path_data_t9m, sheet_name='常温充电')
    stamp = np.asarray(data_xlsx['时间[s]'])[np.isfinite(np.asarray(data_xlsx['时间[s]']))].reshape(-1, 1)

    # 基础存储单元
    # 时间[s]，位置，NTC最高温，NTC最低温，电压-总体，电流，SOC，最高温，最低温
    dataset_base = {
        'stamp': None,
        'location': None,
        'NTC_max': None,
        'NTC_min': None,
        'Voltage': None,
        'Current': None,
        'SOC': None,
        'Temperature_max': None,
        'Temperature_min': None
    }
    dataset = {
        'module-0': list(),
        'module-1': list()
    }

    for cell_name, cell_location in temperature_location.items():
        dataset_base['stamp'] = stamp
        dataset_base['location'] = cell_location * np.ones(stamp.shape)
        dataset_base['NTC_max'] = np.asarray(data_xlsx['最高温-BMS-NTC'])[0:stamp.shape[0]].reshape(-1, 1) + K
        dataset_base['NTC_min'] = np.asarray(data_xlsx['最低温-BMS-NTC'])[0:stamp.shape[0]].reshape(-1, 1) + K
        dataset_base['Voltage'] = np.asarray(data_xlsx['平均电压'])[0:stamp.shape[0]].reshape(-1, 1) * 100
        dataset_base['Current'] = np.asarray(data_xlsx['电流'])[0:stamp.shape[0]].reshape(-1, 1)
        dataset_base['SOC'] = np.asarray(data_xlsx['SOC'])[0:stamp.shape[0]].reshape(-1, 1)
        dataset_base['Temperature_max'] = np.max(np.asarray(data_xlsx[temperature_measure_points[cell_name]])[0:stamp.shape[0]] + K, axis=1, keepdims=True).reshape(-1, 1)
        dataset_base['Temperature_min'] = np.min(np.asarray(data_xlsx[temperature_measure_points[cell_name]])[0:stamp.shape[0]] + K, axis=1, keepdims=True).reshape(-1, 1)
        if cell_name.split('-')[0] == 'M1':
            dataset[f'module-0'].append(dataset_base.copy())
        elif cell_name.split('-')[0] == 'M2':
            dataset[f'module-1'].append(dataset_base.copy())
        else:
            module = -1
    # 保存数据
    with open(path_data_origin_pkl_real, 'wb') as pkl:
        pickle.dump(dataset, pkl)
        pkl.close()
    print(f"pkl文件已保存至【{path_data_origin_pkl_real}】")
