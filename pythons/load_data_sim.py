import pickle
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from api.base.paths import path_data_t9m_sim, path_data_origin_pkl_sim
from api.base.paras import K


def interpolation_duration_to_stamp(stamp, duration, x, desc=''):
    y = np.zeros([stamp.shape[0], x.shape[1]])
    # 通过时间相距最近的20个点进行拟合
    for t in tqdm(range(y.shape[0]), desc=f'插值 {desc} ', leave=False, ncols=100, disable=False):
        ids = np.abs(stamp[t, 0] - duration[:, 0]).argsort(axis=0)
        ids = np.sort(ids[0:20], axis=0)
        fun_x = duration[ids, 0]
        for num in range(y.shape[1]):
            fun_y = x[ids, num]
            func = np.poly1d(np.polyfit(fun_x, fun_y, 4))
            y[t, num] = func(stamp[t, 0])
    return y


if __name__ == '__main__':
    # 基础存储单元
    # 时间[s]，位置，NTC，电压-总体，电流，SOC，最高温，最低温
    dataset_base = {
        'stamp': None,
        'location': None,
        'NTC': None,
        'Voltage': None,
        'Current': None,
        'SOC': None,
        'Temperature_max': None,
        'Temperature_min': None
    }
    dataset = dict()

    # 从data文件夹中提取数据文件
    battery_files = os.listdir(path_data_t9m_sim)
    print(f'从【{path_data_t9m_sim}】提取数据，'
          f'\n其包含：')
    for battery_file in battery_files:
        print(f'[{battery_file}, {os.path.getsize(path_data_t9m_sim + battery_file) / 1024} KB]')

    # 遍历工况
    tbar_file = tqdm(battery_files, leave=False, ncols=100, disable=False)
    for battery_file in tbar_file:
        if battery_file[0] == '~':
            continue
        path_battery_file = path_data_t9m_sim + battery_file

        # 在dataset中加载工况
        charging_condition = battery_file.split('_')[1]
        tbar_file.set_description(charging_condition)
        if charging_condition == '低温充电':
            continue
        dataset.update({charging_condition: dict()})

        # 从工况内提取数据
        data_xlsx_state = pd.read_excel(path_battery_file, sheet_name='电压电流SOC')
        data_xlsx_ntc = pd.read_excel(path_battery_file, sheet_name='NTC')
        data_xlsx_temperature_max = pd.read_excel(path_battery_file, sheet_name='电芯大面最高温')
        data_xlsx_temperature_min = pd.read_excel(path_battery_file, sheet_name='电芯大面最低温')

        # 遍历模组
        num_module, num_cell_in_group, num_group = 2, 4, 25  # 100个电芯，4个电芯为一组，总共25组
        tbar_module = tqdm(range(num_module), leave=False, ncols=100, disable=False)
        for module in tbar_module:
            # 在工况中加载模组
            dataset[charging_condition].update({f'module-{module}': dict()})
            tbar_module.set_description(f'模组 {module} ')

            # 提取基础数据
            stamp = np.asarray(data_xlsx_state['时间']).reshape(-1, 1)
            duration = np.asarray(data_xlsx_ntc['物理时间 [s]']).reshape(-1, 1)
            current = np.asarray(data_xlsx_state['电流']).reshape(-1, 1)
            voltage = np.asarray(data_xlsx_state['电压']).reshape(-1, 1)
            soc = np.asarray(data_xlsx_state['SOC']).reshape(-1, 1)
            if module == 0:
                ntc = np.asarray(data_xlsx_ntc[data_xlsx_ntc.columns[1::3][0:4]])
            else:
                ntc = np.asarray(data_xlsx_ntc[data_xlsx_ntc.columns[1::3][4:]])
            ntc = interpolation_duration_to_stamp(stamp, duration, ntc, desc='ntc')

            # 遍历电芯组
            if module == 0:
                tbar_group = tqdm(range(0, num_group, 1), desc='Group', leave=False, ncols=100, disable=False)
            else:
                tbar_group = tqdm(range(num_group - 1, -1, -1), desc='Group', leave=False, ncols=100, disable=False)
            for group in tbar_group:
                location = np.ones([stamp.shape[0], 1]) * group / (num_group - 1)
                cell_start, cell_end = group * num_cell_in_group + module * 100, (group + 1) * num_cell_in_group - 1 + module * 100
                tbar_group.set_description(f'电芯组 {cell_start}~{cell_end} ')

                # 分组提取数据
                temperature_max_origin = np.asarray(data_xlsx_temperature_max[data_xlsx_temperature_max.columns[1::3][cell_start:(cell_end + 1)]])
                temperature_min_origin = np.asarray(data_xlsx_temperature_min[data_xlsx_temperature_min.columns[1::3][cell_start:(cell_end + 1)]])

                # 根据时间戳进行转换
                temperature_max = interpolation_duration_to_stamp(stamp, duration, temperature_max_origin, desc='temperature_max')
                temperature_min = interpolation_duration_to_stamp(stamp, duration, temperature_min_origin, desc='temperature_min')

                # 赋值给dataset_base
                dataset_base['stamp'] = stamp.copy()
                dataset_base['location'] = location.copy()
                dataset_base['NTC'] = ntc.copy()
                dataset_base['Voltage'] = voltage.copy()
                dataset_base['Current'] = current.copy()
                dataset_base['SOC'] = soc.copy()
                dataset_base['Temperature_max'] = temperature_max
                dataset_base['Temperature_min'] = temperature_min

                # 赋值给dataset中对应的charging_condition
                dataset[charging_condition][f'module-{module}'].update({f'{cell_start}~{cell_end}': dataset_base.copy()})

    with open(path_data_origin_pkl_sim, 'wb') as pkl:
        pickle.dump(dataset, pkl)
        pkl.close()
    print(f"pkl文件已保存至【{path_data_origin_pkl_sim}】，{os.path.getsize(path_data_origin_pkl_sim) / 1024} KB。")
