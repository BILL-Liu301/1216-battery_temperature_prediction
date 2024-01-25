import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from api.base.paths import path_data_t9m_sim, path_data_origin_pkl
from api.base.ids import temperature_measure_points, temperature_location
from api.base.paras import K, num_measure_point


def interpolation_duration_to_stamp(stamp, duration, x):
    y = np.zeros([stamp.shape[0], x.shape[1]])
    # 通过时间相距最近的20个点进行拟合
    for t in range(y.shape[0]):
        ids = np.abs(stamp[t, 0] - duration[:, 0]).argsort(axis=0)
        ids = np.sort(ids[0:20], axis=0)
        fun_x = duration[ids, 0]
        for num in range(y.shape[1]):
            fun_y = x[ids, num]
            func = np.poly1d(np.polyfit(fun_x, fun_y, 4))
            y[t, num] = func(stamp[t, 0])
    return y


if __name__ == '__main__':
    # 从xlsx中提取数据
    print(f'从【{path_data_t9m_sim}】提取数据，正在提取...')
    data_xlsx_state = pd.read_excel(path_data_t9m_sim, sheet_name='常温充-电流电压SOC')
    data_xlsx_ntc = pd.read_excel(path_data_t9m_sim, sheet_name='常温充-NTC')
    data_xlsx_temperature = pd.read_excel(path_data_t9m_sim, sheet_name='常温充-电芯大面最高、低温')

    # 基础存储单元
    # 时间[s]，位置，NTC最高温，NTC最低温，电压-总体，电流，SOC，测点1，测点2，测点3，测点4，测点5，测点6，测点7
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

    # 数据中只有两个模组，每个模组内100个电芯，每四个电芯为1组，所以每个模组内25组
    for module in tqdm(range(2), desc='Module', leave=False, ncols=100, disable=False):
        # 提取基本的时间戳、电压、电流和soc
        stamp, voltage, current, soc = (np.asarray(data_xlsx_state['时间']).reshape(-1, 1), np.asarray(data_xlsx_state['电压-总']).reshape(-1, 1),
                                        np.asarray(data_xlsx_state['电流']).reshape(-1, 1), np.asarray(data_xlsx_state['SOC']).reshape(-1, 1))

        # 提取所有组共享的时间、NTC最高温和NTC最低温，每个模组对应四个ntc
        num_NTC = 4
        duration = np.asarray(data_xlsx_temperature['物理时间 [s]']).reshape(-1, 1)
        NTC = np.zeros([duration.shape[0], num_NTC])
        for ntc in range(num_NTC):
            NTC[:, ntc] = np.asarray(data_xlsx_ntc[f'PG 温度（固体） {ntc + module * num_NTC + 1}'])
        NTC = interpolation_duration_to_stamp(stamp, duration, NTC)

        dataset_base['stamp'], dataset_base['NTC_max'], dataset_base['NTC_min'], dataset_base['Voltage'], dataset_base['Current'], dataset_base['SOC'] \
            = stamp, np.max(NTC, axis=1, keepdims=True), np.min(NTC, axis=1, keepdims=True), voltage, current, soc

        # 提取每组电芯的温度，start_id = [最低温起始id, 最高温起始id]
        num_cell = 4
        num_group = round(100 / num_cell)
        temperature_max, temperature_min = np.zeros([duration.shape[0], num_cell]), np.zeros([duration.shape[0], num_cell])
        if module == 0:
            start_id = [209, 9]
            direction = 1
        elif module == 1:
            start_id = [408, 208]
            direction = -1
        else:
            start_id = [0, 0]
            direction = 0
        for group in tqdm(range(num_group), desc='Group', leave=False, ncols=100, disable=False):
            location = np.ones([stamp.shape[0], 1]) * (num_cell * group) / (num_cell * (num_group - 1))
            for cell in range(num_cell):
                temperature_min[:, cell] = np.asarray(data_xlsx_temperature[f'SG 最小值 温度（固体） {start_id[0] + direction * (cell + num_cell * group)}'])
                temperature_max[:, cell] = np.asarray(data_xlsx_temperature[f'SG 最大值 温度（固体） {start_id[1] + direction * (cell + num_cell * group)}'])
            dataset_base['location'] = location
            dataset_base['Temperature_min'] = interpolation_duration_to_stamp(stamp, duration, temperature_min)
            dataset_base['Temperature_max'] = interpolation_duration_to_stamp(stamp, duration, temperature_max)

            dataset[f'module-{module}'].append(dataset_base.copy())

    with open(path_data_origin_pkl, 'wb') as pkl:
        pickle.dump(dataset, pkl)
        pkl.close()
    print(f"pkl文件已保存至【{path_data_origin_pkl}】")
