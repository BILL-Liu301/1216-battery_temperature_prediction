import pickle
import numpy as np
import pandas as pd

from api.base.paths import path_data_t9m, path_data_origin_pkl_single
from api.base.ids import temperature_measure_points, temperature_location
from api.base.paras import K, num_measure_point


if __name__ == '__main__':
    # 从xlsx中提取数据
    print(f'从【{path_data_t9m}】提取数据，正在提取...')
    data_xlsx = pd.read_excel(path_data_t9m, sheet_name='常温充电')
    duration = np.asarray(data_xlsx['时间[s]'])[np.isfinite(np.asarray(data_xlsx['时间[s]']))]

    # 基础存储单元
    # 时间[s]，位置，NTC最高温，NTC最低温，电压，电流，SOC，测点1，测点2，测点3，测点4，测点5，测点6，测点7
    #         上
    #       1    2
    # 内  3    4   5  外
    #       6    7
    #         下

    data_np_origin = np.zeros([duration.shape[0], 7 + num_measure_point])
    data_dict_origin = dict()
    for cell_name, cell_location in temperature_location.items():
        data_np_origin[:, 0] = duration
        data_np_origin[:, 1] = cell_location
        data_np_origin[:, 2:7] = np.asarray(data_xlsx[['最高温-BMS-NTC', '最低温-BMS-NTC', '平均电压', '电流', 'SOC']])[0:duration.shape[0]]
        data_np_origin[:, 2:4] = data_np_origin[:, 2:4] + K
        data_np_origin[:, 7:] = np.asarray(data_xlsx[temperature_measure_points[cell_name]])[0:duration.shape[0]] + K
        data_dict_origin.update({cell_name: data_np_origin.copy().transpose()})
    # 保存数据
    with open(path_data_origin_pkl_single, 'wb') as pkl:
        pickle.dump(data_dict_origin, pkl)
        pkl.close()
    print(f"pkl文件已保存至【{path_data_origin_pkl_single}】")
