import matplotlib.pyplot as plt

from api.system.compare import Compare_System
from api.base.path import path_map_tabel, path_map_tabel_new, path_ckpt_best_version

if __name__ == '__main__':
    # 创建可视化窗口
    fig = plt.figure(figsize=(20, 11.25))

    # 加载对比系统，并初始化工况
    compare_system = Compare_System(path_map_tabel, path_map_tabel_new, path_ckpt_best_version)
    compare_system.init_condition()

    # 循环充电
    compare_system.charging_both(soc_end=90)

    # 充电结果动态表示
    compare_system.plot_both(plot_speed=20)

    # 最终展示
    plt.show()
