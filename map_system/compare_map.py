import matplotlib.pyplot as plt

from api.system.compare import Compare_System
from api.base.path import path_map_tabel, path_map_tabel_new, path_ckpt_best_version
from charging_system.pythons.api.base.paths import path_figs_test

if __name__ == '__main__':
    # 创建可视化窗口
    fig = plt.figure()

    # 加载对比系统，并初始化工况
    compare_system = Compare_System(path_map_tabel, path_map_tabel_new, path_ckpt_best_version)
    compare_system.init_condition()

    # 循环充电
    compare_system.charging_both(soc_end=90)

    print('开始展示')
    # 充电结果动态表示
    compare_system.plot_both(plot_speed=20)
    plt.pause(1)

    # # 基于原始的MAP出的策略，重新预测，并绘图查看差距
    # condition_record, condition_record_re = compare_system.re_charging(compare_system.condition_record_origin, compare_system.tabel_origin)
    # compare_system.plot_both(plot_speed=50, condition_record_origin=condition_record, condition_record_new=condition_record_re, name_origin='condition_record', name_new='condition_record_re')
    # plt.pause(1)
    #
    # # 基于优化后的MAP出的策略，重新预测，并绘图查看差距
    # condition_record, condition_record_re = compare_system.re_charging(compare_system.condition_record_new, compare_system.tabel_new)
    # compare_system.plot_both(plot_speed=50, condition_record_origin=condition_record, condition_record_new=condition_record_re, name_origin='condition_record', name_new='condition_record_re')
    # plt.pause(1)

    # 最终展示
    plt.savefig(path_figs_test + 'compare.jpg')
    plt.show()
