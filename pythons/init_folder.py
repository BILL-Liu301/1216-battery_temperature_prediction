import os
import shutil

from api.base.paths import path_base, path_dataset, path_pts, path_figs


def mkdir(path):
    if not os.path.exists(path):
        print(f'已创建【{path}】')
        os.mkdir(path)
    else:
        print(f'【{path}】已存在')


if __name__ == '__main__':
    if os.path.exists(path_base):
        print(f'已删除【{path_base}】')
        shutil.rmtree(path_base)
    mkdir(path_base)
    mkdir(path_dataset)
    mkdir(path_pts)
    mkdir(path_figs)
