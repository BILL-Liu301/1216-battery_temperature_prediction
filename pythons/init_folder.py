import os

from api.base.paths import path_output


def mkdir(path):
    if not os.path.exists(path):
        print(f'已创建【{path}】')
        os.mkdir(path)
    else:
        print(f'【{path}】已存在')


if __name__ == '__main__':
    mkdir(path_output)
