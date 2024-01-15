import os
import shutil

from api.base.paths import path_base, path_dataset, path_ckpts
from api.base.paths import path_figs, path_figs_train, path_figs_test, path_figs_val

print(f"警告！警告！即将删除“{path_base}”！！！")
print(f"请进行确认！！！")
flag = input("请进行确认（True/确认删除，False/不删除）：")

if flag:
    print("正在删除...")
    shutil.rmtree(path_base)

    os.mkdir(path_base)

    os.mkdir(path_dataset)
    os.mkdir(path_ckpts)

    os.mkdir(path_figs)
    os.mkdir(path_figs_train)
    os.mkdir(path_figs_test)
    os.mkdir(path_figs_val)
else:
    print("已停止删除程序...")
