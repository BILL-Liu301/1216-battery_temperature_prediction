import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import warnings
from tqdm import tqdm

from lightning_fabric.utilities.warnings import PossibleUserWarning

from api.base.paras import paras_Prediction_Temperature
from api.datasets.prediction_temperature import paras_Prediction_Temperature_dataset
from api.models.prediction_temperature import Prediction_Temperature_LightningModule
from api.base.paths import path_ckpt_best_version, path_ckpts, path_figs_test, path_result_store_temperature
from api.util.plots import plot_for_prediction_temperature_val_test

if __name__ == '__main__':
    pl.seed_everything(2024)
    fig = plt.figure(figsize=(20, 11.25))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置显示中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号

    # 为了好看，屏蔽warning，没事别解注释这个
    warnings.filterwarnings('ignore', category=PossibleUserWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # 找到ckpt
    # path_version = path_ckpts + 'lightning_logs/version_0/checkpoints/'
    path_version = path_ckpt_best_version + 'temperature/version_0/checkpoints/'
    ckpt = path_version + os.listdir(path_version)[0]

    # 设置训练器
    trainer = pl.Trainer(default_root_dir=path_ckpts, accelerator='gpu', devices=1)
    model = Prediction_Temperature_LightningModule.load_from_checkpoint(checkpoint_path=ckpt)
    dataloaders = paras_Prediction_Temperature_dataset['dataset_loader_test']
    trainer.test(model=model, dataloaders=dataloaders)

    # 结果展示
    loss_mean, loss_max, loss_min = torch.cat(model.test_losses['mean'], dim=0), torch.cat(model.test_losses['max'], dim=0), torch.cat(model.test_losses['min'], dim=0)
    print(f'总计有{len(dataloaders.dataset)}组数据')
    print(f'平均均值误差：{loss_mean.mean()}(℃)')
    print(f'平均最大误差：{loss_max.mean()}(℃)')
    print(f'平均最小误差：{loss_min.mean()}(℃)')
    for i in tqdm(range(0, len(model.test_results), 1), desc='Test', leave=False, ncols=100, disable=False):
        test_results = model.test_results[i]
        # plot_for_prediction_temperature_val_test(fig, i, test_results, paras_Prediction_Temperature)  # 用于日常debug
        # plt.savefig(path_figs_test + f'{i}.png')  # 用于日常debug

        plot_for_prediction_temperature_val_test(fig, f'{dataloaders.dataset.data_condition[i]}_{i % 25}', test_results, paras_Prediction_Temperature)  # 用于保存最终结果
        plt.savefig(path_result_store_temperature + f'{i}.png')  # 用于保存最终结果
