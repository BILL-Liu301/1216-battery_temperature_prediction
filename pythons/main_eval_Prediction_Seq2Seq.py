import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import warnings
from tqdm import tqdm

from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from torch.utils.data import DataLoader, random_split

from api.base.paras import paras_Prediction_Seq2Seq, device, num_measure_point, paras_Prediction_Seq2Seq_dataset
from api.datasets.prediction_seq2seq import Prediction_Seq2Seq_Dataset
from api.models.prediction_seq2seq import Prediction_Seq2seq_LightningModule
from api.base.paths import path_data_origin_pkl, path_ckpts, path_figs_test, path_figs_train, path_figs_val
from api.util.plots import plot_for_prediction_seq2seq_val_test

if __name__ == '__main__':
    pl.seed_everything(2024)
    plt.figure(figsize=(20, 11.25))

    # 为了好看，屏蔽warning，没事别解注释这个
    warnings.filterwarnings('ignore', category=PossibleUserWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # 找到ckpt
    path_version = path_ckpts + 'lightning_logs/version_0/checkpoints/'
    ckpt = path_version + os.listdir(path_version)[0]

    # 设置训练器
    trainer = pl.Trainer(accelerator='gpu', devices=1)
    model = Prediction_Seq2seq_LightningModule.load_from_checkpoint(checkpoint_path=ckpt)
    trainer.test(model=model, dataloaders=paras_Prediction_Seq2Seq_dataset['dataset_loader_test'])
    for i in tqdm(range(0, len(model.test_results), 1), desc='Test', leave=False, ncols=100, disable=False):
        test_results = model.test_results[i:i + 1]
        plot_for_prediction_seq2seq_val_test(test_results, num_measure_point)
        plt.savefig(path_figs_test + f'{i}.png')
