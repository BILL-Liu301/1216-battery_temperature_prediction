import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import warnings
from tqdm import tqdm

from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, GradientAccumulationScheduler, Timer
from torch.utils.data import DataLoader, random_split

from api.base.paras import paras_Prediction_Seq2Seq, paras_Prediction_Seq2Seq_dataset, num_measure_point
from api.datasets.prediction_seq2seq import Prediction_Seq2Seq_Dataset
from api.models.prediction_seq2seq import Prediction_Seq2seq_LightningModule
from api.base.paths import path_data_origin_pkl, path_ckpts, path_figs_test, path_figs_train, path_figs_val
from api.util.plots import plot_for_prediction_seq2seq_val_test

if __name__ == '__main__':
    pl.seed_everything(2024)
    plt.figure(figsize=(20, 11.25))

    # # 为了好看，屏蔽warning，没事别解注释这个
    # warnings.filterwarnings('ignore', category=PossibleUserWarning)
    # warnings.filterwarnings('ignore', category=UserWarning)

    # 加载pytorch_lighting模型
    prediction_seq2seq = Prediction_Seq2seq_LightningModule(paras=paras_Prediction_Seq2Seq)

    # 设置训练器
    early_stop_callback = EarlyStopping(monitor='loss_val_nll', min_delta=0.01, patience=5, verbose=False, mode="min")
    model_checkpoint = ModelCheckpoint(monitor='loss_val_nll', save_top_k=1, mode='min')
    model_summery = ModelSummary(max_depth=-1)
    gradient_accumulation_scheduler = GradientAccumulationScheduler({4: 2, 9: 3})
    timer = Timer(duration='00:00:10:00', verbose=True)
    trainer = pl.Trainer(log_every_n_steps=1, max_epochs=paras_Prediction_Seq2Seq['max_epochs'],
                         default_root_dir=path_ckpts, accelerator='gpu', devices=1, callbacks=[early_stop_callback, model_checkpoint, model_summery, timer])
    trainer.fit(model=prediction_seq2seq, train_dataloaders=paras_Prediction_Seq2Seq_dataset['dataset_loader_train'], val_dataloaders=paras_Prediction_Seq2Seq_dataset['dataset_loader_val'])
