import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import warnings
from tqdm import tqdm

from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from torch.utils.data import DataLoader, random_split

from api.base.paras import paras_Prediction_Seq2Seq, device, num_measure_point
from api.datasets.prediction_seq2seq import Prediction_Seq2Seq_Dataset
from api.models.prediction_seq2seq import Prediction_Seq2seq_LightningModule
from api.base.paths import path_data_origin_pkl, path_ckpts, path_figs_test, path_figs_train
from api.util.plots import plot_for_prediction_seq2seq_val_test

if __name__ == '__main__':
    pl.seed_everything(2024)
    plt.figure(figsize=(20, 11.25))

    # 为了好看，屏蔽warning，没事别解注释这个
    warnings.filterwarnings('ignore', category=PossibleUserWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # 模式选择
    mode_switch = {
        'Train_Validation_Test': False,
        'Sample': True
    }

    # 加载数据集
    dataset_base = Prediction_Seq2Seq_Dataset(path_data_origin_pkl=path_data_origin_pkl,
                                              paras=paras_Prediction_Seq2Seq,
                                              device=device)

    # 分割train, test, val，并进行数据加载
    train_valid_set_size = int(len(dataset_base) * 0.9)
    test_set_size = len(dataset_base) - train_valid_set_size
    train_valid_set, test_set = random_split(dataset_base, [train_valid_set_size, test_set_size])

    train_set_size = int(len(train_valid_set) * 0.9)
    valid_set_size = len(train_valid_set) - train_set_size
    train_set, valid_set = random_split(train_valid_set, [train_set_size, valid_set_size])

    dataset_loader_train = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=True, num_workers=2)
    dataset_loader_val = DataLoader(valid_set, batch_size=8, pin_memory=True, num_workers=2)
    dataset_loader_test = DataLoader(test_set, batch_size=1, pin_memory=True, num_workers=1)

    # 加载pytorch_lighting模型
    prediction_seq2seq = Prediction_Seq2seq_LightningModule(paras=paras_Prediction_Seq2Seq)

    # 设置训练器
    early_stop_callback = EarlyStopping(monitor='loss_val_null', min_delta=0.01, patience=5, verbose=False, mode="min")
    model_checkpoint = ModelCheckpoint(monitor='loss_val_null', save_top_k=1, mode='min')
    model_summery = ModelSummary(max_depth=-1)
    trainer = pl.Trainer(log_every_n_steps=1, max_epochs=paras_Prediction_Seq2Seq['max_epochs'],
                         default_root_dir=path_ckpts, accelerator='gpu', devices=1, callbacks=[early_stop_callback, model_checkpoint, model_summery])
    if mode_switch['Train_Validation_Test']:
        trainer.fit(model=prediction_seq2seq, train_dataloaders=dataset_loader_train, val_dataloaders=dataset_loader_val)
        trainer.test(model=prediction_seq2seq, dataloaders=dataset_loader_test)
        for i in tqdm(range(0, len(prediction_seq2seq.test_results), 1), desc='Test', leave=False, ncols=100, disable=False):
            test_results = prediction_seq2seq.test_results[i:i + 1]
            plot_for_prediction_seq2seq_val_test(test_results, num_measure_point)
            plt.savefig(path_figs_test + f'{i}.png')
    if mode_switch['Sample']:
        path_version = path_ckpts + 'lightning_logs/version_0/checkpoints/'
        ckpt = path_version + os.listdir(path_version)[0]
        model = prediction_seq2seq.load_from_checkpoint(checkpoint_path=ckpt, paras=paras_Prediction_Seq2Seq)
        model.eval()

        _results = list()
        for batches in dataset_loader_test:
            for batch in batches:
                inp_loc_i_soc_his = batch[1:4, 0:paras_Prediction_Seq2Seq['seq_history']]
                inp_temperature_his = batch[4:5, 0:paras_Prediction_Seq2Seq['seq_history']]
                inp_loc_i_soc = batch[1:4, paras_Prediction_Seq2Seq['seq_history']:]
                with torch.no_grad():
                    pre_mean, pre_var, _ = model(inp_loc_i_soc_his.T, inp_temperature_his.T, inp_loc_i_soc.T)
                ref_mean = batch[4:5, paras_Prediction_Seq2Seq['seq_history']:].T
                _results.append({
                    'pre': torch.cat([pre_mean, pre_var], dim=1).T.cpu().numpy(),
                    'ref': ref_mean.T.cpu().numpy(),
                    'origin': batch[5:, paras_Prediction_Seq2Seq['seq_history']:].cpu().numpy()
                })
        for i in tqdm(range(0, len(_results), 1), desc='Sample', leave=False, ncols=100, disable=False):
            _result = _results[i:i + 1]
            plot_for_prediction_seq2seq_val_test(_result, num_measure_point)
            plt.savefig(path_figs_test + f'{i}.png')
            print()
