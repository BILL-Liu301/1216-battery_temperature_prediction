import os
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import copy

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split

from api.models.pre_encoder import PreEncoder_LightningModule
from api.datasets.pre_encoder import PreEncoder_Dataset_Expand, PreEncoder_Dataset_Origin
from api.base.paras import device, paras_PreEncoder, num_measure_point
from api.base.paths import path_data_origin_pkl, path_ckpts, path_figs_test, path_figs_val
from api.util.plots import plot_for_pre_encoder_val_test

if __name__ == '__main__':
    pl.seed_everything(2024)
    plt.figure(figsize=(20, 11.25))

    # 模式选择
    mode_switch = {
        'Train_Validation_Test': True,
        'Sample': False
    }

    # 加载pytorch_lighting模型
    pre_encoder_lm = PreEncoder_LightningModule(paras=paras_PreEncoder)

    # 加载expand和origin数据
    dataset_expand = PreEncoder_Dataset_Expand(path_data_origin_pkl=path_data_origin_pkl,
                                               num_pre_encoder=paras_PreEncoder['num_pre_encoder'],
                                               device=device)
    dataset_origin = PreEncoder_Dataset_Origin(path_data_origin_pkl=path_data_origin_pkl,
                                               num_pre_encoder=paras_PreEncoder['num_pre_encoder'],
                                               device=device)

    # 分割train, test, val
    train_set_size = int(len(dataset_expand) * 0.9)
    valid_set_size = len(dataset_expand) - train_set_size
    train_set, valid_set = random_split(dataset_expand, [train_set_size, valid_set_size])
    test_set = dataset_origin

    dataset_loader_train = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True, num_workers=2)
    dataset_loader_val = DataLoader(valid_set, batch_size=4, pin_memory=True, num_workers=2)
    dataset_loader_test = DataLoader(test_set, batch_size=1, pin_memory=True, num_workers=1)

    # 设置训练器
    early_stop_callback = EarlyStopping(monitor='loss_val', min_delta=0.01, patience=5, verbose=False, mode="min")
    model_checkpoint = ModelCheckpoint(monitor='loss_val', save_top_k=1, mode='min')
    trainer = pl.Trainer(log_every_n_steps=1, max_epochs=paras_PreEncoder['max_epochs'],
                         default_root_dir=path_ckpts, accelerator='gpu', devices=1, callbacks=[early_stop_callback, model_checkpoint])
    if mode_switch['Train_Validation_Test']:
        trainer.fit(model=pre_encoder_lm, train_dataloaders=dataset_loader_train, val_dataloaders=dataset_loader_val)
        trainer.test(model=pre_encoder_lm, dataloaders=dataset_loader_test)
        for i in range(0, len(pre_encoder_lm.test_results), 1):
            test_results = pre_encoder_lm.test_results[i:i + 1]
            plot_for_pre_encoder_val_test(test_results, num_measure_point)
            plt.savefig(path_figs_test + f'{i}.png')
            print(f'已保存{i}.png')
    if mode_switch['Sample']:
        path_version = path_ckpts + 'lightning_logs/version_3/checkpoints/'
        ckpt = path_version + os.listdir(path_version)[0]
        model = pre_encoder_lm.load_from_checkpoint(checkpoint_path=ckpt, paras=paras_PreEncoder)
        model.eval()

        _results = list()
        for batch in dataset_loader_train:
            data = batch[0]
            inp_loc_i_soc = data[0:4, 1:]
            inp_temperature = data[4:, 0:1]
            with torch.no_grad():
                pre_temperature = torch.cat([data[0:4],
                                             torch.cat([inp_temperature, model(inp_loc_i_soc, inp_temperature)], dim=1)],
                                            dim=0)
            ref_temperature = data
            _results.append({
                'pre': pre_temperature.cpu().numpy(),
                'ref': ref_temperature.cpu().numpy()
            })
        for i in range(0, len(_results), 1):
            _result = _results[i:i + 1]
            plot_for_pre_encoder_val_test(_result, num_measure_point)
            plt.savefig(path_figs_test + f'{i}.png')
            print(f'已保存{i}.png')
