import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from api.models.pre_encoder import PreEncoder_LightningModule
from api.datasets.pre_encoder import PreEncoder_Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from api.base.paras import device, paras_PreEncoder
from api.base.paths import path_data_origin_pkl, path_pts

if __name__ == '__main__':
    # 模式选择
    mode_switch = {
        'Train_Validation': False,
        'Test': True
    }

    pre_encoder_lm = PreEncoder_LightningModule(paras=paras_PreEncoder)
    dataset = PreEncoder_Dataset(path_data_origin_pkl=path_data_origin_pkl,
                                 num_pre_encoder=paras_PreEncoder['num_pre_encoder'],
                                 device=device)
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    seed = torch.Generator().manual_seed(2024)
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=seed)
    test_set = dataset

    dataset_loader_train = DataLoader(train_set, batch_size=2, shuffle=True, pin_memory=True, num_workers=1)
    dataset_loader_val = DataLoader(valid_set, batch_size=2, pin_memory=True, num_workers=1)
    dataset_loader_test = DataLoader(test_set, batch_size=2, pin_memory=True, num_workers=1)


    early_stop_callback = EarlyStopping(monitor="loss_max_val", min_delta=0.001, patience=5, verbose=False, mode="min")

    trainer = pl.Trainer(log_every_n_steps=1, max_epochs=paras_PreEncoder['max_epochs'], default_root_dir=path_pts,
                         accelerator='gpu', devices=1, callbacks=[early_stop_callback])
    trainer.fit(model=pre_encoder_lm, train_dataloaders=dataset_loader_train, val_dataloaders=dataset_loader_val)
