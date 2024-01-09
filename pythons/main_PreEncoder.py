import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from api.models.pre_encoder import PreEncoder_LightningModule
from api.datasets.pre_encoder import PreEncoder_Dataset
from api.base.paras import device, paras_PreEncoder
from api.base.paths import path_data_origin_pkl, path_pts

if __name__ == '__main__':
    pre_encoder_lm = PreEncoder_LightningModule(paras=paras_PreEncoder)
    dataset_train = PreEncoder_Dataset(path_data_origin_pkl=path_data_origin_pkl,
                                       num_pre_encoder=paras_PreEncoder['num_pre_encoder'],
                                       device=device)
    dataset_val = PreEncoder_Dataset(path_data_origin_pkl=path_data_origin_pkl,
                                     num_pre_encoder=paras_PreEncoder['num_pre_encoder'],
                                     device=device)

    dataset_loader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, pin_memory=True, num_workers=1)
    dataset_loader_val = DataLoader(dataset_val, batch_size=2, pin_memory=True, num_workers=1)
    trainer = pl.Trainer(log_every_n_steps=1, max_epochs=paras_PreEncoder['max_epochs'], default_root_dir=path_pts, accelerator='gpu', devices=1)
    trainer.fit(model=pre_encoder_lm, train_dataloaders=dataset_loader_train, val_dataloaders=dataset_loader_val)
