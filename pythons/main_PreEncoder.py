import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from api.models.pre_encoder import PreEncoder_LightningModule
from api.datasets.pre_encoder import PreEncoder_Dataset
from api.base.paras import num_measure_point, device, paras_PreEncoder
from api.base.paths import path_data_origin_pkl, path_pts

if __name__ == '__main__':
    pre_encoder_lm = PreEncoder_LightningModule(paras=paras_PreEncoder)
    dataset = PreEncoder_Dataset(path_data_origin_pkl=path_data_origin_pkl,
                                 num_pre_encoder=paras_PreEncoder['num_pre_encoder'],
                                 device=device)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    trainer = pl.Trainer(log_every_n_steps=1, max_epochs=paras_PreEncoder['max_epochs'], default_root_dir=path_pts, accelerator='gpu', devices=1)
    trainer.fit(model=pre_encoder_lm, train_dataloaders=dataset_loader)
