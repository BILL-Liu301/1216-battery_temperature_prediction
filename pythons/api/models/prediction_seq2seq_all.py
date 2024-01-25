import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl

from .prediction_seq2seq import Prediction_Seq2seq_Model


class Prediction_Seq2seq_All_Model(nn.Module):
    def __init__(self, model_single):
        super(Prediction_Seq2seq_All_Model, self).__init__()

        self.model_single = model_single

    def forward(self, inp_info_his, inp_temperature_his, inp_info, h_his=None, c_his=None):
        oup_m_, oup_var, _ = self.model_single(inp_info_his, inp_temperature_his, inp_info, h_his=None, c_his=None)
        print()


class Prediction_Seq2seq_All_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict, model_single: Prediction_Seq2seq_Model):
        super(Prediction_Seq2seq_All_LightningModule, self).__init__()
        self.model_single = model_single.eval()
        self.model_all = Prediction_Seq2seq_All_Model(self.model_single)

        self.criterion_train = nn.GaussianNLLLoss(reduction='mean')
        self.criterion_val = nn.MSELoss(reduction='mean')
        self.criterion_test = nn.L1Loss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), paras['lr_init'])
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=paras['lr_init'], total_steps=paras['max_epochs'], pct_start=0.1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=10, T_mult=1, eta_min=1e-5)

        self.save_hyperparameters('paras')

        self.test_results = list()
        self.test_losses = {
            'mean': list(),
            'max': list(),
            'min': list()
        }

    def run_base(self, batch, batch_idx):
        inp_info_his = batch[:, 0:self.seq_history, 1:7]
        inp_temperature_his = batch[:, 0:self.seq_history, 7:8]
        inp_info = batch[:, self.seq_history:, 1:7]
        pre_mean, pre_var, _ = self.prediction_seq2seq(inp_info_his, inp_temperature_his, inp_info)
        ref_mean = batch[:, self.seq_history:, 7:8]
        return pre_mean, pre_var, ref_mean

    def training_step(self, batch, batch_idx):
        pre_mean, pre_var, ref_mean = self.run_base(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            pre_mean, pre_var, ref_mean = self.run_base(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        pre_mean, pre_var, ref_mean = self.run_base(batch, batch_idx)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
