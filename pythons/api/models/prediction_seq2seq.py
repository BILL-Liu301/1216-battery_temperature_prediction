import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class Prediction_Seq2seq_Model(nn.Module):
    def __init__(self, paras: dict):
        super(Prediction_Seq2seq_Model, self).__init__()

        # 基础参数
        self.bias = False
        self.lstm_num_layers = 2
        self.seq_history = paras['seq_history']
        self.seq_predict = paras['seq_predict']
        self.size_middle = paras['size_middle']

        # 对历史时序数据进行编码，his
        self.his_linear_layer_encoder_h = nn.Linear(in_features=self.seq_history, out_features=self.size_middle, bias=self.bias)
        self.his_linear_layer_encoder_c = nn.Linear(in_features=self.seq_history, out_features=self.size_middle, bias=self.bias)
        self.his_lstm = nn.LSTM(input_size=4, hidden_size=self.size_middle, num_layers=self.lstm_num_layers, bias=self.bias)

        # 对未来时序进行预测，pre
        # 分为两部分：
        #       m_：均值/最大值/最小值
        #       var：方差
        self.pre_norm_h = nn.LayerNorm(normalized_shape=self.size_middle)
        self.pre_norm_c = nn.LayerNorm(normalized_shape=self.size_middle)
        self.pre_norm_m_ = nn.LayerNorm(normalized_shape=self.size_middle)
        self.pre_norm_var = nn.LayerNorm(normalized_shape=self.size_middle)
        self.pre_lstm_m_ = nn.LSTM(input_size=3, hidden_size=self.size_middle, num_layers=self.lstm_num_layers, bias=self.bias)
        self.pre_linear_layer_decoder_m_ = nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=1, bias=self.bias))
        self.pre_lstm_var = nn.LSTM(input_size=3, hidden_size=self.size_middle, num_layers=self.lstm_num_layers, bias=self.bias)
        self.pre_linear_layer_decoder_var = nn.Sequential(nn.Linear(in_features=self.size_middle, out_features=self.size_middle, bias=self.bias),
                                                          nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=1, bias=self.bias),
                                                          nn.ReLU())

    def forward(self, inp_loc_i_soc_his, inp_temperature_his, inp_loc_i_soc, h_his=None, c_his=None):
        # T1：seq_history，T2：seq_prediction
        # inp_loc_i_soc_his: [T1, 3]
        # inp_temperature_his: [T1, 1]
        # inp_loc_i_soc: [T2, 3]

        if (h_his is None) and (c_his is None):
            # 对历史数据进行编码
            h0 = self.his_linear_layer_encoder_h(inp_temperature_his.T.repeat(self.lstm_num_layers, 1))
            c0 = self.his_linear_layer_encoder_c(inp_temperature_his.T.repeat(self.lstm_num_layers, 1))
            _, (h_his, c_his) = self.his_lstm(torch.cat([inp_loc_i_soc_his, inp_temperature_his], dim=1), (h0, c0))

        # 对未来数据进行解码，并生成h_pre和c_pre
        temperature_last_his = inp_temperature_his[-1]
        h_his, c_his = self.pre_norm_h(h_his), self.pre_norm_c(c_his)
        lstmed_m_, (h_pre, c_pre) = self.pre_lstm_m_(inp_loc_i_soc, (h_his, c_his))
        oup_m_ = self.pre_linear_layer_decoder_m_(self.pre_norm_m_(lstmed_m_)) + temperature_last_his
        lstmed_var, _ = self.pre_lstm_var(inp_loc_i_soc, (h_his, c_his))
        oup_var = self.pre_linear_layer_decoder_var(self.pre_norm_var(lstmed_var))

        return oup_m_, oup_var, (h_pre, c_pre)


class Prediction_Seq2seq_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict):
        super(Prediction_Seq2seq_LightningModule, self).__init__()
        self.prediction_seq2seq = Prediction_Seq2seq_Model(paras)
        self.criterion_train = nn.GaussianNLLLoss(reduction='mean')
        self.criterion_val = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.parameters(), paras['lr_init'])
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=paras['lr_init'], total_steps=paras['max_epochs'], pct_start=0.1)

        self.seq_history = paras['seq_history']
        self.seq_predict = paras['seq_predict']
        self.test_results = list()

    def forward(self, inp_loc_i_soc_his, inp_temperature_his, inp_loc_i_soc, h_his=None, c_his=None):
        return self.prediction_seq2seq(inp_loc_i_soc_his, inp_temperature_his, inp_loc_i_soc)

    def training_step(self, batches, batch_idx):
        loss_nll_batches = list()
        for batch in batches:
            inp_loc_i_soc_his = batch[1:4, 0:self.seq_history]
            inp_temperature_his = batch[4:5, 0:self.seq_history]
            inp_loc_i_soc = batch[1:4, self.seq_history:]
            pre_mean, pre_var, _ = self.prediction_seq2seq(inp_loc_i_soc_his.T, inp_temperature_his.T, inp_loc_i_soc.T)
            ref_mean = batch[4:5, self.seq_history:].T

            loss_null = self.criterion_train(pre_mean, ref_mean, pre_var).unsqueeze(0)
            loss_nll_batches.append(loss_null)
        loss_train = torch.cat(loss_nll_batches).abs().max()
        self.log('loss_train', loss_train, prog_bar=True, batch_size=len(batches))
        self.log('lr', self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=len(batches))
        return loss_train

    def test_step(self, batches, batch_idx):
        for batch in batches:
            inp_loc_i_soc_his = batch[1:4, 0:self.seq_history]
            inp_temperature_his = batch[4:5, 0:self.seq_history]
            inp_loc_i_soc = batch[1:4, self.seq_history:]
            with torch.no_grad():
                pre_mean, pre_var, _ = self.prediction_seq2seq(inp_loc_i_soc_his.T, inp_temperature_his.T, inp_loc_i_soc.T)
            ref_mean = batch[4:5, self.seq_history:].T
            self.test_results.append({
                'pre': torch.cat([pre_mean, pre_var], dim=0).T.cpu().numpy(),
                'ref': ref_mean.T.cpu().numpy(),
                'origin': batch[5:, self.seq_history:].cpu().numpy()
            })

    def validation_step(self, batches, batch_idx):
        loss_nll_batches = list()
        loss_mse_batches = list()
        for batch in batches:
            inp_loc_i_soc_his = batch[1:4, 0:self.seq_history]
            inp_temperature_his = batch[4:5, 0:self.seq_history]
            inp_loc_i_soc = batch[1:4, self.seq_history:]
            pre_mean, pre_var, _ = self.prediction_seq2seq(inp_loc_i_soc_his.T, inp_temperature_his.T, inp_loc_i_soc.T)
            ref_mean = batch[4:5, self.seq_history:].T

            loss_null = self.criterion_train(pre_mean, ref_mean, pre_var).unsqueeze(0)
            loss_mse = self.criterion_val(pre_mean, ref_mean).unsqueeze(0)
            loss_nll_batches.append(loss_null)
            loss_mse_batches.append(loss_mse)
        loss_val_null = torch.cat(loss_nll_batches).abs().max()
        loss_val_mse = torch.cat(loss_mse_batches).abs().max()
        self.log('loss_val_null', loss_val_null, prog_bar=True, batch_size=len(batches))
        self.log('loss_val_mse', loss_val_mse, prog_bar=True, batch_size=len(batches))

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
