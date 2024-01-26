import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from scipy.stats import norm
import pytorch_lightning as pl


class Prediction_Temperature_Model(nn.Module):
    def __init__(self, paras: dict):
        super(Prediction_Temperature_Model, self).__init__()

        # 基础参数
        self.tanh = nn.Tanh()
        self.bias = False
        self.lstm_bidirectional = False
        if self.lstm_bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.seq_history = paras['seq_history']
        self.seq_attention_once = paras['seq_attention_once']
        self.size_middle = paras['size_middle']
        self.num_layers = paras['num_layers']
        self.device = paras['device']
        self.scale = paras['scale']
        self.info_len = paras['info_len']
        self.delta_limit_m_ = 20
        self.delta_limit_var = 5

        # 对未来时序进行预测，pre
        # 分为两部分：
        #       m_：均值/最大值/最小值
        #       var：方差
        self.pre_linear_layer_info = nn.Sequential(nn.Linear(in_features=self.info_len, out_features=self.size_middle, bias=self.bias),
                                                   nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.size_middle, bias=self.bias))
        self.pre_norm_info = nn.LayerNorm(normalized_shape=self.info_len, elementwise_affine=False)
        self.pre_attention_q = nn.Sequential(nn.Tanh(), nn.Linear(in_features=self.info_len, out_features=self.size_middle, bias=self.bias),
                                             nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.size_middle, bias=self.bias))
        self.pre_attention_k = nn.Sequential(nn.Tanh(), nn.Linear(in_features=self.info_len, out_features=self.size_middle, bias=self.bias),
                                             nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.size_middle, bias=self.bias))
        self.pre_attention_v = nn.Sequential(nn.Tanh(), nn.Linear(in_features=self.info_len, out_features=self.size_middle, bias=self.bias),
                                             nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.size_middle, bias=self.bias))
        self.h0 = torch.ones([self.D * self.num_layers, 1, self.size_middle]).to(self.device)
        self.c0 = torch.ones([self.D * self.num_layers, 1, self.size_middle]).to(self.device)
        self.pre_lstm_m_var = nn.LSTM(input_size=self.size_middle, hidden_size=self.size_middle, num_layers=self.num_layers, bidirectional=self.lstm_bidirectional,
                                      bias=self.bias, batch_first=True)
        self.pre_norm_m_var = nn.LayerNorm(normalized_shape=self.D * self.size_middle, elementwise_affine=False)
        self.pre_linear_layer_decoder_m_ = nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.D * self.size_middle, out_features=self.size_middle, bias=self.bias),
                                                         nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=1, bias=self.bias),
                                                         nn.Tanh())
        self.pre_linear_layer_decoder_var = nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.D * self.size_middle, out_features=self.size_middle, bias=self.bias),
                                                          nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=1, bias=self.bias),
                                                          nn.Sigmoid())

    def self_attention(self, qkv):
        q, k, v = qkv
        b = torch.matmul(k.transpose(1, 2), q)
        oup = torch.matmul(v, self.tanh(b))
        return oup

    def forward(self, inp_info_his, inp_temperature_his, inp_info, h_his=None, c_his=None):
        # T1：seq_history，T2：seq_prediction
        # inp_info_his: [B, T1, info_len]
        # inp_temperature_his: [B, T1, 1]
        # inp_info: [B, T2, info_len]

        # if (h_his is None) and (c_his is None):
        #     # 对历史数据进行编码
        #     # his_linear_layer = self.his_linear_layer_encoder(torch.cat([inp_info_his, inp_temperature_his], dim=2))
        #     _, (h_his, c_his) = self.his_lstm(torch.cat([inp_info_his, inp_temperature_his], dim=2),
        #                                       (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1)))

        # 提取尺寸
        batch_size = inp_info.shape[0]
        seq_predict = inp_info.shape[1]

        seqs = (seq_predict // self.seq_attention_once) if (seq_predict % self.seq_attention_once == 0) else (seq_predict // self.seq_attention_once + 1)
        oup_m_, oup_var = list(), list()
        temperature_ref = inp_temperature_his[:, -1:]
        # 分段预测
        for seq in range(seqs):
            # 编码和归一化
            linear_layer_info = self.pre_linear_layer_info(inp_info[:, seq * self.seq_attention_once:(seq + 1) * self.seq_attention_once])
            norm_info = self.pre_norm_info(inp_info[:, seq * self.seq_attention_once:(seq + 1) * self.seq_attention_once])
            # attention
            attention_q, attention_k, attention_v = self.pre_attention_q(norm_info), self.pre_attention_k(norm_info), self.pre_attention_v(norm_info)
            attentioned = self.self_attention([attention_q, attention_k, attention_v]) + linear_layer_info
            # lstm
            lstmed, _ = self.pre_lstm_m_var(attentioned, (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1)))
            encoded = self.pre_norm_m_var(lstmed)
            # 解码
            oup_m_.append(self.pre_linear_layer_decoder_m_(encoded) * self.delta_limit_m_ + temperature_ref)
            oup_var.append(self.pre_linear_layer_decoder_var(encoded) * self.delta_limit_var)
            temperature_ref = oup_m_[-1][:, -1:]
        oup_m_ = torch.cat(oup_m_, dim=1)
        oup_var = torch.cat(oup_var, dim=1)

        return oup_m_, oup_var, (h_his, c_his)


class Prediction_Temperature_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict):
        super(Prediction_Temperature_LightningModule, self).__init__()
        self.model = Prediction_Temperature_Model(paras)

        self.criterion_train = nn.GaussianNLLLoss(reduction='mean')
        self.criterion_val = nn.MSELoss(reduction='mean')
        self.criterion_test = nn.L1Loss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), paras['lr_init'])
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=paras['lr_init'], total_steps=paras['max_epochs'], pct_start=0.1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=10, T_mult=1, eta_min=1e-5)

        self.save_hyperparameters('paras')

        self.seq_history = paras['seq_history']
        self.scale = paras['scale']
        self.test_results = list()
        self.test_losses = {
            'mean': list(),
            'max': list(),
            'min': list()
        }

    def forward(self, inp_info_his, inp_temperature_his, inp_info, h_his=None, c_his=None):
        return self.model(inp_info_his, inp_temperature_his, inp_info)

    def run_base(self, batch, batch_idx):
        inp_info_his = batch[:, 0:self.seq_history, 1:7]
        inp_temperature_his = batch[:, 0:self.seq_history, 7:8]
        inp_info = batch[:, self.seq_history:, 1:7]
        pre_mean, pre_var, _ = self.model(inp_info_his, inp_temperature_his, inp_info)
        ref_mean = batch[:, self.seq_history:, 7:8]
        return pre_mean, pre_var, ref_mean

    def training_step(self, batch, batch_idx):
        pre_mean, pre_var, ref_mean = self.run_base(batch, batch_idx)
        loss_train = self.criterion_train(pre_mean, ref_mean, pre_var)

        self.log('loss_train', loss_train, prog_bar=True)
        self.log('lr', self.scheduler.get_last_lr()[0], prog_bar=True)
        return loss_train

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            pre_mean, pre_var, ref_mean = self.run_base(batch, batch_idx)
        losses = self.criterion_test(pre_mean, ref_mean)
        for b in range(len(batch)):
            prob = np.abs(norm.cdf(ref_mean[b].cpu().numpy(), pre_mean[b].cpu().numpy(), torch.sqrt(pre_var[b]).cpu().numpy()) - 0.5) * 2
            self.test_results.append({
                'origin': torch.cat([batch[b, self.seq_history:, 0:1], batch[b, self.seq_history:, 8:]], dim=1).T.cpu().numpy(),
                'pre': torch.cat([pre_mean[b], pre_var[b]], dim=1).T.cpu().numpy(),
                'ref': torch.cat([batch[b, :, 0:1], batch[b, :, 7:8]], dim=1).T.cpu().numpy(),
                'loss': losses[b].T.cpu().numpy()[0],
                'prob': prob * 100,
                'info': batch[b, self.seq_history:, 1:7].T.cpu().numpy()
            })
            loss = losses[b]
            self.test_losses['mean'].append(loss.mean().unsqueeze(0))
            self.test_losses['max'].append(loss.max().unsqueeze(0))
            self.test_losses['min'].append(loss.min().unsqueeze(0))

    def validation_step(self, batch, batch_idx):
        pre_mean, pre_var, ref_mean = self.run_base(batch, batch_idx)
        loss_nll = self.criterion_train(pre_mean, ref_mean, pre_var)
        loss_mse = self.criterion_val(pre_mean, ref_mean)

        self.log('loss_val_nll', loss_nll, prog_bar=True)
        self.log('loss_val_mse', loss_mse, prog_bar=True)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
