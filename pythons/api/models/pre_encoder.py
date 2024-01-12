import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl


class PreEncoder_Model(nn.Module):
    def __init__(self, paras: dict):
        super(PreEncoder_Model, self).__init__()

        # 基础参数
        self.bias = False
        self.lstm_num_layers = 2
        self.softmax_switch = False
        self.softmax = nn.Softmax(dim=0)
        self.num_measure_point = paras['num_measure_point']
        self.num_pre_encoder = paras['num_pre_encoder'] - 1
        self.size_middle = paras['size_middle']

        # 预测均值与标准差，mean_var -> ms
        self.ms_linear_layer_h = nn.Linear(in_features=2, out_features=self.size_middle, bias=self.bias)
        self.ms_linear_layer_c = nn.Linear(in_features=2, out_features=self.size_middle, bias=self.bias)
        self.ms_lstm = nn.LSTM(input_size=3, hidden_size=self.size_middle, num_layers=self.lstm_num_layers, bias=self.bias)
        self.ms_linear_layer_decoder = nn.Sequential(nn.Linear(in_features=self.size_middle, out_features=self.size_middle, bias=self.bias), nn.ReLU(),
                                                     nn.Linear(in_features=self.size_middle, out_features=2, bias=self.bias), nn.ReLU())

    def cross_attention(self, qkv):
        q, k, v = qkv
        v = v.repeat(1, k.shape[1])
        b = torch.matmul(k.T, q)
        if self.softmax_switch:
            d_k = b.shape[1]
            b = self.softmax(b) / math.sqrt(d_k)
        oup = torch.matmul(v, b)
        return oup

    def forward(self, inp_loc_i_soc, inp_temperature):
        # 计算均值与标准差
        init_mean_var = torch.cat([torch.mean(inp_temperature, dim=1), torch.var(inp_temperature, dim=1)]).unsqueeze(0)
        h0 = self.ms_linear_layer_h(init_mean_var.repeat(self.lstm_num_layers, 1))
        c0 = self.ms_linear_layer_c(init_mean_var.repeat(self.lstm_num_layers, 1))
        lstmed, _ = self.ms_lstm(inp_loc_i_soc, (h0, c0))
        oup = self.ms_linear_layer_decoder(lstmed) + init_mean_var
        return oup.T


class PreEncoder_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict):
        super(PreEncoder_LightningModule, self).__init__()
        self.pre_encoder = PreEncoder_Model(paras)
        self.criterion_train = nn.GaussianNLLLoss(reduction='mean')
        self.criterion_val = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.parameters(), paras['lr_init'])
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=paras['lr_init'], total_steps=paras['max_epochs'], pct_start=0.1)
        self.test_results = list()

    def forward(self, location_i_soc, init_temperature):
        return self.pre_encoder(location_i_soc, init_temperature)

    def training_step(self, batches, batch_idx):
        loss_nll_batches = list()
        loss_mse_batches = list()
        for batch in batches:
            # 主训练过程
            inp1 = batch[1:4, 1:]  # Location, I, SOC
            inp2 = batch[4:, 0:1]  # Initial Temperature
            pre_mean_var = self.pre_encoder(inp1.T, inp2.T)

            ref_temperature = batch[4:, 1:]
            ref_mean = ref_temperature.mean(dim=0)
            ref_var = ref_temperature.var(dim=0)
            ref_mean_var = torch.cat([ref_mean.unsqueeze(0), ref_var.unsqueeze(0)], dim=0)

            loss_nll = self.criterion_train(pre_mean_var[0], ref_mean_var[0], pre_mean_var[1]).unsqueeze(0)
            loss_mse = self.criterion_val(pre_mean_var, ref_mean_var).unsqueeze(0)
            loss_nll_batches.append(loss_nll)
            loss_mse_batches.append(loss_mse)
        loss_train = torch.cat(loss_nll_batches).abs().max()
        self.log('loss_train', loss_train, prog_bar=True, batch_size=1)
        return loss_train

    def test_step(self, batches, batch_idx):
        for batch in batches:
            # 主训练过程
            inp1 = batch[1:4, 1:]  # Location, I, SOC
            inp2 = batch[4:, 0:1]  # Initial Temperature
            init_mean_var = torch.cat([torch.mean(inp2, dim=0), torch.var(inp2, dim=0)]).unsqueeze(1)
            pre_mean_var = self.pre_encoder(inp1.T, inp2.T)
            pre_mean_var = torch.cat([batch[0:4], torch.cat([init_mean_var, pre_mean_var], dim=1)], dim=0)

            ref_temperature = batch

            self.test_results.append({
                'pre': pre_mean_var.cpu().numpy(),
                'ref': ref_temperature.cpu().numpy()
            })

    def validation_step(self, batches, batch_idx):
        loss_nll_batches = list()
        loss_mse_batches = list()
        for batch in batches:
            # 主训练过程
            inp1 = batch[1:4, 1:]  # Location, I, SOC
            inp2 = batch[4:, 0:1]  # Initial Temperature
            pre_mean_var = self.pre_encoder(inp1.T, inp2.T)

            ref_temperature = batch[4:, 1:]
            ref_mean = ref_temperature.mean(dim=0)
            ref_var = ref_temperature.var(dim=0)
            ref_mean_var = torch.cat([ref_mean.unsqueeze(0), ref_var.unsqueeze(0)], dim=0)

            loss_nll = self.criterion_train(pre_mean_var[0], ref_mean_var[0], pre_mean_var[1]).unsqueeze(0)
            loss_mse = self.criterion_val(pre_mean_var[0], ref_mean_var[0]).unsqueeze(0)
            loss_nll_batches.append(loss_nll)
            loss_mse_batches.append(loss_mse)
        loss_nll_val = torch.cat(loss_nll_batches).abs().max()
        loss_mse_val = torch.cat(loss_mse_batches).abs().max()
        self.log('loss_nll_val', loss_nll_val, prog_bar=True, batch_size=1)
        self.log('loss_mse_val', loss_mse_val, prog_bar=True, batch_size=1)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
