import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class PreEncoder_Model(nn.Module):
    def __init__(self, paras):
        super(PreEncoder_Model, self).__init__()

        # 基础参数
        self.bias = False
        self.softmax_switch = False
        self.softmax = nn.Softmax(dim=0)
        self.num_measure_point = paras['num_measure_point']
        self.num_pre_encoder = paras['num_pre_encoder']
        self.size_middle = paras['size_middle']

        # 预测均值与标准差，mean_std -> ms
        self.ms_norm_v = nn.LayerNorm(self.size_middle)
        self.ms_norm_k = nn.LayerNorm(self.size_middle)
        self.ms_norm_q = nn.LayerNorm(self.size_middle)
        self.ms_linear_layer_encoder_v = nn.Linear(in_features=2, out_features=self.size_middle, bias=self.bias)
        self.ms_linear_layer_encoder_k = nn.Linear(in_features=4, out_features=self.size_middle, bias=self.bias)
        self.ms_linear_layer_encoder_q = nn.Linear(in_features=4, out_features=self.size_middle, bias=self.bias)
        self.ms_linear_layer_decoder = nn.Linear(in_features=self.size_middle, out_features=2, bias=self.bias)
        self.ms_w_qkv = nn.Parameter(torch.normal(mean=0, std=2, size=(3, self.size_middle, self.size_middle)))

        # 预测分布变化
        self.pre_mean_std = None
        self.distri_linear_layer_encoder_v = nn.Linear(in_features=self.num_measure_point, out_features=self.size_middle, bias=self.bias)
        self.distri_linear_layer_decoder = nn.Linear(in_features=self.size_middle, out_features=self.num_measure_point, bias=self.bias)
        self.distri_norm = nn.LayerNorm(self.num_measure_point)

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
        init_mean_std = torch.cat([torch.mean(inp_temperature, dim=0), torch.std(inp_temperature, dim=0)]).unsqueeze(1)

        # 分别计算qkv，q~mean_std，k/v~loc_i_soc，最终解析成mean和std的变化
        v = self.ms_norm_v(self.ms_linear_layer_encoder_v(init_mean_std.T)).T
        k = self.ms_norm_k(self.ms_linear_layer_encoder_k(inp_loc_i_soc.T)).T
        q = self.ms_norm_q(self.ms_linear_layer_encoder_q(inp_loc_i_soc.T)).T
        attentioned = self.cross_attention(qkv=[q, k, v])
        self.pre_mean_std = self.ms_linear_layer_decoder(attentioned.T).T

        # 对分布进行预测
        v = (inp_temperature - init_mean_std[0]) / init_mean_std[1]
        v = self.distri_linear_layer_encoder_v(v.T).T
        distribution = self.cross_attention(qkv=[q, k, v])
        distribution = self.distri_linear_layer_decoder(distribution.T).T
        oup = self.distri_norm(distribution.T).T * self.pre_mean_std[1] + self.pre_mean_std[0]
        return oup


class PreEncoder_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict):
        super(PreEncoder_LightningModule, self).__init__()
        self.pre_encoder = PreEncoder_Model(paras)
        self.criterion = nn.L1Loss(reduction='mean')
        self.optimizer = optim.Adam(self.parameters(), paras['lr_init'])
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=paras['lr_init'], total_steps=paras['max_epochs'], pct_start=0.1)

    def training_step(self, batches):
        loss_batches = list()
        for batch in batches:
            # 主训练过程
            inp1 = batch[0:4]  # Position Embedding, Location, I, SOC
            inp2 = batch[4:, 0:1]  # Initial Temperature
            pre_temperature = self.pre_encoder(inp1, inp2)
            pre_mean_std = self.pre_encoder.pre_mean_std

            ref_temperature = batch[4:]
            ref_mean = ref_temperature.mean(dim=0)
            ref_std = ref_temperature.std(dim=0)
            ref_mean_std = torch.cat([ref_mean.unsqueeze(0), ref_std.unsqueeze(0)], dim=0)

            loss_temperature = self.criterion(pre_temperature, ref_temperature).unsqueeze(0)
            loss_mean_std = self.criterion(pre_mean_std, ref_mean_std).unsqueeze(0)
            loss_batches.append((loss_temperature + loss_mean_std) / 2)
        loss_torch = torch.cat(loss_batches)
        self.log('loss_max', loss_torch.max(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('loss_min', loss_torch.min(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('loss_mean', loss_torch.mean(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss_torch.mean()

    def configure_optimizers(self):
        return self.optimizer
