import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from scipy.stats import norm
import pytorch_lightning as pl


class Prediction_Temperature_Module(nn.Module):
    def __init__(self, paras: dict):
        super(Prediction_Temperature_Module, self).__init__()

        # 基础参数
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
        self.delta_limit_mean = 5
        self.delta_limit_var = 10

        # 对未来时序进行预测，pre
        # 分为两部分：
        #       m_：均值/最大值/最小值
        #       var：方差

        model_encode = nn.ModuleDict({
            'for_init': nn.Linear(in_features=1 + self.info_len, out_features=self.size_middle, bias=self.bias),
            # 'for_lstm': nn.Linear(in_features=self.info_len, out_features=self.size_middle, bias=self.bias)
        })
        model_attention = nn.ModuleDict({
            'q': nn.Sequential(nn.Linear(in_features=self.info_len, out_features=self.size_middle, bias=self.bias)),
            'k': nn.Sequential(nn.Linear(in_features=self.info_len, out_features=self.size_middle, bias=self.bias)),
            'v': nn.Sequential(nn.Linear(in_features=self.info_len, out_features=self.size_middle, bias=self.bias)),
            'attention': nn.MultiheadAttention(embed_dim=self.size_middle, num_heads=1, batch_first=True, bias=self.bias)
        })
        model_lstm = nn.ModuleDict({
            'lstm': nn.LSTM(input_size=self.size_middle, hidden_size=self.size_middle, num_layers=self.num_layers,
                            bidirectional=self.lstm_bidirectional, bias=self.bias, batch_first=True)
        })
        self.h0 = torch.ones([self.D * self.num_layers, 1, self.size_middle]).to(self.device)
        self.c0 = torch.ones([self.D * self.num_layers, 1, self.size_middle]).to(self.device)
        model_decode = nn.ModuleDict({
            'for_norm': nn.LayerNorm(normalized_shape=self.D * self.size_middle, elementwise_affine=False),
            'for_mean': nn.Sequential(nn.Linear(in_features=self.D * self.size_middle, out_features=1, bias=self.bias),
                                      nn.Tanh()),
            'for_var': nn.Sequential(nn.Linear(in_features=self.D * self.size_middle, out_features=1, bias=self.bias),
                                     nn.Sigmoid())
        })
        self.model = nn.ModuleDict({
            'model_encode': model_encode,
            'model_attention': model_attention,
            'model_lstm': model_lstm,
            'model_decode': model_decode,
        })

    def forward(self, inp_info_his, inp_temperature_his, inp_info, h_his=None, c_his=None):
        # inp_info_his: [B, 1, [location, current, soc, voltage, ntc_max, ntc_min]]
        # inp_temperature_his: [B, 1, [temperature_max]]
        # inp_info: [B, seq_prediction, [location, current, soc, voltage, ntc_max, ntc_min]]

        # 提取尺寸
        batch_size = inp_info.shape[0]
        seq_predict = inp_info.shape[1]

        # 计算sequence的组数
        seqs = (seq_predict // self.seq_attention_once) if (seq_predict % self.seq_attention_once == 0) else (seq_predict // self.seq_attention_once + 1)

        # 数据初始化
        info_ref = inp_info_his
        temperature_ref = inp_temperature_his
        oup_mean, oup_var = list(), list()

        # 分段预测
        for seq in range(seqs):
            # 截取段落
            inp_info_seq = inp_info[:, seq * self.seq_attention_once:(seq + 1) * self.seq_attention_once]

            # 模型核心（LSTM）初始化
            encode_for_lstm = self.model['model_encode']['for_init'](torch.cat((info_ref, temperature_ref), dim=-1))
            _, (h1, c1) = self.model['model_lstm']['lstm'](encode_for_lstm, (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1)))

            # attention
            q, k, v = (self.model['model_attention']['q'](inp_info_seq),
                       self.model['model_attention']['k'](inp_info_seq),
                       self.model['model_attention']['v'](inp_info_seq))
            attentioned, _ = self.model['model_attention']['attention'](q, k, v)

            # lstm
            # lstmed, _ = self.model['model_lstm']['lstm'](self.model['model_encode']['for_lstm'](inp_info_seq), (h1, c1))
            lstmed, _ = self.model['model_lstm']['lstm'](attentioned, (h1, c1))

            # 解码生成均值和方差
            cumsum = torch.cumsum(lstmed, dim=1)
            # normed = self.model['model_decode']['for_norm'](lstmed)
            mean = self.model['model_decode']['for_mean'](cumsum) * self.delta_limit_mean + temperature_ref
            var = self.model['model_decode']['for_var'](cumsum) * self.delta_limit_var

            # 数据拼接和参考更新
            oup_mean.append(mean)
            oup_var.append(var)
            info_ref = inp_info_seq[:, -1:].clone()
            temperature_ref = mean[:, -1:].clone()

        oup_mean = torch.cat(oup_mean, dim=1)
        oup_var = torch.cat(oup_var, dim=1)

        return oup_mean, oup_var, (h_his, c_his)


class Prediction_Temperature_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict):
        super(Prediction_Temperature_LightningModule, self).__init__()
        self.model_temperature = Prediction_Temperature_Module(paras)

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
        return self.model_temperature(inp_info_his, inp_temperature_his, inp_info, h_his=h_his, c_his=c_his)

    def run_base(self, batch, batch_idx):
        inp_info_his = batch[:, 0:self.seq_history, 1:7]  # 历史的定位、电流、SOC、电压、NTC_max、NTC_min
        inp_temperature_his = batch[:, 0:self.seq_history, 7:8]  # 初始Temperature_max
        inp_info = batch[:, self.seq_history:, 1:7]  # 未来的定位、电流、SOC、电压、NTC_max、NTC_min
        pre_mean, pre_var, _ = self.model_temperature(inp_info_his, inp_temperature_his, inp_info)
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
                'info': batch[b, self.seq_history:, 1:7].T.cpu().numpy()  # 定位、电流、SOC、电压、NTC_max、NTC_min
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
