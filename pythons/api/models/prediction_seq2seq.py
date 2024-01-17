import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl


class Prediction_Seq2seq_Model(nn.Module):
    def __init__(self, paras: dict):
        super(Prediction_Seq2seq_Model, self).__init__()

        # 基础参数
        self.bias = False
        self.lstm_num_layers = 1
        self.lstm_bidirectional = False
        if self.lstm_bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.seq_history = paras['seq_history']
        self.seq_predict = paras['seq_predict']
        self.size_middle = paras['size_middle']
        self.device = paras['device']
        self.scale = paras['scale']
        self.delta_limit = 1
        self.info_len = 6

        # 对历史时序数据进行编码，his
        self.h0 = torch.ones([self.D * self.lstm_num_layers, 1, self.size_middle]).to(self.device)
        self.c0 = torch.ones([self.D * self.lstm_num_layers, 1, self.size_middle]).to(self.device)
        self.his_norm = nn.LayerNorm(normalized_shape=self.size_middle, elementwise_affine=False)
        self.his_lstm = nn.LSTM(input_size=(self.info_len + 1), hidden_size=self.size_middle, num_layers=self.lstm_num_layers,
                                bidirectional=self.lstm_bidirectional, bias=self.bias, batch_first=True)

        # 对未来时序进行预测，pre
        # 分为两部分：
        #       m_：均值/最大值/最小值
        #       var：方差
        # self.pre_norm_m_ = nn.LayerNorm(normalized_shape=self.D * self.size_middle, elementwise_affine=True)
        # self.pre_norm_var = nn.LayerNorm(normalized_shape=self.D * self.size_middle, elementwise_affine=True)
        self.pre_lstm_m_var = nn.LSTM(input_size=self.info_len, hidden_size=self.size_middle, num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional,
                                      bias=self.bias, batch_first=True)
        self.pre_linear_layer_decoder_m_ = nn.Sequential(nn.Tanh(), nn.Linear(in_features=self.D * self.size_middle, out_features=self.D * self.size_middle, bias=self.bias),
                                                         nn.ReLU(), nn.Linear(in_features=self.D * self.size_middle, out_features=1, bias=self.bias),
                                                         nn.Tanh())
        self.pre_linear_layer_decoder_var = nn.Sequential(nn.Tanh(), nn.Linear(in_features=self.D * self.size_middle, out_features=self.D * self.size_middle, bias=self.bias),
                                                          nn.ReLU(), nn.Linear(in_features=self.D * self.size_middle, out_features=1, bias=self.bias),
                                                          nn.ReLU())

    def forward(self, inp_info_his, inp_temperature_his, inp_info, h_his=None, c_his=None):
        # T1：seq_history，T2：seq_prediction
        # inp_info_his: [B, T1, info_len]
        # inp_temperature_his: [B, T1, 1]
        # inp_info: [B, T2, info_len]

        batch_size = inp_info_his.shape[0]
        if (h_his is None) and (c_his is None):
            # 对历史数据进行编码
            _, (h_his, c_his) = self.his_lstm(torch.cat([inp_info_his, inp_temperature_his], dim=2),
                                              (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1)))

        # 对未来数据进行解码，并生成h_pre和c_pre
        lstmed_m_var, _ = self.pre_lstm_m_var(inp_info, (h_his, c_his))
        lstmed_m_var = self.his_norm(lstmed_m_var)
        oup_m_ = self.pre_linear_layer_decoder_m_(lstmed_m_var) * self.delta_limit * self.scale
        oup_var = self.pre_linear_layer_decoder_var(lstmed_m_var) * self.scale * self.scale

        return oup_m_, oup_var, (h_his, c_his)


class Prediction_Seq2seq_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict):
        super(Prediction_Seq2seq_LightningModule, self).__init__()
        self.prediction_seq2seq = Prediction_Seq2seq_Model(paras)
        self.criterion_train = nn.GaussianNLLLoss(reduction='mean')
        self.criterion_val = nn.MSELoss(reduction='mean')
        self.criterion_test = nn.L1Loss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), paras['lr_init'])
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=paras['lr_init'], total_steps=paras['max_epochs'], pct_start=0.02)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=10, T_mult=1, eta_min=1e-5)

        self.save_hyperparameters('paras')

        self.seq_history = paras['seq_history']
        self.seq_predict = paras['seq_predict']
        self.scale = paras['scale']
        self.test_results = list()
        self.test_losses = {
            'mean': list(),
            'max': list(),
            'min': list()
        }

    def forward(self, inp_info_his, inp_temperature_his, inp_info, h_his=None, c_his=None):
        return self.prediction_seq2seq(inp_info_his, inp_temperature_his, inp_info)

    def run_base(self, batch, batch_idx):
        inp_info_his = batch[:, 0:self.seq_history, 1:7]
        inp_temperature_his = batch[:, 0:self.seq_history, 7:8] - torch.cat([batch[:, 0:1, 7:8], batch[:, 0:(self.seq_history - 1), 7:8]], dim=1)
        inp_info = batch[:, self.seq_history:, 1:7]
        pre_mean, pre_var, _ = self.prediction_seq2seq(inp_info_his, inp_temperature_his, inp_info)
        ref_mean = batch[:, self.seq_history:, 7:8] - batch[:, (self.seq_history - 1):-1, 7:8]
        return pre_mean, pre_var, ref_mean

    def training_step(self, batch, batch_idx):
        pre_mean, pre_var, ref_mean = self.run_base(batch, batch_idx)
        loss_train = self.criterion_train(pre_mean, ref_mean, pre_var)

        self.log('loss_train', loss_train, prog_bar=True, batch_size=len(batch))
        self.log('lr', self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=len(batch))
        return loss_train

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            pre_mean, pre_var, ref_mean = self.run_base(batch, batch_idx)

        pre_mean = pre_mean.cumsum(dim=1) + batch[:, (self.seq_history - 1):self.seq_history, 7:8]
        ref_mean = batch[:, self.seq_history:, 7:8]
        losses = self.criterion_test(pre_mean / self.scale, ref_mean / self.scale)
        for b in range(len(batch)):
            self.test_results.append({
                'origin': torch.cat([batch[b, self.seq_history:, 0:1], batch[b, self.seq_history:, 8:]], dim=1).T.cpu().numpy(),
                'pre': torch.cat([pre_mean[b], pre_var[b]], dim=1).T.cpu().numpy() / self.scale,
                'ref': torch.cat([batch[b, :, 0:1], batch[b, :, 7:8] / self.scale], dim=1).T.cpu().numpy()
            })
            loss = losses[b]
            self.test_losses['mean'].append(loss.mean().unsqueeze(0))
            self.test_losses['max'].append(loss.max().unsqueeze(0))
            self.test_losses['min'].append(loss.min().unsqueeze(0))

    def validation_step(self, batch, batch_idx):
        pre_mean, pre_var, ref_mean = self.run_base(batch, batch_idx)
        loss_nll = self.criterion_train(pre_mean, ref_mean, pre_var)
        pre_mean = pre_mean.cumsum(dim=1) + batch[:, (self.seq_history - 1):self.seq_history, 7:8]
        ref_mean = batch[:, self.seq_history:, 7:8]
        loss_mse = self.criterion_val(pre_mean / self.scale, ref_mean / self.scale)

        self.log('loss_val_nll', loss_nll, prog_bar=True, batch_size=len(batch))
        self.log('loss_val_mse', loss_mse, prog_bar=True, batch_size=len(batch))

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
