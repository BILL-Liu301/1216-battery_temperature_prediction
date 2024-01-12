import math
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
        self.lstm_num_layers = 2
        self.lstm_bidirectional = False
        if self.lstm_bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.seq_history = paras['seq_history']
        self.seq_predict = paras['seq_predict']
        self.size_middle = paras['size_middle']
        self.device = paras['device']

        # 对历史时序数据进行编码，his
        self.h0 = torch.ones([self.D * self.lstm_num_layers, 1, self.size_middle]).to(self.device)
        self.c0 = torch.ones([self.D * self.lstm_num_layers, 1, self.size_middle]).to(self.device)
        self.his_lstm = nn.LSTM(input_size=4, hidden_size=self.size_middle, num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional,
                                bias=self.bias, batch_first=True)

        # 对未来时序进行预测，pre
        # 分为两部分：
        #       m_：均值/最大值/最小值
        #       var：方差
        # self.pre_norm_m_ = nn.LayerNorm(normalized_shape=self.D * self.size_middle, elementwise_affine=True)
        # self.pre_norm_var = nn.LayerNorm(normalized_shape=self.D * self.size_middle, elementwise_affine=True)
        self.pre_lstm_m_var = nn.LSTM(input_size=3, hidden_size=self.size_middle, num_layers=self.lstm_num_layers, bidirectional=self.lstm_bidirectional,
                                      bias=self.bias, batch_first=True)
        self.pre_linear_layer_decoder_m_ = nn.Sequential(nn.Linear(in_features=self.D * self.size_middle, out_features=self.size_middle, bias=self.bias),
                                                         nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=1, bias=self.bias))
        self.pre_linear_layer_decoder_var = nn.Sequential(nn.Linear(in_features=self.D * self.size_middle, out_features=self.size_middle, bias=self.bias),
                                                          nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=1, bias=self.bias),
                                                          nn.ReLU())

    def forward(self, inp_loc_i_soc_his, inp_temperature_his, inp_loc_i_soc, h_his=None, c_his=None):
        # T1：seq_history，T2：seq_prediction
        # inp_loc_i_soc_his: [B, T1, 3]
        # inp_temperature_his: [B, T1, 1]
        # inp_loc_i_soc: [B, T2, 3]

        batch_size = inp_loc_i_soc_his.shape[0]
        temperature_start_his = inp_temperature_his.mean(1).unsqueeze(1)
        if (h_his is None) and (c_his is None):
            # 对历史数据进行编码
            _, (h_his, c_his) = self.his_lstm(torch.cat([inp_loc_i_soc_his, inp_temperature_his], dim=2),
                                              (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1)))

        # 对未来数据进行解码，并生成h_pre和c_pre
        lstmed_m_var, (h_pre, c_pre) = self.pre_lstm_m_var(inp_loc_i_soc, (h_his, c_his))
        oup_m_ = self.pre_linear_layer_decoder_m_(lstmed_m_var) + temperature_start_his
        oup_var = self.pre_linear_layer_decoder_var(lstmed_m_var)

        return oup_m_, oup_var, (h_pre, c_pre)


class Prediction_Seq2seq_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict):
        super(Prediction_Seq2seq_LightningModule, self).__init__()
        self.prediction_seq2seq = Prediction_Seq2seq_Model(paras)
        self.criterion_train = nn.GaussianNLLLoss(reduction='mean')
        self.criterion_val = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.parameters(), paras['lr_init'])
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=paras['lr_init'], total_steps=paras['max_epochs'], pct_start=0.1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=10, T_mult=1, eta_min=1e-5)

        self.save_hyperparameters('paras')

        self.seq_history = paras['seq_history']
        self.seq_predict = paras['seq_predict']
        self.test_results = list()

    def forward(self, inp_loc_i_soc_his, inp_temperature_his, inp_loc_i_soc, h_his=None, c_his=None):
        return self.prediction_seq2seq(inp_loc_i_soc_his, inp_temperature_his, inp_loc_i_soc)

    def training_step(self, batch, batch_idx):
        inp_loc_i_soc_his = batch[:, 0:self.seq_history, 1:4]
        inp_temperature_his = batch[:, 0:self.seq_history, 4:5]
        inp_loc_i_soc = batch[:, self.seq_history:, 1:4]
        pre_mean, pre_var, _ = self.prediction_seq2seq(inp_loc_i_soc_his, inp_temperature_his, inp_loc_i_soc)
        ref_mean = batch[:, self.seq_history:, 4:5]
        loss_train = self.criterion_train(pre_mean, ref_mean, pre_var)

        self.log('loss_train', loss_train, prog_bar=True, batch_size=len(batch))
        self.log('lr', self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=len(batch))
        return loss_train

    def test_step(self, batches, batch_idx):
        inp_loc_i_soc_his = batches[:, 0:self.seq_history, 1:4]
        inp_temperature_his = batches[:, 0:self.seq_history, 4:5]
        inp_loc_i_soc = batches[:, self.seq_history:, 1:4]
        with torch.no_grad():
            pre_mean, pre_var, _ = self.prediction_seq2seq(inp_loc_i_soc_his, inp_temperature_his, inp_loc_i_soc)
        ref_mean = batches[:, self.seq_history:, 4:5]
        for batch in range(len(batches)):
            self.test_results.append({
                'ref': ref_mean[batch].T.cpu().numpy(),
                'origin': batches[batch, self.seq_history:, 5:].T.cpu().numpy(),
                'pre': torch.cat([pre_mean[batch], pre_var[batch]], dim=1).T.cpu().numpy()
            })

    def validation_step(self, batch, batch_idx):
        inp_loc_i_soc_his = batch[:, 0:self.seq_history, 1:4]
        inp_temperature_his = batch[:, 0:self.seq_history, 4:5]
        inp_loc_i_soc = batch[:, self.seq_history:, 1:4]
        pre_mean, pre_var, _ = self.prediction_seq2seq(inp_loc_i_soc_his, inp_temperature_his, inp_loc_i_soc)
        ref_mean = batch[:, self.seq_history:, 4:5]
        loss_nll = self.criterion_train(pre_mean, ref_mean, pre_var)
        loss_mse = self.criterion_val(pre_mean, ref_mean)

        self.log('loss_val_nll', loss_nll, prog_bar=True, batch_size=len(batch))
        self.log('loss_val_mse', loss_mse, prog_bar=True, batch_size=len(batch))

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
