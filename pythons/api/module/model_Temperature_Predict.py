import torch
import torch.nn as nn


class TemperaturePrediction(nn.Module):
    def __init__(self, device, paras):
        super(TemperaturePrediction, self).__init__()
        self.encoder = Encoder(paras=paras['encoder'], device=device).to(device)
        self.decoder = Decoder(paras=paras['decoder']).to(device)

        self.seq_predict = 0
        self.seq_all_t = torch.zeros([paras['num_measure_point'], paras['seq_concern']]).to(device)
        self.seq_all_i_soc = torch.zeros([2, paras['seq_concern']]).to(device)
        self.mode = ''
        self.mode_for_train = 1

    def set_train(self):
        self.mode = 'train'

    def set_test(self):
        self.mode = 'test'

    def reinit_seq(self):
        self.seq_all_t = torch.zeros_like(self.seq_all_t)
        self.seq_all_i_soc = torch.zeros_like(self.seq_all_i_soc)

    def mode_train(self, inp1, inp2):
        oup = list()
        for seq in range(self.seq_predict):
            self.seq_all_i_soc[:, 0:min((seq + 2), self.seq_all_i_soc.shape[1])] = inp1[1:, max(0, seq + 2 - self.seq_all_i_soc.shape[1]):(seq + 2)]
            if seq == 0:
                self.seq_all_t[:, 0:1] = inp2[:, 0:1]
            else:
                self.seq_all_t[:, 0:min((seq + 1), self.seq_all_t.shape[1])] = torch.cat([inp2[:, 0:1], torch.cat(oup, dim=1)], dim=1)[:, max(0, seq + 1 - self.seq_all_t.shape[1]):(seq + 1)]
            if torch.isnan(self.seq_all_i_soc).any() or torch.isnan(self.seq_all_t).any():
                print(f'seq_all_i_soc: {torch.isnan(self.seq_all_i_soc).any()}, seq_all_t: {torch.isnan(self.seq_all_t).any()}')
                raise Exception
            encoded = self.encoder(self.seq_all_i_soc.T)
            decoded = self.decoder(encoded, self.seq_all_t)
            oup.append(decoded)
            self.reinit_seq()
        return torch.cat(oup, dim=1)

    def mode_test(self, inp1, inp2):
        oup = list()
        for seq in range(self.seq_predict):
            self.seq_all_i_soc[:, 0:(seq + 2)] = inp1[1:, 0:(seq + 2)]
            if seq == 0:
                self.seq_all_t[:, 0:(seq + 1)] = inp2
            else:
                self.seq_all_t[:, 0:(seq + 1)] = torch.cat([inp2, torch.cat(oup, dim=1)], dim=1)
            encoded = self.encoder(self.seq_all_i_soc.T)
            decoded = self.decoder(encoded, self.seq_all_t)
            oup.append(decoded)
            self.reinit_seq()
        return torch.cat(oup, dim=1)

    def forward(self, inp1, inp2, seq_predict):
        # 若mode为train，则说明是训练模式，输入数据inp2需包含全过程的温度变化
        # 若mode为test，则说明是测试模式，输入数据inp2只需包含初始温度
        self.seq_predict = seq_predict
        if self.mode == 'train':
            oup = self.mode_train(inp1=inp1, inp2=inp2)
        elif self.mode == 'test':
            oup = torch.cat([inp1, torch.cat([inp2, self.mode_test(inp1=inp1, inp2=inp2)], dim=1)], dim=0)
        else:
            print(f'当前模式为【{self.mode}】，但输入数据的尺寸分别为inp1：【{inp1.shape}】，inp2：【{inp2.shape}】')
            raise Exception
        return oup


class Encoder(nn.Module):
    def __init__(self, device, paras):
        super(Encoder, self).__init__()
        size_inp, size_middle, size_oup, size_state = paras['size_inp'], paras['size_middle'], paras['size_oup'], paras['size_state']
        self.bias = False
        self.tanh_switch = False
        self.tanh = nn.Tanh()
        self.num_layers = 1
        self.size_inp = size_inp
        self.size_middle = size_middle
        self.size_oup = size_oup
        self.device = device

        self.linear_layer = nn.Sequential(nn.Linear(size_inp, size_middle, bias=self.bias),
                                          nn.ReLU(), nn.Linear(size_middle, size_oup, bias=self.bias))
        self.lstm = nn.LSTM(input_size=size_oup, hidden_size=size_oup, num_layers=self.num_layers, batch_first=False)
        self.w_qkv = nn.Parameter(torch.normal(mean=0, std=2, size=(3, size_oup, size_oup)))
        self.norm = nn.LayerNorm(size_state, eps=1e-6, elementwise_affine=False)

    def self_attention(self, w_qkv, inp):
        q = torch.matmul(w_qkv[0], inp)
        k = torch.matmul(w_qkv[1], inp)
        v = torch.matmul(w_qkv[2], inp)

        b = torch.matmul(k.T, q)
        if self.tanh_switch:
            b = self.tanh(b)
        oup = torch.matmul(v, b)
        return oup, q, k, v

    def forward(self, inp):
        # 初始化lstm
        h0 = torch.zeros(self.num_layers, self.size_oup).to(self.device)
        c0 = torch.zeros(self.num_layers, self.size_oup).to(self.device)

        lineared = self.linear_layer(inp)
        oup, _ = self.lstm(lineared, (h0, c0))
        _, q, _, _ = self.self_attention(w_qkv=self.w_qkv, inp=oup.T)
        oup = self.norm(q.T)
        return oup.T


class Decoder(nn.Module):
    def __init__(self, paras):
        super(Decoder, self).__init__()
        size_multi_head, size_state, seq_inp, size_middle, seq_oup = paras['size_multi_head'], paras['size_state'], paras['seq_inp'], paras['size_middle'], paras['seq_oup']
        self.bias = False
        self.tanh_switch = False
        self.softmax_switch = False
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.norm = nn.LayerNorm(size_state, eps=1e-6, elementwise_affine=False)
        self.size_multi_head = size_multi_head
        self.mean_std = torch.tensor([[0.0, 0.0]])

        self.w_qkv = nn.Parameter(torch.normal(mean=0, std=2, size=(3, size_state, size_state)))
        self.multi_head_layers = nn.ModuleList([])
        for _ in range(size_multi_head):
            multi_head_layers_temp = nn.ModuleList([])
            for _ in range(3):
                multi_head_layers_temp.append(nn.Sequential(nn.Linear(seq_inp, size_middle, bias=self.bias),
                                                            nn.ReLU(), nn.Linear(size_middle, seq_inp, bias=self.bias)))
            self.multi_head_layers.append(multi_head_layers_temp)
        self.concat_layers = nn.Sequential(nn.Linear(size_state * size_multi_head, size_middle, bias=self.bias),
                                           nn.ReLU(), nn.Linear(size_middle, size_state, bias=self.bias))
        self.linear_layers_1 = nn.Sequential(nn.Linear(seq_inp, size_middle, bias=self.bias),
                                             nn.ReLU(), nn.Linear(size_middle, seq_oup, bias=self.bias))
        self.linear_layers_2 = nn.Sequential(nn.Linear(2, size_middle, bias=self.bias),
                                             nn.ReLU(), nn.Linear(size_middle, 2, bias=self.bias))

    def self_attention(self, w_qkv, inp):
        q = torch.matmul(w_qkv[0], inp)
        k = torch.matmul(w_qkv[1], inp)
        v = torch.matmul(w_qkv[2], inp)

        b = torch.matmul(k.T, q)
        if self.tanh_switch:
            b = self.tanh(b)
        oup = torch.matmul(v, b)
        return oup, q, k, v

    def cross_attention(self, qkv):
        q, k, v = qkv
        b = torch.matmul(k.T, q)
        if self.tanh_switch:
            b = self.tanh(b)
        oup = torch.matmul(v, b)
        return oup

    def multi_head_attention(self, qkv, size_multi_head, multi_head_layers, concat_layers):
        scaled_qkv = []
        for multi_head in multi_head_layers:
            scaled_qkv_temp = []
            for index_qkv, linear_layers in enumerate(multi_head):
                scaled_qkv_temp.append(linear_layers(qkv[index_qkv]))
            scaled_qkv.append(scaled_qkv_temp)
        scaled_attention = self.dot_production_attention(inp=scaled_qkv, size_multi_head=size_multi_head)
        oup = self.concat(inp=scaled_attention, concat_layers=concat_layers)
        return oup

    def dot_production_attention(self, inp, size_multi_head):
        oup = []
        for i in range(size_multi_head):
            if self.softmax_switch:
                oup.append(torch.matmul(inp[i][2], self.softmax(torch.matmul(inp[i][1].T, inp[i][0]))))
            else:
                oup.append(torch.matmul(inp[i][2], torch.matmul(inp[i][1].T, inp[i][0])))
        return oup

    def concat(self, inp, concat_layers):
        inp = torch.concat(inp, dim=0)
        oup = concat_layers(inp.T)
        return oup.T

    def forward(self, encoded, inp):
        q = encoded
        nonzero_id = torch.nonzero(inp.mean(axis=0))[-1]
        mean = inp.mean(axis=0)[nonzero_id]
        std = torch.sqrt(inp.var(axis=0))[nonzero_id]

        mean_std = self.linear_layers_2(torch.cat([mean, std]).unsqueeze(0)) + torch.cat([mean, std]).unsqueeze(0)
        _, _, k, v = self.self_attention(w_qkv=self.w_qkv, inp=self.norm(inp.T).T)
        attentioned = self.multi_head_attention(qkv=[q, k, v], size_multi_head=self.size_multi_head,
                                                multi_head_layers=self.multi_head_layers, concat_layers=self.concat_layers)
        # attentioned = self.cross_attention(qkv=[q, k, v])
        oup = self.norm(self.linear_layers_1(attentioned).T).T * mean_std[0, 1] + mean_std[0, 0]
        return oup
