import torch
import torch.nn as nn


class TemperaturePrediction(nn.Module):
    def __init__(self, device, paras):
        super(TemperaturePrediction, self).__init__()
        self.encoder = Encoder(paras=paras['encoder']).to(device)
        self.decoder = Decoder(paras=paras['decoder']).to(device)

        self.seq_predict = paras['seq_predict']
        self.seq_all = torch.zeros([paras['num_measure_point'], paras['seq_all']]).to(device)
        self.mode = ''

    def set_train(self):
        self.mode = 'train'

    def set_test(self):
        self.mode ='test'

    def mode_train(self, inp1, inp2):
        oup = list()
        for seq in range(self.seq_predict):
            self.seq_all[:, 0:(seq+1)] = inp2[:, 0:(seq+1)]
            encoded = self.encoder(inp1)
            decoded = self.decoder(encoded, self.seq_all)
            oup.append(decoded)
        return oup


    def mode_test(self, inp1, inp2):
        return 0.0

    def forward(self, inp1, inp2):
        # 若mode为train，则说明是训练模式，输入数据inp2需包含全过程的温度变化
        # 若mode为test，则说明是测试模式，输入数据inp2只需包含初始温度
        if self.mode == 'train' and inp2.shape[1] != 1:
            oup = self.mode_train(inp1=inp1, inp2=inp2)
        elif self.mode == 'test' and inp2.shape[1] == 1:
            oup = self.mode_test(inp1=inp1, inp2=inp2)
        else:
            print(f'当前模式为【{self.mode}】，但输入数据的尺寸分别为inp1：【{inp1.shape}】，inp2：【{inp2.shape}】')
            raise Exception
        return oup


class Encoder(nn.Module):
    def __init__(self, paras):
        super(Encoder, self).__init__()
        size_inp, size_middle, size_oup = paras['size_inp'], paras['size_middle'], paras['size_oup']
        self.bias = False
        self.linear_layer = nn.Sequential(nn.Linear(size_inp, size_middle, bias=self.bias),
                                          nn.ReLU(), nn.Linear(size_middle, size_middle),
                                          nn.ReLU(), nn.Linear(size_middle, size_oup))

    def forward(self, inp):
        oup = self.linear_layer(inp)
        return oup.T


class Decoder(nn.Module):
    def __init__(self, paras):
        super(Decoder, self).__init__()
        size_multi_head, size_state, seq_inp, size_middle, seq_oup = paras['size_multi_head'], paras['size_state'], paras['seq_inp'], paras['size_middle'], paras['seq_oup']
        self.bias = False
        self.softmax_switch = True
        self.softmax = nn.Softmax()
        self.norm = nn.LayerNorm(seq_oup, eps=1e-6)
        self.size_multi_head = size_multi_head

        self.w_qkv = nn.Parameter(torch.normal(mean=0, std=2, size=(3, size_state, size_state)))
        self.multi_head_layers = nn.ModuleList([])
        for _ in range(size_multi_head):
            multi_head_layers_temp = nn.ModuleList([])
            for _ in range(3):
                multi_head_layers_temp.append(nn.Sequential(nn.ReLU(), nn.Linear(seq_inp, size_middle, bias=self.bias),
                                                            nn.ReLU(), nn.Linear(size_middle, seq_inp, bias=self.bias)))
            self.multi_head_layers.append(multi_head_layers_temp)
        self.concat_layers = nn.Sequential(nn.ReLU(), nn.Linear(size_state * size_multi_head, size_middle, bias=self.bias),
                                           nn.ReLU(), nn.Linear(size_middle, size_state, bias=self.bias))
        self.linear_layers = nn.Sequential(nn.ReLU(), nn.Linear(seq_inp, size_middle, bias=self.bias),
                                           nn.ReLU(), nn.Linear(size_middle, seq_oup, bias=self.bias))

    def self_attention(self, w_qkv, inp):
        q = torch.matmul(w_qkv[0], inp)
        k = torch.matmul(w_qkv[1], inp)
        v = torch.matmul(w_qkv[2], inp)

        b = torch.matmul(k.T, q)
        if self.softmax_switch:
            b = self.softmax(b)
        oup = torch.matmul(v, b)
        return oup, q, k, v

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
        _, _, k, v = self.self_attention(w_qkv=self.w_qkv, inp=inp)
        attentioned = self.multi_head_attention(qkv=[q, k, v], size_multi_head=self.size_multi_head,
                                                multi_head_layers=self.multi_head_layers, concat_layers=self.concat_layers)
        oup = self.linear_layers(attentioned)
        return oup
