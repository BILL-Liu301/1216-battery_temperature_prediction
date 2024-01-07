import math
import numpy as np
from tqdm import tqdm
from threading import Thread


class ModeTrain(Thread):
    def __init__(self, model, batch_size, dataset, criterion, optimizer):
        Thread.__init__(self)
        self.model = model
        self.batch_size = batch_size
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.flag_finish = False
        self.loss = np.array([0])
        self.loss_min, self.loss_max = 1e10, 0
        self.grad_max = 0.0
        self.grad_max_name = ''

    def read_grad_max(self, model, prefix):
        grad_max, grad_max_name = 0.0, ''
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_max = max(grad_max, param.grad.abs().max())
                    grad_max_name = prefix + '-' + name
        return grad_max, grad_max_name

    def run(self):
        self.model.set_train()
        for dataset_cell in tqdm(self.dataset.values(), desc='Cell', leave=False, ncols=100, disable=False):
            for i in tqdm(range(math.floor(len(dataset_cell) / self.batch_size) + 1), desc='Batch', leave=False, ncols=100, disable=False):
                temperature_prediction = list()
                temperature_reference = list()
                # 训练主过程
                for j in tqdm(range(min(self.batch_size, len(dataset_cell) - i * self.batch_size)), desc='Group', leave=False, ncols=100):
                    inp1 = dataset_cell[i * self.batch_size + j][0:3]
                    inp2 = dataset_cell[i * self.batch_size + j][3:]
                    temperature_prediction.append(self.model(inp1, inp2))
                    temperature_reference.append(inp2[:, 1:])
                loss_mean, self.loss = self.criterion(pre=temperature_prediction, ref=temperature_reference)
                self.optimizer.zero_grad()
                loss_mean.backward()
                self.optimizer.step()
                self.loss_min = min(self.loss_min, self.loss.min())
                self.loss_max = max(self.loss_max, self.loss.max())

                # 梯度检测
                self.grad_max, self.grad_max_name = 0.0, ''
                self.grad_max, self.grad_max_name = self.read_grad_max(self.model.encoder, 'encoder')
                self.grad_max, self.grad_max_name = self.read_grad_max(self.model.decoder, 'decoder')
        self.flag_finish = True


class ModeTest:
    def __init__(self, model, criterion, dataset):
        self.model = model
        self.criterion = criterion
        self.dataset = dataset
        self.loss = None
        self.temperature_prediction = None

    def run(self):
        self.model.set_test()
        self.temperature_prediction = dict()
        for dataset_cell_key, dataset_cell in tqdm(self.dataset.items(), desc='Cell', leave=False, ncols=100, disable=False):
            temperature_prediction_temp = list()
            for seq in tqdm(dataset_cell, desc='Group', leave=False, ncols=100, disable=False):
                inp1 = seq[0:3]
                inp2 = seq[3:, 0:1]
                temperature_prediction_temp.append(self.model(inp1, inp2))
            self.temperature_prediction.update({dataset_cell_key: temperature_prediction_temp})
        self.loss = self.criterion(pre=self.temperature_prediction, ref=self.dataset)
