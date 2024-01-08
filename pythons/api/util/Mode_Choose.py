import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from threading import Thread

from pythons.api.util.plots import plot_during_train


class ModeTrain(Thread):
    def __init__(self, model, batch_size, dataset, criterion, optimizer, scheduler,
                 epoch_max, sequence_predict, num_measure_point,
                 path_pts, path_pt_best, path_figs):
        Thread.__init__(self)
        self.model = model
        self.batch_size = batch_size
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_state_dict = scheduler.state_dict()
        self.epoch_max = epoch_max
        self.flag_finish = False

        self.path_pts = path_pts
        self.path_pt_best = path_pt_best
        self.path_figs = path_figs

        self.loss = np.array([0])
        self.loss_min, self.loss_max = 1e10, 0
        self.loss_all = np.zeros([epoch_max, 2])
        self.lr_all = np.zeros([epoch_max, 1])
        self.epoch_now = 0
        self.point_now = 0

        self.grad_max_origin, self.grad_max_limit = 0.0, 0.0
        self.grad_max_name_origin, self.grad_max_name_limit = '', ''
        self.sequence_predict = sequence_predict
        self.num_measure_point = num_measure_point
        self.judge = 0.0

    def read_grad_max(self, model, prefix):
        grad_max, grad_max_name = 0.0, ''
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    if param.grad.abs().max() > grad_max or torch.isnan(param.grad.abs().max()):
                        grad_max = param.grad.abs().max()
                        grad_max_name = prefix + '-' + name
        return [grad_max, grad_max_name]

    def cal_grad(self):
        grad_max, grad_max_name = 0.0, ''
        grad_max_1, grad_max_name_1 = self.read_grad_max(self.model.encoder, 'encoder')
        grad_max_2, grad_max_name_2 = self.read_grad_max(self.model.decoder, 'decoder')
        if grad_max_1 >= grad_max_2 or torch.isnan(grad_max_1):
            grad_max = grad_max_1
            grad_max_name = grad_max_name_1
        if grad_max_2 > grad_max_1 or torch.isnan(grad_max_2):
            grad_max = grad_max_2
            grad_max_name = grad_max_name_2
        return grad_max, grad_max_name

    def shuffle(self, dataset_cell):
        # Fisher-Yates算法
        for i in range(len(dataset_cell)-1, 0, -1):
            j = random.randint(0, i)
            dataset_cell[i], dataset_cell[j] = dataset_cell[j], dataset_cell[i]

    def run(self):
        self.model.set_train()
        for point in tqdm(range(0, self.sequence_predict, 12), desc='Points', leave=False, ncols=100, disable=False):
            self.judge = (1 - 0.1) / (self.sequence_predict - 1) * point + 0.1
            self.point_now = point
            self.epoch_now = 0
            self.scheduler.load_state_dict(self.scheduler_state_dict)
            for epoch in tqdm(range(self.epoch_max), desc='Epoch', leave=False, ncols=100, disable=False):
                self.epoch_now = epoch
                self.loss_min, self.loss_max = 1e10, 0
                for dataset_cell in tqdm(self.dataset.values(), desc='Cell', leave=False, ncols=100, disable=False):
                    # 打乱数据集
                    self.shuffle(dataset_cell)
                    for batch in tqdm(range(math.floor(len(dataset_cell) / self.batch_size) + 1), desc='Batch', leave=False, ncols=100, disable=False):
                        temperature_prediction = list()
                        temperature_reference = list()
                        # 训练主过程
                        for group in tqdm(range(min(self.batch_size, len(dataset_cell) - batch * self.batch_size)), desc='Group', leave=False, ncols=100):
                            inp1 = dataset_cell[batch * self.batch_size + group][0:3, 0:(point + 2)]
                            inp2 = dataset_cell[batch * self.batch_size + group][3:, 0:1]
                            if torch.isnan(inp1).any() or torch.isnan(inp2).any():
                                print(f'inp1: {torch.isnan(inp1).any()}, inp2: {torch.isnan(inp1).any()}')
                                raise Exception
                            temperature_prediction.append(self.model(inp1, inp2, inp1.shape[1]-1))
                            temperature_reference.append(dataset_cell[batch * self.batch_size + group][3:, 1:(point + 2)])
                        loss_mean, self.loss = self.criterion(pre=temperature_prediction, ref=temperature_reference)
                        self.optimizer.zero_grad()
                        loss_mean.backward()
                        self.grad_max_origin, self.grad_max_name_origin = self.cal_grad()
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=1)
                        # nn.utils.clip_grad_value_(self.model.parameters(), clip_value=10)
                        self.grad_max_limit, self.grad_max_name_limit = self.cal_grad()
                        self.optimizer.step()
                        self.loss_min = min(self.loss_min, self.loss.min())
                        self.loss_max = max(self.loss_max, self.loss.max())

                self.loss_all[epoch, :] = np.array([self.loss.min(), self.loss.max()])
                self.lr_all[epoch, 0] = self.scheduler.get_last_lr()[0]
                self.scheduler.step()
                torch.save(self.model, os.path.join(self.path_pts, f'{point + 1}_{epoch + 1}_{self.loss_all[epoch, 1]:.4f}.pt'))
                if self.loss_all[epoch, 1] == self.loss_all[0:(epoch + 1), 1].min():
                    torch.save(self.model, self.path_pt_best)
                plot_during_train(epoch, self.loss_all, self.lr_all)
                plt.savefig(f'{self.path_figs}/train.png')
                # 判断是否提前结束
                # 若最后一个batch的最后一个group的loss max小于某个值，进入下一个点
                if self.loss.max() <= self.judge:
                    break

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
            seq_id = 0
            for seq in tqdm(dataset_cell, desc='Group', leave=False, ncols=100, disable=False):
                if seq_id % seq.shape[1] != 0:
                    seq_id += 1
                    continue
                seq_id += 1
                inp1 = seq[0:3]
                inp2 = seq[3:, 0:1]
                temperature_prediction_temp.append(self.model(inp1, inp2, inp1.shape[1]-1))
            self.temperature_prediction.update({dataset_cell_key: temperature_prediction_temp})
        self.loss = self.criterion(pre=self.temperature_prediction, ref=self.dataset)
