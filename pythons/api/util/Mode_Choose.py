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

    def run(self):
        self.model.set_train()
        for dataset_cell in tqdm(self.dataset.values(), desc='Cell', leave=False, ncols=80, disable=False):
            for i in tqdm(range(math.floor(len(dataset_cell) / self.batch_size) + 1), desc='Batch', leave=False, ncols=80, disable=False):
                temperature_prediction = list()
                temperature_reference = list()
                for j in tqdm(range(min(self.batch_size, len(dataset_cell) - i * self.batch_size)), desc='Seq', leave=False, ncols=80):
                    inp1 = dataset_cell[i * self.batch_size + j][1:3]
                    inp2 = dataset_cell[i * self.batch_size + j][3:]
                    temperature_prediction.append(self.model(inp1, inp2))
                    temperature_reference.append(inp2[:, 1:])
                loss_mean, self.loss = self.criterion(pre=temperature_prediction, ref=temperature_reference)
                self.optimizer.zero_grad()
                loss_mean.backward()
                self.optimizer.step()
                self.loss_min = min(self.loss_min, self.loss.min())
                self.loss_max = max(self.loss_max, self.loss.max())
        self.flag_finish = True
