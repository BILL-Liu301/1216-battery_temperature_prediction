import numpy as np
import torch
import torch.nn as nn


class CriterionTrain(nn.Module):
    def __init__(self):
        super(CriterionTrain, self).__init__()
        self.l1loss = nn.L1Loss(reduction='mean')
        self.l2loss = nn.MSELoss(reduction='mean')

    def forward(self, pre, ref):
        loss = list()
        for i in range(len(pre)):
            loss.append(self.l2loss(pre[i], ref[i]).unsqueeze(0))
        loss = torch.cat(loss)
        return torch.mean(loss), np.asarray(loss.cpu().detach().numpy())


class CriterionTest(nn.Module):
    def __init__(self):
        super(CriterionTest, self).__init__()
        self.l1loss = nn.L1Loss(reduction='mean')
        self.l2loss = nn.MSELoss(reduction='mean')

    def forward(self, pre, ref):
        loss = dict()
        for cell in pre:
            pre_list, ref_list = pre[cell], ref[cell]
            loss_temp = list()
            for i in range(len(pre_list)):
                loss_temp.append(self.l1loss(pre_list[i][3:, 1:], ref_list[i][3:, 1:]).unsqueeze(0))
            loss.update({cell: torch.cat(loss_temp).cpu().numpy()})
        return loss
