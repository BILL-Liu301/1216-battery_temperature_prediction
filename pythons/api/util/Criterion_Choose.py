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
