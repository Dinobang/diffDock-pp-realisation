import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rpp_dock.diff.transform import Noise
from rpp_dock.utils.geom_utils import score_norm


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.noise_scheduler = Noise()

    def forward(self, data, outputs):
        eps = 4e-3
        t_pred, rot_pred = outputs
        current_t = data.time_steps
        tr, rot = self.noise_scheduler(current_t)

        tr_right = data.tr_score
        rot_right = data.rot_score

        tr = tr.unsqueeze(-1)
        rot_score_norm = score_norm(rot).unsqueeze(-1)

        tr_loss = ((t_pred - tr_right) ** 2 * tr ** 2).mean(dim=1)

        # tr_loss = ((t_pred - tr_right) ** 2).mean(dim=1)
        rot_loss = (((rot_pred - rot_right) / (rot_score_norm + eps)) ** 2).mean(dim=1)

        loss = tr_loss + rot_loss

        return loss.mean()
