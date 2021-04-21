# -*- encoding: utf-8 -*-
import torch.nn as nn
from Loss.SegLoss.DiceLoss import SoftDiceLoss, DC_and_CE_loss


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        # self.loss = DC_and_CE_loss()
        self.loss = SoftDiceLoss()

    def forward(self, prediction, gt):
        pred_A = prediction[0]
        pred_B = prediction[1]
        gt_mask = gt[0]

        pred_A_loss = self.loss(pred_A, gt_mask)  # negative value
        pred_B_loss = self.loss(pred_B, gt_mask)  # negative value

        loss = 0.5 * pred_A_loss + pred_B_loss

        return loss
