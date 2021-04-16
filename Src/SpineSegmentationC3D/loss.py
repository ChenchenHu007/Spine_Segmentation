# -*- encoding: utf-8 -*-
import torch.nn as nn
from Loss import SoftDiceLoss
import torch


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.ce_loss_func = nn.CrossEntropyLoss()  need help
        self.soft_dice = SoftDiceLoss.SoftDiceLoss(num_classes=20, eps=1e-10)

    def forward(self, prediction, gt):
        with torch.no_grad():
            pred_A = prediction[0].cpu().numpy()
            pred_B = prediction[1].cpu().numpy()
            gt_mask = gt[0]

            pred_A_loss = self.soft_dice(pred_A, gt_mask)
            pred_B_loss = self.soft_dice(pred_B, gt_mask)

            L1_loss = 0.5 * pred_A_loss + pred_B_loss

        return L1_loss
