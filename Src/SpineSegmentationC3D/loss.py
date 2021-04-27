# -*- encoding: utf-8 -*-
import torch.nn as nn
from Loss.SegLoss.DiceLoss import SoftDiceLoss, DC_and_CE_loss


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.dc_and_ce = DC_and_CE_loss(soft_dice_kwargs={'batch_dice': False,
                                                          'do_bg': True,
                                                          'smooth': 1.,
                                                          'square': False},
                                        ce_kwargs={})

        # self.loss = SoftDiceLoss()

    def forward(self, prediction, gt):
        pred_A = prediction[0]  # tensor: (b, num_classes, D, H, W)
        pred_B = prediction[1]  # tensor: (b, num_classes, D, H, W)
        gt_mask = gt[0]
        # gt_A = gt[0]  # tensor: (b, C, D, H, W)
        # gt_B = gt[1]

        pred_A_loss = self.dc_and_ce(pred_A, gt_mask)  # negative value
        pred_B_loss = self.dc_and_ce(pred_B, gt_mask)  # negative value

        loss = 0.5 * pred_A_loss + pred_B_loss

        return loss
