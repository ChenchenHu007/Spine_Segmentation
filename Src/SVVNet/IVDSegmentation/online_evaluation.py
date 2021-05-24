# -*- encoding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm
from Loss.SegLoss.DiceLoss import SoftDiceLoss


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = SoftDiceLoss()

    def forward(self, prediction, gt):
        pred_A = prediction  # tensor: (b, num_classes, D, H, W)
        gt_mask = gt[0]
        # gt_A = gt[0]  # tensor: (b, C, D, H, W)
        # gt_B = gt[1]

        loss = self.loss(pred_A, gt_mask)  # negative value

        return loss


def online_evaluation(trainer):

    list_val_loss = []
    val_loss_function = Loss()

    with torch.no_grad():
        trainer.setting.network.eval()

        for batch_idx, case in tqdm(enumerate(trainer.setting.val_loader)):
            input_ = case[0].to(trainer.setting.device)  # tensor: (batch_size, C, D, H, W)
            target = case[1:]
            for target_i in range(len(target)):
                target[target_i] = target[target_i].to(trainer.setting.device)

            pred_Mask = trainer.setting.network(input_)
            val_loss = val_loss_function(pred_Mask, target).cpu().numpy()
            list_val_loss.append(val_loss)

    try:
        trainer.print_log_to_file('===============================================> mean val DSC: %12.12f'
                                  % (np.mean(list_val_loss)), 'a')
    except:
        pass
    # Evaluation score is the lower the better
    return np.mean(list_val_loss)
