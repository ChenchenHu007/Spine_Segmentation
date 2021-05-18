# -*- encoding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm


def online_evaluation(trainer):

    list_val_loss = []

    with torch.no_grad():
        trainer.setting.network.eval()

        for batch_idx, case in tqdm(enumerate(trainer.setting.val_loader)):
            input_ = case[0].to(trainer.setting.device)  # tensor: (batch_size, C, D, H, W)
            target = case[1:]
            for target_i in range(len(target)):
                target[target_i] = target[target_i].to(trainer.setting.device)

            pred_spine_heatmap = trainer.setting.network(input_)
            val_loss = trainer.setting.loss_function(pred_spine_heatmap, target)
            val_loss = val_loss.cpu()
            list_val_loss.append(val_loss)

    try:
        trainer.print_log_to_file('===============================================> mean val loss %12.12f'
                                  % (np.mean(list_val_loss)), 'a')
    except:
        pass
    # Evaluation score is the lower the better
    return np.mean(list_val_loss)
