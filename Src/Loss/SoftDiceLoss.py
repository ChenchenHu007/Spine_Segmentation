from torch import nn
import numpy as np
import torch

# modified from https://www.spinesegmentation-challenge.com/?page_id=34
class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes=20, eps=1e-10):
        super(SoftDiceLoss, self).__init__()

        self.num_classes = num_classes
        self.eps = eps

    def forward(self, prediction, target):
        with torch.no_grad():
            prediction = prediction.cpu().numpy()
            target = target.cpu().numpy()

        empty_value = -1.0
        dscs = empty_value * np.ones((self.num_classes,), dtype=np.float32)
        for i in range(0, self.num_classes):
            if i not in target and i not in prediction:
                continue
            target_per_class = np.where(target == i, 1, 0).astype(np.float32)
            prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

            tp = np.sum(prediction_per_class * target_per_class)
            fp = np.sum(prediction_per_class) - tp
            fn = np.sum(target_per_class) - tp
            dsc = 2 * tp / (2 * tp + fp + fn + self.eps)
            dscs[i] = dsc
        dscs = np.where(dscs == -1.0, np.nan, dscs)
        subject_level_dice = np.nanmean(dscs[1:])  # class 0 is excluded
        return torch.tensor(np.asarray(subject_level_dice)).to('cuda:0')
