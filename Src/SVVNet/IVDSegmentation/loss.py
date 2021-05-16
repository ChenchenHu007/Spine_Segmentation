import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, prediction, gt):
        spine_heatmap_gt = gt[0]

        loss = self.loss(prediction, spine_heatmap_gt)

        return loss
