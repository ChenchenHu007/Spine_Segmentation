import torch


def post_processing(pred_spine_heatmap, target_D, device):
    assert pred_spine_heatmap.dim() == 5, 'should be 5D tensor'
    batch_size, _, d, h, w = pred_spine_heatmap.shape
    # td, th, tw = target.shape[-3:]
    td = target_D
    th = h
    tw = w

    pred = torch.zeros(1, 2, td, th, tw).to(device)

    pred_a = pred_spine_heatmap[0]
    pred_b = pred_spine_heatmap[1]
    pred[:, :, :d, :, :] += pred_a
    pred[:, :, td - d:, :, :] += pred_b
    pred[:, :, td - d:d, :, :] /= 2.

    return pred