"""
This function modified from https://github.com/pangshumao/SpineParseNet/blob/master/augment/transforms.py

"""

import torch
import numpy as np
import pandas as pd


def expand_as_one_hot(input_, num_channels, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW or NxHxW label image to NxCxHxW, where each label gets converted to
    its corresponding one-hot vector
    :param input_: 4D input image (NxDxHxW) or 3D input image (NxHxW)
    :param num_channels: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW) or 4D output image (NxCxHxW)
    """
    assert input_.dim() in [3, 4]

    shape = input_.size()
    shape = list(shape)
    shape.insert(1, num_channels)
    shape = tuple(shape)

    # expand the input tensor to 1xNxDxHxW
    # index = input.unsqueeze(0)

    # expand the input tensor to Nx1xDxHxW
    index = input_.unsqueeze(1)

    if ignore_index is not None:
        # create ignore_index mask for the result
        expanded_index = index.expand(shape)
        mask = expanded_index == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        index = index.clone()
        index[index == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input_.device).scatter_(1, index, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input_.device).scatter_(1, index, 1)


def csv_to_catalogue(csv_path):
    catalogue = pd.read_csv(csv_path)

    return catalogue
