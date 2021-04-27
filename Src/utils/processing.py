import numpy
import torch
import numpy as np
import math
import cv2


def normalize(img, eps=1e-4):
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    mean and std parameter have to be provided explicitly.
    """
    mean = np.mean(img)
    std = np.std(img)

    return (img - mean) / (std + eps)


def resize_image(img, dsize):
    """
    :param img: numpy array, shape of (C, D, H, W)
    """
    if len(img.shape) == 3:
        D, H, W = img.shape
    else:
        _, D, H, W = img.shape

    img_ = np.empty(img.shape)
    for slice_i in range(D):
        img_[slice_i] = cv2.resize(img[slice_i], dsize=dsize, interpolation=cv2.INTER_NEAREST)

    return img_


# Crop
def crop(img, start, end, axis='x'):
    """
    crop an image along with the given axis
    :param img: torch or numpy image whose shape must be C*D*H*W or C*Z*Y*X,
    :param start: the index where cropping starts
    :param end: the index where cropping ends
    :param axis: which axis cropping along with
    """
    assert axis.lower() in ['z', 'y', 'x', 'd', 'h', 'w'], str(axis) + 'is not in (D, H, W) or (z, y, x) !'

    if axis.lower() in ['z', 'd']:
        img = img[:, start:end, :, :]
    elif axis.lower() in ['h', 'y']:
        img = img[:, :, start:end, :]
    else:
        img = img[:, :, :, start:end]

    return img


def pad_to_size(img, dsize):
    """
    modified from
    https://github.com/pangshumao/SpineParseNet/blob/e069246e4e430d6e5bc73112f9eaedbde0555f6c/test_coarse.py
    :param img: (C, D, H, W) or (C, z, y, x)
    :param dsize: 3D (D, H, W) or 4D (C, D, H, W)
    """
    assert len(dsize) == 3 or len(dsize) == 4, 'invalid dsize, only 3D or 4D'
    _, ori_z, ori_h, ori_w = img.shape

    if len(dsize) == 3:
        dz, dh, dw = dsize
    else:
        _, dz, dh, dw = dsize

    pad_z = dz - ori_z
    pad_h = dh - ori_h
    pad_w = dw - ori_w
    assert pad_z >= 0 and pad_h >= 0 and pad_w >= 0, str(img.shape) + ', but ' + str(dsize)

    before_z = int(math.ceil(pad_z / 2.))
    after_z = int(pad_z - before_z)

    before_h = int(math.ceil(pad_h / 2.))
    after_h = int(pad_h - before_h)

    before_w = int(math.ceil(pad_w / 2.))
    after_w = int(pad_w - before_w)

    assert isinstance(img, np.ndarray) or isinstance(img, torch.Tensor), 'wrong type of img'
    if isinstance(img, np.ndarray):
        img = np.pad(img,
                     pad_width=((0, 0), (before_z, after_z), (before_h, after_h), (before_w, after_w)),
                     mode='constant',
                     constant_values=0)
    else:
        img = torch.nn.functional.pad(img,
                                      pad=(before_z, after_z, before_h, after_h, before_w, after_w)[::-1],
                                      # the order of pad is (left_x, right_x, top_y, bottom_y, front_z, back_z)
                                      mode='constant',
                                      value=0)

    return img


def remove_padding_z(img, target_z):
    """
    modified from
    https://github.com/pangshumao/SpineParseNet/blob/e069246e4e430d6e5bc73112f9eaedbde0555f6c/test_coarse.py
    :param img: (C, D, H, W) or (C, z, y, x)

    """

    _, z, h, w = img.shape
    num_paddings = abs(target_z - z)
    start_index = int(math.ceil(num_paddings / 2.))
    end_index = int(num_paddings - start_index)
    if end_index == 0:
        img = crop(img, start_index, z, axis='z')
    else:
        img = crop(img, start_index, -end_index, axis='z')

    return img

