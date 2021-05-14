"""
These functions are from https://github.com/LSL000UD/RTDosePrediction
"""

import numpy as np
import random
import cv2
import torch
from scipy import ndimage


def random_scale(list_images,):
    pass


# Random flip
def random_flip_3d(list_images, list_axis=(0, 1, 2), p=0.5):
    if random.random() <= p:
        if 0 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, ::-1, :, :]
        if 1 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, ::-1, :]
        if 2 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, :, ::-1]

    return list_images


def random_elastic_deformation(list_images, p=0.3, spline_order=3, alpha=15, sigma=3):
    if random.random() <= p:
        for image_i in range(len(list_images)):
            shape = list_images[image_i][0].shape
            dz = ndimage.gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha
            dy = ndimage.gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha
            dx = ndimage.gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha

            z_dim, y_dim, x_dim = shape[0], shape[1], shape[2]
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx
            list_images[image_i][0] = ndimage.map_coordinates(list_images[image_i][0],
                                                              indices,
                                                              order=spline_order,
                                                              mode='reflect')

    return list_images


# Random rotation using OpenCV
def random_rotate_around_z_axis(list_images,
                                list_angles,
                                list_interp,
                                list_border_value,
                                p=0.5):
    if random.random() <= p:
        # Randomly pick an angle list_angles
        _angle = random.choice(list_angles)
        # Do not use random scaling, set scale factor to 1
        _scale = 1.

        for image_i in range(len(list_images)):
            for channel_i in range(list_images[image_i].shape[0]):
                for slice_i in range(list_images[image_i].shape[1]):
                    rows, cols = list_images[image_i][channel_i, slice_i, :, :].shape
                    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), _angle, scale=_scale)
                    list_images[image_i][channel_i, slice_i, :, :] = \
                        cv2.warpAffine(list_images[image_i][channel_i, slice_i, :, :],
                                       M,
                                       (cols, rows),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=list_border_value[image_i],
                                       flags=list_interp[image_i])
    return list_images


# Random translation
def random_translate(list_images, p, max_shift=20, list_pad_value=(0, 0, 0)):
    if random.random() <= p:
        ori_z, ori_h, ori_w = list_images[0].shape[1:]  # MR

        bw = max_shift - 1
        ew = ori_w - 1 - max_shift

        for image_i in range(len(list_images)):
            list_images[image_i] = list_images[image_i][:, :, :, bw:ew + 1]

        # Pad to original size
        list_images = random_pad_to_size_3d(list_images,
                                            target_size=[ori_z, ori_h, ori_w],
                                            list_pad_value=list_pad_value)
    return list_images


# To tensor, images should be C*Z*H*W
def to_tensor(list_images):
    for image_i in range(len(list_images)):
        list_images[image_i] = torch.from_numpy(list_images[image_i].copy()).float()
    return list_images


# Pad
def random_pad_to_size_3d(list_images, target_size, list_pad_value=(0, 0)):
    _, ori_z, ori_h, ori_w = list_images[0].shape[:]
    new_z, new_h, new_w = target_size[:]

    pad_z = new_z - ori_z
    pad_h = new_h - ori_h
    pad_w = new_w - ori_w

    pad_z_1 = random.randint(0, pad_z)
    pad_h_1 = random.randint(0, pad_h)
    pad_w_1 = random.randint(0, pad_w)

    pad_z_2 = pad_z - pad_z_1
    pad_h_2 = pad_h - pad_h_1
    pad_w_2 = pad_w - pad_w_1

    output = []
    for image_i in range(len(list_images)):
        _image = list_images[image_i]
        output.append(np.pad(_image,
                             ((0, 0), (pad_z_1, pad_z_2), (pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
                             mode='constant',
                             constant_values=list_pad_value[image_i])
                      )
    return output
