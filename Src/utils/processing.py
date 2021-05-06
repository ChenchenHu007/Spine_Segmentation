import torch
import numpy as np
import math
import cv2
import SimpleITK as sitk


def normalize(img, eps=1e-4):
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    """
    mean = np.mean(img)
    std = np.std(img)

    return (img - mean) / (std + eps)


def resize_3Dimage(img, dsize, mode='nearest'):
    """
    :param img: numpy array, shape of (C, D, H, W)
    """
    interpolation = {'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR, 'area': cv2.INTER_AREA,
                     'cubic': cv2.INTER_CUBIC}
    assert mode in interpolation.keys(), 'only support nearest, linear, area and cubic.'
    if len(img.shape) == 3:
        D, H, W = img.shape
    else:
        _, D, H, W = img.shape

    img_ = np.empty((D, dsize[0], dsize[1]))
    for slice_i in range(D):
        img_[slice_i] = cv2.resize(img[slice_i], dsize=dsize, interpolation=interpolation[mode])

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
    assert len(dsize) == 3 or len(dsize) == 4, 'invalid dsize, only 3D or 4D supported.'
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


def get_sitk_interpolator(interpolator):
    """
    Return an sitk interpolator object for the given string.
    :param interpolator: Interpolator type as string.
                         'nearest': sitk.sitkNearestNeighbor
                         'linear': sitk.sitkLinear
                         'cubic': sitk.sitkBSpline
                         'label_gaussian': sitk.sitkLabelGaussian
                         'gaussian': sitk.sitkGaussian
                         'lanczos': sitk.sitkLanczosWindowedSinc
    :return: The sitk interpolator object.
    """
    if interpolator == 'nearest':
        return sitk.sitkNearestNeighbor
    elif interpolator == 'linear':
        return sitk.sitkLinear
    elif interpolator == 'cubic':
        return sitk.sitkBSpline
    elif interpolator == 'label_gaussian':
        return sitk.sitkLabelGaussian
    elif interpolator == 'gaussian':
        return sitk.sitkGaussian
    elif interpolator == 'lanczos':
        return sitk.sitkLanczosWindowedSinc
    else:
        raise Exception('invalid interpolator type')


# modified from https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html?highlight=resample
def resize_sitk_image(image, dsize=(256, 256, None), interpolator='nearest'):  # note that order in sitk is (x, y, z)
    """
    image:sitk.SimpleITK.Image
    dsize: The order of dsize is similar with sitk
    """
    assert isinstance(image, sitk.SimpleITK.Image)
    interpolator = get_sitk_interpolator(interpolator=interpolator)
    old_size = np.array(image.GetSize())
    old_spacing = np.array(image.GetSpacing())

    if None in dsize:
        dsize = list(dsize)
        index = dsize.index(None)
        dsize[index] = int(old_size[index])

    scale_factor = np.array(dsize) / old_size

    new_spacing = old_spacing / scale_factor

    image = sitk.Resample(image1=image, size=dsize,
                          transform=sitk.Transform(),
                          interpolator=interpolator,
                          outputOrigin=image.GetOrigin(),
                          outputSpacing=new_spacing,
                          outputDirection=image.GetDirection(),
                          defaultPixelValue=0,
                          outputPixelType=image.GetPixelID())

    return image


def resize_to_spacing(image, new_spacing, interpolator='nearest'):
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = [int(old_sp * old_si / new_sp) for old_sp, old_si, new_sp in zip(old_spacing, old_size, new_spacing)]

    interpolator = get_sitk_interpolator(interpolator)
    image = sitk.Resample(image1=image, size=new_size,
                          transform=sitk.Transform(),
                          interpolator=interpolator,
                          outputOrigin=image.GetOrigin(),
                          outputSpacing=new_spacing,
                          outputDirection=image.GetDirection(),
                          defaultPixelValue=0,
                          outputPixelType=image.GetPixelID())

    return image
