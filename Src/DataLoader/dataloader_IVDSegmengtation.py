import torch.utils.data as data
import os
import SimpleITK as sitk
import numpy as np
import random
import cv2
import pandas as pd
from utils.processing import normalize, crop, crop_center
from utils.heatmap_generator import HeatmapGenerator

from DataAugmentation.augmentation_3D import \
    random_flip_3d, random_rotate_around_z_axis, random_translate, to_tensor


def landmark_extractor(landmarks):
    """
    return a list of the landmarks
    :param landmarks: pandas.Dataframe
    """
    labels = landmarks.columns[1:].tolist()  # exclude the 'axis' column
    list_landmarks = []
    for label in labels:
        list_landmarks.append(np.array(landmarks[label]))

    return list_landmarks


def read_data(case_dir):
    """
    read data from the given path
    """
    dict_images = {}
    list_files = ['MR_512.nii.gz', 'landmarks_512.csv', ]

    for file_name in list_files:
        file_path = case_dir + '/' + file_name
        assert os.path.exists(file_path), case_dir + ' do not exist!'

        if file_name.split('.')[-1] == 'csv':
            landmarks = pd.read_csv(file_path)
            dict_images['list_landmarks'] = landmark_extractor(landmarks)
        else:
            dtype = sitk.sitkFloat32
            dict_images['MR'] = sitk.ReadImage(file_path, dtype)
            dict_images['MR'] = sitk.GetArrayFromImage(dict_images['MR'])[np.newaxis, :, :, :]

    return dict_images


def pre_processing(dict_images):
    MR = dict_images['MR']
    MR = normalize(MR)
    _, D, H, W = MR.shape

    heatmap_generator = HeatmapGenerator(image_size=(D, H, W),
                                         sigma=2.,
                                         scale_factor=1.,
                                         normalize=True,
                                         size_sigma_factor=3,
                                         sigma_scale_factor=1,
                                         dtype=np.float32)
    list_landmarks = dict_images['list_landmarks']
    heatmap = heatmap_generator.generate_heatmaps(list_landmarks)

    if D > 12:
        start = random.choice([i for i in range(D - 12 + 1)])
        MR = crop(MR, start=start, end=start + 12, axis='z')
        heatmap = crop(heatmap, start=start, end=start + 12, axis='z')

    # select a lable in random,if it is none,select again!
    index = random.randint(10, 18)
    while np.isnan(list_landmarks[int(index)][0]):
        index = random.randint(10, 18)

    MR = crop_center(MR, list_landmarks[index], (128, 128))
    heatmap = crop_center(heatmap, list_landmarks[index], (128, 128))# the heatmap here is a tensor of (19, 12, 128,128),but only one is useful.
    heatmap = heatmap[index][np.newaxis, :, :, :]

    return [MR, heatmap]


def train_transform(list_images):
    # list_images = [Input, Label(gt_dose), possible_dose_mask]
    # Random flip
    # list_images = random_flip_3d(list_images, list_axis=(0, 2), p=0.8)

    # Random rotation
    list_images = random_rotate_around_z_axis(list_images,
                                              list_angles=(0, 5, 10, 15, -5, -10, -15),
                                              list_border_value=(0, 0, 0),
                                              list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                              p=0.3)

    list_images = random_translate(list_images,  # [MR, Mask]
                                   p=0.8,
                                   max_shift=20,
                                   list_pad_value=[0, 0, 0])

    # To torch tensor
    list_images = to_tensor(list_images)
    return list_images
