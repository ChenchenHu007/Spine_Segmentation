import torch.utils.data as data
import os
import SimpleITK as sitk
import numpy as np
import random
import cv2
import pandas as pd
from utils.processing import crop
from utils.heatmap_generator import HeatmapGenerator

from DataAugmentation.augmentation_spinelocation import \
    random_rotate_around_z_axis, random_translate, random_elastic_deformation, to_tensor


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
    read data from a given path
    """
    dict_images = dict()
    list_files = ['MR_512.nii.gz', 'landmarks_512.csv', ]

    for file_name in list_files:
        file_path = case_dir + '/' + file_name
        assert os.path.exists(file_path), case_dir + ' does not exist!'

        if file_name.split('.')[-1] == 'csv':
            landmarks = pd.read_csv(file_path)
            dict_images['list_landmarks'] = landmark_extractor(landmarks)
        else:
            dict_images['MR'] = sitk.ReadImage(file_path, sitk.sitkFloat32)
            dict_images['MR'] = sitk.GetArrayFromImage(dict_images['MR'])[np.newaxis, :, :, :]

    return dict_images


def pre_processing(dict_images):
    MR = dict_images['MR']
    MR = np.clip(MR / 2048, a_max=1, a_min=0)
    _, D, H, W = MR.shape

    heatmap_generator = HeatmapGenerator(image_size=(D, H, W),
                                         sigma=2.,
                                         spine_heatmap_sigma=20,
                                         scale_factor=1.,
                                         normalize=True,
                                         size_sigma_factor=6,
                                         sigma_scale_factor=1,
                                         dtype=np.float32)
    spine_heatmap = heatmap_generator.generate_spine_heatmap(list_landmarks=dict_images['list_landmarks'])

    if D > 12:
        start = random.choice([i for i in range(D - 12 + 1)])
        MR = crop(MR, start=start, end=start + 12, axis='z')
        spine_heatmap = crop(spine_heatmap, start=start, end=start + 12, axis='z')

    return [MR, spine_heatmap]


def train_transform(list_images):

    list_images = random_elastic_deformation(list_images, p=0.3)

    # Random rotation
    list_images = random_rotate_around_z_axis(list_images,
                                              list_angles=(0, 5, 10, 15, -5, -10, -15),
                                              list_border_value=(0, 0, 0),
                                              list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                              p=0.3)

    list_images = random_translate(list_images,  # [MR, spine_heatmap]
                                   p=0.8,
                                   max_shift=30)

    # To torch tensor
    list_images = to_tensor(list_images)
    return list_images


def val_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images


class SpineLocationDataset(data.Dataset):
    def __init__(self, catalogue, num_samples_per_epoch, phase, path):

        self.num_samples_per_epoch = num_samples_per_epoch
        self.transform = {'train': train_transform, 'val': val_transform}[phase]

        self.cases = catalogue[phase].dropna()

        self.list_case_id = [os.path.join(path, self.cases[i]) for i in range(len(self.cases))]

        random.shuffle(self.list_case_id)
        self.num_case = len(self.list_case_id)

    def __getitem__(self, index_):
        if index_ <= self.num_case - 1:
            case_id = self.list_case_id[index_]
        else:
            new_index_ = index_ - (index_ // self.num_case) * self.num_case
            case_id = self.list_case_id[new_index_]

        dict_images = read_data(case_id)
        list_images = pre_processing(dict_images)

        list_images = self.transform(list_images)
        return list_images  # [MR, spine_heatmap]

    def __len__(self):
        return self.num_samples_per_epoch


def get_loader(catalogue, batch_size=2,
               num_samples_per_epoch=1, num_works=4,
               phase='train',
               path='../../../Data/Spine_Segmentation'):
    dataset = SpineLocationDataset(catalogue=catalogue,
                                   num_samples_per_epoch=num_samples_per_epoch,
                                   phase=phase, path=path)

    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_works,
                             pin_memory=True)

    return loader
