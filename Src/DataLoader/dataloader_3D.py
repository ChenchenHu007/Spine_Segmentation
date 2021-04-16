import torch.utils.data as data
import os
import SimpleITK as sitk
import numpy as np
import random
import cv2

from DataAugmentation.augmentation_3D import \
    random_flip_3d, random_rotate_around_z_axis, random_translate, to_tensor


def read_data(case_dir):
    dict_images = {}
    list_MR_Mask = ['MR', 'Mask']

    for img_name in list_MR_Mask:
        img = case_dir + '/' + img_name + '.nii.gz'
        assert os.path.exists(img)

        if img_name == 'MR':
            dtype = sitk.sitkFloat32

        else:
            dtype = sitk.sitkUInt8

        dict_images[img_name] = sitk.ReadImage(img, dtype)

    return dict_images


def pre_processing(dict_images):

    MR = dict_images['MR']  # （0, 2500+）HU
    # MR = np.clip(MR, a_min=-1024)
    MR = MR / 1000.  # naive normalization
    Mask = dict_images['Mask']

    list_images = [MR, Mask]

    return list_images


def train_transform(list_images):
    # list_images = [Input, Label(gt_dose), possible_dose_mask]
    # Random flip
    list_images = random_flip_3d(list_images, list_axis=(0, 2), p=0.8)

    # Random rotation
    list_images = random_rotate_around_z_axis(list_images,
                                              list_angles=(0, 40, 80, 120, 160, 200, 240, 280, 320),
                                              list_boder_value=(0, 0, 0),
                                              list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                              p=0.3)

    # Random translation, but make use the region can receive dose is remained
    list_images = random_translate(list_images,  # [MR, Mask]
                                   mask=list_images[1][0, :, :, :],  # Mask
                                   p=0.8,
                                   max_shift=20,
                                   list_pad_value=[0, 0, 0])

    # To torch tensor
    list_images = to_tensor(list_images)
    return list_images


def val_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images


class SpineDataset(data.Dataset):
    def __init__(self, num_samples_per_epoch, phase):
        self.phase = phase
        self.num_samples_per_epoch = num_samples_per_epoch
        self.transform = {'train': train_transform, 'val': val_transform}

        Spine_Segmentation = '../../Data/Spine_Segmentation'
        cases = sorted(os.listdir(Spine_Segmentation))

        self.list_case_id = {'train': [os.path.join(Spine_Segmentation, cases[i]) for i in range(0, 120)],
                             'val': [os.path.join(Spine_Segmentation, cases[i]) for i in range(120, 151)]}[phase]

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

        list_images = self.transform[self.phase](list_images)
        return list_images

    def __len__(self):
        return self.num_samples_per_epoch


def get_loader(train_bs=1, val_bs=1, train_num_samples_per_epoch=1, val_num_samples_per_epoch=1, num_works=2):
    train_dataset = SpineDataset(num_samples_per_epoch=train_num_samples_per_epoch, phase='train')
    val_dataset = SpineDataset(num_samples_per_epoch=val_num_samples_per_epoch, phase='val')

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_works,
                                   pin_memory=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works,
                                 pin_memory=True)

    return train_loader, val_loader

