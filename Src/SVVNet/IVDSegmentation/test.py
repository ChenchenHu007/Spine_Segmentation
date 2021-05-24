# -*- encoding: utf-8 -*-
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

if os.path.abspath('../..') not in sys.path:
    sys.path.insert(0, os.path.abspath('../..'))

from Evaluate.evaluate import *
from model import *
from NetworkTrainer.network_trainer import *
from DataLoader.dataloader_IVDsegmentation import landmark_extractor
from utils.heatmap_generator import HeatmapGenerator
from utils.tools import csv_to_catalogue
from utils.processing import crop
from post_processing import post_processing


def evaluate_IVDs(prediction_dir, gt_dir):
    """
    This is a demo for calculating the mean dice of all subjects.
    modified from https://www.spinesegmentation-challenge.com/?page_id=34
    """
    dscs = []
    list_case_ids = os.listdir(prediction_dir)
    for case_id in tqdm(list_case_ids):
        pred_mask = sitk.ReadImage(os.path.join(prediction_dir, case_id, 'pred_IVDMask.nii.gz'))
        pred = sitk.GetArrayFromImage(pred_mask)

        gt_mask = sitk.ReadImage(os.path.join(gt_dir, case_id, 'IVDs_512.nii.gz'))
        gt = sitk.GetArrayFromImage(gt_mask)

        dsc = cal_subject_level_dice(pred, gt, num_classes=20)
        dscs.append(dsc)
    return np.mean(dscs)


def crop_to_center(img, landmark=(0, 0, 0), dsize=(12, 128, 128)):
    """
    :param img 4D image with shape (C, D, H, W)
    """
    _, D, H, W = img.shape
    # bz = max(landmark[0] - dsize[0] // 2, 0)
    # ez = min(bz + dsize[0], D)
    pad_h_1, pad_h_2, pad_w_1, pad_w_2 = 0, 0, 0, 0

    bh = landmark[1] - dsize[1] // 2
    eh = bh + dsize[1]
    if bh < 0:
        pad_h_1 = abs(bh)
        bh = 0
    if eh > H:
        pad_h_2 = eh - H
        eh = H

    bw = landmark[2] - dsize[2] // 2
    ew = bw + dsize[2]
    if bw < 0:
        pad_w_1 = abs(bw)
        bw = 0
    if ew > W:
        pad_w_2 = ew - W
        ew = W
    # img = crop(img, bz, ez, axis='z')
    img = crop(img, bh, eh, axis='y')
    img = crop(img, bw, ew, axis='x')
    img = np.pad(img,
                 ((0, 0), (0, 0), (pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
                 mode='constant',
                 constant_values=0)

    return img, [bh, eh, bw, ew], [pad_h_1, pad_h_2, pad_w_1, pad_w_2]


def read_data(case_dir):
    """
    read data from a given path
    """
    dict_images = dict()
    list_files = ['MR_512.nii.gz', 'landmarks_512.csv', ]
    # In fact, there is no Mask during inference, so we cannot load it.

    for file_name in list_files:
        file_path = case_dir + '/' + file_name
        assert os.path.exists(file_path), case_dir + ' does not exist!'

        if file_name.split('.')[-1] == 'csv':
            landmarks = pd.read_csv(file_path)
            dict_images['list_landmarks'] = landmark_extractor(landmarks)
        elif file_name.split('.')[0].split('_')[0] == 'MR':
            dict_images['MR'] = sitk.ReadImage(file_path, sitk.sitkFloat32)
            dict_images['MR'] = sitk.GetArrayFromImage(dict_images['MR'])[np.newaxis, :, :, :]
        elif file_name.split('.')[0].split('_')[0] == 'Mask':
            dict_images['Mask'] = sitk.ReadImage(file_path, sitk.sitkInt16)
            dict_images['Mask'] = sitk.GetArrayFromImage(dict_images['Mask'])[np.newaxis, :, :, :]

    return dict_images


def pre_processing(dict_images):
    MR = dict_images['MR']
    MR = np.clip(MR / 2048, a_min=0, a_max=1)

    list_IVD_landmarks = dict_images['list_landmarks'][10:]

    return [MR, list_IVD_landmarks]


def copy_sitk_imageinfo(image1, image2):
    image2.SetSpacing(image1.GetSpacing())
    image2.SetDirection(image1.GetDirection())
    image2.SetOrigin(image1.GetOrigin())

    return image2


# Input is B*C*Z*H*W
def flip_3d(input_, list_axes):

    input_ = torch.flip(input_, list_axes)

    return input_


def test_time_augmentation(trainer, input_, TTA_mode):
    """
    :param input_: 5D tensor (B, C, D, H, W)
    """
    list_pred = []

    for list_flip_axes in TTA_mode:
        # Do Augmentation before forward
        augmented_input = flip_3d(input_, list_flip_axes)
        pred = trainer.setting.network(augmented_input)  # (B, num_classes, D, H, W) probability distribution

        # Aug back to original order
        pred = flip_3d(pred, list_flip_axes)  # probability distribution
        # print(pred.cpu().numpy().shape)

        list_pred.append(pred)  # probability distribution

    list_pred = torch.stack(list_pred)

    return torch.mean(list_pred, dim=0)


def inference(trainer, list_case_dirs, save_path, do_TTA=False):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if do_TTA:
      TTA_mode = [[], [2], [4], [2, 4]]
    else:
      TTA_mode = [[]]

    with torch.no_grad():
        trainer.setting.network.eval()
        for case_dir in tqdm(list_case_dirs):
            assert os.path.exists(case_dir), case_dir + 'does not exist!'
            case_id = case_dir.split('/')[-1]

            dict_images = read_data(case_dir)
            list_images = pre_processing(dict_images)
            MR = list_images[0]
            # MR = torch.from_numpy(MR)
            list_IVD_landmarks = list_images[1]

            C, D, H, W = MR.shape
            dsize = (12, 64, 96)
            # all pred_IVDMask will be insert into this tensor
            pred_Mask = torch.zeros(C, D, H, W).to(trainer.setting.device)
            heatmap_generator = HeatmapGenerator(image_size=(D, H, W),
                                                 sigma=2.,
                                                 scale_factor=1.,
                                                 normalize=True,
                                                 size_sigma_factor=8,
                                                 sigma_scale_factor=2,
                                                 dtype=np.float32)

            for index, landmark in enumerate(list_IVD_landmarks):
                if True in np.isnan(landmark):
                    continue

                temp = torch.zeros(C, D, H, W).to(trainer.setting.device)
                heatmap = heatmap_generator.generate_heatmap(landmark)[np.newaxis, :, :, :]  # (1, D, H, W)
                # heatmap = torch.from_numpy(heatmap)
                input_ = np.concatenate((MR, heatmap), axis=0)  # (2, D, H, W)

                if D > 12:
                    input_, patch, pad = crop_to_center(input_, landmark=landmark, dsize=dsize)
                    input_ = np.stack((input_[:, :12, :, :], input_[:, -12:, :, :]), axis=0)  # (2, 2, 12, H, W)

                    input_ = torch.from_numpy(input_).to(trainer.setting.device)
                    # pred_IVDMask = trainer.setting.network(input_)  # (2, 2, 12, 128, 128)
                    pred_IVDMask = test_time_augmentation(trainer, input_, TTA_mode)
                    pred_IVDMask = post_processing(pred_IVDMask, D, device=trainer.setting.device)  # (1, 2, D, 128, 128)
                    pred_IVDMask = torch.argmax(pred_IVDMask, dim=1)  # (1, D, 128, 128)

                else:
                    input_, patch, pad = crop_to_center(input_, landmark=landmark, dsize=dsize)
                    input_ = torch.from_numpy(input_).unsqueeze(0).to(trainer.setting.device)
                    # pred_IVDMask = trainer.setting.network(input_)  # (1, 2, 12, 128, 128)
                    pred_IVDMask = test_time_augmentation(trainer, input_, TTA_mode)
                    pred_IVDMask = torch.argmax(pred_IVDMask, dim=1)  # (1, 12, 128, 128)

                bh, eh, bw, ew = patch
                pad_h_1, pad_h_2, pad_w_1, pad_w_2 = pad
                if pad_h_1 > 0:
                    pred_IVDMask = pred_IVDMask[:, :, pad_h_1:, :]
                if pad_h_2 > 0:
                    pred_IVDMask = pred_IVDMask[:, :, :-pad_h_2, :]
                if pad_w_1 > 0:
                    pred_IVDMask = pred_IVDMask[:, :, :, pad_w_1:]
                if pad_w_2 > 0:
                    pred_IVDMask = pred_IVDMask[:, :, :, :-pad_w_2]

                pred_IVDMask = torch.where(pred_IVDMask > 0, index + 11, 0)
                temp[:, :, bh:eh, bw:ew] = pred_IVDMask
                pred_Mask += temp

            pred_Mask = pred_Mask.cpu().numpy()  # (1, 12, 128, 128)

            # Save prediction to nii image
            template_nii = sitk.ReadImage(case_dir + '/MR_512.nii.gz')

            prediction_nii = sitk.GetImageFromArray(pred_Mask[0])
            prediction_nii = copy_sitk_imageinfo(template_nii, prediction_nii)
            if not os.path.exists(save_path + '/' + case_id):
                os.mkdir(save_path + '/' + case_id)
            sitk.WriteImage(prediction_nii, save_path + '/' + case_id + '/pred_IVDMask.nii.gz')


if __name__ == "__main__":
    if not os.path.exists('../../../Data/Spine_Segmentation'):  # this is base dataset
        raise Exception('Spine_Segmentation should be prepared before testing, ' +
                        'please run prepare_3D.py and landmark generation.py')

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_id', type=int, default=0,
                        help='GPU id used for testing (default: 0)')
    parser.add_argument('--model_path', type=str,
                        default='../../../Output/IVD_Segmentation/best_val_evaluation_index.pkl')
    parser.add_argument('--TTA', type=int, default=1,
                        help='do test-time augmentation, default True')

    parser.add_argument('--model_type', type=str, default='Unet_base')
    parser.add_argument('--catalogue', type=int, default=0)
    args = parser.parse_args()

    trainer = NetworkTrainer()
    trainer.setting.project_name = 'IVD_Segmentation'
    trainer.setting.output_dir = '../../../Output/IVD_Segmentation'

    if args.model_type == 'Unet_base':
        trainer.setting.network = Model(in_ch=2, out_ch=2,
                                        list_ch=[-1, 16, 32, 64, 128, 256])
        print('Loading Unet_base !')
    else:
        trainer.setting.network = Model(in_ch=2, out_ch=2,
                                        list_ch=[-1, 32, 64, 128, 256, 512])
        print('Loading Unet_large !')

    # Load model weights
    print(args.model_path)
    trainer.init_trainer(ckpt_file=args.model_path,
                         list_GPU_ids=[args.GPU_id],
                         only_network=True)

    # Start inference
    print('\n\n# Start inference !')

    csv_path = '../../Catalogue' + '/' + str(args.catalogue) + '.csv'
    catalogue = csv_to_catalogue(csv_path)
    path = '../../../Data/Spine_Segmentation'
    cases = catalogue['test'].dropna()
    list_case_dirs = [os.path.join(path, cases[i]) for i in range(len(cases))]

    inference(trainer, list_case_dirs, save_path=os.path.join(trainer.setting.output_dir, 'Prediction'),
              do_TTA=args.TTA)

    # print('\n\n# IVD prediction completed !')
    print('\n\n# Start evaluation !')
    Dice_score = evaluate_IVDs(prediction_dir=os.path.join(trainer.setting.output_dir, 'Prediction'),
                               gt_dir=path)

    print('\n\nDice score is: ' + str(Dice_score))