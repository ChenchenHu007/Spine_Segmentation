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
from loss import Loss


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


# Input is C*Z*H*W
def flip_3d(input_, list_axes):
    if 'Z' in list_axes:
        input_ = input_[:, ::-1, :, :]
    if 'W' in list_axes:
        input_ = input_[:, :, :, ::-1]

    return input_


def test_time_augmentation(trainer, input_, TTA_mode):
    list_prediction_B = []

    for list_flip_axes in TTA_mode:
        # Do Augmentation before forward
        augmented_input = flip_3d(input_.copy(), list_flip_axes)
        augmented_input = torch.from_numpy(augmented_input.astype(np.float32))
        augmented_input = augmented_input.unsqueeze(0).to(trainer.setting.device)
        [_, prediction_B] = trainer.setting.network(augmented_input)

        # Aug back to original order
        prediction_B = flip_3d(np.array(prediction_B.cpu().data[0, :, :, :, :]), list_flip_axes)
        # numpy: (num_classes, D, H, W)

        list_prediction_B.append(prediction_B)

    return np.mean(list_prediction_B, axis=0)


def inference(trainer, list_case_dirs, save_path, do_TTA=False):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dict_loss = dict()
    loss_function = Loss()
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

            for label, landmark in enumerate(list_IVD_landmarks):
                if True in np.isnan(landmark):
                    continue

                heatmap = heatmap_generator.generate_heatmap(landmark)[np.newaxis, :, :, :]  # (1, D, H, W)
                # heatmap = torch.from_numpy(heatmap)
                input_ = np.concatenate((MR, heatmap), axis=0)  # (2, D, H, W)

                if D > 12:
                    input_, patch, pad = crop_to_center(input_, landmark=landmark, dsize=dsize)
                    input_ = np.stack((input_[:, :12, :, :], input_[:, -12:, :, :]), axis=0)  # (2, 2, 12, H, W)

                    input_ = torch.from_numpy(input_).to(trainer.setting.device)
                    pred_IVDMask = trainer.setting.network(input_)  # (2, 2, 12, 128, 128)
                    pred_IVDMask = post_processing(pred_IVDMask, D, device=trainer.setting.device)  # (1, 2, D, 128, 128)
                    pred_IVDMask = torch.argmax(pred_IVDMask, dim=1)  # (1, D, 128, 128)

                else:
                    input_, patch, pad = crop_to_center(input_, landmark=landmark, dsize=dsize)
                    input_ = torch.from_numpy(input_).unsqueeze(0).to(trainer.setting.device)
                    pred_IVDMask = trainer.setting.network(input_)  # (1, 2, 12, 128, 128)
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

                pred_IVDMask = torch.where(pred_IVDMask > 0, label + 11, 0)
                pred_Mask[:, :, bh:eh, bw:ew] = pred_IVDMask

            # FIXME TTA, only elastic_deformation. rotate_around_z_axis, translate are complex
            # Test-time augmentation
            # if do_TTA:
            #     TTA_mode = [[], ['Z'], ['W'], ['Z', 'W']]
            # else:
            #     TTA_mode = [[]]
            # prediction = test_time_augmentation(trainer, input_, TTA_mode)
            # prediction = one_hot_to_img(prediction)

            # dict_loss[case_id] = loss_function(pred_Mask, [target]).cpu()
            pred_Mask = pred_Mask.cpu().numpy()

            # Save prediction to nii image
            templete_nii = sitk.ReadImage(case_dir + '/MR_512.nii.gz')

            prediction_nii = sitk.GetImageFromArray(pred_Mask[0])
            prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
            if not os.path.exists(save_path + '/' + case_id):
                os.mkdir(save_path + '/' + case_id)
            sitk.WriteImage(prediction_nii, save_path + '/' + case_id + '/pred_IVDMask.nii.gz')
    return dict_loss


if __name__ == "__main__":
    if not os.path.exists('../../../Data/Spine_Segmentation'):  # this is base dataset
        raise Exception('Spine_Segmentation should be prepared before testing, ' +
                        'please run prepare_3D.py and landmark generation.py')

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_id', type=int, default=0,
                        help='GPU id used for testing (default: 0)')
    parser.add_argument('--model_path', type=str,
                        default='../../../Output/IVD_Segmentation/best_val_evaluation_index.pkl')
    parser.add_argument('--TTA', type=bool, default=True,
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

    dict_loss = inference(trainer, list_case_dirs, save_path=os.path.join(trainer.setting.output_dir, 'Prediction'),
                          do_TTA=args.TTA)

    """
    Owing to the incomplete prediction, evaluation only for IVD is useless
    Official evaluation function, cal_subject_level_dice needed 
    after assembling the IVD, Vertebrae and Coccyx Segmentation
    """
    print('\n\n# IVD prediction completed !')
    # print('\n\n# Start evaluation !')
    # for key, value in dict_loss.items():
    #     print(key + ': %12.12f' % value)
    #
    # print('\n\nmean loss is: ' + str(np.mean(list(dict_loss.values()))))
