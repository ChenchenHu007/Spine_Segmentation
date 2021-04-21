# -*- encoding: utf-8 -*-
import os
import sys
import argparse

# from tqdm import tqdm
import numpy as np

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

from Evaluate.evaluate import *
from model import *
from NetworkTrainer.network_trainer import *
from utils.tools import one_hot_to_img
from DataLoader.dataloader_3D import val_transform


def read_data(case_dir):
    dict_images = {}
    list_MR_Mask = ['MR']

    for img_name in list_MR_Mask:
        img = case_dir + '/' + img_name + '.nii.gz'
        assert os.path.exists(img)

        if img_name == 'MR':
            dtype = sitk.sitkFloat32

        else:
            dtype = sitk.sitkUInt8

        dict_images[img_name] = sitk.ReadImage(img, dtype)
        dict_images[img_name] = sitk.GetArrayFromImage(dict_images[img_name])[np.newaxis, :, :, :]

    return dict_images


def pre_processing(dict_images):

    MR = dict_images['MR']  # （0, 2500+）HU
    # MR = np.clip(MR, a_min=-1024)
    MR = MR / 1000.  # naive normalization
    # Mask = dict_images['Mask']

    list_images = [MR]

    return list_images


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

        list_prediction_B.append(prediction_B[0, :, :, :])

    return np.mean(list_prediction_B, axis=0)


def inference(trainer, list_case_dirs, save_path, do_TTA=False):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        trainer.setting.network.eval()
        for case_dir in tqdm(list_case_dirs):
            case_id = case_dir.split('/')[-1]

            dict_images = read_data(case_dir)  # {'MR': MR_Image_array}
            list_images = pre_processing(dict_images)  # [MR]

            input_ = list_images[0]  # tensor: (b==1, C, D, H, W)
            # gt_mask = list_images[1]

            # Test-time augmentation
            # if do_TTA:
            #     TTA_mode = [[], ['Z'], ['W'], ['Z', 'W']]
            # else:
            #     TTA_mode = [[]]
            # prediction = test_time_augmentation(trainer, input_, TTA_mode)
            # prediction = one_hot_to_img(prediction)
            [input_] = val_transform([input_])  # [input_] -> [torch.tensor()]
            input_ = input_.unsqueeze(0).to(trainer.setting.device)  # (1, 1, 16, 256, 256)
            [_, prediction_B] = trainer.setting.network(input_)  # tensor: (1, 20, 16, 256, 256)
            prediction_B = np.array(prediction_B.cpu().data[0, :, :, :, :])  # numpy: (20, 16, 256, 256)
            prediction_B = one_hot_to_img(prediction_B)

            # Pose-processing
            # prediction[np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)] = 0
            # prediction = 70. * prediction

            # Save prediction to nii image
            templete_nii = sitk.ReadImage(case_dir + '/MR.nii.gz')
            prediction_nii = sitk.GetImageFromArray(prediction_B)
            prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
            if not os.path.exists(save_path + '/' + case_id):
                os.mkdir(save_path + '/' + case_id)
            sitk.WriteImage(prediction_nii, save_path + '/' + case_id + '/pred_mask.nii.gz')


if __name__ == "__main__":
    if not os.path.exists('../../Data/Spine_Segmentation'):
        raise Exception('Spine_Segmentation should be prepared before testing, please run prepare_3D.py')

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_id', type=int, default=0,
                        help='GPU id used for testing (default: 0)')
    parser.add_argument('--model_path', type=str,
                        default='../../Output/Spine_Segmentation_C3D/best_val_evaluation_index.pkl')
    parser.add_argument('--TTA', type=bool, default=True,
                        help='do test-time augmentation, default True')

    parser.add_argument('--model_type', type=str, default='C3D_base')
    args = parser.parse_args()

    trainer = NetworkTrainer()
    trainer.setting.project_name = 'Spine_Segmentation_C3D'
    trainer.setting.output_dir = '../../Output/Spine_Segmentation_C3D'

    if args.model_type == 'C3D_base':
        trainer.setting.network = Model(in_ch=1, out_ch=20,
                                        list_ch_A=[-1, 16, 32, 64, 128, 256],
                                        list_ch_B=[-1, 32, 64, 128, 256, 512])
    else:
        trainer.setting.network = Model(in_ch=1, out_ch=20,
                                        list_ch_A=[-1, 16, 32, 64, 128, 256],
                                        list_ch_B=[-1, 16, 32, 64, 128, 256])

    # Load model weights
    print(args.model_path)
    trainer.init_trainer(ckpt_file=args.model_path,
                         list_GPU_ids=[args.GPU_id],
                         only_network=True)

    # Start inference
    print('\n\n# Start inference !')
    Spine_Segmentation = '../../Data/Spine_Segmentation'
    cases = sorted(os.listdir(Spine_Segmentation))
    list_case_dirs = [os.path.join(Spine_Segmentation, cases[i]) for i in range(150, 172)]
    inference(trainer, list_case_dirs, save_path=os.path.join(trainer.setting.output_dir, 'Prediction'),
              do_TTA=args.TTA)

    # Evaluation
    print('\n\n# Start evaluation !')
    Dice_score = evaluate_demo(prediction_dir=os.path.join(trainer.setting.output_dir, 'Prediction'),
                               gt_dir=Spine_Segmentation)

    print('\n\nDise score is: ' + str(Dice_score))

