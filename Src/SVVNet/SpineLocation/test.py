# -*- encoding: utf-8 -*-
import os
import sys
import argparse
import pandas as pd

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

if os.path.abspath('../..') not in sys.path:
    sys.path.insert(0, os.path.abspath('../..'))

from Evaluate.evaluate import *
from model import *
from NetworkTrainer.network_trainer import *
from DataLoader.dataloader_spinelocation import val_transform
from utils.heatmap_generator import HeatmapGenerator
from utils.tools import csv_to_catalogue
from post_processing import post_processing
from loss import Loss


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
    dict_images = dict()
    list_files = ['MR_512.nii.gz', 'landmarks_512.csv']

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
    MR = np.clip(MR / 2048, a_min=0, a_max=1)
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

    return [MR, spine_heatmap]


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
            list_images = pre_processing(dict_images)  # [MR]
            list_images = val_transform(list_images)

            C, D, H, W = list_images[0].shape
            target = list_images[1].unsqueeze(0).to(trainer.setting.device)
            if D > 12:
                input_ = torch.stack((list_images[0][:, :12, :, :], list_images[0][:, -12:, :, :]), dim=0) \
                    .to(trainer.setting.device)
                pred_spine_heatmap = trainer.setting.network(input_)
                pred_spine_heatmap = post_processing(pred_spine_heatmap, target, device=trainer.setting.device)

            else:
                input_ = list_images[0].unsqueeze(0).to(trainer.setting.device)
                pred_spine_heatmap = trainer.setting.network(input_)



            # FIXME TTA
            # Test-time augmentation
            # if do_TTA:
            #     TTA_mode = [[], ['Z'], ['W'], ['Z', 'W']]
            # else:
            #     TTA_mode = [[]]
            # prediction = test_time_augmentation(trainer, input_, TTA_mode)
            # prediction = one_hot_to_img(prediction)

            dict_loss[case_id] = loss_function(pred_spine_heatmap, [target]).cpu()
            pred_spine_heatmap = pred_spine_heatmap.cpu()

            # Save prediction to nii image
            templete_nii = sitk.ReadImage(case_dir + '/MR_512.nii.gz')

            prediction_nii = sitk.GetImageFromArray(pred_spine_heatmap[0][0])
            prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
            if not os.path.exists(save_path + '/' + case_id):
                os.mkdir(save_path + '/' + case_id)
            sitk.WriteImage(prediction_nii, save_path + '/' + case_id + '/pred_mask.nii.gz')
    return dict_loss


if __name__ == "__main__":
    if not os.path.exists('../../../Data/Spine_Segmentation'):
        raise Exception('Spine_Segmentation should be prepared before testing, please run prepare_3D.py')

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_id', type=int, default=0,
                        help='GPU id used for testing (default: 0)')
    parser.add_argument('--model_path', type=str,
                        default='../../../Output/Spine_Location/best_val_evaluation_index.pkl')
    parser.add_argument('--TTA', type=bool, default=True,
                        help='do test-time augmentation, default True')

    parser.add_argument('--model_type', type=str, default='C3D_base')
    parser.add_argument('--catalogue', type=int, default=0)
    args = parser.parse_args()

    trainer = NetworkTrainer()
    trainer.setting.project_name = 'Spine_Location'
    trainer.setting.output_dir = '../../../Output/Spine_Location'

    if args.model_type == 'C3D_base':
        trainer.setting.network = Model(in_ch=1, out_ch=1,
                                        list_ch=[-1, 16, 32, 64, 128, 256])
        print('Loading Unet_base !')
    else:
        trainer.setting.network = Model(in_ch=1, out_ch=1,
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

    # Evaluation
    print('\n\n# Start evaluation !')
    for key, value in dict_loss.items():
        print(key + ': %12.12f' % value)

    print('\n\nmean loss is: ' + str(np.mean(dict_loss.values())))
