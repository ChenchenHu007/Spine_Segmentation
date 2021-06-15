# -*- encoding: utf-8 -*-
import os
import sys
import argparse
import pandas as pd
from scipy import ndimage
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

    with torch.no_grad():
        trainer.setting.network.eval()
        for case_dir in tqdm(list_case_dirs):
            assert os.path.exists(case_dir), case_dir + 'does not exist!'
            case_id = case_dir.split('/')[-1]

            dict_images = read_data(case_dir)
            list_images = pre_processing(dict_images)  # [MR]
            list_images = val_transform(list_images)

            input_ = list_images[0]
            C, D, H, W = input_.shape
            target = list_images[1].unsqueeze(0).to(trainer.setting.device)
            if D > 12:
                input_ = torch.stack((input_[:, :12, :, :], input_[:, -12:, :, :]), dim=0).to(trainer.setting.device)
                pred_heatmap = trainer.setting.network(input_)
                pred_heatmap = post_processing(pred_heatmap, target, device=trainer.setting.device)
            else:
                input_ = list_images[0].unsqueeze(0).to(trainer.setting.device)
                pred_heatmap = trainer.setting.network(input_)

            # Test-time augmentation
            # if do_TTA:
            #     TTA_mode = [[], ['Z'], ['W'], ['Z', 'W']]
            # else:
            #     TTA_mode = [[]]
            # prediction = test_time_augmentation(trainer, input_, TTA_mode)
            # prediction = one_hot_to_img(prediction)

            pred_heatmap = pred_heatmap.cpu().numpy()
            # pred_heatmap = np.where(pred_heatmap > 0, pred_heatmap, 0)  # final pred

            # Save prediction to nii image
            template_nii = sitk.ReadImage(case_dir + '/MR_512.nii.gz')

            prediction_nii = sitk.GetImageFromArray(pred_heatmap[0][0])
            prediction_nii = copy_sitk_imageinfo(template_nii, prediction_nii)
            if not os.path.exists(save_path + '/' + case_id):
                os.mkdir(save_path + '/' + case_id)
            sitk.WriteImage(prediction_nii, save_path + '/' + case_id + '/pred_heatmap.nii.gz')


def evaluate(prediction_dir, gt_dir):

    list_errors = []
    list_case_ids = os.listdir(prediction_dir)
    for case_id in list_case_ids:
        pred = sitk.ReadImage(os.path.join(prediction_dir, case_id, 'pred_heatmap.nii.gz'))
        pred = sitk.GetArrayFromImage(pred)
        D, H, W = pred.shape

        landmarks = pd.read_csv(os.path.join(gt_dir, case_id, 'landmarks_512.csv'))
        list_landmarks = landmark_extractor(landmarks)
        heatmap_generator = HeatmapGenerator(image_size=(D, H, W),
                                             sigma=2.,
                                             spine_heatmap_sigma=20,
                                             scale_factor=1.,
                                             normalize=True,
                                             size_sigma_factor=6,
                                             sigma_scale_factor=1,
                                             dtype=np.float32)
        gt = heatmap_generator.generate_spine_heatmap(list_landmarks=list_landmarks)  # 4D

        pred_centroid = ndimage.center_of_mass(pred)
        gt_centroid = ndimage.center_of_mass(gt[0])
        error = [abs(pred_centroid[i] - gt_centroid[i]) for i in range(3)]
        print(f"\n{case_id}:,\npred: {pred_centroid},\ngt: {gt_centroid},\nerror: {error}\n")
        list_errors.append(error)

    return np.mean(list_errors, axis=0)


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

    parser.add_argument('--model_type', type=str, default='Unet_base')
    parser.add_argument('--catalogue', type=int, default=0)
    args = parser.parse_args()

    trainer = NetworkTrainer()
    trainer.setting.project_name = 'Spine_Location'
    trainer.setting.output_dir = '../../../Output/Spine_Location'

    if args.model_type == 'Unet_base':
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
    cases = catalogue['test1'].dropna()
    list_case_dirs = [os.path.join(path, cases[i]) for i in range(len(cases))]

    inference(trainer, list_case_dirs, save_path=os.path.join(trainer.setting.output_dir, 'Prediction'),
              do_TTA=args.TTA)

    # Evaluation
    print('\n\n# Start evaluation !')
    mean_error = evaluate(prediction_dir=os.path.join(trainer.setting.output_dir, 'Prediction'),
                          gt_dir=path)

    print('\n\nmean error: {}'.format(mean_error))
