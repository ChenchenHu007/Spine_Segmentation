import os
import sys
import SimpleITK as sitk
import numpy as np

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

from utils.processing import resize_sitk_image
from utils.tools import csv_to_catalogue
from tqdm import tqdm
from Evaluate.evaluate import cal_subject_level_dice


def copy_sitk_imageinfo(template, target_img):
    target_img.SetSpacing(template.GetSpacing())
    target_img.SetDirection(template.GetDirection())
    target_img.SetOrigin(template.GetOrigin())

    return target_img


def evaluate(prediction_dir, gt_dir):
    """
    This is a demo for calculating the mean dice of all subjects.
    modified from https://www.spinesegmentation-challenge.com/?page_id=34
    """
    dscs = []
    list_seg = os.listdir(prediction_dir)
    if '.DS_Store' in list_seg:  # macos
        list_seg.remove('.DS_Store')
    for seg in tqdm(list_seg):
        case_id = seg.split('seg_case')[-1].split('.nii.gz')[0]
        pred_mask = sitk.ReadImage(os.path.join(prediction_dir, seg), sitk.sitkUInt8)
        pred = sitk.GetArrayFromImage(pred_mask)

        gt_mask = sitk.ReadImage(os.path.join(gt_dir, 'Case' + str(case_id) + '/raw_Mask.nii.gz'), sitk.sitkUInt8)
        gt = sitk.GetArrayFromImage(gt_mask)

        dsc = cal_subject_level_dice(pred, gt, num_classes=20)
        dscs.append(dsc)
    return np.mean(dscs)


if __name__ == "__main__":
    if not os.path.exists('../../Data/Spine_Segmentation'):  # this is base dataset
        raise Exception('Spine_Segmentation should be prepared before testing, ' +
                        'please run prepare_3D.py and landmark generation.py')
    assert os.path.exists('../../Output/IVD_Segmentation'), 'IVD_Segmentation does not exist !'
    assert os.path.exists('../../Output/Coccyx_Segmentation'), 'Coccyx_Segmentation does not exist !'
    assert os.path.exists('../../Output/Vertebrae_Segmentation'), 'IVD_Segmentation does not exist !'

    pred_IVDs_dir = '../../Output/IVD_Segmentation/Prediction'
    pred_Coccyx_dir = '../../Output/Coccyx_Segmentation/Prediction'
    pred_Vertebrae_dir = '../../Output/Vertebrae_Segmentation/Prediction'

    csv_path = '../Catalogue/0.csv'
    catalogue = csv_to_catalogue(csv_path)
    cases = catalogue['test1'].dropna()
    list_case_dirs = dict()

    path = "../../Data/Spine_Segmentation"
    output_dir = '../../Output/segmentation_results'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for case in tqdm(cases):
        raw_MR_path = os.path.join(path, case, 'raw_MR.nii.gz')
        pred_IVD_path = os.path.join(pred_IVDs_dir, case, 'pred_IVDMask.nii.gz')
        pred_Coccyx_path = os.path.join(pred_Coccyx_dir, case, 'pred_CoccyxMask.nii.gz')
        pred_Vertebrae_path = os.path.join(pred_Vertebrae_dir, case, 'pred_VertebraeMask.nii.gz')

        raw_MR = sitk.ReadImage(raw_MR_path)  # template
        W, H, D = raw_MR.GetSize()
        # print('D:{}, H:{}, W:{}'.format(D, H, W))

        pred_IVD = sitk.ReadImage(pred_IVD_path, sitk.sitkUInt8)
        pred_IVD = sitk.GetArrayFromImage(pred_IVD)
        pred_Coccyx = sitk.ReadImage(pred_Coccyx_path, sitk.sitkUInt8)
        pred_Coccyx = sitk.GetArrayFromImage(pred_Coccyx)
        pred_Vertebrae = sitk.ReadImage(pred_Vertebrae_path, sitk.sitkUInt8)
        pred_Vertebrae = sitk.GetArrayFromImage(pred_Vertebrae)

        prediction = pred_IVD + pred_Coccyx + pred_Vertebrae
        # print(prediction.dtype)
        prediction_nii = sitk.GetImageFromArray(prediction)
        prediction_nii = resize_sitk_image(prediction_nii, dsize=(W, H, None), interpolator='nearest')
        prediction_nii = copy_sitk_imageinfo(template=raw_MR, target_img=prediction_nii)
        sitk.WriteImage(prediction_nii, os.path.join(output_dir, 'seg_' + case.lower() + '.nii.gz'))

    print('\n\n# Start evaluation !')
    Dice_score = evaluate(prediction_dir=output_dir, gt_dir=path)

    print('\n\nDice score is: ' + str(Dice_score))
