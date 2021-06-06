import os
import SimpleITK as sitk
import numpy as np
import argparse
from utils.tools import csv_to_catalogue
from Evaluate.evaluate import *

def copy_sitk_imageinfo(image1, image2):
    image2.SetSpacing(image1.GetSpacing())
    image2.SetDirection(image1.GetDirection())
    image2.SetOrigin(image1.GetOrigin())

    return image2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--catalogue', type=int, default=0)
    args = parser.parse_args()

    csv_path = 'Catalogue' + '/' + str(args.catalogue) + '.csv'
    catalogue = csv_to_catalogue(csv_path)
    cases = catalogue['test'].dropna()
    list_case_dirs = dict()

    data_path = '../../../Data/Spine_Segmentation'
    save_path = '../../../Output/Joint'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    Coccyx_path = '../../../Output/Coccyx_Segmentation/Prediction'
    list_case_dirs['Coccyx'] = [os.path.join(Coccyx_path, cases[i]) for i in range(len(cases))]

    IVD_path = '../../../Output/IVD_Segmentation/Prediction'
    list_case_dirs['IVD'] = [os.path.join(IVD_path, cases[i]) for i in range(len(cases))]

    Vertebrae_path = '../../../Output/Vertebrae_Segmentation/Prediction'
    list_case_dirs['Vertebrae'] = [os.path.join(Vertebrae_path, cases[i]) for i in range(len(cases))]

    dict_images = dict()
    files_name = ['Coccyx', 'IVD', 'Vertebrae']
    S = (12, 512, 512)
    #joint
    for num in range(len(cases)):
        Joint_nii = np.zeros(S)

        pre_Coccyx = sitk.ReadImage(list_case_dirs[files_name[0]][num] + '/' + 'pred_' + files_name[0] + 'Mask.nii.gz',sitk.sitkFloat32)
        pre_Coccyx = sitk.GetArrayFromImage(pre_Coccyx)

        pre_IVD = sitk.ReadImage(list_case_dirs[files_name[1]][num] + '/' + 'pred_' + files_name[1] + 'Mask.nii.gz',sitk.sitkFloat32)
        pre_IVD = sitk.GetArrayFromImage(pre_IVD)

        pre_Vertebrae = sitk.ReadImage(list_case_dirs[files_name[2]][num] + '/' + 'pred_' + files_name[2] + 'Mask.nii.gz', sitk.sitkFloat32)
        pre_Vertebrae = sitk.GetArrayFromImage(pre_Vertebrae)

        Joint_nii = pre_Coccyx + pre_IVD + pre_Vertebrae

        template_nii = sitk.ReadImage(data_path + '/' + cases[num] + '/MR_512.nii.gz')
        Joint_nii = sitk.GetImageFromArray(Joint_nii)
        Joint_nii = copy_sitk_imageinfo(template_nii, Joint_nii)
        if not os.path.exists(save_path + '/' + cases[num]):
            os.mkdir(save_path + '/' + cases[num])
        sitk.WriteImage(Joint_nii, save_path + '/' + cases[num] + '/pred_JointMask.nii.gz')
    print('Joint Done!')

    # joint evaluation
    Joint_path = '../../../Output/Output/Joint'
    list_case_dirs['Joint'] = [os.path.join(Joint_path, cases[i]) for i in range(len(cases))]
    dscs = []
    for case_dir in range(len(list_case_dirs['Joint'])):
        pred_mask = sitk.ReadImage(list_case_dirs['Joint'][case_dir] + '/' + 'pred_JointMask.nii.gz')
        pred = sitk.GetArrayFromImage(pred_mask)

        gt_mask = sitk.ReadImage(data_path + '/' + cases[case_dir] + '/Mask_512.nii.gz')
        gt = sitk.GetArrayFromImage(gt_mask)
        dsc = cal_subject_level_dice(pred, gt, num_classes=20)
        dscs.append(dsc)
    Dice_score = np.mean(dscs)
    print('\n\nDice score is: ' + str(Dice_score))








