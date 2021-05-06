import os
import sys
import numpy as np
import SimpleITK as sitk
from shutil import copyfile

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

from utils.processing import resize_sitk_image

if __name__ == '__main__':

    train_path = '../../Data/train'
    MR_path = os.path.join(train_path, 'MR')
    Mask_path = os.path.join(train_path, 'Mask')
    Spine_Segmentation = '../../Data/Spine_Segmentation'
    MRs = os.listdir(MR_path)
    Masks = os.listdir(Mask_path)

    if not os.path.exists(Spine_Segmentation):
        os.mkdir(Spine_Segmentation)

    # FIXME resize and crop
    for MR in MRs:
        case_id = MR.split('.nii.gz')[0]  # Case*
        case_path = os.path.join(MR_path, MR)  #
        dst_path = os.path.join(Spine_Segmentation, case_id)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        copyfile(case_path, os.path.join(dst_path, 'raw_MR.nii.gz'))

        img = sitk.ReadImage(case_path)

        img_ = resize_sitk_image(img, dsize=(256, 256, None), interpolator='linear')
        sitk.WriteImage(img_, dst_path + '/' + 'MR.nii.gz')

        img_ = resize_sitk_image(img, dsize=(512, 512, None), interpolator='linear')
        sitk.WriteImage(img_, dst_path + '/' + 'MR_512.nii.gz')

    for Mask in Masks:
        case_id = Mask.split('.nii.gz')[0].split('mask_case')[-1]
        case_id = 'Case' + case_id

        case_path = os.path.join(Mask_path, Mask)
        dst_path = os.path.join(Spine_Segmentation, case_id)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        copyfile(case_path, os.path.join(dst_path, 'raw_Mask.nii.gz'))

        img = sitk.ReadImage(case_path)

        img_ = resize_sitk_image(img, dsize=(256, 256, None), interpolator='nearest')
        sitk.WriteImage(img_, dst_path + '/' + 'Mask.nii.gz')

        img_ = resize_sitk_image(img, dsize=(512, 512, None), interpolator='nearest')
        sitk.WriteImage(img_, dst_path + '/' + 'Mask_512.nii.gz')

    print('Done!')
