import os
import numpy as np
from glob import glob
import SimpleITK as sitk
from shutil import copyfile


# modified from https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html?highlight=resample
def resize_image(image, dsize=(512, 512, 16)):  # note that sequence of array in sitk is (x, y, z)
    original_size = np.array(image.GetSize())
    original_spacing = np.array(image.GetSpacing())

    scale_factor = np.array(dsize) / original_size

    new_spacing = original_spacing / scale_factor

    image_ = sitk.Resample(image1=image, size=dsize,
                           transform=sitk.Transform(),
                           interpolator=sitk.sitkLinear,
                           outputOrigin=image.GetOrigin(),
                           outputSpacing=new_spacing,
                           outputDirection=image.GetDirection(),
                           defaultPixelValue=0,
                           outputPixelType=image.GetPixelID())

    return image_


# def FileTreeRefactor():
#
#     MRs = glob(os.path.join(MR_path, '*.nii.gz'))
#     Masks = glob(os.path.join(Mask_path, '*.nii.gz'))
#
#     for i in range(len(MRs)):
#         case_index = MRs[i].split('/')[-1].split('.nii.gz')[0]
#         case_dir = os.path.join(Spine_Segmentation, case_index)
#
#         if not os.path.exists(case_dir):
#             os.mkdir(case_dir)
#
#         os.rename(MRs[i], os.path.join(case_dir, 'MR.nii.gz'))
#
#     for i in range(len(Masks)):
#         case_index = Masks[i].split('/')[-1].split('.nii.gz')[0].split('mask_case')[-1]
#         case_index = 'Case' + case_index
#         case_dir = os.path.join(Spine_Segmentation, case_index)
#
#         assert os.path.exists(case_dir)
#
#         os.rename(Masks[i], os.path.join(case_dir, 'Mask.nii.gz'))


if __name__ == '__main__':

    train_path = '../../Data/train'
    MR_path = os.path.join(train_path, 'MR')
    Mask_path = os.path.join(train_path, 'Mask')
    Spine_Segmentation = '../../Data/Spine_Segmentation'
    MRs = os.listdir(MR_path)
    Masks = os.listdir(Mask_path)

    if not os.path.exists(Spine_Segmentation):
        os.mkdir(Spine_Segmentation)

    for MR in MRs:
        case_id = MR.split('.nii.gz')[0]  # Case*
        case_path = os.path.join(MR_path, MR)  #
        dst_path = os.path.join(Spine_Segmentation, case_id)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        img = sitk.ReadImage(case_path)
        img = resize_image(img, dsize=(512, 512, 16))
        sitk.WriteImage(img, dst_path + '/' + 'MR.nii.gz')

    for Mask in Masks:
        case_id = Mask.split('.nii.gz')[0].split('mask_case')[-1]
        case_id = 'Case' + case_id

        case_path = os.path.join(Mask_path, Mask)
        dst_path = os.path.join(Spine_Segmentation, case_id)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        img = sitk.ReadImage(case_path)
        img = resize_image(img, dsize=(512, 512, 16))
        sitk.WriteImage(img, dst_path + '/' + 'Mask.nii.gz')

    for Mask in Masks:
        case_id = 'Case' + Mask.split('.nii.gz')[0].split('mask_case')[-1]
        case_path = os.path.join(Spine_Segmentation, case_id)
        src = os.path.join(Mask_path, Mask)
        dst = os.path.join(case_path, 'Mask_original.nii.gz')
        copyfile(src, dst)
