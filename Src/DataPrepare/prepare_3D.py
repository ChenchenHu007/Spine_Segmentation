import os
import numpy as np
from glob import glob
import SimpleITK as sitk
from shutil import copyfile
from scipy.ndimage import zoom


def resize_image(image, dsize=(16, 256, 256), order=3):
    """
    resizes for each slice then adds zeros slice to dsize[0]
    image:sitk.SimpleITK.Image
    dsize: The order of dsize is (z, y, x)
    """
    assert isinstance(image, sitk.SimpleITK.Image)

    image = sitk.GetArrayFromImage(image)
    num_classes = np.amax(image)
    original_size = np.array(image.shape)
    if original_size[0] >= dsize[0]:
        dsize_ = np.array(dsize)
    else:
        dsize_ = np.array([original_size[0], dsize[1], dsize[2]])
    resize_factor = dsize_ / original_size
    image = zoom(image, resize_factor, order=order)
    for i in range(dsize[0] - original_size[0]):
        if not i % 2:
            image = np.concatenate((np.zeros((1, dsize[1], dsize[2])), image), axis=0)
        else:
            image = np.concatenate((image, np.zeros((1, dsize[1], dsize[2]))), axis=0)

    return image, num_classes


# modified from https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html?highlight=resample
def resize_image_sitk(image, dsize=(256, 256, 16)):  # note that order in sitk is (x, y, z)
    """
    image:sitk.SimpleITK.Image
    dsize: The order of dsize is similar with sitk
    """
    assert isinstance(image, sitk.SimpleITK.Image)
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

        copyfile(case_path, os.path.join(dst_path, 'MR_raw.nii.gz'))

        img = sitk.ReadImage(case_path)
        img, _ = resize_image(img, dsize=(16, 256, 256), order=3)
        img = sitk.GetImageFromArray(img)
        sitk.WriteImage(img, dst_path + '/' + 'MR.nii.gz')

    for Mask in Masks:
        case_id = Mask.split('.nii.gz')[0].split('mask_case')[-1]
        case_id = 'Case' + case_id

        case_path = os.path.join(Mask_path, Mask)
        dst_path = os.path.join(Spine_Segmentation, case_id)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        copyfile(case_path, os.path.join(dst_path, 'Mask_raw.nii.gz'))

        img = sitk.ReadImage(case_path)
        img, num_classes = resize_image(img, dsize=(16, 256, 256), order=0)
        img = img.astype(np.uint16)
        img = np.where(img > num_classes, num_classes, img)
        img = np.where(img < 0, 0, img)
        img = sitk.GetImageFromArray(img)
        sitk.WriteImage(img, dst_path + '/' + 'Mask.nii.gz')

    print('Done!')
