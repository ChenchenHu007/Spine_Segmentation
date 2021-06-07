import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


def generate_centroids_from_array(image):
    """
    generate the centroids of all labels in a given ndarray image
    :param image: ndarray image
    return a dictionary of centroids
    """
    assert isinstance(image, np.ndarray)
    labels = np.unique(image)
    labels = np.delete(labels, 0)
    landmarks_ = dict()
    landmarks_['axis'] = ['z', 'y', 'x']
    for label_i in range(1, 20):
        if label_i in labels:
            tmp = np.where(image == label_i, label_i, 0)
            landmark_i = ndimage.center_of_mass(tmp)
            landmark_i = [round(x) for x in landmark_i]
            landmarks_[str(label_i)] = landmark_i
        else:
            landmarks_[str(label_i)] = [np.nan, np.nan, np.nan]

    return landmarks_


if __name__ == '__main__':
    dataset = '../../Data/Spine_Segmentation'
    cases = os.listdir(dataset)
    for case in cases:
        case_path = os.path.join(dataset, case)
        Mask = os.path.join(case_path, 'Mask.nii.gz')
        Mask_512 = os.path.join(case_path, 'Mask_512.nii.gz')
        raw_Mask = os.path.join(case_path, 'raw_Mask.nii.gz')

        # for Mask
        img = sitk.ReadImage(Mask)
        img_arr = sitk.GetArrayFromImage(img)
        landmarks = generate_centroids_from_array(img_arr)
        df = pd.DataFrame(data=landmarks)
        df.to_csv(os.path.join(case_path, 'landmarks.csv'), index=False)

        # for Mask_512
        img = sitk.ReadImage(Mask_512)
        img_arr = sitk.GetArrayFromImage(img)
        landmarks = generate_centroids_from_array(img_arr)
        df = pd.DataFrame(data=landmarks)
        df.to_csv(os.path.join(case_path, 'landmarks_512.csv'), index=False)

        # for raw_Mask
        img = sitk.ReadImage(raw_Mask)
        img_arr = sitk.GetArrayFromImage(img)
        landmarks = generate_centroids_from_array(img_arr)
        df = pd.DataFrame(data=landmarks)
        df.to_csv(os.path.join(case_path, 'raw_landmarks.csv'), index=False)

    print('Done!')
