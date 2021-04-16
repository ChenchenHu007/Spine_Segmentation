import numpy as np
import os
import SimpleITK as sitk
from tqdm import tqdm

"""
These codes are modified from https://github.com/ababier/open-kbp
"""


def cal_subject_level_dice(prediction, target, num_classes=20):
    """
    step1: calculate the dice of each category
    step2: remove the dice of the empty category and background, and then calculate the mean of the remaining dices.
    :param prediction: the automated segmentation result, a numpy array with shape of (h, w, d)
    :param target: the ground truth mask, a numpy array with shape of (h, w, d)
    :param num_classes: total number of categories
    :return:
    """
    eps = 1e-10
    empty_value = -1.0
    dscs = empty_value * np.ones((num_classes,), dtype=np.float32)
    for i in range(0, num_classes):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp
        dsc = 2 * tp / (2 * tp + fp + fn + eps)
        dscs[i] = dsc
    dscs = np.where(dscs == -1.0, np.nan, dscs)
    subject_level_dice = np.nanmean(dscs[1:])
    return subject_level_dice


def evaluate_demo(prediction_dir, gt_dir):
    """
    This is a demo for calculating the mean dice of all subjects.
    modified from https://www.spinesegmentation-challenge.com/?page_id=34
    """
    dscs = []
    list_case_ids = tqdm(os.listdir(prediction_dir))
    for case_id in list_case_ids:
        pred_mask = sitk.ReadImage(os.path.join(prediction_dir, case_id, 'pred_mask.nii.gz'))
        pred = sitk.GetArrayFromImage(pred_mask)

        gt_mask = sitk.ReadImage(os.path.join(gt_dir, case_id, 'Mask.nii.gz'))
        gt = sitk.GetArrayFromImage(gt_mask)

        dsc = cal_subject_level_dice(pred, gt, num_classes=20)
        dscs.append(dsc)
    return np.mean(dscs)
