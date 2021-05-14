"""
This file is modified from  https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe
"""

import math
import numpy as np
from scipy import ndimage


class HeatmapGenerator:
    """
        Generates numpy arrays of Gaussian landmark images for the given parameters.
        :param image_size: Output image size
        :param sigma: Sigma of Gaussian
        :param scale_factor: the value of the landmark is multiplied with this value
        :param normalize: if true, the value on the center is set to scale_factor
                     otherwise, the default gaussian normalization factor is used
        :param size_sigma_factor: the region size for which values are being calculated
                                  Note: if the value is too large, you cannot see the complete area without an
                                  appropriate value of sigma_scale_factor in generate_heatmap method
                                  the number of nonzero values = (sigma * size_sigma_factor + 1) ^ 2 * 5

        :param sigma_scale_factor: the values of the gaussian heatmap are multiplied by this value.

        """
    def __init__(self,
                 image_size=(12, 256, 256),  # (12, 512, 512)
                 sigma=2.,
                 spine_heatmap_sigma=10,  # 20
                 scale_factor=1.,
                 normalize=True,
                 size_sigma_factor=3,  # 6
                 sigma_scale_factor=1,
                 dtype=np.float32):
        self.image_size = image_size
        self.sigma = sigma
        self.spine_heatmap_sigma = spine_heatmap_sigma
        self.scale_factor = scale_factor
        self.dim = len(image_size)
        self.normalize = normalize
        self.size_sigma_factor = size_sigma_factor
        self.sigma_scale_factor = sigma_scale_factor
        self.dtype = dtype

    def generate_heatmap(self, landmark):

        """
        Generates a numpy array of the landmark image for the specified point and parameters.
        :param landmark: numpy coordinates ([x], [y, x] or [z, y, x]) of the point.
        :return: numpy array of the landmark image.
        """

        assert isinstance(landmark, np.ndarray), 'landmark must be a numpy.ndarray'
        heatmap = np.zeros(self.image_size, dtype=self.dtype)

        # if the centroid is invalid
        if True in np.isnan(landmark):
            return heatmap

        region_start = (landmark - self.sigma * self.size_sigma_factor / 2).astype(int)
        region_end = (landmark + self.sigma * self.size_sigma_factor / 2).astype(int) + 1
        region_start[0] = landmark[0] - 2
        region_end[0] = landmark[0] + 2 + 1

        region_start = np.maximum(0, region_start).astype(int)
        region_end = np.minimum(self.image_size, region_end).astype(int)

        if np.any(region_start >= region_end):
            return heatmap

        region_size = (region_end - region_start).astype(int)

        sigma = self.sigma * self.sigma_scale_factor
        scale = self.scale_factor

        # if the distribution is not Standard normal distribution
        if not self.normalize:
            scale /= math.pow(math.sqrt(2 * math.pi) * sigma, self.dim)

        if self.dim == 1:
            dx = np.meshgrid(range(region_size[0]))
            x_diff = dx + region_start[0] - landmark[0]

            squared_distances = x_diff * x_diff

            cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))

            heatmap[region_start[0]:region_end[0]] = cropped_heatmap[:]

        if self.dim == 2:
            dy, dx = np.meshgrid(range(region_size[1]), range(region_size[0]))
            y_diff = dx + region_start[0] - landmark[0]
            x_diff = dy + region_start[1] - landmark[1]

            squared_distances = x_diff * x_diff + y_diff * y_diff

            cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))

            heatmap[region_start[0]:region_end[0], region_start[1]:region_end[1]] = cropped_heatmap[:, :]

        elif self.dim == 3:
            dz, dy, dx = np.meshgrid(range(region_size[0]), range(region_size[1]), range(region_size[2]), indexing='ij')
            z_diff = dz + region_start[0] - landmark[0]
            y_diff = dy + region_start[1] - landmark[1]
            x_diff = dx + region_start[2] - landmark[2]

            squared_distances = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff

            cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))

            heatmap[region_start[0]:region_end[0], region_start[1]:region_end[1], region_start[2]:region_end[2]] \
                = cropped_heatmap[:, :, :]

        return heatmap

    # return one-hot style heatmaps of landmarks for a given image
    def generate_heatmaps(self, list_landmarks, stack_axis=0):
        """
        Generates a 4d numpy array landmark images for the specified points and parameters.
        """
        list_heatmaps = []

        for landmark in list_landmarks:
            list_heatmaps.append(self.generate_heatmap(landmark))

        heatmaps = np.stack(list_heatmaps, axis=stack_axis)

        return heatmaps

    def generate_spine_heatmap(self, list_landmarks):
        """Generates a 4d numpy array image of spine heatmap"""
        landmark_heatmaps = self.generate_heatmaps(list_landmarks)
        landmark_heatmaps = np.sum(landmark_heatmaps, axis=0)
        spine_heatmap = ndimage.gaussian_filter(landmark_heatmaps, sigma=self.spine_heatmap_sigma)
        return spine_heatmap[np.newaxis, :, :, :]

