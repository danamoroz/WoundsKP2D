# Copyright 2020 Toyota Research Institute.  All rights reserved.

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os, os.path


class WoundsDataset(Dataset):
    """
    Wounds dataset class.
    Note: output_shape = (output_width, output_height)
    Note: this returns Pytorch tensors, resized to output_shape (if specified)
    Note: the homography will be adjusted according to output_shape.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    use_color : bool
        Return color images or convert to grayscale.
    data_transform : Function
        Transformations applied to the sample
    output_shape: tuple
        If specified, the images and homographies will be resized to the desired shape.
    """
    def __init__(self, root_dir, use_color=True, data_transform=None, output_shape=None):

        super().__init__()
        self.type = type
        self.root_dir = root_dir
        self.data_transform = data_transform
        self.output_shape = output_shape
        self.use_color = use_color
        base_path = Path(root_dir)
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        orig_path = os.path.join(root_dir, 'Original')
        warp_path = os.path.join(root_dir, 'output_images')
        homography_path = os.path.join(root_dir, 'H')
        num_images = len([name for name in os.listdir(orig_path) if os.path.isfile(os.path.join(orig_path, name))])

        for i in range(num_images):
            image_paths.append(str(Path(orig_path, "orig_" + str(i) + ".jpg")))
            warped_image_paths.append(str(Path(warp_path, "warp_" + str(i) + ".jpg")))
            homographies.append(np.load(str(Path(homography_path, "H_" + str(i) + ".npy"))))

        self.files = {'image_paths': image_paths, 'warped_image_paths': warped_image_paths, 'homography': homographies}

    @staticmethod
    def scale_homography(homography, original_scale, new_scale, pre):
        scales = np.divide(new_scale, original_scale)
        if pre:
            s = np.diag(np.append(scales, 1.))
            homography = np.matmul(s, homography)
        else:
            sinv = np.diag(np.append(1. / scales, 1.))
            homography = np.matmul(homography, sinv)
        return homography

    def __len__(self):
        return len(self.files['image_paths'])

    def __getitem__(self, idx):

        def _read_image(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if self.use_color:
                return img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return gray

        image = _read_image(self.files['image_paths'][idx])

        warped_image = _read_image(self.files['warped_image_paths'][idx])
        homography = self.files['homography'][idx]
        sample = {'image': image, 'warped_image': warped_image, 'homography': homography, 'index' : idx}

        # Apply transformations
        if self.output_shape is not None:
            sample['homography'] = self.scale_homography(sample['homography'],
                                                         sample['image'].shape[:2][::-1],
                                                         self.output_shape,
                                                         pre=False)
            sample['homography'] = self.scale_homography(sample['homography'],
                                                         sample['warped_image'].shape[:2][::-1],
                                                         self.output_shape,
                                                         pre=True)

            for key in ['image', 'warped_image']:
                sample[key] = cv2.resize(sample[key], self.output_shape)
                if self.use_color is False:
                    sample[key] = np.expand_dims(sample[key], axis=2)

        transform = transforms.ToTensor()
        for key in ['image', 'warped_image']:
            sample[key] = transform(sample[key]).type('torch.FloatTensor')
        return sample
