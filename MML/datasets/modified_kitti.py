import sys
sys.path.append('.')

import os
import cv2
import random
from PIL import Image
import numpy as np
import zipfile

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from MML.utils.inherit import DatasetInherit
from MML.preprocessing import Transform


class TrainDataset(Dataset, DatasetInherit):

    def __init__(self, transform_dict=None, img_size=(320, 960)):
        root_dir = 'kitti_train_dataset'

        # custom transform
        self.img_size = img_size
        if transform_dict:
            transform_dict['img_size'] = self.img_size if not transform_dict['img_size'] else transform_dict['img_size']
            self.custom_transform = Transform(transform_dict)
        else:
            self.custom_transform = None

        # if the dataset doesn't exist on the disk, download it
        if not os.path.exists(root_dir):
            print('Downloading Dataset...')
            os.system(
                'wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip')
            with zipfile.ZipFile("data_semantics.zip","r") as zipped_file:
                zipped_file.extractall()
            os.rename('training', root_dir)
            os.rename('testing', 'kitti_test_dataset')
            os.system('rm -r data_semantics.zip')

        # create image and mask filenames
        self.image_filenames = []
        self.mask_filenames = []
        for filename in os.listdir(os.path.join(root_dir, 'image_2')):
            self.image_filenames.append(
                os.path.join(root_dir, 'image_2', filename))
            self.mask_filenames.append(
                os.path.join(root_dir, 'semantic', filename))

        # Total number of classes in the mask
        self.num_classes = 34

        # Number of classes to keep including the background
        self.req_classes = 5

        # Indices of the required classes (excluding the background)
        self.required_classes = [7, 21, 23, 26]

    def transform(self, image, mask, img_size):
        # Resize
        image = TF.resize(image, size=img_size)
        mask = TF.resize(mask, size=img_size)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask))

        # Normalize with mean 0 and stddev 1
        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return image, mask

    def get_req_classes(self, mask):
        """Extract selected classes from the dataset"""
        background = np.zeros(mask.shape)
        for idx, cls in enumerate(self.required_classes):
            cls_val = np.ones(mask.shape) * (idx+1)
            background = np.where(mask == cls, cls_val,
                                  background).astype(np.uint8)
        return background

    def __len__(self):
        return len(self.image_filenames)

    @property
    def name(self):
        # specifies the name of the class
        return 'Modified Kitti'

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 1)
        image = Image.fromarray(image)

        mask = cv2.imread(self.mask_filenames[index], 0)
        mask = self.get_req_classes(mask)
        mask = Image.fromarray(mask)

        # if custom transform in not available
        # just normalize and convert to tensor
        if self.custom_transform is None:
            image, mask = self.transform(image, mask, self.img_size)
        else:
            image, mask = self.custom_transform(image, mask)
        return image, mask

    def __str__(self):
        info = 'Modified Kitti Dataset \nNo. of Images: {} \nImage Ch: 3  Mask Ch: 5\nImage Size: (320, 960)'.format(
            self.__len__())
        return info


class TestDataset(Dataset):

    def __init__(self, img_size=(320, 960)):
        root_dir = 'kitti_test_dataset'
        self.img_size = img_size
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        # create image filenames
        self.image_filenames = []
        for filename in os.listdir(os.path.join(root_dir, 'image_2')):
            self.image_filenames.append(
                os.path.join(root_dir, 'image_2', filename))

        self.num_classes = 34
        self.req_classes = 5

    def transform(self, image, img_size):
        """Basic Transforms"""

        # Resize
        image = TF.resize(image, size=img_size)

        # Transform to tensor
        image = TF.to_tensor(image)

        # Normalize with mean 0 and std 1
        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return image

    def __len__(self):
        return len(self.image_filenames)

    def __str__(self):
        info = 'Modified Kitti Dataset \nNo. of Images: {} \nImage Ch: 3 \nImage Size: (320, 960)'.format(
            self.__len__())
        return info

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 1)
        image = Image.fromarray(image)
        image = self.transform(image, self.img_size)
        return image
