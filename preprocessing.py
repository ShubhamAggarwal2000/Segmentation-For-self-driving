import cv2
import random
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

from MML.utils.affine import RandomAffine


class Transform():
    """
    Transformation Class

    Creating an instance of this class requires transform_dict to be passed.
    The instance of this class needs to be passed to the TrainDataset Class. (see datasets)

    Parameters
    ----------
    transform_dict : dict
        dict specifying the parameters for presprocessing

    transform_dict = {
        # Resize                  
        'img_size': (272, 400),   If None, original image size will be used

        # Zoom In
        'zoom_in_prob': 0.5,       
        'zoom_in_scale': (1, 1.1),   # 1 implies no zoom in

        # Zoom Out
        'zoom_out_prob': 0.5,
        'zoom_out_scale': (1, 1.1),  # 1 implies no zoom out
        # Pad the border with the mean of the image, if False then pad with 0
        'fill_mean': True,           

        # Select Both zoom in and zoom out
        # If False, either zoomed in or zoomed out (not both at the same time)
        # If True, both zoom in and zoom out are applied together
        'zoom_in_out': True,

        # Flip  
        'h_flip': True,
        'v_flip': False,

        # Rotation (degrees)
        # Random rotation between -angle to +angle
        'angle': 5,

        # Color Jitter
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': 0.3,
        'hue': 0.1,         # must be within (-0.5, 0.5)

        # gamma transform (intensity scaling)
        # 1 implies no intensity scaling
        # less than 1: brighten 
        # greater than 1: darken
        # must be greater than 0
        # suggested range: (0.6, 1.5)  
        'gamma': (0.8, 1.1),         

        # pixelwise noise
        # suggested range: (less than 0.2)
        'additive_noise': 0.01,
        # suggested range (less than 0.2)
        'multiplicative_noise': 0,

        # random erasing
        'erase_prob': 1,             # Probabiity of erasing
        'erase_count': 5,            # Iterations
        'erase_scale': (0.05, 0.05), # Scale wrt the MMLal image 
        'erase_ratio': (1, 1),       # aspect ratio of the erased box
        'erase_value': 0,            # value to be filled in the erased box

        # Normalization
        'mean': [0.5],
        'std': [0.5],
    }
    """

    def __init__(self, transform_dict):
        self.transform_dict = transform_dict

    def zoom_in(self, image, mask):
        # Random Resize and Crop
        if random.random() < self.transform_dict['zoom_in_prob']:

            original_h = np.array(image).shape[0]
            original_w = np.array(image).shape[1]
            old_size = min(original_h, original_w)

            scale_low = min(self.transform_dict['zoom_in_scale'])
            scale_high = max(self.transform_dict['zoom_in_scale'])

            # set the minimum value of scale_low as 1.1
            scale_low = scale_low if scale_low > 1.1 else 1.1

            scale = random.uniform(scale_low, scale_high)
            new_size = int(scale * old_size)

            # smaller side after resizing preserving the aspect ratio
            image = TF.resize(image, size=new_size,
                              interpolation=Image.NEAREST)
            mask = TF.resize(mask, size=new_size, interpolation=Image.NEAREST)

            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.transform_dict['img_size'])
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
        else:
            image = TF.resize(
                image, size=self.transform_dict['img_size'], interpolation=Image.NEAREST)
            mask = TF.resize(
                mask, size=self.transform_dict['img_size'], interpolation=Image.NEAREST)
        return image, mask

    def horizontal_flip(self, image, mask):
        # Random horizontal flipping
        if random.random() < 0.5 and self.transform_dict['h_flip']:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def vertical_flip(self, image, mask):
        # Random horizontal flipping
        if random.random() < 0.5 and self.transform_dict['v_flip']:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask

    def zoom(self, image, mask):
        '''
        zoom calls zoom_in and zoom_out
        if transform_dict['zoom_in_out'] is True then it will call both zoom_in
        and zoom_out at the same time else one at a time
        '''
        if self.transform_dict['zoom_in_out']:
            image, mask = self.zoom_in(image, mask)
            image, mask = self.zoom_out(image, mask)
        else:
            if random.random() < 0.5:
                image, mask = self.zoom_in(image, mask)
            else:
                image, mask = self.zoom_out(image, mask)
        return image, mask

    def additive_noise(self, image, mask):
        # pixelwise additive noise
        if not self.transform_dict['additive_noise']:
            return image, mask
        else:
            image = np.array(image)
            max_val = np.max(image)
            val = self.transform_dict['additive_noise']

            # to prevent the program from crashing
            try:
                noise = np.random.randint(size=image.shape, 
                                          low=int((-val) * max_val), 
                                          high=int((+val)*max_val))
            except:
                noise = 0

            image = np.add(image, noise)
            image = cv2.normalize(image, None, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
            return Image.fromarray(image), mask

    def zoom_out(self, image, mask):
        # Random zoom out
        if random.random() < self.transform_dict['zoom_out_prob']:

            original_h = np.array(image).shape[0]
            original_w = np.array(image).shape[1]

            scale_low = min(self.transform_dict['zoom_out_scale'])
            scale_high = max(self.transform_dict['zoom_out_scale'])
            scale = random.uniform(scale_low, scale_high)

            new_h = int(scale * original_h)
            new_w = int(scale * original_w)

            left = random.randint(0, new_w - original_w)
            right = new_w - (left + original_w)
            top = random.randint(0, new_h - original_h)
            bottom = new_h - (top + original_h)
            padding = (left, top, right, bottom)

            if self.transform_dict['fill_mean']:
                if len(np.array(image).shape) == 3:
                    fill_value = tuple(
                        int(self.transform_dict['mean'][i] * 255) for i in range(3))
                else:
                    fill_value = int(self.transform_dict['mean'][0] * 255)
            else:
                fill_value = 0

            image = TF.pad(image, padding, fill=fill_value,
                           padding_mode='constant')
            mask = TF.pad(mask, padding, fill=0, padding_mode='constant')

            image = TF.resize(
                image, size=self.transform_dict['img_size'], interpolation=Image.NEAREST)
            mask = TF.resize(
                mask, size=self.transform_dict['img_size'], interpolation=Image.NEAREST)

        else:
            image = TF.resize(
                image, size=self.transform_dict['img_size'], interpolation=Image.NEAREST)
            mask = TF.resize(
                mask, size=self.transform_dict['img_size'], interpolation=Image.NEAREST)
        return image, mask

    def adjust_gamma(self, image, mask):
        # randomly change the brightness
        gamma_low = min(self.transform_dict['gamma'])
        gamma_high = max(self.transform_dict['gamma'])
        gamma = random.uniform(gamma_low, gamma_high)
        image = TF.adjust_gamma(image, gamma, gain=1)
        return image, mask

    def rotate(self, image, mask):
        # rotate an image randomly in range 
        # (-self.transform_dict['angle'],self.transform_dict['angle'])
        angle = random.uniform(-self.transform_dict['angle'],
                               self.transform_dict['angle'])

        if self.transform_dict['fill_mean']:
            if len(np.array(image).shape) == 3:
                fill_value = tuple(
                    int(self.transform_dict['mean'][i] * 255) for i in range(3))
            else:
                fill_value = int(self.transform_dict['mean'][0] * 255)
        else:
            fill_value = 0

        rotate_image = RandomAffine(angle, fillcolor=fill_value)
        rotate_mask = RandomAffine(angle, fillcolor=0)

        image = rotate_image(image)
        mask = rotate_mask(mask)
        return image, mask

    def color_jitter(self, image, mask):
        # randomly change the brightness, hue, saturation and contrast
        brightness = self.transform_dict['brightness']
        contrast = self.transform_dict['contrast']
        saturation = self.transform_dict['saturation']
        hue = self.transform_dict['hue']
        color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                              saturation=saturation, hue=hue)
        image = color_jitter(image)
        return image, mask

    def random_erasing(self, image, mask):
        # randomly erase patches from the image
        erase_prob = self.transform_dict['erase_prob']
        erase_scale = self.transform_dict['erase_scale']
        erase_ratio = self.transform_dict['erase_ratio']
        erase_value = self.transform_dict['erase_value']

        random_erase = transforms.RandomErasing(p=erase_prob, scale=erase_scale, ratio=erase_ratio, 
                                         value=erase_value, inplace=False)

        for count in range(self.transform_dict['erase_count']):
            image = random_erase(image)
        return image, mask

    def sanity_check(self, image):
        # sanity check
        if len(self.transform_dict['mean']) != len(self.transform_dict['std']):
            raise AssertionError(
                'Mean and Standard Deviation must be lists of same length')
        if len(np.array(image).shape) == 3:
            if len(self.transform_dict['mean']) != np.array(image).shape[2]:
                raise AssertionError(
                    'Mean and Image Color Channels must be of same length')
        else:
            if len(self.transform_dict['mean']) != 1:
                raise AssertionError(
                    'Mean and Image Color Channels must be of same length')

    def multiplicative_noise(self, image , mask):
        # pixelwise multiplicative noise
        if not self.transform_dict['multiplicative_noise']:
            return image, mask
        else:
            image = np.array(image)
            max_val = np.max(image)
            val = self.transform_dict['multiplicative_noise']
            noise = np.random.uniform(size=image.shape, low=1-val, high=1+val)
            image = image * noise
            image = cv2.normalize(image, None, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
            return Image.fromarray(image), mask

    def __call__(self, image, mask):

        # Sanity Check
        self.sanity_check(image)

        # Transforms(image in PIL format)
        image, mask = self.zoom(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.color_jitter(image, mask)
        image, mask = self.adjust_gamma(image, mask)
        image, mask = self.additive_noise(image, mask)
        image, mask = self.multiplicative_noise(image, mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask))

        # Random erasing requires image in tensor format
        image, mask = self.random_erasing(image, mask)

        # Normalize with mean 0 and stddev 1
        image = TF.normalize(
            image, self.transform_dict['mean'], self.transform_dict['std'])

        return image, mask
