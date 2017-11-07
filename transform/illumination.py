#!/usr/bin/env python3
# coding: utf-8

from __future__ import print_function
from __future__ import division

import cv2
import numpy as np

import torch
import torch.utils as utils

from torchvision import datasets, transforms
from PIL import Image

class Illumination(object):
    """ Illumination transform
    Example usage:
        illumination_transform = transforms.Compose([
           transforms.Scale(224),
           Illumination(enable_illumination=True, enable_log=True),
           transforms.ToTensor(),
        ])
    """
    def __init__(self, enable_log=False, enable_illumination=False):
        """ Init
        Args:
            enable_log: enable log transform
            enable_illumination: enable illumination transform
        """
        self.enable_log = enable_log
        self.enable_illumination = enable_illumination

    def _adjust_illumination(self, img):
        """ Adjust illumination
        Args:
            img (PIL.Image): Image to be transformed
        Returns:
            PIL.Image: transformed image
        """
        # convert PIL.Image to nparray
        img = np.asarray(img)

        # updated channels value using scale parameters (0.6 - 1.4)
        if self.enable_illumination:
            channel_scale = torch.rand(3) * 0.8 + 0.6
            img = np.clip(img * channel_scale.numpy(), a_min=0, a_max=255)

        # log
        if self.enable_log:
            img = np.log(img + 1e-10)

        # convert nparray back to PIL.Image and return
        return Image.fromarray(np.uint8(img))

    def __call__(self, img):
        """ Call in torchvision transforms
        Args:
            img (PIL.Image): Image to be transformed
        Returns:
            PIL.Image: transformed image
        """
        return self._adjust_illumination(img)
