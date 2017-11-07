#!/usr/bin/env python3
# coding: utf-8

from __future__ import print_function
from __future__ import division

import cv2
import numpy as np

import torch.utils as utils

from torchvision import datasets, transforms
from PIL import Image

class Illumination(object):
    """ Illumination transform """
    def __init__(self, r=1.0, g=1.0, b=1.0, enable_log=False, enable_illumination=False):
        """ Init """
        self.r = r
        self.g = g
        self.b = b
        self.enable_log = enable_log
        self.enable_illumination = enable_illumination

    def _adjust_illumination(self, img):
        """ Adjust illumination """
        # convert PIL.Image to nparray
        img = np.asarray(img)

        # updated channels value using r, g, b parameters
        if self.enable_illumination:
            img = np.clip(img * [self.r, self.g, self.b], a_min=0, a_max=255)

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
            PIL.Image: image that with illumination adjusted
        """
        return self._adjust_illumination(img)

# Example uasage:
# illumination_transform = transforms.Compose([
#     transforms.Scale(224),
#     Illumination(r=1.5, enable_log=True),
#     transforms.ToTensor(),
# ])

# dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=illumination_transform)
# dataloader = utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# for batch_idx, (inputs, targets) in enumerate(dataloader):
#     img = inputs[0].numpy()
#     img = np.uint8(np.stack([img[0], img[1], img[2]], axis=-1))
#     cv2.imwrite('test/illumination-%f.jpg' % batch_idx, img)
#     if batch_idx == 10:
#         break
