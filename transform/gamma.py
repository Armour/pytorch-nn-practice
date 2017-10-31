#!/usr/bin/env python3
# coding: utf-8

from __future__ import print_function

import cv2
import numpy as np

import torch.utils as utils

from torchvision import datasets, transforms

from PIL import Image

class Gamma(object):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def _adjust_gamma(self, img):
        # convert PIL.Image to nparray
        img = np.asarray(img)
        # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        invGamma = 1.0 / self.gamma
        table = np.uint8(np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]))
        # apply gamma correction using the lookup table
        img = cv2.LUT(img, table)
        # convert nparray to PIL.Image and return
        return Image.fromarray(np.uint8(img))

    def __call__(self, img):
        """ Call in torchvision transforms
        Args:
            img (PIL.Image): Image to be gamma corrected.
        Returns:
            PIL.Image: gamma corrected image.
        """
        return self._adjust_gamma(img)

# gamma_transform = transforms.Compose([
#     transforms.Scale(224),
#     Gamma(0.5),
#     transforms.ToTensor(),
# ])

# gammaset = datasets.CIFAR10(root='data', train=True, download=True, transform=gamma_transform)
# gammaloader = utils.data.DataLoader(gammaset, batch_size=1, shuffle=True, num_workers=2)

# for batch_idx, (inputs, targets) in enumerate(gammaloader):
#     img = inputs[0].numpy()
#     img = np.uint8(np.stack([img[0], img[1], img[2]], axis=-1) * 255)
#     cv2.imwrite('test/gamma-correction-%f.jpg' % batch_idx, img)
#     if batch_idx == 50:
#         break
