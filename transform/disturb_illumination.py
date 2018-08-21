#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DisturbIllumination transformer."""

__author__ = 'Chong Guo <armourcy@gmail.com>'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'MIT'

import torch

class DisturbIllumination(object):
    """ Transform to randomly distrub ilumination
    Example usage:
        transform = transforms.Compose([
           transforms.ToTensor(),
           DisturbIllumination(),
        ])
    """
    def __init__(self):
        """ Init """
        pass

    def __call__(self, img):
        """ Call in torchvision transforms
        Args:
            img (PIL.Image): Image to be transformed
        Returns:
            PIL.Image: transformed image
        """

        rgb_scale = torch.rand(3) * 0.8 + 0.6  # 0.6 ~ 1.4

        img[0,:,:] = img[0,:,:] * rgb_scale[0]  # 0.0 ~ 1.4
        img[1,:,:] = img[1,:,:] * rgb_scale[1]  # 0.0 ~ 1.4
        img[2,:,:] = img[2,:,:] * rgb_scale[2]  # 0.0 ~ 1.4

        return img
