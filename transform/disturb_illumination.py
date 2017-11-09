#!/usr/bin/env python3
# coding: utf-8

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
        channel_scale = torch.rand(3) * 0.8 + 0.6  # 0.6 ~ 1.4

        img = img * channel_scale

        img[img > 1] = 1
        img[img < 0] = 0

        return img
