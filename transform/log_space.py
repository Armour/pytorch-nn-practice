#!/usr/bin/env python3
# coding: utf-8

import torch

class LogSpace(object):
    """ Transform to log space
    Example usage:
        transform = transforms.Compose([
           transforms.ToTensor(),
           LogSpace(),
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
        img = img * 255 + 1 # 1 ~ 256
        img = torch.log(img) # 0 ~ 8
        return img
