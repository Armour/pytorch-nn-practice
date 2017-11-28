#!/usr/bin/env python3
# coding: utf-8

import torch
import math
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
            img (Tensor Image): Image to be transformed
        Returns:
            Tensor Image: transformed image
        """
        img = img * 255 + 1  # 1 ~ 256
        img = torch.log(img) / math.log(2)  # 0 ~ 8
        return img
