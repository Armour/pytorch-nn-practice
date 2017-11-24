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
        img.mul_(255).add_(1)  # 1 ~ 256
        img.log_().div_(math.log(2)) # 0 ~ 8
        img.div_(torch.max(img)) # 0 ~ 1

        shape = img.shape
        offset0 = img[:,1:,:]
        offset0 = torch.cat((offset0[:,0:1,:] , offset0), 1)
        offset1 = img[:,:,1:]
        offset1 = torch.cat((offset1[:,:,0:1] , offset1), 2)
        offset2 = img[:,:-1,:]
        offset2 = torch.cat((offset2, offset2[:,-1:,:]), 1)
        offset3 = img[:,:,:-1]
        offset3 = torch.cat((offset3, offset3[:,:,-1:]), 2)

        img = img.neg() + (offset0 + offset1 + offset2 + offset3) / 4

        img.sub_(torch.min(img))
        maxv = torch.max(img)
        if maxv > 0:
            img.div_(maxv) # 0 ~ 1

        return img
