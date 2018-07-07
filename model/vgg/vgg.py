#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG model."""

__author__ = 'Chong Guo'
__copyright__ = 'Copyright 2017, Chong Guo'
__license__ = 'MIT'
__email__ = 'armourcy@email.com'

import math
import torch.nn as nn

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """VGG model."""
    def __init__(self, vgg_name, batch_norm=True, num_classes=1000):
        """ Init
        Args:
            vgg_name (string): VGG identifier, map to different config parameters
            batch_norm (bool): If True, use batch normalization
            num_classes (int): The number of output classes
        """
        super(VGG, self).__init__()
        if vgg_name in cfg:
            self.features = make_layers(cfg[vgg_name], batch_norm)
            self.classifier = nn.Sequential(
                # [n, 512 * 7 * 7]
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                # [n, 4096]
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                # [n, 4096]
                nn.Linear(4096, num_classes),
                # [n, num_classes]
            )
            self._initialize_weights()
        else:
            print('vgg_name doesn\'t exist, please choose from vgg11, vgg13, vgg16 and vgg19')

    def forward(self, x):
        """Pytorch forward function implementation."""
        x = self.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Init weight parameters."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    """Generate network layers based on configs."""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    sample_data = torch.ones(12, 3, 224, 224)
    sample_input = Variable(sample_data)
    net = VGG("vgg16")
    print(net)
    print(net(sample_input))
    print(net(sample_input).shape)
