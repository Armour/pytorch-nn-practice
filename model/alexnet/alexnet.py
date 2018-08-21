#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Alexnet model."""

__author__ = 'Chong Guo <armourcy@gmail.com>'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'MIT'

import math
import torch.nn as nn


class AlexNet(nn.Module):
    """Alexnet model."""
    def __init__(self, num_classes=1000):
        """ Init
        Args:
            num_classes (int): The number of output classes
        """
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # [n, 3, 224, 224]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # [n, 96, 55, 55]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [n, 96, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # [n, 256, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [n, 256, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [n, 384, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [n, 384, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [n, 256, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [n, 256, 6, 6]
        )
        self.classifier = nn.Sequential(
            # [n, 256 * 6 * 6]
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # [n, 4096]
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # [n, 4096]
            nn.Linear(4096, num_classes),
            # [n, num_classes]
        )
        self._initialize_weights()

    def forward(self, x):
        """Pytorch forward function implementation."""
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Init weight parameters."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    sample_data = torch.ones(12, 3, 224, 224)
    sample_input = Variable(sample_data)
    net = AlexNet()
    print(net)
    print(net(sample_input))
    print(net(sample_input).shape)
