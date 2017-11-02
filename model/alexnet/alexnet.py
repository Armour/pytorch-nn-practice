#!/usr/bin/env python3
# coding: utf-8

import math
import torch.nn as nn

class AlexNet(nn.Module):
    """ Alexnet model """
    def __init__(self, num_classes=1000):
        """ Init
        Args:
            num_classes (int): The number of output classes
        """
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # [n, 3, 224, 224]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            # [n, 96, 55, 55]
            nn.ReLU(inplace=True),
            # [n, 96, 55, 55]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [n, 96, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            # [n, 256, 27, 27]
            nn.ReLU(inplace=True),
            # [n, 256, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [n, 256, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            # [n, 384, 13, 13]
            nn.ReLU(inplace=True),
            # [n, 384, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            # [n, 384, 13, 13]
            nn.ReLU(inplace=True),
            # [n, 384, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # [n, 256, 13, 13]
            nn.ReLU(inplace=True),
            # [n, 256, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [n, 256, 6, 6]
        )
        self.classifier = nn.Sequential(
            # [n, 256 * 6 * 6]
            nn.Linear(256 * 6 * 6, 4096),
            # [n, 4096]
            nn.ReLU(inplace=True),
            # [n, 4096]
            nn.Dropout(),
            # [n, 4096]
            nn.Linear(4096, 4096),
            # [n, 4096]
            nn.ReLU(inplace=True),
            # [n, 4096]
            nn.Dropout(),
            # [n, 4096]
            nn.Linear(4096, num_classes),
            # [n, num_classes]
        )
        self._initialize_weights()

    def forward(self, x):
        """ Pytorch forward function implementation """
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """ Init weight parameters """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
