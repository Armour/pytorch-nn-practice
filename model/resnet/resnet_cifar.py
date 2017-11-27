#!/usr/bin/env python3
# coding: utf-8

import math
import torch.nn as nn

cfg = {
    'res18': {
        'block': 'BasicBlock',
        'num_blocks': [2, 2, 2, 2]
    },
    'res34': {
        'block': 'BasicBlock',
        'num_blocks': [3, 4, 6, 3]
    },
    'res50': {
        'block': 'Bottleneck',
        'num_blocks': [3, 4, 6, 3]
    },
    'res101': {
        'block': 'Bottleneck',
        'num_blocks': [3, 4, 23, 3]
    },
    'res152': {
        'block': 'Bottleneck',
        'num_blocks': [3, 8, 36, 3]
    },
}

class BasicBlock(nn.Module):
    """ Basic residual block """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """ Init
        Args:
            in_channels: channel number for input image
            out_channels: channel number for output image
            stride: stride number for Conv2d
            downsample: downsample function (default: None)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """ Pytorch forward function implementation """
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

    expansion = 1

class Bottleneck(nn.Module):
    """ Bottleneck residual block """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """ Init
        Args:
            in_channels: channel number for input image
            out_channels: channel number for output image
            stride: stride number for Conv2d
            downsample: downsample function (default: None)
        """
        super(Bottleneck, self).__init__()
        expanded_channel = out_channels * Bottleneck.expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, expanded_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expanded_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """ Pytorch forward function implementation """
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

    expansion = 4

class ResNet(nn.Module):
    """ Resnet model """
    def __init__(self, resnet_name, num_classes=1000):
        """ Init
        Args:
            resnet_name (string): resnet identifier, map to different config parameters
            num_classes (int): The number of output classes
        """
        super(ResNet, self).__init__()
        if resnet_name in cfg:
            self.block = globals()[cfg[resnet_name]['block']]
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(64, 64, cfg[resnet_name]['num_blocks'][0], stride=1)
            self.layer2 = self._make_layer(64, 128, cfg[resnet_name]['num_blocks'][1], stride=2)
            self.layer3 = self._make_layer(128, 256, cfg[resnet_name]['num_blocks'][2], stride=2)
            self.layer4 = self._make_layer(256, 512, cfg[resnet_name]['num_blocks'][3], stride=2)
            self.avgpool = nn.AvgPool2d(4)
            self.fc = nn.Linear(512 * self.block.expansion, num_classes)
            self._initialize_weights()
        else:
            print('resnet_name doesn\'t exist, please choose from res18, res34, res50, res101 and res152')

    def forward(self, x):
        """ Pytorch forward function implementation """
        # [n, 3, 32, 32]
        x = self.conv1(x)
        # [n, 64, 32, 32]
        x = self.bn1(x)
        # [n, 64, 32, 32]
        x = self.relu(x)
        # [n, 64, 32, 32]
        x = self.layer1(x)
        # [n, 64 * self.block.expansion, 32, 32]
        x = self.layer2(x)
        # [n, 128 * self.block.expansion, 16, 16]
        x = self.layer3(x)
        # [n, 256 * self.block.expansion, 8, 8]
        x = self.layer4(x)
        # [n, 512 * self.block.expansion, 4, 4]
        x = self.avgpool(x)
        # [n, 512 * self.block.expansion, 1, 1]
        x = x.view(x.size(0), -1)
        # [n, 512 * self.block.expansion]
        x = self.fc(x)
        # [n, num_classes]
        return x

    def _initialize_weights(self):
        """ Init weight parameters """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """ Generate network layers based on configs """
        downsample = None
        expanded_in_channels = in_channels * self.block.expansion
        expanded_out_channels = out_channels * self.block.expansion
        if expanded_in_channels != expanded_out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(expanded_in_channels, expanded_out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expanded_out_channels),
            )

        layers = []
        layers.append(self.block(in_channels, out_channels, stride, downsample))
        for i in range(1, num_blocks):
            layers.append(self.block(expanded_out_channels, out_channels))

        return nn.Sequential(*layers)


if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    sample_data = torch.ones(12, 3, 32, 32)
    sample_input = Variable(sample_data)
    net = ResNet("res18")
    print(net(sample_input))
