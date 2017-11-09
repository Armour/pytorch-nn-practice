#!/usr/bin/env python3
# coding: utf-8

import math
import torch.nn as nn

class BasicConv2d(nn.Module):
    """ Basic convolution layer with batch normalizetion """
    def __init__(self, in_channels, out_channels, **kwargs):
        """ Init convolution layer """
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """ Pytorch forward function implementation """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionA(nn.Module):
    """ Inception module A """
    def __init__(self, in_channels, pool_features):
        """ Init """
        super(InceptionA, self).__init__()

        self.branch1_1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch2_1x1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch2_5x5 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3_1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3_3x3_1 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3_3x3_2 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch4_avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_1x1 = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        """ Pytorch forward function implementation """
        branch1 = self.branch1_1x1(x)

        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_5x5(branch2)

        branch3 = self.branch3_1x1(x)
        branch3 = self.branch3_3x3_1(branch3)
        branch3 = self.branch3_3x3_2(branch3)

        branch4 = self.branch4_avgpool(x)
        branch4 = self.branch4_1x1(branch4)

        # out_channels = 64 + 64 + 96 + pool_features
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class InceptionB(nn.Module):
    """ Inception module B """
    def __init__(self, in_channels):
        """ Init """
        super(InceptionB, self).__init__()

        self.branch1_3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch2_1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch2_3x3_1 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch2_3x3_2 = BasicConv2d(96, 96, kernel_size=3, stride=2)

        self.branch3_maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        """ Pytorch forward function implementation """
        branch1 = self.branch1_3x3(x)

        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_3x3_1(branch2)
        branch2 = self.branch2_3x3_2(branch2)

        branch3 = self.branch3_maxpool(x)

        # out_channels = 384 + 96 + in_channels
        return torch.cat([branch1, branch2, branch3], 1)

class InceptionC(nn.Module):
    """ Inception module C """
    def __init__(self, in_channels, channels_7x7):
        """ Init """
        super(InceptionC, self).__init__()

        self.branch1_1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch2_1x1 = BasicConv2d(in_channels,channels_7x7, kernel_size=1)
        self.branch2_1x7 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3))
        self.branch2_7x1 = BasicConv2d(channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch3_1x1 = BasicConv2d(in_channels, channels_7x7, kernel_size=1)
        self.branch3_7x1_1 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0))
        self.branch3_1x7_1 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3))
        self.branch3_7x1_2 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0))
        self.branch3_1x7_2 = BasicConv2d(channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch4_avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        """ Pytorch forward function implementation """
        branch1 = self.branch1_1x1(x)

        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_1x7(branch2)
        branch2 = self.branch2_7x1(branch2)

        branch3 = self.branch3_1x1(x)
        branch3 = self.branch3_7x1_1(branch3)
        branch3 = self.branch3_1x7_1(branch3)
        branch3 = self.branch3_7x1_2(branch3)
        branch3 = self.branch3_1x7_2(branch3)

        branch4 = self.branch4_avgpool(x)
        branch4 = self.branch4_1x1(branch4)

        # out_channels = 192 + 192 + 192 + 192 = 768
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class InceptionD(nn.Module):
    """ Inception module D """
    def __init__(self, in_channels):
        """ Init """
        super(InceptionD, self).__init__()

        self.branch1_1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch1_3x3 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch2_1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch2_1x7 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch2_7x1 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch2_3x3 = BasicConv2d(192, 192, kernel_size=3, stride=2)

        self.branch3_maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        """ Pytorch forward function implementation """
        branch1 = self.branch1_1x1(x)
        branch1 = self.branch1_3x3(branch1)

        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_1x7(branch2)
        branch2 = self.branch2_7x1(branch2)
        branch2 = self.branch2_3x3(branch2)

        branch3 = self.branch3_maxpool(x)

        # out_channels = 320 + 192 + in_channel
        return torch.cat([branch1, branch2, branch3], 1)

class InceptionE(nn.Module):
    """ Inception module E """
    def __init__(self, in_channels):
        """ Init """
        super(InceptionE, self).__init__()

        self.branch1_1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch2_1x1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch2_1x3 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch2_3x1 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3_1x1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3_3x3 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3_1x3 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3_3x1 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch4_avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        """ Pytorch forward function implementation """
        branch1 = self.branch1_1x1(x)

        branch2 = self.branch2_1x1(x)
        branch2 = [
            self.branch2_1x3(branch2),
            self.branch2_3x1(branch2),
        ]
        branch2 = torch.cat(branch2, 1)

        branch3 = self.branch3_1x1(x)
        branch3 = self.branch3_3x3(branch3)
        branch3 = [
            self.branch3_1x3(branch3),
            self.branch3_3x1(branch3),
        ]
        branch3 = torch.cat(branch3, 1)

        branch4 = self.branch4_avgpool(x)
        branch4 = self.branch4_1x1(branch4)

        # out_channels = 320 + 384 * 2 + 384 * 2 + 192 = 2048
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class InceptionAux(nn.Module):
    """ Inception auxiliary classifiers """
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1x1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv5x5 = BasicConv2d(128, 768, kernel_size=5)
        self.conv5x5.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # [n, 768, 17, 17]
        x = self.avgpool(x)
        # [n, 768, 5, 5]
        x = self.conv1x1(x)
        # [n, 128, 5, 5]
        x = self.conv5x5(x)
        # [n, 768, 1, 1]
        x = x.view(x.size(0), -1)
        # [n, 768]
        x = self.fc(x)
        # [n, 1000]
        return x

class InceptionV3(nn.Module):
    """ Inception V3 model """
    def __init__(self, num_classes=1000, aux_logits=True):
        """ Init
        Args:
            num_classes (int): The number of output classes
        """
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.Conv2d_1_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_3_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool2d_1_3x3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_4_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_5_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.MaxPool2d_2_3x3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.InceptionA1 = InceptionA(192, pool_features=32)
        self.InceptionA2 = InceptionA(256, pool_features=64)
        self.InceptionA3 = InceptionA(288, pool_features=64)
        self.InceptionB = InceptionB(288)
        self.InceptionC1 = InceptionC(768, channels_7x7=128)
        self.InceptionC2 = InceptionC(768, channels_7x7=160)
        self.InceptionC3 = InceptionC(768, channels_7x7=160)
        self.InceptionC4 = InceptionC(768, channels_7x7=192)
        if self.aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.InceptionD = InceptionD(768)
        self.InceptionE1 = InceptionE(1280)
        self.InceptionE2 = InceptionE(2048)
        self.AvgPool2d_8x8 = nn.AvgPool2d(kernel_size=8)
        self.Dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)

        self._initialize_weights()

    def forward(self, x):
        """ Pytorch forward function implementation """
        # [n, 3, 299, 299]
        x = self.Conv2d_1_3x3(x)
        # [n, 32, 149, 149]
        x = self.Conv2d_2_3x3(x)
        # [n, 32, 147, 147]
        x = self.Conv2d_3_3x3(x)
        # [n, 64, 147, 147]
        x = self.MaxPool2d_1_3x3(x)
        # [n, 64, 73, 73]
        x = self.Conv2d_4_1x1(x)
        # [n, 80, 73, 73]
        x = self.Conv2d_5_3x3(x)
        # [n, 192, 71, 71]
        x = self.MaxPool2d_2_3x3(x)
        # [n, 192, 35, 35]
        x = self.InceptionA1(x)
        # [n, 256, 35, 35]
        x = self.InceptionA2(x)
        # [n, 288, 35, 35]
        x = self.InceptionA3(x)
        # [n, 288, 35, 35]
        x = self.InceptionB(x)
        # [n, 768, 17, 17]
        x = self.InceptionC1(x)
        # [n, 768, 17, 17]
        x = self.InceptionC2(x)
        # [n, 768, 17, 17]
        x = self.InceptionC3(x)
        # [n, 768, 17, 17]
        x = self.InceptionC4(x)
        # [n, 768, 17, 17]
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # [n, 768, 17, 17]
        x = self.InceptionD(x)
        # [n, 1280, 8, 8]
        x = self.InceptionE1(x)
        # [n, 2048, 8, 8]
        x = self.InceptionE2(x)
        # [n, 2048, 8, 8]
        x = self.AvgPool2d_8x8(x)
        # [n, 2048, 1, 1]
        x = self.Dropout(x)
        # [n, 2048, 1, 1]
        x = x.view(x.size(0), -1)
        # [n, 2048]
        x = self.fc(x)
        # [n, num_classes]
        if self.training and self.aux_logits:
            return x, aux
        else:
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
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    sample_data = torch.ones(12, 3, 224, 224)
    sample_input = Variable(sample_data)
    net = InceptionV3()
    print(net(sample_input))
