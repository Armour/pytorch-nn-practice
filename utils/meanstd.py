#!/usr/bin/env python3
# coding: utf-8

from __future__ import print_function
from __future__ import division

import os
import numpy as np

from torchvision import datasets, transforms

train_transform = transforms.Compose([transforms.ToTensor()])

# cifar10
train_set = datasets.CIFAR10(root='../data/', train=True, download=True, transform=train_transform)
print(train_set.train_data.shape)
print(train_set.train_data.mean(axis=(0,1,2))/255)
print(train_set.train_data.std(axis=(0,1,2))/255)
# (50000, 32, 32, 3)
# [0.49139968  0.48215841  0.44653091]
# [0.24703223  0.24348513  0.26158784]

# cifar100
train_set = datasets.CIFAR100(root='../data/', train=True, download=True, transform=train_transform)
print(train_set.train_data.shape)
print(train_set.train_data.mean(axis=(0,1,2))/255)
print(train_set.train_data.std(axis=(0,1,2))/255)
# (50000, 32, 32, 3)
# [0.50707516  0.48654887  0.44091784]
# [0.26733429  0.25643846  0.27615047]

# mnist
train_set = datasets.MNIST(root='../data/', train=True, download=True, transform=train_transform)
print(list(train_set.train_data.size()))
print(train_set.train_data.float().mean()/255)
print(train_set.train_data.float().std()/255)
# [60000, 28, 28]
# 0.1306604762738429
# 0.30810780717887876
