#!/usr/bin/env python3
# coding: utf-8

from __future__ import print_function
from __future__ import division

import os
import math
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import torch.utils as utils

from model.vgg import vgg
from model.alexnet import alexnet
from model.inception import inception
from model.resnet import resnet
from transform.illumination import Illumination

from torch.autograd import Variable
from torchvision import models, datasets, transforms

def calculate_mean_and_std():
    transform_train = transforms.Compose([
        Illumination(enable_illumination=args.enable_training_illumination_transform, enable_log=args.enable_training_log_transform),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    dataloader = utils.data.DataLoader(dataset)
    data = np.stack([inputs[0].numpy() for inputs, targets in dataloader])
    mean = data.mean(axis=(0,2,3))
    std = data.std(axis=(0,2,3))
    return torch.from_numpy(mean), torch.from_numpy(std)

def train(epoch):
    """Traning epoch."""
    print('==> Training Epoch: %d' % epoch)
    net.train()
    total_train_loss = 0
    total_correct = 0
    total_size = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        batch_correct = predicted.eq(targets.data).cpu().sum()
        total_correct += batch_correct
        total_size += targets.size(0)

        # Print traning loss
        if batch_idx % args.log_interval == 0:
            print('%f/%f ==> Training loss: %f    Correct number: %f/%f' % (batch_idx, len(trainloader), loss.data[0], batch_correct, targets.size(0)))

    print("==> Total training loss: %f    Total correct: %f/%f" % (total_train_loss, total_correct, total_size))

def test(epoch):
    """Testing epoch."""
    global best_accuracy
    print('==> Testing Epoch: %d' % epoch)
    net.eval()
    total_test_loss = 0
    total_correct = 0
    total_size = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        total_test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        batch_correct = predicted.eq(targets.data).cpu().sum()
        total_correct += batch_correct
        total_size += targets.size(0)

        # Print testing loss
        if batch_idx % args.log_interval == 0:
            print('%f/%f ==> Testing loss: %f    Correct number: %f/%f' % (batch_idx, len(testloader), loss.data[0], batch_correct, targets.size(0)))

        # Save output image
        if batch_idx % args.save_interval == 0:
            img = inputs.data.numpy()[0]
            img = img * data_std.numpy()[0] + data_mean.numpy()[0] # denormalize
            img = np.clip(img, a_min=0.0, a_max=1.0) # clip
            img = np.uint8(np.stack([img[0], img[1], img[2]], axis=-1) * 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, classes[predicted[0]] + ':' + classes[targets.data[0]], (5,len(img[0]) - 15), font, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imwrite('test/illumination-%f.jpg' % (batch_idx // args.save_interval), img)

    print("==> Total testing loss: %f    Total correct: %f/%f" % (total_test_loss, total_correct, total_size))

    # Save checkpoint.
    accuracy = 100.*total_correct/total_size
    if accuracy > best_accuracy:
        print('==> Saving checkpoint..')
        state = {
            'epoch': epoch,
            'accuracy': accuracy,
            'state_dict': net.state_dict(),
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_accuracy = accuracy

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial learning rate decayed by 10 every args.lr_decay_interval epochs."""
    learning_rate = args.learning_rate * (0.1 ** (epoch // args.lr_decay_interval))
    print('==> Set learning rate: %f' % learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


# Setup args
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-lr','--learning-rate', type=float, default=0.01,
                    help='initial learning rate (default: 0.01)')
parser.add_argument('-e', '--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--train-batch-size', type=int, default=50,
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=100,
                    help='input batch size for testing (default: 100)')
parser.add_argument('--lr-decay-interval', type=int, default=50,
                    help='number of epochs to decay the learning rate (default: 50)')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of workers (default: 4)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status (default: 50)')
parser.add_argument('--save-interval', type=int, default=50,
                    help='how many batches to wait before saving testing output image (default: 50)')
parser.add_argument('-i1', '--enable-training-illumination-transform', action='store_true', default=False,
                    help='enable illumination transform for traning')
parser.add_argument('-i2', '--enable-testing-illumination-transform', action='store_true', default=False,
                    help='enable illumination transform for testing')
parser.add_argument('-l1', '--enable-training-log-transform', action='store_true', default=False,
                    help='enable log transform for traning')
parser.add_argument('-l2', '--enable-testing-log-transform', action='store_true', default=False,
                    help='enable log transform for testing')
parser.add_argument('-t', '--only-testing', action='store_true', default=False,
                    help='only run testing')
parser.add_argument('-r', '--resume', action='store_true', default=False,
                    help='resume from checkpoint')
args = parser.parse_args()

# Init variables
print('==> Init variables..')
use_cuda = cuda.is_available()
best_accuracy = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Init seed
print('==> Init seed..')
torch.manual_seed(args.seed)
if use_cuda:
    cuda.manual_seed(args.seed)

# Download data
print('==> Download data..')
datasets.CIFAR10(root='data', train=True, download=True)

# Calculate mean and std
print('==> Calculate mean and std..')
data_mean, data_std = calculate_mean_and_std()
print('mean = ', data_mean)
print('std = ', data_std)

# Prepare training transform
print('==> Prepare training transform..')
transform_train = transforms.Compose([
    transforms.Scale(224),
    transforms.RandomHorizontalFlip(),
    Illumination(enable_illumination=args.enable_training_illumination_transform, enable_log=args.enable_training_log_transform),
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std),
])

# Prepare testing transform
print('==> Prepare testing transform..')
transform_test = transforms.Compose([
    transforms.Scale(224),
    Illumination(enable_illumination=args.enable_testing_illumination_transform, enable_log=args.enable_testing_log_transform),
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std),
])

# Init dataloader
print('==> Init dataloader..')
trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
trainloader = utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

testset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
testloader = utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

# Model
print('==> Building model..')
net = resnet.ResNet('res34', num_classes=10)
# net = vgg.VGG('vgg16', num_classes=10)
# net = alexnet.AlexNet(num_classes=10)
# net = inception.InceptionV3(num_classes=10)

if use_cuda:
    net = net.cuda()

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['accuracy']
    net.load_state_dict(checkpoint['state_dict'])

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
if use_cuda:
    criterion = criterion.cuda()
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=5e-4)

for epoch in range(start_epoch, start_epoch + args.epochs):
    if args.only_testing:
        test(epoch)
        break
    else:
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
