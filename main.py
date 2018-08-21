#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main function."""

__author__ = 'Chong Guo <armourcy@gmail.com>'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'MIT'

import os
import math
import os.path as osp
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import torch.utils as utils
from torchvision import models, datasets, transforms

from model.resnet import resnet_cifar

from transform.log_space import LogSpace
from transform.disturb_illumination import DisturbIllumination


def calculate_mean_and_std(enable_log_transform):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if enable_log_transform:
        transform = transforms.Compose([
            transform,
            LogSpace(),
        ])
    dataset = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
    dataloader = utils.data.DataLoader(dataset)
    data = np.stack([inputs[0].numpy() for inputs, targets in dataloader])
    mean = data.mean(axis=(0,2,3))
    std = data.std(axis=(0,2,3))
    return mean, std

if __name__ == '__main__':
    # Setup args
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('-lr','--learning-rate', type=float, default=0.1,
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('-e', '--epochs', type=int, default=150,
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--train-batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 64)')

    parser.add_argument('--lr-decay-interval', type=int, default=50,
                        help='number of epochs to decay the learning rate (default: 50)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='how many batches to wait before saving testing output image (default: 50)')
    parser.add_argument('-s', '--save-directory', type=str, default='checkpoint',
                        help='checkpoint save directory (default: checkpoint)')
    parser.add_argument('-n', '--env-name', type=str, default=None,
                        help='experiment name (default: None)')

    parser.add_argument('-dtest', '--enable-disturb-illumination-test', action='store_true', default=False,
                        help='enable disturb illumination for testing data')
    parser.add_argument('-dtrain', '--enable-disturb-illumination-train', action='store_true', default=False,
                        help='enable disturb illumination for training data')

    parser.add_argument('-l', '--enable-log-transform', action='store_true', default=False,
                        help='enable log transform for both traning and testing data')
    parser.add_argument('-t', '--only-testing', action='store_true', default=False,
                        help='only run testing')
    parser.add_argument('-r', '--resume', action='store_true', default=False,
                        help='resume from checkpoint')

    args = parser.parse_args()

    # Init variables
    print('==> Init variables..')
    use_cuda = cuda.is_available()
    best_accuracy = 0  # best testing accuracy
    best_epoch = 0  # epoch with the best testing accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.env_name is None:
        args.env_name = "Log:%s-Train:%s-Test:%s" % (args.enable_log_transform, \
                                                     args.enable_disturb_illumination_train, \
                                                     args.enable_disturb_illumination_test)
        args.save_directory = osp.join(args.save_directory, args.env_name)

    # Init seed
    print('==> Init seed..')
    torch.manual_seed(args.seed)
    if use_cuda:
        cuda.manual_seed(args.seed)

    # Calculate mean and std
    print('==> Prepare mean and std..')
    print("\t Log : %s" % args.enable_log_transform)
    if not args.enable_log_transform:
        # mean_log, std_log = (0.50707543, 0.48655024, 0.44091907), (0.26733398, 0.25643876, 0.27615029)
        mean_log, std_log = calculate_mean_and_std(enable_log_transform=False)
    else:
        # mean_log, std_log = (6.69928741, 6.65900993, 6.40947819), (1.2056427,  1.15127575, 1.31597221)
        mean_log, std_log = calculate_mean_and_std(enable_log_transform=True)
    print('\tmean_log = ', mean_log)
    print('\tstd_log = ', std_log)

    data_mean = torch.FloatTensor(mean_log)
    data_std = torch.FloatTensor(std_log)


    # Prepare training transform
    print('==> Prepare training transform..')
    t_trans = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    if args.enable_disturb_illumination_train:
        print("\tDisturbance on train")
        t_trans += [DisturbIllumination(), ]

    if args.enable_log_transform:
        print("\tLogSpace on train")
        t_trans += [LogSpace(), ]

    traning_transform = transforms.Compose([
        *t_trans,
        transforms.Normalize(data_mean, data_std),
    ])
    print(traning_transform)
    # Prepare testing transform
    print('==> Prepare testing transform..')
    t_trans = [
        transforms.ToTensor(),
    ]
    if args.enable_disturb_illumination_test:
        print("\tDisturbance on test")
        t_trans += [DisturbIllumination(), ]

    if args.enable_log_transform:
        print("\tLogSpace on test")
        t_trans += [LogSpace(), ]

    testing_transform = transforms.Compose([
        *t_trans,
        transforms.Normalize(data_mean, data_std),
    ])
    print(testing_transform)

    # Init dataloaderenable_log_transform
    print('==> Init dataloader..')
    trainset = datasets.CIFAR100(root='data', train=True, download=True, transform=traning_transform)
    trainloader = utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

    testset = datasets.CIFAR100(root='data', train=False, download=True, transform=testing_transform)
    testloader = utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    print('==> Building model..')
    net = resnet_cifar.ResNet('res34', num_classes=100)
    if use_cuda:
        net = net.cuda()

    # Resume if required
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.save_directory), 'Error: no checkpoint directory found!'
        if use_cuda:
            checkpoint = torch.load(args.save_directory + '/ckpt.t7')
        else:
            checkpoint = torch.load(args.save_directory + '/ckpt.t7', map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['start_epoch']
        best_epoch = checkpoint['best_epoch']
        best_accuracy = checkpoint['best_accuracy']
        net.load_state_dict(checkpoint['state_dict'])

    # Loss function and Optimizer
    print('==> Setup loss function and optimizer..')
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=1e-4,
                          nesterov=True)

    # Training
    print('==> Init trainer..')
    from trainer import Trainer
    train = Trainer(net, trainloader, testloader, optimizer, start_epoch=start_epoch,
                 best_accuracy=best_accuracy, best_epoch=best_epoch, base_lr=args.learning_rate,
                 criterion=criterion, lr_decay_interval=args.lr_decay_interval, use_cuda=use_cuda, save_dir=args.save_directory)
    print('==> Start training..')
    train.execute(args.epochs)

