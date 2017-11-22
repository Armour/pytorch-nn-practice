#!/usr/bin/env python3
# coding: utf-8

from __future__ import print_function
from __future__ import division

import os
import math
import argparse
import numpy as np
import os.path as osp
import json

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import torch.utils as utils

from model.vgg import vgg
from model.alexnet import alexnet
from model.inception import inception
from model.resnet import resnet_cifar
from transform.log_space import LogSpace
from transform.disturb_illumination import DisturbIllumination

from torch.autograd import Variable
from torchvision import models, datasets, transforms

from tools.logger import Logger

class Trainer():
    def __init__(self, net, train_loader, test_loader, optimizer,
                 baselr = 0.1, criterion=nn.CrossEntropyLoss(),
                 lr_decay_interval = 50,
                 use_cuda=True, savedir="checkpoint"):

        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer

        self.baselr = baselr
        self.lr_decay_interval = lr_decay_interval
        self.criterion = criterion
        self.use_cuda = use_cuda

        self.best_accuracy = 0
        self.best_epoch = 0
        self.output = savedir

        self.tflog_writer = None

        try:
            from tools.logger import Logger
        except ImportError as e:
            print("fail to import tensorboard: {} ".format(e))
        else:
            self.tflog_writer = Logger(self.output, restart=True)

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        self.jsonlog_writer_train = open(osp.join(self.output, "train.log"), 'w+')
        self.jsonlog_writer_test = open(osp.join(self.output, "test.log"), 'w+')



    def __del__(self):
        self.jsonlog_writer_train.close()
        self.jsonlog_writer_test.close()

    def train(self, epoch):
        """ Traning epoch """
        print('==> Training Epoch: %d' % epoch)
        self.net.train()
        total_train_loss = 0
        total_correct = 0
        total_size = 0

        n_train = len(self.train_loader.dataset)
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = predicted.eq(targets.data).cpu().sum()
            total_correct += batch_correct
            error_rate = 100. * (1 - batch_correct / len(inputs))
            total_size += targets.size(0)

            partial_epoch = epoch + batch_idx / len(self.train_loader)
            print('Epoch: [{}] Train:[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
                epoch, total_size, n_train, 100. * batch_idx / len(self.train_loader),
                loss.data[0], error_rate))

            info = {
                'epoch' : partial_epoch,
                'train-loss': loss.data[0],
                'train-top1-error': error_rate
            }
            self.jsonlog_writer_train.write(json.dumps(info) + "\n")

            if self.tflog_writer is not None:
                info.pop('epoch', None)
                for tag, value in info.items():
                    self.tflog_writer.scalar_summary(tag, value, partial_epoch)

    def test(self, epoch):
        """ Testing epoch """
        print('==> Testing Epoch: %d' % epoch)
        self.net.eval()
        total_test_loss = 0
        total_correct = 0
        total_size = 0
        n_train = len(self.test_loader.dataset)
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            total_test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = predicted.eq(targets.data).cpu().sum()
            total_correct += batch_correct
            total_size += targets.size(0)
            error_rate = 100. * (1 - batch_correct / len(inputs))

            partial_epoch = epoch + batch_idx / len(self.train_loader)
            print('Epoch: [{}]\tTest: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
                epoch, total_size, n_train, 100. * batch_idx / len(self.train_loader),
                loss.data[0], error_rate))

        print('Epoch %d Total testing loss: %f    Total testing error rate: %f' % (
            epoch, total_test_loss, (total_size - total_correct) / total_size * 100))
        accuracy = total_correct / total_size * 100
        loss = total_test_loss / total_size

        # writing logs into files
        info = {
            'epoch': epoch,
            'test-loss': loss,
            'test-top1-error': 100 - accuracy
        }
        self.jsonlog_writer_train.write(json.dumps(info) + "\n")

        if self.tflog_writer is not None:
            info.pop('epoch', None)
            for tag, value in info.items():
                self.tflog_writer.scalar_summary(tag, value, partial_epoch)

        return accuracy, loss

    def adjust_learning_rate(self, epoch):
        """ Sets the learning rate to the initial learning rate decayed by 10 every args.lr_decay_interval epochs """
        learning_rate = self.baselr * (0.1 ** (epoch // self.lr_decay_interval))
        print('==> Set learning rate: %f' % learning_rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def execute(self, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch):
            self.adjust_learning_rate(epoch)
            self.train(epoch)
            acc, loss = self.test(epoch)

            # Save checkpoint.
            accuracy = acc
            if accuracy > self.best_accuracy:
                print('==> Saving checkpoint..')
                state = {
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'state_dict': self.net.state_dict(),
                }
                torch.save(state, osp.join(self.output, 'ckpt.t7'))
                self.best_accuracy = accuracy
                self.best_epoch = epoch

            print('Epoch [%d], Best accuracy : %.2f from Epoch [%d]' % (
                epoch, self.best_accuracy, self.best_epoch))