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
from torch.autograd import Variable

from tools.logger import Logger


class Trainer():
    def __init__(self, net, train_loader, test_loader, optimizer, start_epoch=0,
                 best_accuracy=0, best_epoch=0, base_lr=0.1, criterion=nn.CrossEntropyLoss(),
                 lr_decay_interval=50, use_cuda=True, save_dir='checkpoint'):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer

        self.base_lr = base_lr
        self.criterion = criterion
        self.lr_decay_interval = lr_decay_interval
        self.use_cuda = use_cuda

        self.best_accuracy = best_accuracy
        self.best_epoch = best_epoch
        self.start_epoch = start_epoch
        self.save_dir = save_dir

        self.tflog_writer = None

        try:
            from tools.logger import Logger
        except ImportError as e:
            print("fail to import tensorboard: {} ".format(e))
        else:
            self.tflog_writer = Logger(self.save_dir, restart=True)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.jsonlog_writer_train = open(osp.join(self.save_dir, "train.log"), 'w+')
        self.jsonlog_writer_test = open(osp.join(self.save_dir, "test.log"), 'w+')

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
            print('Epoch: [{}]\tTrain:[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
                epoch, total_size, n_train, 100. * batch_idx / len(self.train_loader),
                loss.data[0], error_rate))

            info = {
                'epoch': partial_epoch,
                'train-loss': loss.data[0],
                'train-top1-error': error_rate
            }
            self.jsonlog_writer_train.write(json.dumps(info) + "\n")

            if self.tflog_writer is not None:
                info.pop('epoch', None)
                for tag, value in info.items():
                    self.tflog_writer.scalar_summary(tag, value, partial_epoch)

        print('Epoch: [{}]\tTotal training loss: [{:.6f}]\tTotal training error rate: [{:.6f}]'.format(
            epoch, total_train_loss, (total_size - total_correct) / total_size * 100))

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

        print('Epoch: [{}]\tTotal testing loss: [{:.6f}]\tTotal testing error rate: [{:.6f}]'.format(
            epoch, total_test_loss, (total_size - total_correct) / total_size * 100))

        accuracy = total_correct / total_size * 100
        loss = total_test_loss / total_size

        # writing logs into files
        info = {
            'epoch': epoch,
            'test-loss': loss,
            'test-top1-error': 100 - accuracy
        }
        self.jsonlog_writer_test.write(json.dumps(info) + "\n")

        if self.tflog_writer is not None:
            info.pop('epoch', None)
            for tag, value in info.items():
                self.tflog_writer.scalar_summary(tag, value, partial_epoch)

        return accuracy, loss

    def adjust_learning_rate(self, epoch):
        """ Sets the learning rate to the initial learning rate decayed by 10 every args.lr_decay_interval epochs """
        learning_rate = self.base_lr * (0.1 ** (epoch // self.lr_decay_interval))
        print('==> Set learning rate: %f' % learning_rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def execute(self, end_epoch):
        for epoch in range(self.start_epoch, end_epoch):
            self.adjust_learning_rate(epoch)
            self.train(epoch)
            accuracy, loss = self.test(epoch)

            # Save checkpoint.
            if accuracy > self.best_accuracy:
                print('==> Saving checkpoint..')
                self.best_accuracy = accuracy
                self.best_epoch = epoch
                state = {
                    'start_epoch': epoch,
                    'best_epoch': self.best_epoch,
                    'best_accuracy': self.best_accuracy,
                    'state_dict': self.net.state_dict(),
                }
                torch.save(state, osp.join(self.save_dir, 'ckpt.t7'))

            print('Epoch [%d], Best accuracy : %.2f from Epoch [%d]' % (
                epoch, self.best_accuracy, self.best_epoch))
