#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Drawing utilities."""

__author__ = 'Chong Guo <armourcy@gmail.com>'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'MIT'

from __future__ import print_function
from __future__ import division

import os
import re
import argparse

from matplotlib import pyplot as plt


def get_data_from_log_file(log_file):
    """ Get loss and error rate data from log file """
    traning_loss = []
    traning_error_rate = []
    testing_loss = []
    testng_error_rate = []

    with open(log_file) as f:
        for line in f:
            r = re.match('==> Total training loss: (.*)   Total training error rate: (.*)', line)
            if r is not None:
                traning_loss.append(r.groups()[0].strip())
                traning_error_rate.append(r.groups()[1].strip())
            r = re.match('==> Total testing loss: (.*)    Total testing error rate: (.*)', line)
            if r is not None:
                testing_loss.append(r.groups()[0].strip())
                testng_error_rate.append(r.groups()[1].strip())

    return (traning_loss, testing_loss), (traning_error_rate, testng_error_rate)


def draw_image(loss, error_rate, saved_name):
    """ Draw and save loss and error rate image """
    f, axs = plt.subplots(2, sharex=True)
    axs[0].plot([i + 1 for i in range(len(loss[0]))], loss[0], label="train")
    axs[0].plot([i + 1 for i in range(len(loss[1]))], loss[1], label="test")
    axs[0].set_ylabel('loss')
    axs[0].legend()
    axs[1].plot([i + 1 for i in range(len(error_rate[0]))], error_rate[0], label="train")
    axs[1].plot([i + 1 for i in range(len(error_rate[1]))], error_rate[1], label="test")
    axs[1].set_ylabel('error rate')
    axs[1].legend()
    plt.xlabel('epoch')
    plt.savefig(saved_name)


if __name__ == '__main__':
    # Setup args
    parser = argparse.ArgumentParser(description='Draw error and loss graph from log')
    parser.add_argument('-f', '--log-file', type=str, default='test/log',
                        help='the log file that we read data from')
    parser.add_argument('-s', '--saved-name', type=str, default='test/result.png',
                        help='the output image name')
    args = parser.parse_args()

    loss, error_rate = get_data_from_log_file(args.log_file)

    draw_image(loss=loss, error_rate=error_rate, saved_name=args.saved_name)
