#!/usr/bin/env bash
source activate torch

# no log, normal train, normal test
CUDA_VISIBLE_DEVICES=0 python main.py -e 150 --lr-decay-interval 50 -s checkpoint_0

# no log, disturb train, normal test
CUDA_VISIBLE_DEVICES=1 python main.py -e 150 --lr-decay-interval 50 -dtrain  -s checkpoint_1

# yes log, normal train, normal test
CUDA_VISIBLE_DEVICES=2 python main.py -e 150 --lr-decay-interval 50 -l  -s checkpoint_2

# yes log, disturb on train, normal test
CUDA_VISIBLE_DEVICES=3 python main.py -e 150 --lr-decay-interval 50 -l -dtrain  -s checkpoint_3
