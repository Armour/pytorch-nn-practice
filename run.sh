#!/usr/bin/env bash
source activate torch

# no log, normal train, normal test
CUDA_VISIBLE_DEVICES=0 python main.py -e 150 --lr-decay-interval 50

# no log, disturb train, normal test
CUDA_VISIBLE_DEVICES=1 python main.py -e 150 --lr-decay-interval 50 -dtrain

# yes log, normal train, normal test
CUDA_VISIBLE_DEVICES=2 python main.py -e 150 --lr-decay-interval 50 -l

# yes log, disturb on train, normal test
CUDA_VISIBLE_DEVICES=3 python main.py -e 150 --lr-decay-interval 50 -l -dtrain
