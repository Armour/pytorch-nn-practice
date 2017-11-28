#!/usr/bin/env bash
source activate torch
# global enviroments
epochs=300
lr_decay_interval=75

# no log, normal train, normal test
GPUid=0

CUDA_VISIBLE_DEVICES=${GPUid}  python main.py -epochs ${epochs} \
                               --lr-decay-interval ${lr_decay_interval}

# no log, disturb train, normal test
GPUid=1

CUDA_VISIBLE_DEVICES=${GPUid}  python main.py --epochs ${epochs} \
                               --lr-decay-interval ${lr_decay_interval} -dtrain

# yes log, normal train, normal test
GPUid=2

CUDA_VISIBLE_DEVICES=${GPUid}  python main.py --epochs ${epochs} \
                               --lr-decay-interval${lr_decay_interval} -l
# yes log, disturb on train, normal test
GPUid=3

CUDA_VISIBLE_DEVICES=${GPUid}  python main.py --epochs ${epochs} \
                               --lr-decay-interval${lr_decay_interval} -l -dtrain
