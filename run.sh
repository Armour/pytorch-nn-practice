source activate torch
# global enviroments
epochs=300
lr_decay_interval=75

# no log, normal train, normal test
GPUid=0

CUDA_VISIBLE_DEVICES=${GPUid}  python main.py --epochs 300 --lr-decay-interval 75


# no log, disturb train, normal test
GPUid=1

CUDA_VISIBLE_DEVICES=${GPUid}  python main.py --epochs 300 --lr-decay-interval 75  \
                                             -dtrain

# yes log, normal train, normal test
GPUid=2

CUDA_VISIBLE_DEVICES=${GPUid}  python main.py --epochs 300 --lr-decay-interval 75  \
                                             -l
# yes log, disturb on train, normal test
GPUid=3

CUDA_VISIBLE_DEVICES=${GPUid}  python main.py --epochs 300 --lr-decay-interval 75  \
                                             -l -dtrain
