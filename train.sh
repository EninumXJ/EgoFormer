#!/bin/bash
DATA_DIR=/data1/lty/dataset/egopose_dataset/datasets
CONFIG_PATH=/data1/lty/dataset/egopose_dataset/datasets/meta/meta_subject_01.yml
CUDA_VISIBLE_DEVICES=3,4 python train.py \
    --dataset_path $DATA_DIR \
    --config_path $CONFIG_PATH \
    --exp_name train02 \
    --epochs 10 \
    --lr 0.001 \
    --batch_size 16 \
    --snapshot_pref baseline_stage1 \
    --gpus 0 1 \
    --eval-freq=1 \
    --clip-gradient=0.5 \