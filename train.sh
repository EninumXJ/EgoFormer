#!/bin/bash
DATA_DIR=/data1/lty/dataset/egopose_dataset/datasets
CONFIG_PATH=/data1/lty/dataset/egopose_dataset/datasets/meta/meta_subject_01.yml
CUDA_VISIBLE_DEVICES=4,5 python train.py \
    --dataset_path $DATA_DIR \
    --config_path $CONFIG_PATH \
    --exp_name train05 \
    --epochs 10 \
    --lr 0.01 \
    --batch_size 16 \
    --snapshot_pref baseline_stage1 \
    --gpus 0 1 \
    --eval-freq=1 \
    --clip-gradient=10 \
    # --resume logs/train05/baseline_stage1_model_best.pth.tar \