#!/bin/bash
DATA_DIR=/home/litianyi/dataset/egopose_dataset/datasets
CONFIG_PATH=/home/litianyi/dataset/egopose_dataset/datasets/meta/meta_subject_01.yml
CUDA_VISIBLE_DEVICES=1,2,3 python train.py \
    --dataset_path $DATA_DIR \
    --config_path $CONFIG_PATH \
    --exp_name train02 \
    --epochs 20 \
    --lr 0.005 \
    --batch_size 32 \
    --snapshot_pref baseline_stage1 \
    --gpus 0 1 2 \
    --eval-freq=1 \
    --clip-gradient=20 \
    --resume logs/train01/baseline_stage1_model_best.pth.tar \