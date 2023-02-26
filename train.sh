#!/bin/bash
DATA_DIR=/home/liumin/litianyi/workspace/data/datasets
CONFIG_PATH=/home/liumin/litianyi/workspace/data/datasets/meta/meta_subject_01.yml
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_path $DATA_DIR \
    --config_path $CONFIG_PATH \
    --exp_name train04 \
    --epochs 150 \
    --lr 0.01 \
    --batch_size 48 \
    --snapshot_pref transformer \
    --gpus 0  \
    --eval-freq=1 \
    --clip-gradient=30 \
    --L 30
    --h 10
    --dff 720
    --N 8
    --lr_steps 50 100
    # --resume logs/train01/baseline_stage1_model_best.pth.tar \