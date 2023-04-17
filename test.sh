DATA_DIR=/home/litianyi/data/EgoMotion/
CONFIG_PATH=/home/litianyi/data/EgoMotion/meta_remy.yml
# DATA_DIR=/home/liumin/litianyi/workspace/data/datasets
# CONFIG_PATH=/home/liumin/litianyi/workspace/data/datasets/meta/meta_subject_01.yml
CUDA_VISIBLE_DEVICES=2 python test.py \
    --dataset_path $DATA_DIR \
    --config_path $CONFIG_PATH \
    --exp_name figure5 \
    --dataset EgoMotion \
    --batch_size 1 \
    --L 20 \
    --h 10 \
    --pose_dim 48 \
    --dff 1440 \
    --N 16 \
    --dropout 0.1 \
    --resume logs/train02_baseline/baseline_stage1_model_best.pth.tar \