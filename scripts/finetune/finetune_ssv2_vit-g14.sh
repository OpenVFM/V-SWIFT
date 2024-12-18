#!/usr/bin/env bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_NTHREADS=32
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_ALGO=Ring
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export GLOO_SOCKET_IFNAME=eth0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export NNODES=1
export RANK=0
export ADDR="127.0.0.1"
export PORT="32500"
all_threads=$(nproc --all)
echo "Total CPU cores: $all_threads"
threads_per_gpu=$((all_threads / NUM_GPUS))
echo "Threads per GPU: $threads_per_gpu"

EXP_NAME=finetune_ssv2_vit-g14_v4
OUTPUT_DIR="work_dir_dali/finetune/${EXP_NAME}"

ROOT_PATH='video_dataset'
TRAIN_DATA_PATH="video_dataset/ssv2_train_new.csv"
VAL_DATA_PATH="video_dataset/ssv2_val_new.csv"
TEST_DATA_PATH="video_dataset/ssv2_val_new.csv"
PRETRAIN_MODEL_PATH="work_dir_dali/checkpoint-19.pth"

# update --batch_size ------>  Speedup
# update --dali_py_num_workers (80%+ * threads_per_gpu) ------>  Speedup
# FLASH=1 ------> open 'flash-attn' Speedup
# DALI=1 ------> DALI-aug
# TORCH=1 ------> torch-aug

TORCH=1 FLASH=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
        finetune_main_deepspeed.py \
        --model vit_giant_patch14_224 \
        --data_set SSV2 \
        --nb_classes 174 \
        --train_data_path ${TRAIN_DATA_PATH} \
        --val_data_path ${VAL_DATA_PATH} \
        --test_data_path ${TEST_DATA_PATH} \
        --data_root ${ROOT_PATH} \
        --finetune ${PRETRAIN_MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --epochs 50 \
        --save_ckpt_freq 2 \
        --batch_size 12 \
        --num_frames 16 \
        --tubelet_size 2 \
        --sparse_sampling \
        --dali_num_threads 4 \
        --dali_py_num_workers 12 \
        --input_size 224 \
        --short_side_size 224 \
        --smoothing 0.1 \
        --use_mean_pooling \
        --lr 2.3e-4 \
        --min_lr 1e-6 \
        --clip_grad 5.0 \
        --drop_path 0.25 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --warmup_epochs 5 \
        --layer_decay 0.9 \
        --test_tta_num_segment 2 \
        --test_tta_num_crop 3