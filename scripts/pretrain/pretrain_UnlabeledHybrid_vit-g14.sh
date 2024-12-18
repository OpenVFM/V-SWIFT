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
export PORT="32501"
all_threads=$(nproc --all)
echo "Total CPU cores: $all_threads"
threads_per_gpu=$((all_threads / NUM_GPUS))
echo "Threads per GPU: $threads_per_gpu"


ADDR=$1
NNODES=$2
RANK=$3
echo "NNODES: $NNODES"
echo "ADDR: $ADDR"
echo "RANK: $RANK"

EXP_NAME=pretrain_UnlabeledHybrid_vit-g14
OUTPUT_DIR="work_dir_dali/pretrain/${EXP_NAME}"
ROOT_PATH='video_dataset'
#DATA_PATH="video_dataset/all_videos_train.txt"
DATA_PATH="video_dataset/ssv2_train_new.csv"

# update --batch_size ------>  Speedup
# update --dali_py_num_workers (90% * threads_per_gpu) ------>  max Speedup
# FLASH=1 ------> open 'flash-attn' Speedup

FLASH=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
        pretrain_main_bf16.py \
        --model pixel_pretrain_videomae_giant_patch14_224 \
        --data_path ${DATA_PATH} \
        --data_root ${ROOT_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --epochs 20 \
        --save_ckpt_freq 1 \
        --batch_size 60 \
        --gpus_not_equal_num_shards \
        --set_max_sample 160000 \
        --num_frames 16 \
        --tubelet_size 2 \
        --sampling_rate 4 \
        --dali_num_threads 4 \
        --dali_py_num_workers 8 \
        --mask_ratio 0.9 \
        --decoder_depth 4 \
        --input_size 224 \
        --drop_path 0.0 \
        --lr 2e-3 \
        --min_lr 1e-6 \
        --clip_grad 0.05 \
        --opt_betas 0.9 0.98 \
        --warmup_epochs 5 \
        --weight_decay 0.05