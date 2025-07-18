#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7


export OMP_NUM_THREADS=8

export TORCH_NCCL_TRACE_BUFFER_SIZE=1 # This is for debugging NCCL issues


# Implements DinoFlow to train single tube model

uv run torchrun --standalone --nnodes=1 --nproc-per-node=4  src/dinoflow/dino.py \
    $1 \
    --tube-type b \
    --run-name $2 \
