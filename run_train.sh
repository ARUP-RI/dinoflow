#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=4,5


export OMP_NUM_THREADS=8

export TORCH_NCCL_TRACE_BUFFER_SIZE=1 # This is for debugging NCCL issues


# Implements DinoFlow to train single tube model

torchrun --standalone --nnodes=1 --nproc-per-node=2  src/dinoflow/dino.py \
    $1 \
    --tube-type t \
    --run-name $2 \
