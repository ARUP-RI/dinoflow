#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7


export OMP_NUM_THREADS=8

export TORCH_NCCL_TRACE_BUFFER_SIZE=1 # This is for debugging NCCL issues

torchrun --standalone --nnodes=1 --nproc-per-node=4  src/dinoflow/dino.py \
    $1 \
    --tube-type t \
    --run-name $2 \
    --checkpoint /home/22319/src/dinoflow/tcells_csq2_lr0025/tcells_csq2_lr0025_epoch40.pt
