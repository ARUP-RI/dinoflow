#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,6,7


export OMP_NUM_THREADS=8

export TORCH_NCCL_TRACE_BUFFER_SIZE=1 # This is for debugging NCCL issues


# Produces a single unified model from checkpoints/backbones for each of the three tubes 

python src/dinoflow/eval.py train3tubes \
    b_p4k_fix_koleo/b_p4k_fix_koleo_epoch99.pt \
    t_p4k_fix_koleo/t_p4k_fix_koleo_epoch99.pt \
    m_p4k_fix_koleo/m_p4k_fix_koleo_epoch99.pt \
    /data2/brendan/flow/casedx_2024-08-21_train_valid_3xlowpct.csv \
    /data2/brendan/flow/casedx_2024-08-21_test_valid.csv \
    $1 \
    --labelkey 'CLL' \
    --positive-repeat-factor 1 \
    --dataroot /data2/brendan/flow/ \
    --events 8192 \
    --max-lr 0.0005 \
    --batch-size 256 \
    --epochs 30

