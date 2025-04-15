#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,6,7


export OMP_NUM_THREADS=8

export TORCH_NCCL_TRACE_BUFFER_SIZE=1 # This is for debugging NCCL issues


uv run src/dinoflow/eval.py train \
    $1 \
    /home/22319/data/brendan/dinoflow/casedx_2024-08-21_noreport_train_with_m_projections.csv \
    /home/22319/data/brendan/dinoflow/casedx_2024-08-21_noreport_test_with_m_projections.csv \
    /home/22319/data/brendan/dinoflow/checkpoints/b_p4k_fix_koleo/b_p4k_fix_koleo_epoch99.pt \
    conf.yaml \
    --labelkey 'AML' \
    --mode 'binary' \
    --tube-type m \
    --dataroot /data2/brendan/flow/ \
    --events 4096 \
    --batch-size 64 \
    --epochs 50

