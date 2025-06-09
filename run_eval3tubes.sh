#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3


export OMP_NUM_THREADS=8

export TORCH_NCCL_TRACE_BUFFER_SIZE=1 # This is for debugging NCCL issues


# Produces a single unified model from checkpoints/backbones for each of the three tubes 

uv run src/dinoflow/eval.py train3tubes \
    $1 \
    /home/22319/data/brendan/dinoflow/casedx_2024-08-21_noreport_train_with_m_projections.csv \
    /home/22319/data/brendan/dinoflow/casedx_2024-08-21_noreport_test_with_m_projections.csv \
    /home/22319/data/brendan/dinoflow/checkpoints/b_sml_moreaug/b_sml_moreaug_epoch99.pt \
    /home/22319/data/brendan/dinoflow/checkpoints/t_sml_moreaug/t_sml_moreaug_epoch99.pt \
    /home/22319/data/brendan/dinoflow/checkpoints/m_sml_moreaug/m_sml_moreaug_epoch99.pt \
    --labelkey 'NORMAL' \
    --positive-repeat-factor 1 \
    --dataroot /data2/brendan/flow/ \
    --freeze-backbone-layers 2 \
    --events 4096 \
    --max-lr 0.00025 \
    --batch-size 32 \
    --epochs 50 \
    --comet-workspace brendan \
    --comet-project dinoflow-normal

