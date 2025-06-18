#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=4


export OMP_NUM_THREADS=8


TUBE=t

uv run src/dinoflow/emitembeds.py \
    /mnt/ri_share/Data/flow2/dinoflow_checkpoints/${TUBE}_sml_moreaug/${TUBE}_sml_moreaug_epoch99.pt \
    /mnt/ri_share/Data/flow2/casedx_2024-08-21_noreport_train_with_m_projections.csv \
    /mnt/ri_share/Data/combined_flow_bm/dinoflow_embeddings/${TUBE}_sml_moreaug \
    ${TUBE} \
    --dataroot /data2/brendan/flow/ \
    --events 8192 \
    --batch-size 128 \

