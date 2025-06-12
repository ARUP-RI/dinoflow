#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7


export OMP_NUM_THREADS=8

export TORCH_NCCL_TRACE_BUFFER_SIZE=1 # This is for debugging NCCL issues

# This script is for training a classifier head using a *single* dinoflow trained backbone
# The first script argument is the run name, used for comet logs and saved checkpoint data
# CSVs must contain paths to the tube data under the 'path' column
# The targets to train for are given by the --labelkey argument (AML below), which must also be in the CSV
# --mode can be binary, multiclass, or regression (for viability training)

# These CSVs have binary labels for the usual 30+ different standard diagnoses
# /home/22319/data/brendan/dinoflow/casedx_2024-08-21_noreport_train_with_m_projections.csv \
# /home/22319/data/brendan/dinoflow/casedx_2024-08-21_noreport_test_with_m_projections.csv \

# These CSVs are for viability training only - mode must be 'regression' and the labelkey must be 'viability' to use these
# /home/22319/data/brendan/dinoflow/viabs_250304_train_with_projections.csv \
#    /home/22319/data/brendan/dinoflow/viabs_250304_test_with_projections.csv \
# Batch size 128
    
uv run src/dinoflow/eval.py train3tubes \
    $1 \
    /mnt/ri_share/Data/flow2/casedx_2024-08-21_noreport_train_with_m_projections.csv \
    /mnt/ri_share/Data/flow2/casedx_2024-08-21_noreport_test_with_m_projections.csv \
    /mnt/ri_share/Data/flow2/dinoflow_checkpoints/b_sml_moreaug/b_sml_moreaug_epoch99.pt \
    /mnt/ri_share/Data/flow2/dinoflow_checkpoints/m_sml_moreaug/m_sml_moreaug_epoch99.pt \
    /mnt/ri_share/Data/flow2/dinoflow_checkpoints/t_sml_moreaug/t_sml_moreaug_epoch99.pt \
    conf.yaml \
    --labelkey 'NORMAL' \
    --mode 'binary' \
    --dataroot /data2/brendan/flow/ \
    --events 4096 \
    --batch-size 32 \
    --epochs 25 \
    --comet-workspace "r-i" \
    --comet-project "test-dino" \
    # --freeze-backbone \
    # --freeze-backbone-layers 2 \


