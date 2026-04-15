#!/usr/bin/bash

export TORCH_NCCL_TRACE_BUFFER_SIZE=1 # This is for debugging NCCL issues
export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=8

# This script is for training a classifier head using a *single* dinoflow trained backbone
# The first script argument is the run name, used for comet logs and saved checkpoint data
# CSVs must contain paths to the tube data under the 'path' column
# The targets to train for are given by the --labelkey argument (AML below), which must also be in the CSV
# --mode can be binary, multiclass, or regression (for viability training)

# These CSVs are for viability training only - mode must be 'regression' and the labelkey must be 'viability' to use these
# /home/22319/data/brendan/dinoflow/viabs_250304_train_with_projections.csv \
#    /home/22319/data/brendan/dinoflow/viabs_250304_test_with_projections.csv \
# Batch size 128
    
uv run torchrun --standalone --nproc_per_node=4 src/dinoflow/eval.py train3tubesmulti \
    $1 \
    /home/31792/dinoflow_ds/20260314_dinoflow_multi_train_wdb_noaddon.csv \
    /home/31792/dinoflow_ds/20260314_dinoflow_multi_test_wdb_noaddon.csv \
    /mnt/ri_share/Data/flow2/dinoflow_checkpoints/b_sml_moreaug/b_sml_moreaug_epoch99.pt \
    /mnt/ri_share/Data/flow2/dinoflow_checkpoints/m_sml_moreaug/m_sml_moreaug_epoch99.pt \
    /mnt/ri_share/Data/flow2/dinoflow_checkpoints/t_sml_moreaug/t_sml_moreaug_epoch99.pt \
    /home/31792/repos/dinoflow/src/dinoflow/multitask_train_v2.yaml \
    --labelkey 'ACTION_REQUIRED' \
    --dataroot /mnt/ri_share/Data/flow_data/ \
    --events 16384 \
    --batch-size 16 \
    --epochs 100 \
    --comet-workspace "alex-rangel" \
    --comet-project "dinoflow-multi" \
    #--freeze-backbone \
    #--freeze-backbone-layers 2 \


