#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export OMP_NUM_THREADS=8

export TORCH_NCCL_TRACE_BUFFER_SIZE=1 # This is for debugging NCCL issues



# Produces a single unified model from checkpoints/backbones for each of the three tubes 
    # --freeze-backbone \
    # --freeze-backbone-layers 2 \
# READ:
# remove --pos_weight to get standard BCE loss without class eighting
#--pos-weight 0.0365 \ for weak33
# use:
# --checkpoint_monitor val_loss \
# --checkpoint_mode min \
# to use ragular validation loss as the checkpoint monitor metric
#--max-lr 0.00025 \ for the blanket
#--max-lr 0.00005 \ for the unbalances specialized models
#--max-lr 0.0001  \
#/mnt/ri_share/Data/flow2/new_labels_nopl_2025-08-26_casedx_2024-08-21_noreport_train_with_m_projections_PB_only.csv \
#/mnt/ri_share/Data/flow2/new_labels_nopl_2025-08-26_casedx_2024-08-21_noreport_test_with_m_projections_PB_only.csv \
#--dataroot /data2/brendan/flow/ \
#--batch-size 8 \ for L40 GPUS


uv run src/dinoflow/eval.py \
    $1 \
    /home/32210/test_things/lmd_conversion/updated_train_set_09052025_pb.csv \
    /home/32210/test_things/lmd_conversion/updated_test_set_09052025_pb.csv \
    /mnt/ri_share/Data/flow2/dinoflow_checkpoints/b_sml_moreaug/b_sml_moreaug_epoch99.pt \
    /mnt/ri_share/Data/flow2/dinoflow_checkpoints/m_sml_moreaug/m_sml_moreaug_epoch99.pt \
    /mnt/ri_share/Data/flow2/dinoflow_checkpoints/t_sml_moreaug/t_sml_moreaug_epoch99.pt \
    --labelkey 'ACTION_REQUIRED' \
    --mode 'binary' \
    --positive-repeat-factor 1 \
    --dataroot /mnt/ri_share/Data/flow_data/ \
    --events 16384 \
    --max-lr 0.00025 \
    --batch-size 16 \
    --epochs 100 \
    --comet-workspace r-i \
    --comet-project dinoflow-action-required-sept2025 \
    --checkpoint-monitor specificity_at_recall_0.99 \
    --checkpoint-mode max \
    #--pos-weight 0.0722 \
 
