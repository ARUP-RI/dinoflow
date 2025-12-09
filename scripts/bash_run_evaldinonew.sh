#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=3
export UV_SSL_NO_VERIFY=1  # Add this line


#/home/32210/git/dinoflow/dinoflow_eval_dinoflow_ar_16kev_PB_lr0001/last.ckpt
#/home/32210/test_things/lmd_conversion/updated_test_set_09052025.csv 
#/home/32210/git/dinoflow/dinoflow_eval_dinoflow_ar_16kev_PB_lr00025_100epochs_bs16/dinoflow_ar_16kev_PB_lr00025_100epochs_bs16_specificity_at_recall_0.99=0.210__epoch=57.ckpt 

uv run evalute_dinoflow_new.py \
/home/32210/git/dinoflow/dinoflow_eval_dinoflow_ar_16kev_PB_lr00025_100epochs_bs16/dinoflow_ar_16kev_PB_lr00025_100epochs_bs16_specificity_at_recall_0.99=0.210__epoch=57.ckpt \
/home/32210/test_things/lmd_conversion/updated_test_set_09052025_pb.csv \
--dataroot /mnt/ri_share/Data/flow_data/ \
--labelkey ACTION_REQUIRED \
--events 16384 \
--output-prefix my_evaluation_dinoflow_repo_11132025 \
--num-subsamples 3