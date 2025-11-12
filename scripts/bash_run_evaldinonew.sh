#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=7


#/home/32210/git/dinoflow/dinoflow_eval_dinoflow_ar_16kev_PB_lr0001/last.ckpt
#/home/32210/test_things/lmd_conversion/updated_test_set_09052025.csv 

uv run evalute_dinoflow_new.py \
/home/32210/git/dinoflow/dinoflow_eval_dinoflow_ar_16kev_PB_lr00025_100epochs_bs16/dinoflow_ar_16kev_PB_lr00025_100epochs_bs16_specificity_at_recall_0.99=0.210__epoch=57.ckpt \
/home/32210/test_things/lmd_conversion/updated_test_set_09052025_pb.csv \
--dataroot /mnt/ri_share/Data/flow_data/ \
--labelkey ACTION_REQUIRED \
--events 16384 \
--output-prefix my_evaluation_oct2325 \
--num-subsamples 3