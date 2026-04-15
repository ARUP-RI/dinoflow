#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=8

uv run src/dinoflow/emit_3tube_embs.py \
  --ckpt-path /home/31792/repos/dinoflow/dinoflow_eval_dinoflow_multi_bs16_8kev_lr0.00025_trunkmult0.2_noaddononly_ar_rvm/dinoflow_multi_bs16_8kev_lr0.00025_trunkmult0.2_noaddononly_ar_rvm_val/specificity_at_recall_0.99=0.312__epoch=39.ckpt \
  --input-csv /home/31792/dinoflow_ds/updated_test_set_09052025_pb_with_reports_for_alex.csv \
  --output-dir /mnt/ri_share/Data/alexr/flow3/dinoflow_multi_embs \
  --dataroot /mnt/ri_share/Data/flow_data/ \
  --events 8192 \
  --batch-size 128 \
  --num-workers 4 \
  --labelkey "ACTION_REQUIRED" \
  --save-mode per