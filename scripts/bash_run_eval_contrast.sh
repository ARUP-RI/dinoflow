#!/usr/bin/bash

# Evaluate a ContrastClassificationModel checkpoint trained with train3tubes --mode contrast-binary
# 
# Usage: ./bash_run_eval_contrast.sh
#
# Modify the paths below to point to your checkpoint and test CSV

export CUDA_VISIBLE_DEVICES=1

# Path to your trained checkpoint
CHECKPOINT="/mnt/ri_share/Data/alexr/flow3/ckpts_to_test/bs16_lr0.00025_ev16k_cw0.01+warmup_unfreeze_specificity_at_recall_0.99=0.168__epoch=57.ckpt"

# Path to your test CSV
TEST_CSV="/mnt/ri_share/Data/flow2/updated_test_set_09052025_pb_with_reports_for_alex.csv"

# Data root directory
DATAROOT="/mnt/ri_share/Data/flow_data/"

# Label column in CSV
LABELKEY="ACTION_REQUIRED"

# Number of events per sample (should match training)
EVENTS=16384

# Output prefix for results
OUTPUT_PREFIX="contrast_eval_results_epoch57_v2"

# Number of subsamples (1 = single prediction, >1 = average multiple random subsamples)
NUM_SUBSAMPLES=3

# Run evaluation
uv run evaluate_contrast_model.py \
    "$CHECKPOINT" \
    "$TEST_CSV" \
    --dataroot "$DATAROOT" \
    --labelkey "$LABELKEY" \
    --events "$EVENTS" \
    --output-prefix "$OUTPUT_PREFIX" \
    --num-subsamples "$NUM_SUBSAMPLES"

