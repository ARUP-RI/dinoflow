#!/usr/bin/bash
set -e

# Example script for running single-tube DinoFlow inference
# Adjust paths according to your setup

# Configuration
CHECKPOINT="/mnt/ri_share/Data/flow2/dinoflow_checkpoints/dinoflow_eval_b_moreaug_viab_unfrz_8k/b_moreaug_viab_unfrz_8k_rmse=4.281__epoch=41.ckpt"
TEST_CSV="/home/32210/test_things/lmd_conversion/updated_test_set_09052025_pb.csv"
TUBE_TYPE="b"  # or 't' or 'm'
LABELKEY="label"  # or 'DISEASE_FREE' depending on your CSV
DATAROOT="/mnt/ri_share/Data/flow_data/"
EVENTS=16384
BATCH_SIZE=16
TASK_TYPE="regression"  # 'auto', 'binary', or 'regression'
OUTPUT_CSV="predictions_$(date +%Y%m%d_%H%M%S).csv"

# Resolve this script's directory so it works from any CWD
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Run inference (use system python to avoid uv TLS issues)
python "$DIR/one_tube_viability_eval.py" \
    "$CHECKPOINT" \
    "$TEST_CSV" \
    --tube-type "$TUBE_TYPE" \
    --labelkey "$LABELKEY" \
    --dataroot "$DATAROOT" \
    --events "$EVENTS" \
    --batch-size "$BATCH_SIZE" \
    --task-type "$TASK_TYPE" \
    --output-csv "$OUTPUT_CSV"

echo ""
echo "=========================================="
echo "Inference complete!"
echo "Results saved to: $OUTPUT_CSV"
echo "=========================================="

