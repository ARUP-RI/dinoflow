#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

CKPT="/home/31792/repos/dinoflow/dinoflow_eval_dinoflow_multi_bs16_8kev_lr0.00025_trunkmult0.2_allheads_noaddononly_apgate/dinoflow_multi_bs16_8kev_lr0.00025_trunkmult0.2_allheads_noaddononly_apgate_val/specificity_at_recall_0.99=0.310__epoch=23.ckpt"
CSV="/home/31792/dinoflow_ds/updated_test_set_09052025_pb_with_reports_for_alex.csv"
DATAROOT="/mnt/ri_share/Data/flow_data/"
OUTDIR="./eval_outputs"
OUTPREFIX="${OUTDIR}/20260413_allheads_e23_16kev_arpreds"

# Eval knobs
EVENTS=16384
NUM_SUBSAMPLES=3

mkdir -p "${OUTDIR}"

echo "Running multitask eval"
echo "Checkpoint: ${CKPT}"
echo "CSV: ${CSV}"
echo "Dataroot: ${DATAROOT}"
echo "Outprefix: ${OUTPREFIX}"
echo "Events: ${EVENTS}"
echo "Subsamples: ${NUM_SUBSAMPLES}"

uv run evaluate_multi_model.py \
  "${CKPT}" \
  "${CSV}" \
  --dataroot "${DATAROOT}" \
  --events "${EVENTS}" \
  --num-subsamples "${NUM_SUBSAMPLES}" \
  --output-prefix "${OUTPREFIX}" \

echo "Done."
echo "Predictions: ${OUTPREFIX}_predictions.csv"
echo "Metrics: ${OUTPREFIX}_metrics_by_task.csv"
