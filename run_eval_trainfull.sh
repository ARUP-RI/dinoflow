#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7


export OMP_NUM_THREADS=8

export TORCH_NCCL_TRACE_BUFFER_SIZE=1 # This is for debugging NCCL issues

# Continue training a 3-tube model 
# First arg is a checkpoint as produced by the train3tubes function
# Second arg is run name

python src/dinoflow/eval.py continue-training \
    $1 \
    /data2/brendan/flow/casedx_2024-08-21_train_valid_3xlowpct.csv \
    /data2/brendan/flow/casedx_2024-08-21_test_valid.csv \
    $2 \
    --model-class BinaryClassificationModel \
    --labelkey 'CLL' \
    --positive-repeat-factor 1 \
    --freeze-backbone-layers 6 \
    --dataroot /data2/brendan/flow/ \
    --events 4096 \
    --batch-size 32 \
    --epochs 50

