#!/bin/sh

export CUDA_VISIBLE_DEVICES=2

uv run src/dinoflow/somtrain.py infer \
    ~/src/dinoflow/som_conf.yaml \
    /data2/brendan/flow/converted_flowtubes/ \
    t  \
    /home/22319/data/brendan/dinoflow/checkpoints/t_longer_som_checkpoints/t_longer_som_m32_n32_dim13_epoch99.p \
    --destdir ~/data/brendan/dinoflow/som_projections/t_longer_e99 
