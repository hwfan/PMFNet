#!/usr/bin/env bash

model_name="e2e_pmf_net_R-50-FPN_1x"
EXP=$3
mkdir -p ./Outputs/logs/${EXP}

CUDA_VISIBLE_DEVICES=$1 python -u tools/test_net.py --dataset vcoco_test \
        --cfg configs/baselines/$model_name.yaml \
        --use_precomp_box \
        --mlp_head_dim 256 \
        --part_crop_size 5 --use_kps17 \
        --net_name $4 \
        --load_ckpt $2 |tee ./Outputs/logs/${EXP}/test-log.out
