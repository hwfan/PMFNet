#!/usr/bin/env bash

model_name="e2e_pmf_net_R-50-FPN_1x"
EXP="final_trainval_DEBUG"
mkdir -p ./Outputs/logs/${EXP}

CUDA_VISIBLE_DEVICES=0,1 python -u tools/test_net.py --dataset vcoco_test \
        --cfg configs/baselines/$model_name.yaml \
        --use_precomp_box \
        --mlp_head_dim 256 \
        --part_crop_size 5 --use_kps17 \
        --net_name PMFNet_Final \
        --load_ckpt ./Outputs/exp/final_trainval/ckpt/model_step47999.pth |tee ./Outputs/logs/${EXP}/test-log.out
