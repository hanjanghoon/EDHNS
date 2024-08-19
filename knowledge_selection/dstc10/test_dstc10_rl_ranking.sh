#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
port=7878


finedata="dstc10_data"
model_path="roberta-large"
pre_model="roberta"
load_ckpt="runs/dstc10_sample/rlm_05_gpu8_random/checkpoint-9690/pred_test_part.json"
# load_ckpt="temp"

nohup python3 -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=${port} baseline/main.py \
    --eval_only \
    --checkpoint ${load_ckpt}/ \
    --max_candidates_per_forward_eval 256 \
    --task "selection" \
    --dataroot ${finedata} \
    --model_name_or_path ${model_path} \
    --exp_name ${load_ckpt}_test > ${load_ckpt}/test_logs.txt &