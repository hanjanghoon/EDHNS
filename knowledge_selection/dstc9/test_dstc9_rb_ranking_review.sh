#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
port=7172


finedata="dstc9_data"
model_path="roberta-base"
pre_model="roberta"
load_ckpt="runs/dstc9_test_part/rb_1_10_sbert_cr045_150_gpu4_rs16_n3_seed404/checkpoint-886"
# load_ckpt="temp"

nohup python3 -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=${port} baseline/main.py \
    --eval_only \
    --checkpoint ${load_ckpt} \
    --max_candidates_per_forward_eval 256 \
    --task "selection" \
    --dataroot ${finedata} \
    --model_name_or_path ${model_path} \
    --exp_name ${load_ckpt}_test > ${load_ckpt}/test_logs.txt &