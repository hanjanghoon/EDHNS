#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
port=8353

train_path=e-com/data/train_SNDS_2000_001.jsonl
val_path=e-com/data/valid_mdns.pkl
test_path=e-com/data/test_mdns.pkl
res_path=e-com/data/train_res_2000_001.pkl


load_ckpt="runs/2312/ecom_FT_2000_001_seed713"

# --multi_task \
# --seed 1 \
# --pareto \
# --max_grad_norm 3 \
# 46 max
#--checkpoint ${load_ckpt} \
nohup python3 -m torch.distributed.launch --nproc_per_node=16 \
    --master_port=${port} main_dtb_FP_ecom.py \
    --negative_sample_method "ranking" \
    --train_path=${train_path} \
    --val_path=${val_path} \
    --test_path=${test_path} \
    --res_path=${res_path} \
    --premodel "bert_ch" \
    --learning_rate 1e-5 \
    --negative_num 2 \
    --candidates_num 10 \
    --ranking_step 16 \
    --max_seq 512 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --epochs 5 \
    --eval_only \
    --checkpoint ${load_ckpt} \
    --exp_name test > ${load_ckpt}/test_score.txt &