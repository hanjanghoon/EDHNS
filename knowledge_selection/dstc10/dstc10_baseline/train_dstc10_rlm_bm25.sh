#!/bin/bash
export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
port=8491

finedata="dstc10_data"
model_path="roberta-large"
pre_model="roberta"
save_ckpt="dstc10_sum/rlm_05_gpu8_bm25_first"

# load_ckpt="runs/post9_large/rr2/ckpt-2464"
# --checkpoint ${load_ckpt} \
# --pareto \
# --eval_with_test \
#  --max_grad_norm 1 \
#--batch_size 2
#--knowledge_file test/knowledge.json

nohup python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=${port} baseline_scale/main.py \
    --task "selection" \
    --dataroot ${finedata} \
    --negative_sample_method "bm25" \
    --premodel ${pre_model} \
    --learning_rate 5e-6 \
    --negative_num 1 \
    --gradient_accumulation_steps 1 \
    --batch_size 4 \
    --num_train_epochs 10 \
    --eval_with_test \
    --multi_task \
    --seed 0 \
    --model_name_or_path ${model_path} \
    --exp_name ${save_ckpt} > acl_log_dstc10/rlm_05_gpu8_bm25_first.txt &
