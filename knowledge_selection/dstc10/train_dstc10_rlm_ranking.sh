#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
port=8467

finedata="dstc10_data_sample"
model_path="roberta-large"
pre_model="roberta"
save_ckpt="dstc10_sum/rlm_05_256_gpu8_rs2048_sam_margin"

# load_ckpt="runs/post9_large/rr2/ckpt-2464"
# --checkpoint ${load_ckpt} \
# --pareto \
#  --max_grad_norm 1 \
#--batch_size 2
#--knowledge_file test/knowledge.json
#--eval_with_test \

nohup python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=${port} baseline/main.py \
    --task "selection" \
    --dataroot ${finedata} \
    --negative_sample_method "ranking" \
    --premodel ${pre_model} \
    --ranking_step 2048 \
    --learning_rate 5e-6 \
    --history_max_tokens 384 \
    --eval_with_test \
    --negative_num 1 \
    --candidate_num 256 \
    --gradient_accumulation_steps 1 \
    --batch_size 4 \
    --num_train_epochs 10 \
    --multi_task \
    --seed 0 \
    --model_name_or_path ${model_path} \
    --exp_name ${save_ckpt} > acl_log_dstc10/rlm_05_256_gpu8_rs2048_sam_margin.txt &

