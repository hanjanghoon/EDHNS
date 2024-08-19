#!/bin/bash
export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
port=7299

finedata="dstc10_data_sample_wtest"
model_path="roberta-large"
pre_model="roberta"
save_ckpt="dstc10_sample_wtest/rlm_05_gpu8_random_n3_seed713"

# load_ckpt="runs/post9_large/rr2/ckpt-2464"
# --checkpoint ${load_ckpt} \
# --pareto \
#  --max_grad_norm 1 \
#--batch_size 2
#--knowledge_file test/knowledge.json
#--eval_with_test \

nohup python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=${port} baseline_2step_eff/main.py \
    --task "selection" \
    --dataroot ${finedata} \
    --negative_sample_method "random" \
    --premodel ${pre_model} \
    --ranking_step 2048 \
    --learning_rate 5e-6 \
    --history_max_tokens 384 \
    --eval_with_test \
    --negative_num 3 \
    --candidate_num 64 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --multi_task \
    --seed 713 \
    --model_name_or_path ${model_path} \
    --exp_name ${save_ckpt} > acl_log_dstc10_wtest/rlm_05_gpu8_random_n3_seed713.txt &

