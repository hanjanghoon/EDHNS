#!/bin/bash
export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
port=8923

finedata="dstc9_data"
model_path="roberta-large"
pre_model="roberta"
save_ckpt="dstc9_test_part/rl_05_gpu8_static10_n3_seed713"

# load_ckpt="runs/post9_large/rr2/ckpt-2464"
# --checkpoint ${load_ckpt} \
# --pareto \
#  --max_grad_norm 1 \
#--batch_size 2
#--knowledge_file test/knowledge.json
#--eval_with_test \

nohup python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=${port} baseline_static/main.py \
    --task "selection" \
    --dataroot ${finedata} \
    --negative_sample_method "ranking" \
    --premodel ${pre_model} \
    --learning_rate 5e-6 \
    --history_max_tokens 384 \
    --negative_num 3 \
    --candidate_num 10 \
    --gradient_accumulation_steps 2 \
    --batch_size 2 \
    --num_train_epochs 5 \
    --seed 713 \
    --model_name_or_path ${model_path} \
    --exp_name ${save_ckpt} > acl_log_dstc9/rl_05_gpu8_static10_n3_seed713.txt &

