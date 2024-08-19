#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
port=8332

finedata="dstc10_data_sample_wtest"
model_path="roberta-large"
pre_model="roberta"
save_ckpt="dstc10_sample_wtest/rlm_1_50_sbert_562_gpu4_rs16_n3_seed713"

# load_ckpt="runs/post9_large/rr2/ckpt-2464"
# --checkpoint ${load_ckpt} \
# --eval_with_test \
#  --max_grad_norm 1 \
#--batch_size 2
#--knowledge_file test/knowledge.json

nohup python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port=${port} baseline_2step_eff/main.py \
    --task "selection" \
    --dataroot ${finedata} \
    --negative_sample_method "ranking" \
    --premodel ${pre_model} \
    --ranking_step 16 \
    --learning_rate 1e-5 \
    --negative_num 3 \
    --candidate_num 50 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --eval_with_test \
    --seed 713 \
    --multi_task \
    --model_name_or_path ${model_path} \
    --exp_name ${save_ckpt} > acl_log_dstc10_wtest/rlm_1_50_sbert_562_gpu4_rs16_n3_seed713.txt &
