#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
port=9291

finedata="dstc10_data_sample_wtest"
model_path="roberta-base"
pre_model="roberta"
save_ckpt="dstc10_sample_wtest/rbm_1_gpu4_random_n3_seed713"

# load_ckpt="runs/post9_large/rr2/ckpt-2464"
# --checkpoint ${load_ckpt} \
# --pareto \
#  --max_grad_norm 1 \
#--batch_size 2
#--knowledge_file test/knowledge.json
#--eval_with_test \

nohup python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port=${port} baseline_2step_eff/main.py \
    --task "selection" \
    --dataroot ${finedata} \
    --negative_sample_method "random" \
    --premodel ${pre_model} \
    --ranking_step 2048 \
    --learning_rate 1e-5 \
    --history_max_tokens 384 \
    --eval_with_test \
    --negative_num 3 \
    --candidate_num 64 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --multi_task \
    --seed 713 \
    --model_name_or_path ${model_path} \
    --exp_name ${save_ckpt} > acl_log_dstc10_wtest/rbm_1_gpu4_random_n3_seed713.txt &

