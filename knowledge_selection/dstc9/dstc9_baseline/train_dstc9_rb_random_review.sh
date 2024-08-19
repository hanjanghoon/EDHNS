#!/bin/bash
export CUDA_VISIBLE_DEVICES=8,9,10,11
port=8113

finedata="dstc9_data"
model_path="roberta-base"
pre_model="roberta"
save_ckpt="dstc9_test_part/rb_1_gpu4_random_n3_seed713"

# load_ckpt="runs/post9_large/rr2/ckpt-2464"
# --checkpoint ${load_ckpt} \
# --pareto \
# --eval_with_test \
#  --max_grad_norm 1 \
#--batch_size 2
#--knowledge_file test/knowledge.json

nohup python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port=${port} baseline_2step_eff/main.py \
    --task "selection" \
    --dataroot ${finedata} \
    --negative_sample_method "random" \
    --premodel ${pre_model} \
    --learning_rate 1e-5 \
    --negative_num 3 \
    --gradient_accumulation_steps 2 \
    --batch_size 4 \
    --num_train_epochs 5 \
    --seed 713 \
    --eval_with_test \
    --model_name_or_path ${model_path} \
    --exp_name ${save_ckpt} > acl_log_dstc9/rb_1_gpu4_random_n3_seed713.txt &
