#!/bin/bash
finedata="dstc10_data_sample_wtest"
load_ckpt="runs/dstc10_sample_wtest/rbm_1_gpu4_random_n3_seed713/checkpoint-7412/pred_test_part.json"
# load_ckpt="temp"
score_file="rb_score.json"


python3 scripts/scores_sl.py \
    --dataset "test" \
    --dataroot ${finedata} \
    --outfile ${load_ckpt} \
    --scorefile $score_file &&

cat $score_file