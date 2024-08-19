#!/bin/bash

gpu_num="2"
finedata="dstc9_data"
load_ckpt="runs/dstc9_test_part/rb_1_gpu4_random_n3_seed713/checkpoint-2995/pred_test.json"
# load_ckpt="temp"
score_file="rb_score.json"


python3 scripts/scores_sl.py \
    --dataset "test" \
    --dataroot ${finedata} \
    --outfile ${load_ckpt} \
    --scorefile $score_file &&


cat $score_file