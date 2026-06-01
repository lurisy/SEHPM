#!/bin/bash
for((i=0;i<=9;i++));
do
    echo $i
    TXT_PATH='./results/test_'"$i"'.txt'
    CKPT_PATH='./minigpt4/output/20260528112/'"$i"'/checkpoint_6.pth'
    JSON_DIR="./datasets/imagenet-r/20_20_order3_2000exp"

    python batch_eval.py \
        --cfg-path eval_configs/minigpt4_eval_all_tasks_imgr.yaml \
        --gpu-id 0 \
        --task-id $i \
        --txt-path "$TXT_PATH" \
        --ckpt-path "$CKPT_PATH" \
        --json-dir "$JSON_DIR"
done
python get_score_all.py



