#!/usr/bin/env bash
export PYTHONPATH=/home/cwh/coding/TrackViz
PYTHON2=/home/cwh/anaconda2/bin/python
VISION_DIR=/home/cwh/coding/taudl_pyt/taudl_iter/eval

source_names=("duke" "market"  "grid" "viper" "cuhk")
#source_names=("cuhk")
target_names=("market" "duke")


for j in "${!source_names[@]}"
do
    for i in "${!target_names[@]}"
    do
        train_list="../data/${target_names[$i]}/train.txt"
        transfer_feature="${VISION_DIR}/${source_names[$j]}_${target_names[$i]}-r-train/train_ft.mat"
        $PYTHON2 st_cluster.py --train_list $train_list --transfer_feature $transfer_feature --transfer ${source_names[$j]}_${target_names[$i]}-iter
    done
done