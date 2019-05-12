#!/usr/bin/env bash
export PYTHONPATH=/home/cwh/coding/TrackViz
PYTHON2=/home/cwh/anaconda2/bin/python
PICKLE_DIR=/home/cwh/coding/TrackViz/pre_process

source_names=("duke" "market"  "grid" "viper" "cuhk")
target_names=("market" "duke")

for j in "${!source_names[@]}"
do
    for i in "${!target_names[@]}"
    do
        train_list="../data/${target_names[$i]}/train.txt"
        $PYTHON2 cluster_eval.py --track_path ${PICKLE_DIR}/${source_names[$j]}_${target_names[$i]}-iter_cluster.pck
    done
done