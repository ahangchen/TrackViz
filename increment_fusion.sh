#!/usr/bin/env bash
export PYTHONPATH=/home/cwh/coding/TrackViz
PYTHON2=/home/cwh/anaconda2/bin/python
PYTHON3=/home/cwh/anaconda3/bin/python
LD_LIBRAR_PATH=/usr/local/cuda/lib64
VISION_DIR=/home/cwh/coding/taudl_pyt/transfer/eval

source_names=("duke" "market"  "grid" "viper" "cuhk")
#source_names=("grid")
target_names=("market" "duke")
#target_names=("market")

for j in "${!source_names[@]}"
do
    for i in "${!target_names[@]}"
    do
        if [ ${target_names[$i]} != ${source_names[$j]} ] ; then
            data_folder_path="${source_names[$j]}_${target_names[$i]}-r-train"
            vision_folder_path="${VISION_DIR}/${source_names[$j]}_${target_names[$i]}-r-train"
            $PYTHON2 img_st_fusion.py --data_folder_path $data_folder_path --vision_folder_path $vision_folder_path &
        fi
    done


#    for i in {0..1}
#    for i in {0..9}
#    do
#        data_folder_path="${source_names[$j]}_grid-cv-${i}-r-train"
#        vision_folder_path="${VISION_DIR}/${source_names[$j]}_grid-cv-${i}-r-train"
#        $PYTHON2 img_st_fusion.py --data_folder_path $data_folder_path --vision_folder_path $vision_folder_path
#    done
done