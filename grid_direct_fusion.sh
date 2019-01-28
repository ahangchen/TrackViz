#!/usr/bin/env bash
export PYTHONPATH=/home/cwh/coding/TrackViz
PYTHON2=/home/cwh/anaconda2/bin/python
PYTHON3=/home/cwh/anaconda3/bin/python
LD_LIBRAR_PATH=/usr/local/cuda/lib64
VISION_DIR=/home/cwh/coding/taudl_pyt/baseline/eval

for i in {0..9}
do
    data_folder_path="grid-cv-${i}_grid-cv-${i}-train"
    vision_folder_path="${VISION_DIR}/grid-cv-${i}_grid-cv-${i}-train"
    $PYTHON2 img_st_fusion.py --data_folder_path $data_folder_path --vision_folder_path $vision_folder_path
done