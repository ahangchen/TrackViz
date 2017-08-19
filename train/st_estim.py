#coding=utf-8
from random import randint

import shutil

from util.file_helper import read_lines, safe_remove
from util.serialize import pickle_save
from os.path import exists


def get_predict_delta_tracks(fusion_param, useful_predict_cnt=10, random=False):
    if 'r-' in fusion_param['renew_pid_path']:
        print 'incremental process, copy origin distribution pickle'
        # 在增量前去做删除
        # safe_remove(fusion_param['distribution_pickle_path'])
        try:
            if not exists(fusion_param['distribution_pickle_path']):
                # 直接使用增量前的时空模型
                shutil.copy(fusion_param['distribution_pickle_path'].replace('r-', ''), fusion_param['distribution_pickle_path'])
                print 'copy train track distribute pickle done'
            else:
                # 多次增量
                print 'already exists'
        except shutil.Error:
            print 'copy error'
        return None

    # 获取左图列表
    answer_path = fusion_param['answer_path']
    answer_lines = read_lines(answer_path)

    real_tracks = list()
    for answer in answer_lines:
        info = answer.split('_')
        if 'bmp' in info[2]:
            #
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            # grid
            real_tracks.append([info[0], int(info[1][0]), int(info[2]), 1])
        else:
            # market
            real_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])
    print 'left image ready'
    # 获取右图列表
    renew_pid_path = fusion_param['renew_pid_path']
    predict_lines = read_lines(renew_pid_path)
    print 'predict images ready'
    camera_cnt = 6
    # 左图中的人在右图可能出现在6个摄像头中
    camera_delta_s = [[list() for j in range(camera_cnt)] for i in range(camera_cnt)]
    person_cnt = len(answer_lines)
    # market1501数据集有六个序列，只有同一个序列才能计算delta
    for i, line in enumerate(predict_lines):
        predict_pids = line.split(' ')
        for j, predict_pid in enumerate(predict_pids):
            if j > useful_predict_cnt:
                break
            if random:
                predict_pid = randint(0, person_cnt - 1)
            else:
                predict_pid = int(predict_pid) - 1
            # same seq
            if real_tracks[i][3] == real_tracks[predict_pid][3]:
                delta = real_tracks[i][2] - real_tracks[predict_pid][2]
                if abs(delta) < 1000000:
                    camera_delta_s[real_tracks[i][1] - 1][real_tracks[predict_pid][1] - 1].append(delta)
    print 'deltas collected'
    for camera_delta in camera_delta_s:
        for delta_s in camera_delta:
            delta_s.sort()
    print 'deltas sorted'
    safe_remove(fusion_param['distribution_pickle_path'])
    safe_remove(fusion_param['rand_distribution_pickle_path'])
    pickle_save(fusion_param['distribution_pickle_path'], camera_delta_s)
    print 'deltas saved'
    return camera_delta_s