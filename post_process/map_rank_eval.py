from __future__ import division, print_function, absolute_import

import os
import numpy as np

def extract_info(dir_path):
    infos = []
    for image_name in sorted(os.listdir(dir_path)):
        arr = image_name.split('_')
        person = int(arr[0])
        camera = int(arr[1][1])
        infos.append((person, camera))

    return infos


def map_rank_eval(result_argsort):
    DATASET = '/home/cwh/coding/Market-1501'
    TEST = os.path.join(DATASET, 'bounding_box_test')
    TEST_NUM = min(19732, len(result_argsort[0]))
    QUERY = os.path.join(DATASET, 'query')
    QUERY_NUM = 3368

    test_info = extract_info(TEST)
    query_info = extract_info(QUERY)
    # for evaluate rank1 and map
    match = []
    junk = []

    for q_index, (qp, qc) in enumerate(query_info):
        tmp_match = []
        tmp_junk = []
        for t_index, (tp, tc) in enumerate(test_info):
            if tp == qp and qc != tc:
                tmp_match.append(t_index)
            elif tp == qp or tp == -1:
                tmp_junk.append(t_index)
        match.append(tmp_match)
        junk.append(tmp_junk)

    rank_1 = 0.0
    mAP = 0.0

    for idx in range(len(query_info)):
        recall = 0.0
        precision = 1.0
        hit = 0.0
        cnt = 0.0
        ap = 0.0
        YES = match[idx]
        IGNORE = junk[idx]
        rank_flag = True
        for i in range(0, TEST_NUM):
        # in unsupervised eval, sort ascend
        # for i in list(reversed(range(0, TEST_NUM))):
            k = result_argsort[idx][i]
            if k in IGNORE:
                continue
            else:
                cnt += 1
                if k in YES:
                    hit += 1
                    if rank_flag:
                        rank_1 += 1
                tmp_recall = hit / len(YES)
                tmp_precision = hit / cnt
                ap = ap + (tmp_recall - recall) * ((precision + tmp_precision) / 2)
                recall = tmp_recall
                precision = tmp_precision
                rank_flag = False
            if hit == len(YES):
                break
        mAP += ap

    print('Rank 1:\t%f' % (rank_1 / QUERY_NUM))
    print('mAP:\t%f' % (mAP / QUERY_NUM))
    return rank_1, mAP

if __name__ == '__main__':
    read_rank_list = np.genfromtxt('../data/viper_test/cross_filter_pid.log', delimiter=' ')
    map_rank_eval(read_rank_list)