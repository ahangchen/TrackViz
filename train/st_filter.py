# coding=utf-8

from post_process.track_prob import track_score
from profile.fusion_param import get_fusion_param, ctrl_msg
from util.file_helper import read_lines, read_lines_and, write, safe_remove
from util.serialize import pickle_load
from viz.delta_track import viz_fusion_curve, viz_heat_map

import numpy as np
import pandas as pd


def real_track(answer_path):
    answer_lines = read_lines(answer_path)
    real_tracks = list()
    for answer in answer_lines:
        info = answer.split('_')
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            real_tracks.append([info[0], int(info[1][0]), int(info[2])])
        else:
            real_tracks.append([info[0], int(info[1][1]), int(info[2])])
    return real_tracks


def smooth_score(c1, c2, time1, time2, camera_delta_s):
    track_interval = 20
    smooth_window_size = 10
    smooth_scores = [
        track_score(camera_delta_s, c1,
                    time1 - (smooth_window_size / 2 - 1) * track_interval + j * track_interval, c2, time2,
                    interval=track_interval)
        for j in range(smooth_window_size)]
    # filter
    for j in range(smooth_window_size):
        if smooth_scores[j] < 0.01:
            smooth_scores[j] = 0
    # smooth
    score = sum(smooth_scores) / len(smooth_scores)
    return score


def predict_track_scores(camera_delta_s, fusion_param, smooth=False):
    # fusion_param = get_fusion_param()
    # persons_deltas_score = pickle_load(fusion_param['persons_deltas_path'])
    # if pickle_load(fusion_param['persons_deltas_path']) is not None:
    #     return persons_deltas_score
    predict_path = fusion_param['renew_pid_path']
    # test_tracks.txt
    answer_path = fusion_param['answer_path']
    answer_lines = read_lines(answer_path)
    # 左图
    real_tracks = list()
    for answer in answer_lines:
        info = answer.split('_')
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            real_tracks.append([info[0], int(info[1][0]), int(info[2])])
        else:
            real_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])
    top_cnt = 10
    persons_deltas_score = list()
    pids4probes = np.genfromtxt(predict_path, delimiter=' ')
    for probe_i, pids4probe in enumerate(pids4probes):
        person_deltas_score = list()
        for pid4probe in pids4probe:
            # todo transfer: if predict by python, start from 0, needn't minus 1
            pid4probe = int(pid4probe)
            # predict_idx = predict_idx - 1
            if len(real_tracks[pid4probe]) > 3:
                s1 = real_tracks[pid4probe][3]
                s2 = real_tracks[probe_i][3]
                if s1 != s2:
                    person_deltas_score.append(-1.0)
                    continue
            time1 = real_tracks[pid4probe][2]
            # if track_score_idx == 3914:
            #    print 'test'
            time2 = real_tracks[probe_i][2]
            c1 = real_tracks[pid4probe][1]
            c2 = real_tracks[probe_i][1]
            if smooth:
                score = smooth_score(c1, c2, time1, time2, camera_delta_s)
            else:
                # 给定摄像头，时间，获取时空评分，这里camera_deltas如果是随机算出来的，则是随机评分
                score = track_score(camera_delta_s, c1, time1, c2, time2, interval=100)
            person_deltas_score.append(score)
        probe_i += 1
        persons_deltas_score.append(person_deltas_score)

    return persons_deltas_score


def predict_img_scores(fusion_param):
    # fusion_param = get_fusion_param()
    # final_persons_scores = pickle_load(fusion_param['persons_ap_path'])
    # if pickle_load(fusion_param['persons_ap_path']) is not None:
    #     return final_persons_scores
    predict_score_path = fusion_param['renew_ac_path']
    final_persons_scores = list()
    persons_scores = read_lines(predict_score_path)
    for person_scores in persons_scores:
        res_score = list()
        scores = person_scores.split()
        for score in scores:
            res_score.append(float(score))
        final_persons_scores.append(res_score)
    # pickle_save(fusion_param['persons_ap_path'], final_persons_scores)
    return final_persons_scores


def predict_pids(fusion_param):
    # fusion_param = get_fusion_param()
    # predict_persons = pickle_load(fusion_param['predict_person_path'])
    # if pickle_load(fusion_param['predict_person_path']) is not None:
    #     return predict_persons
    predict_person_path = fusion_param['renew_pid_path']
    predict_persons = list()
    persons_predicts = read_lines(predict_person_path)
    for person_predict in persons_predicts:
        res_pids = list()
        pids = person_predict.split()
        for pid in pids:
            res_pids.append(int(pid))
        predict_persons.append(res_pids)
    # pickle_save(fusion_param['predict_person_path'], predict_persons)
    return predict_persons


def get_person_pids(predict_path):
    predict_person_path = predict_path
    predict_persons = list()
    persons_predicts = read_lines(predict_person_path)
    for person_predict in persons_predicts:
        res_pids = list()
        pids = person_predict.split()
        for pid in pids:
            res_pids.append(int(pid))
        predict_persons.append(res_pids)
    return predict_persons


def fusion_st_img_ranker(fusion_param):
    # 从renew_pid和renew_ac获取预测的人物id和图像分数
    persons_ap_scores = predict_img_scores(fusion_param)
    persons_ap_pids = predict_pids(fusion_param)
    # 从磁盘获取之前建立的时空模型，以及随机时空模型
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    rand_delta_s = pickle_load(fusion_param['rand_distribution_pickle_path'])
    # 计算时空评分和随机时空评分
    persons_track_scores = predict_track_scores(camera_delta_s, fusion_param)
    rand_track_scores = predict_track_scores(rand_delta_s, fusion_param)

    persons_cross_scores = list()
    log_path = fusion_param['eval_fusion_path']
    map_score_path = fusion_param['fusion_normal_score_path']
    safe_remove(map_score_path)
    safe_remove(log_path)
    line_log_cnt = 10

    for i, person_ap_pids in enumerate(persons_ap_pids):
        cross_scores = list()
        for j, person_ap_pid in enumerate(person_ap_pids):
            if rand_track_scores[i][j] < 0.02:
                cross_score = persons_track_scores[i][j] * persons_ap_scores[i][j] / 0.02
            else:
                cross_score = persons_track_scores[i][j] * persons_ap_scores[i][j] / rand_track_scores[i][j]
            cross_scores.append(cross_score)
        persons_cross_scores.append(cross_scores)
    print 'img score ready'
    max_score = max([max(predict_cross_scores) for predict_cross_scores in persons_cross_scores])

    for i, person_cross_scores in enumerate(persons_cross_scores):
        for j, person_cross_score in enumerate(person_cross_scores):
            if person_cross_score > 0:
                # diff seq not sort and not normalize
                persons_cross_scores[i][j] /= max_score
            else:
                person_cross_score *= -0.02
    person_score_idx_s = list()
    top1_scores = list()
    print 'above person score ready'
    for i, person_cross_scores in enumerate(persons_cross_scores):
        # 单个probe的预测结果中按score排序，得到index，用于对pid进行排序
        sort_score_idx_s = sorted(range(len(person_cross_scores)), key=lambda k: -person_cross_scores[k])
        person_score_idx_s.append(sort_score_idx_s)
        # 统计top1分布，后面计算中位数用
        top1_scores.append(person_cross_scores[sort_score_idx_s[0]])
    # 降序排，取前60%处的分数
    sorted_top1_scores = sorted(top1_scores, reverse=True)
    mid_score = sorted_top1_scores[int(len(sorted_top1_scores) * 0.5)]
    mid_score_path = fusion_param['mid_score_path']
    safe_remove(mid_score_path)
    write(mid_score_path, '%f\n' % mid_score)
    print(str(mid_score))
    for i, person_ap_pids in enumerate(persons_ap_pids):
        # img_score_s = list()
        # img_score_idx_s = list()
        # for j in range(len(person_ap_pids)):
        #     img_score_idx_s.append(person_ap_pids.index(person_ap_pids[person_score_idx_s[i][j]]))
        #     img_score_s.append(persons_ap_scores[i][img_score_idx_s[j]])
        # sort_img_score_s = sorted(img_score_s, reverse=True)
        for j in range(len(person_ap_pids)):
            # write(map_score_path, '%f ' % sort_img_score_s[j])
            # 按score排序得到的index对pid进行排序
            write(map_score_path, '%f ' % (persons_cross_scores[i][person_score_idx_s[i][j]]))
            write(log_path, '%d ' % person_ap_pids[person_score_idx_s[i][j]])
        write(log_path, '\n')
        write(map_score_path, '\n')


def gallery_track_scores(camera_delta_s, fusion_param, smooth=False):
    predict_path = fusion_param['renew_pid_path']
    # answer path is probe path
    answer_path = fusion_param['answer_path']
    answer_lines = read_lines(answer_path)
    query_tracks = list()
    for answer in answer_lines:
        info = answer.split('_')
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            query_tracks.append([info[0], int(info[1][0]), int(info[2])])
        else:
            query_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])

    gallery_path = fusion_param['gallery_path']
    gallery_lines = read_lines(gallery_path)
    gallery_tracks = list()
    for gallery in gallery_lines:
        info = gallery.split('_')
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            gallery_tracks.append([info[0], int(info[1][0]), int(info[2])])
        else:
            gallery_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])

    persons_deltas_score = list()
    pids4probes = np.genfromtxt(predict_path, delimiter=' ')
    for probe_i, pids4probe in enumerate(pids4probes):
        person_deltas_score = list()
        for i, pid4probe in enumerate(pids4probe):
            # if i >= top_cnt:
            #     break
            pid4probe = int(pid4probe)
            probe_i_tmp = probe_i - 1  # (probe_i + 1) % len(pids4probes)
            # todo transfer: if predict by python, start from 0, needn't minus 1
            # predict_idx = predict_idx - 1
            if len(query_tracks[probe_i_tmp]) > 3:
                s1 = query_tracks[probe_i_tmp][3]
                # print predict_idx
                s2 = gallery_tracks[pid4probe][3]
                if s1 != s2:
                    person_deltas_score.append(-1.0)
                    continue
            time1 = query_tracks[probe_i_tmp][2]
            # if track_score_idx == 3914:
            #     print 'test'
            time2 = gallery_tracks[pid4probe][2]
            c1 = query_tracks[probe_i_tmp][1]
            c2 = gallery_tracks[pid4probe][1]
            if smooth:
                score = smooth_score(c1, c2, time1, time2, camera_delta_s)
            else:
                # 给定摄像头，时间，获取时空评分，这里camera_deltas如果是随机算出来的，则是随机评分
                score = track_score(camera_delta_s, c1, time1, c2, time2, interval=100)
            person_deltas_score.append(score)
        probe_i += 1
        persons_deltas_score.append(person_deltas_score)

    return persons_deltas_score


def fusion_st_gallery_ranker(fusion_param):
    log_path = fusion_param['eval_fusion_path']
    map_score_path = fusion_param['fusion_normal_score_path']  # fusion_param = get_fusion_param()
    persons_ap_scores = predict_img_scores(fusion_param)
    persons_ap_pids = predict_pids(fusion_param)
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    rand_delta_s = pickle_load(fusion_param['rand_distribution_pickle_path'])
    persons_track_scores = gallery_track_scores(camera_delta_s, fusion_param)
    rand_track_scores = gallery_track_scores(rand_delta_s, fusion_param)

    persons_cross_scores = list()
    safe_remove(map_score_path)
    safe_remove(log_path)

    for i, person_ap_pids in enumerate(persons_ap_pids):
        cross_scores = list()
        for j, person_ap_pid in enumerate(person_ap_pids):
            cur_track_score = persons_track_scores[i][j]
            # if cur_track_score < 0.02:
            #     cur_track_score = cur_track_score
            rand_track_score = rand_track_scores[i][j]
            if rand_track_score < 0.02:
                rand_track_score = 0.02
            cross_score = cur_track_score * persons_ap_scores[i][j] / rand_track_score
            cross_scores.append(cross_score)
        persons_cross_scores.append(cross_scores)
        # pickle_save(ctrl_msg['data_folder_path']+'viper_r-testpersons_cross_scores.pick', persons_cross_scores)
        # pickle_save(ctrl_msg['data_folder_path']+'viper_r-testpersons_ap_pids.pick', persons_ap_pids)

    max_score_s = [max(predict_cross_scores) for predict_cross_scores in persons_cross_scores]
    min_score_s = [min(predict_cross_scores) for predict_cross_scores in persons_cross_scores]
    max_score = max(max_score_s)
    for i, person_cross_scores in enumerate(persons_cross_scores):
        for j, person_cross_score in enumerate(person_cross_scores):
            if persons_cross_scores[i][j] >= 0:
                # diff seq not sort, not rank for max, and not normalize
                if max_score_s[i] == 0:
                    print i
                persons_cross_scores[i][j] /= max_score_s[i]
                # persons_cross_scores[i][j] /= max_score
                # if persons_cross_scores[i][j] > 0.5:
                #     print 'same'
                #     print persons_cross_scores[i][j]
            else:
                # so diff seq is negative, normalize by minimum
                # persons_cross_scores[i][j] /= min_score_s[i]
                # persons_cross_scores[i][j] *= 1.0
                persons_cross_scores[i][j] *= -0.02
                # print persons_cross_scores[i][j]
    person_score_idx_s = list()

    for i, person_cross_scores in enumerate(persons_cross_scores):
        # 单个probe的预测结果中按score排序，得到index，用于对pid进行排序
        sort_score_idx_s = sorted(range(len(person_cross_scores)), key=lambda k: -person_cross_scores[k])
        person_score_idx_s.append(sort_score_idx_s)

    for i, person_ap_pids in enumerate(persons_ap_pids):
        for j in range(len(person_ap_pids)):
            write(log_path, '%d ' % person_ap_pids[person_score_idx_s[i][j]])
            write(map_score_path, '%.3f ' % persons_cross_scores[i][person_score_idx_s[i][j]])
        write(log_path, '\n')
        write(map_score_path, '\n')
    return person_score_idx_s


def fusion_curve(fusion_param):
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    rand_camera_deltas = pickle_load(fusion_param['rand_distribution_pickle_path'])
    delta_width = 6000.0
    delta_cnt = 30
    interval_width = delta_width/ delta_cnt
    delta_stripe = delta_width / delta_cnt
    delta_range = map(lambda x: x * delta_stripe - delta_width/2, range(delta_cnt))
    # delta_range = map(lambda x: x*1.0 - 60.0, range(120))
    over_probs = [[list() for j in range(6)] for i in range(6)]
    for i in range(6):
        for j in range(i+1, 6):
            for k in range(len(delta_range)):
                match_track_score = track_score(camera_delta_s, i + 1, 0, j + 1, delta_range[k], interval=interval_width)
                rand_track_score = track_score(rand_camera_deltas, i + 1, 0, j + 1, delta_range[k], interval=interval_width)
                if rand_track_score < 0.02:
                    # print rand_track_score
                    rand_track_score = 0.02
                else:
                    print match_track_score / rand_track_score

                # raw_probs[i][j].append(match_track_score)
                # rand_probs[i][j].append(rand_track_score)
                over_probs[i][j].append(match_track_score / rand_track_score)
    return delta_range, over_probs


def fusion_heat(fusion_param):
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    rand_camera_deltas = pickle_load(fusion_param['rand_distribution_pickle_path'])
    delta_width = 3000.0
    delta_cnt = 15
    interval_width = delta_width/ delta_cnt
    delta_stripe = delta_width / delta_cnt
    delta_range = map(lambda x: x * delta_stripe - delta_width/2, range(delta_cnt))
    # delta_range = map(lambda x: x*1.0 - 60.0, range(120))
    viz_deltas = list()
    viz_camera_pairs = list()
    viz_values = list()
    for i in range(6):
        for j in range(i+1, 6):
            for k in range(len(delta_range)):
                match_track_score = track_score(camera_delta_s, i + 1, 0, j + 1, delta_range[k], interval=interval_width)
                rand_track_score = track_score(rand_camera_deltas, i + 1, 0, j + 1, delta_range[k], interval=interval_width)
                if rand_track_score < 0.02:
                    # print rand_track_score
                    rand_track_score = 0.02
                else:
                    print match_track_score / rand_track_score

                # raw_probs[i][j].append(match_track_score)
                # rand_probs[i][j].append(rand_track_score)
                viz_deltas.append(delta_range[k])
                viz_camera_pairs.append('c%d-c%d' % (i + 1, j + 1))
                viz_values.append(match_track_score)
        break
    df = pd.DataFrame({'transfer_time': viz_deltas,
                       'camera_pair': viz_camera_pairs,
                       'values': viz_values})
    pt = df.pivot_table(index='camera_pair', columns='transfer_time', values='values', aggfunc=np.sum)
    return pt

if __name__ == '__main__':
    ctrl_msg['data_folder_path'] = 'market_market-test'
    # fusion_param = get_fusion_param()
    # fusion_st_img_ranker(fusion_param, fusion_param['pos_shot_rate'], fusion_param['neg_shot_rate'])
    # eval_on_train_test(fusion_param, test_mode=True)
    fusion_param = get_fusion_param()
    # fusion_st_gallery_ranker(fusion_param)
    # delta_range, over_probs = fusion_curve(fusion_param)
    # viz_fusion_curve(delta_range, [over_probs])
    pt = fusion_heat(fusion_param)
    viz_heat_map(pt)
