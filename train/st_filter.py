#coding=utf-8

import math

from post_process.predict_eval import eval_on_train_test
from post_process.track_prob import track_score
from profile.fusion_param import get_fusion_param, ctrl_msg
from util.file_helper import read_lines, read_lines_and, write, safe_remove
from util.serialize import pickle_load

line_idx = 0
track_score_idx = 0
data_type = 1


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


def predict_track_scores(camera_delta_s, fusion_param):
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

    global track_score_idx
    track_score_idx = 0

    def predict_judge(line):
        global track_score_idx

        predict_idx_es = line.split()
        person_deltas_score = list()
        for i, predict_idx in enumerate(predict_idx_es):
            # if i >= top_cnt:
            #     break
            if len(real_tracks[int(predict_idx) - 1]) > 3:
                s1 = real_tracks[int(predict_idx) - 1][3]
                s2 = real_tracks[track_score_idx][3]
                if s1 != s2:
                    person_deltas_score.append(-1.0)
                    continue
            time1 = real_tracks[int(predict_idx) - 1][2]
            if track_score_idx == 3914:
                print 'test'
            time2 = real_tracks[track_score_idx][2]
            c1 = real_tracks[int(predict_idx) - 1][1]
            c2 = real_tracks[track_score_idx][1]
            # 给定摄像头，时间，获取时空评分，这里camera_deltas如果是随机算出来的，则是随机评分
            score = track_score(camera_delta_s, c1, time1, c2, time2, interval=100)
            person_deltas_score.append(score)
        track_score_idx += 1
        persons_deltas_score.append(person_deltas_score)

    read_lines_and(predict_path, predict_judge)
    # pickle_save(fusion_param['persons_deltas_path'], persons_deltas_score)
    return persons_deltas_score


def predict_smooth_track_scores(camera_delta_s, fusion_param):
    # fusion_param = get_fusion_param()
    # persons_deltas_score = pickle_load(fusion_param['persons_deltas_path'])
    # if pickle_load(fusion_param['persons_deltas_path']) is not None:
    #     return persons_deltas_score
    predict_path = fusion_param['renew_pid_path']
    answer_path = fusion_param['answer_path']
    answer_lines = read_lines(answer_path)
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

    global track_score_idx
    track_score_idx = 0

    def predict_judge(line):
        global track_score_idx

        predict_idx_es = line.split()
        person_deltas_score = list()
        for i, predict_idx in enumerate(predict_idx_es):
            # if i >= top_cnt:
            #     break
            if len(real_tracks[int(predict_idx) - 1]) > 3:
                s1 = real_tracks[int(predict_idx) - 1][3]
                s2 = real_tracks[track_score_idx][3]
                if s1 != s2:
                    person_deltas_score.append(-1.0)
                    continue
            time1 = real_tracks[int(predict_idx) - 1][2]
            if track_score_idx == 3914:
                print 'test'
            time2 = real_tracks[track_score_idx][2]
            c1 = real_tracks[int(predict_idx) - 1][1]
            c2 = real_tracks[track_score_idx][1]
            track_interval = 20
            smooth_window_size = 10
            smooth_scores = [
                track_score(camera_delta_s, c1, time1-(smooth_window_size/2-1)*track_interval+j*track_interval, c2, time2, interval=track_interval)
                for j in range(smooth_window_size)]
            # filter
            for j in range(smooth_window_size):
                if smooth_scores[j] < 0.01:
                    smooth_scores[j] = 0
            # smooth
            score = sum(smooth_scores)/len(smooth_scores)
            # if score < 0.001:
            #     score = 0
            person_deltas_score.append(score)
        track_score_idx += 1
        persons_deltas_score.append(person_deltas_score)

    read_lines_and(predict_path, predict_judge)
    # pickle_save(fusion_param['persons_deltas_path'], persons_deltas_score)
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



def fusion_st_img_ranker(fusion_param, pos_shot_rate=0.5, neg_shot_rate=0.01):
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
    score_path = fusion_param['fusion_raw_score_path']
    renew_path = fusion_param['fusion_pid_path']
    renew_ac_path = fusion_param['fusion_score_path']
    safe_remove(map_score_path)
    safe_remove(log_path)
    safe_remove(score_path)
    safe_remove(renew_path)
    safe_remove(renew_ac_path)
    line_log_cnt = 10

    print(sum([sum(persons_track_scores[j]) for j in range(len(persons_track_scores))])/len(persons_track_scores)/len(persons_track_scores[0]))
    for i, person_ap_pids in enumerate(persons_ap_pids):
        cross_scores = list()
        for j, person_ap_pid in enumerate(person_ap_pids):
            cur_track_score = persons_track_scores[i][j]
            # if cur_track_score < 0.001:
            #     cur_track_score = 0
            rand_track_score = rand_track_scores[i][j]
            # 平滑分母噪音，否则时空会失效
            if rand_track_score < 0.02:
                rand_track_score = 0.02
            cross_score = cur_track_score * persons_ap_scores[i][j] / rand_track_score
            cross_scores.append(cross_score)
        persons_cross_scores.append(cross_scores)
    for i, person_cross_scores in enumerate(persons_cross_scores):
        for j, person_cross_score in enumerate(person_cross_scores):
            if person_cross_score < 0:
                # 不同序列，时空评分设为1，不过grid上没有这个问题
                print 'diff seq use img score'
                persons_cross_scores[i][j] *= -0.02

    max_score = max([max(predict_cross_scores) for predict_cross_scores in persons_cross_scores])
    for i, person_cross_scores in enumerate(persons_cross_scores):
        for j, person_cross_score in enumerate(person_cross_scores):
            # if j == 1:
                # print('cur_rank1: %f, max: %f' % (person_cross_score, max_score))
            # 归一化
            persons_cross_scores[i][j] /= max_score
            # if persons_cross_scores[i][j] > 0.9:
            #     print '%d,%d' % (i, j)
    person_score_idx_s = list()
    # top1 statics
    top1_scores = list()
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
        write(score_path, '\n')
        write(map_score_path, '\n')


def fusion_curve(fusion_param):
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    rand_camera_deltas = pickle_load(fusion_param['rand_distribution_pickle_path'])

    delta_range = map(lambda x: x*30.0 - 15000.0, range(1000))
    # delta_range = map(lambda x: x*1.0 - 60.0, range(120))
    raw_probs = [[list() for j in range(6)] for i in range(6)]
    rand_probs = [[list() for j in range(6)] for i in range(6)]
    over_probs = [[list() for j in range(6)] for i in range(6)]
    for i in range(6):
        for j in range(6):
            for k in range(len(delta_range)):
                match_track_score = track_score(camera_delta_s, i + 1, 0, j + 1, delta_range[k], interval=100)
                rand_track_score = track_score(rand_camera_deltas, i + 1, 0, j + 1, delta_range[k], interval=100)
                if rand_track_score < 0.02:
                    # print rand_track_score
                    rand_track_score = 0.02
                else:
                    print match_track_score/rand_track_score

                raw_probs[i][j].append(match_track_score)
                rand_probs[i][j].append(match_track_score)
                over_probs[i][j].append(match_track_score/rand_track_score)
    return delta_range, raw_probs, rand_probs, over_probs

if __name__ == '__main__':
    # st_scissors()
    # st_img_ranker()
    ctrl_msg['data_folder_path'] = 'top-m2g-std0-test'
    fusion_param = get_fusion_param()
    # cross_st_img_ranker(fusion_param)
    fusion_st_img_ranker(fusion_param, fusion_param['pos_shot_rate'], fusion_param['neg_shot_rate'])
    eval_on_train_test(fusion_param, test_mode=True)
