from post_process.track_prob import track_score
from profile.fusion_param import get_fusion_param, ctrl_msg
from util.file_helper import read_lines, read_lines_and, write, safe_remove
from util.serialize import pickle_load, pickle_save

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
    persons_deltas_score = pickle_load(fusion_param['persons_deltas_path'])
    if pickle_load(fusion_param['persons_deltas_path']) is not None:
        return persons_deltas_score
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
            real_tracks.append([info[0], int(info[1][1]), int(info[2])])
    top_cnt = 10
    persons_deltas_score = list()

    def predict_judge(line):
        global track_score_idx

        predict_idx_es = line.split()
        person_deltas_score = list()
        for i, predict_idx in enumerate(predict_idx_es):
            # if i >= top_cnt:
            #     break
            time1 = real_tracks[int(predict_idx) - 1][2]
            if track_score_idx == 3914:
                print 'test'
            time2 = real_tracks[track_score_idx][2]
            c1 = real_tracks[int(predict_idx) - 1][1]
            c2 = real_tracks[track_score_idx][1]
            score = track_score(camera_delta_s, c1, time1, c2, time2)
            person_deltas_score.append(score)
        track_score_idx += 1
        persons_deltas_score.append(person_deltas_score)

    read_lines_and(predict_path, predict_judge)
    pickle_save(fusion_param['persons_deltas_path'], persons_deltas_score)
    return persons_deltas_score


def predict_img_scores(fusion_param):
    # fusion_param = get_fusion_param()
    final_persons_scores = pickle_load(fusion_param['persons_ap_path'])
    if pickle_load(fusion_param['persons_ap_path']) is not None:
        return final_persons_scores
    predict_score_path = fusion_param['renew_ac_path']
    final_persons_scores = list()
    persons_scores = read_lines(predict_score_path)
    for person_scores in persons_scores:
        res_score = list()
        scores = person_scores.split()
        for score in scores:
            res_score.append(float(score))
        final_persons_scores.append(res_score)
    pickle_save(fusion_param['persons_ap_path'], final_persons_scores)
    return final_persons_scores


def predict_pids(fusion_param):
    # fusion_param = get_fusion_param()
    predict_persons = pickle_load(fusion_param['predict_person_path'])
    if pickle_load(fusion_param['predict_person_path']) is not None:
        return predict_persons
    predict_person_path = fusion_param['renew_pid_path']
    predict_persons = list()
    persons_predicts = read_lines(predict_person_path)
    for person_predict in persons_predicts:
        res_pids = list()
        pids = person_predict.split()
        for pid in pids:
            res_pids.append(int(pid))
        predict_persons.append(res_pids)
    pickle_save(fusion_param['predict_person_path'], predict_persons)
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


def cross_st_img_ranker(fusion_param):
    # fusion_param = get_fusion_param()
    persons_ap_scores = predict_img_scores(fusion_param)
    persons_ap_pids = predict_pids(fusion_param)
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    persons_track_scores = predict_track_scores(camera_delta_s, fusion_param)

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
    # not limit this count for logging all probability
    line_log_cnt = 10

    for i, person_ap_pids in enumerate(persons_ap_pids):
        cross_scores = list()
        for j, person_ap_pid in enumerate(person_ap_pids):
            cross_score = (persons_track_scores[i][j] * 1) * (persons_ap_scores[i][j] * 1)
            cross_scores.append(cross_score**0.5)
        persons_cross_scores.append(cross_scores)

    max_score = max([max(predict_cross_scores) for predict_cross_scores in persons_cross_scores])
    person_score_idx_s = list()

    for i, person_cross_scores in enumerate(persons_cross_scores):
        sort_score_idx_s = sorted(range(len(person_cross_scores)), key=lambda k: -person_cross_scores[k])
        person_score_idx_s.append(sort_score_idx_s)

    for i, person_ap_pids in enumerate(persons_ap_pids):
        img_score_s = list()
        img_score_idx_s = list()
        for j in range(len(person_ap_pids)):
            if j < line_log_cnt:
                img_score_idx_s.append(person_ap_pids.index(person_ap_pids[person_score_idx_s[i][j]]))
                img_score_s.append(persons_ap_scores[i][img_score_idx_s[j]])
        sort_img_score_s = sorted(img_score_s, reverse=True)
        for j in range(len(person_ap_pids)):
            if j < line_log_cnt:
                write(map_score_path, '%f ' % sort_img_score_s[j])
                write(score_path, '%f ' % (persons_cross_scores[i][person_score_idx_s[i][j]] / max_score))
                write(log_path, '%d ' % person_ap_pids[person_score_idx_s[i][j]])
            write(renew_ac_path, '%f ' % (persons_cross_scores[i][person_score_idx_s[i][j]]))
            write(renew_path, '%d ' % person_ap_pids[person_score_idx_s[i][j]])
        write(log_path, '\n')
        write(score_path, '\n')
        write(map_score_path, '\n')
        write(renew_path, '\n')
        write(renew_ac_path, '\n')


def fusion_st_img_ranker(fusion_param, pos_shot_rate, neg_shot_rate):
    # fusion_param = get_fusion_param()
    persons_ap_scores = predict_img_scores(fusion_param)
    persons_ap_pids = predict_pids(fusion_param)
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    persons_track_scores = predict_track_scores(camera_delta_s, fusion_param)
    rand_delta_s = pickle_load(fusion_param['rand_distribution_pickle_path'])
    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'] + '_rand'
    fusion_param = get_fusion_param()
    rand_track_scores = predict_track_scores(rand_delta_s, fusion_param)

    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-5]
    fusion_param = get_fusion_param()

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
    # not limit this count for logging all probability
    line_log_cnt = 10

    for i, person_ap_pids in enumerate(persons_ap_pids):
        cross_scores = list()
        for j, person_ap_pid in enumerate(person_ap_pids):
            m1 = persons_ap_scores[i][j] * (1 - pos_shot_rate - neg_shot_rate) + \
                 fusion_param['neg_shot_rate']
            p = rand_track_scores[i][j]
            m2 = (persons_track_scores[i][j] - p * pos_shot_rate)/(1 - neg_shot_rate)

            cross_score = m1*m2/(m1*m2+(1-m1)*p)
            if cross_score < 0:
                print('Sv:%f, Sst:%f, Srst:%f, Ep:%f, En:%f, m1:%f, m2:%f' % (
                    persons_ap_scores[i][j], persons_track_scores[i][j], p,
                    pos_shot_rate, neg_shot_rate, m1, m2))
            cross_scores.append(cross_score)
        persons_cross_scores.append(cross_scores)

    max_score = max([max(predict_cross_scores) for predict_cross_scores in persons_cross_scores])
    person_score_idx_s = list()

    for i, person_cross_scores in enumerate(persons_cross_scores):
        sort_score_idx_s = sorted(range(len(person_cross_scores)), key=lambda k: -person_cross_scores[k])
        person_score_idx_s.append(sort_score_idx_s)

    for i, person_ap_pids in enumerate(persons_ap_pids):
        img_score_s = list()
        img_score_idx_s = list()
        for j in range(len(person_ap_pids)):
            if j < line_log_cnt:
                img_score_idx_s.append(person_ap_pids.index(person_ap_pids[person_score_idx_s[i][j]]))
                img_score_s.append(persons_ap_scores[i][img_score_idx_s[j]])
        sort_img_score_s = sorted(img_score_s, reverse=True)
        for j in range(len(person_ap_pids)):
            if j < line_log_cnt:
                write(map_score_path, '%f ' % sort_img_score_s[j])
                write(score_path, '%f ' % (persons_cross_scores[i][person_score_idx_s[i][j]] / max_score))
                write(log_path, '%d ' % person_ap_pids[person_score_idx_s[i][j]])
            write(renew_ac_path, '%f ' % (persons_cross_scores[i][person_score_idx_s[i][j]]))
            write(renew_path, '%d ' % person_ap_pids[person_score_idx_s[i][j]])
        write(log_path, '\n')
        write(score_path, '\n')
        write(map_score_path, '\n')
        write(renew_path, '\n')
        write(renew_ac_path, '\n')


if __name__ == '__main__':
    # st_scissors()
    # st_img_ranker()
    fusion_param = get_fusion_param()
    # cross_st_img_ranker(fusion_param)
    fusion_st_img_ranker(fusion_param, fusion_param['pos_shot_rate'], fusion_param['neg_shot_rate'])