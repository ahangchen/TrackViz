from post_process.track_prob import track_score
from util.file_helper import read_lines, read_lines_and, write, safe_remove
from util.serialize import pickle_load, pickle_save

line_idx = 0
camera_delta_s = pickle_load('data/top10/sorted_deltas.pickle')
interval_scores = pickle_load('data/top10/interval_scores.pickle')


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


def predict_track_scores():
    persons_deltas_score = pickle_load('data/top10/persons_deltas_score.pickle')
    if pickle_load('data/top10/persons_deltas_score.pickle') is not None:
        return persons_deltas_score
    predict_path = 'data/top10/renew_pid.log'
    answer_path = 'data/top10/test_tracks.txt'
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
    pickle_save('data/top10/persons_deltas_score.pickle', persons_deltas_score)
    return persons_deltas_score


def predict_img_scores():
    final_persons_scores = pickle_load('data/top10/persons_ap_scores.pickle')
    if pickle_load('data/top10/persons_ap_score.pickle') is not None:
        return final_persons_scores
    predict_score_path = 'data/top10/renew_ac.log'
    final_persons_scores = list()
    persons_scores = read_lines(predict_score_path)
    for person_scores in persons_scores:
        res_score = list()
        scores = person_scores.split()
        for score in scores:
            res_score.append(float(score))
        final_persons_scores.append(res_score)
    pickle_save('data/top10/persons_ap_scores.pickle', final_persons_scores)
    return final_persons_scores


def predict_pids():
    predict_persons = pickle_load('data/top10/predict_persons.pickle')
    if pickle_load('data/top10/predict_persons.pickle') is not None:
        return predict_persons
    predict_person_path = 'data/top10/renew_pid.log'
    predict_persons = list()
    persons_predicts = read_lines(predict_person_path)
    for person_predict in persons_predicts:
        res_pids = list()
        pids = person_predict.split()
        for pid in pids:
            res_pids.append(int(pid))
        predict_persons.append(res_pids)
    pickle_save('data/top10/predict_persons.pickle', predict_persons)
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


def st_filter_s():
    predict_path = 'data/top10/renew_pid.log'
    answer_path = 'data/top10/test_tracks.txt'
    filter_path = 'data/top10/filter_pid.log'
    sure_path = 'data/top10/sure_pid.log'
    safe_remove(filter_path)
    safe_remove(sure_path)
    answer_lines = read_lines(answer_path)
    real_tracks = list()
    for answer in answer_lines:
        info = answer.split('_')
        real_tracks.append([info[0], int(info[1][1]), int(info[2])])

    top_cnt = 5

    def predict_judge(line):
        global line_idx

        predict_idx_es = line.split()

        for i, predict_idx in enumerate(predict_idx_es):
            if i < top_cnt:
                write(filter_path, predict_idx + ' ')
            else:
                print(line_idx)
                print(len(real_tracks))
                time1 = real_tracks[int(predict_idx) - 1][2]
                time2 = real_tracks[line_idx][2]
                c1 = real_tracks[int(predict_idx) - 1][1]
                c2 = real_tracks[line_idx][1]
                score = track_score(camera_delta_s, c1, time1, c2, time2)
                # score, [0, 0.6] shot rate up, [0.6, 0.85], shot rate keep the same
                if score > 0.9:
                    write(filter_path, predict_idx + ' ')
                # score [0.85, 0.9], all shot up, line shot down
                # score [0.9, 1], show rate down
                # all shot max is 53%, line shot max is 83%
                if score > 0.98:
                    write(sure_path, predict_idx + ' ')
        line_idx += 1
        write(filter_path, '\n')
        write(sure_path, '\n')
    global line_idx
    line_idx = 0
    read_lines_and(predict_path, predict_judge)


def st_img_filter():
    persons_ap_scores = predict_img_scores()
    persons_ap_pids = predict_pids()
    persons_track_scores = predict_track_scores()
    log_path = 'data/top10/double_filter_pid.log'
    safe_remove(log_path)
    for i, person_ap_pids in enumerate(persons_ap_pids):
        for j, person_ap_pid in enumerate(person_ap_pids):
            # top 5 will be a more accurate filter
            if j >= 5:
                break
            # more strict track filter lead to more accurate result, but 0.5 will be more available
            if persons_track_scores[i][j] > 0.9 and persons_ap_scores[i][j] > 0.9:
                write(log_path, '%d ' % person_ap_pid)
        write(log_path, '\n')


def cross_st_img_ranker():
    persons_ap_scores = predict_img_scores()
    persons_ap_pids = predict_pids()
    persons_track_scores = predict_track_scores()

    persons_cross_scores = list()
    log_path = 'data/top10/cross_filter_pid.log'
    map_score_path = 'data/top10/cross_filter_score.log'
    score_path = 'data/top10/raw_cross_filter_score.log'
    renew_path = 'data/top10/renew_pid1.log'
    renew_ac_path = 'data/top10/renew_ac1.log'
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


if __name__ == '__main__':
    # st_scissors()
    # st_img_ranker()
    cross_st_img_ranker()