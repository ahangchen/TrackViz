from delta_track import viz_data_for_market
from file_helper import read_lines_and, write
from file_helper import write_line
from raw_data import train_track_path
from serialize import pickle_save
from st_filter import predict_img_scores, predict_pids

predict_path = 'top10/predict_pid.log'
test_path = 'top10/test_tracks.txt'


last_class = 0


def get_person_idx():
    ids = list()

    def count_id(line):
        global last_class
        cur_class = int(line.split('_')[0])
        if cur_class != last_class:
            ids.append(cur_class)
            last_class = cur_class

    read_lines_and(train_track_path, count_id)
    print(ids)
    return ids


def get_tracks():
    tracks = list()

    def add_track(line):
        tracks.append(line)

    read_lines_and(test_path, add_track)
    return tracks

renew_pid_path = 'top10/renew_pid.log'
renew_score_path= 'top10/renew_ac.log'


def renew_tracks():
    persons_ap_scores = predict_img_scores()
    persons_ap_pids = predict_pids()
    new_ap_scores = [list() for _ in range(3914)]
    new_ap_pids = [list() for _ in range(3914)]
    for i in range(3913):
        new_ap_scores[i] = persons_ap_scores[i][:]
        new_ap_pids[i] = persons_ap_pids[i][:]
    new_ap_pids.append(list())
    new_ap_scores.append(list())
    for i, person_ap_pids in enumerate(persons_ap_pids):
        for j, person_ap_pid in enumerate(person_ap_pids):
            append_score = persons_ap_scores[i][j]
            if person_ap_pid == 1:
                break
            new_ap_pids[person_ap_pid - 1].append(i + 1)
            new_ap_scores[person_ap_pid - 1].append(append_score)
    sort_idxes = list()
    for i, person_ap_scores in enumerate(new_ap_scores):
        sort_idx = sorted(range(len(person_ap_scores)), key=lambda k: -person_ap_scores[k])
        sort_idxes.append(sort_idx)

    for i, person_ap_pids in enumerate(new_ap_pids):

        for j, person_ap_pid in enumerate(person_ap_pids):
            if j >= 10:
                break
            write(renew_pid_path, '%d ' % person_ap_pids[sort_idxes[i][j]])
            write(renew_score_path, '%f ' % new_ap_scores[i][sort_idxes[i][j]])
        write(renew_pid_path, '\n')
        write(renew_score_path, '\n')

predict_line_idx = 0
predict_track_path = 'top10/predict_tracks.txt'


def get_predict_tracks():
    # renew_tracks()
    origin_tracks = get_tracks()
    # person_ids = get_person_idx()

    def add_predict_track(line):
        global predict_line_idx
        # print predict_line_idx
        if line == '\n':
            predict_line_idx += 1
            return
        if origin_tracks[predict_line_idx].startswith('-1'):
            tail = origin_tracks[predict_line_idx][2:-1]
        else:
            tail = origin_tracks[predict_line_idx][4: -1]
        if predict_line_idx == 499:
            print(predict_line_idx)
        if 'jpe' in tail:
            camera = tail[1]
        else:
            camera = tail[2]
        track_time = tail.split('_')[2]
        mids = line.split()
        write_line(predict_track_path,
                   ('%04d_c%d_%d_n.jpg' % (int(predict_line_idx) + 1, int(camera), int(track_time))))
        write_line('top10/predict_c%d.txt' % int(camera),
                   ('%04d_c%d_%d_n.jpg' % (int(predict_line_idx) + 1, int(camera), int(track_time))))
        for i, mid in enumerate(mids):
            if i >= 10:
                break
            write_line(predict_track_path,
                       ('%04d_c%d_%d_n.jpg' % (int(mid), int(camera), int(track_time))))
            write_line('top10/predict_c%d.txt' % int(camera),
                       ('%04d_c%d_%d_n.jpg' % (int(mid), int(camera), int(track_time))))
        predict_line_idx += 1
        # print('done')
    read_lines_and(renew_pid_path, add_predict_track)


def store_sorted_deltas():
    camera_delta_s = viz_data_for_market()
    for camera_delta in camera_delta_s:
        for delta_s in camera_delta:
            delta_s.sort()
    # for matlab
    for camera_delta in camera_delta_s:
        for delta_s in camera_delta:
            per_camera_deltas = ' '.join(map(str, delta_s))
            write_line('top10/sorted_deltas.txt', per_camera_deltas)
    # for python
    pickle_save('top10/sorted_deltas.pickle', camera_delta_s)


def interval_scores():
    camera_delta_s = viz_data_for_market()
    for camera_delta in camera_delta_s:
        for delta_s in camera_delta:
            delta_s.sort()
    gap_cnt = 5
    camera_pair_travel_probs = [[list() for _ in range(len(camera_delta_s[0]))] for _ in range(len(camera_delta_s))]
    for i, camera_delta in enumerate(camera_delta_s):
        for j, delta_s in enumerate(camera_delta):
            gap_width = (delta_s[-1] - delta_s[0])/float(gap_cnt)
            for k in range(gap_cnt):
                left_bound = delta_s[0] + gap_width * k
                right_bound = delta_s[1] + gap_width * (k + 1)
                total_cnt = sum(map(len, camera_delta))
                sp_cnt = len(delta_s)
                camera_pair_travel_probs[i][j].append({
                    'left': left_bound,
                    'right': right_bound,
                    'prob': sp_cnt / float(total_cnt)
                    # 'prob': (binary_search(delta_s, right_bound) - binary_search(delta_s, left_bound)) / float(total_cnt)
                })
    pickle_save('top10/interval_scores.pickle', camera_pair_travel_probs)
    return camera_pair_travel_probs


if __name__ == '__main__':
    get_predict_tracks()
    store_sorted_deltas()
    # scores = interval_scores()
