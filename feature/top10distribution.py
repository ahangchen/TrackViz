from profile.fusion_param import get_fusion_param
from train.delta_track import viz_data_for_market
from util.file_helper import read_lines_and, safe_remove
from util.file_helper import write_line
from util.serialize import pickle_save


def get_tracks():
    fusion_param = get_fusion_param()
    test_path = fusion_param['answer_path']
    tracks = list()

    def add_track(line):
        tracks.append(line)

    read_lines_and(test_path, add_track)
    return tracks


predict_line_idx = 0


def get_predict_tracks():
    fusion_param = get_fusion_param()
    renew_pid_path = fusion_param['renew_pid_path']
    predict_track_path = fusion_param['predict_track_path']
    # renew_tracks()
    origin_tracks = get_tracks()
    # person_ids = get_person_idx()
    safe_remove(predict_track_path)
    camera_cnt = 8
    for i in range(camera_cnt):
        safe_remove(fusion_param['predict_camera_path'] + str(i) + '.txt')

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
        if 's' in tail:
            s_num = int(tail[4])
        else:
            s_num = 1
        if predict_line_idx == 499:
            print(predict_line_idx)
        if 'jpe' in tail:
            camera = tail[1]
        else:
            camera = tail[2]
        track_time = tail.split('_')[2]
        mids = line.split()
        write_line(predict_track_path,
                   ('%04d_c%ds%d_%d_n.jpg' % (int(predict_line_idx) + 1, int(camera), s_num, int(track_time))))
        write_line(fusion_param['predict_camera_path'] + str(camera) + '.txt',
                   ('%04d_c%ds%d_%d_n.jpg' % (int(predict_line_idx) + 1, int(camera), s_num, int(track_time))))
        for i, mid in enumerate(mids):
            if i >= 10:
                break
            write_line(predict_track_path,
                       ('%04d_c%ds%d_%d_n.jpg' % (int(mid), int(camera), s_num, int(track_time))))
            write_line(fusion_param['predict_camera_path'] + str(camera) + '.txt',
                       ('%04d_c%ds%d_%d_n.jpg' % (int(mid), int(camera), s_num, int(track_time))))
        predict_line_idx += 1
        # print('done')
    read_lines_and(renew_pid_path, add_predict_track)


def store_sorted_deltas():
    fusion_param = get_fusion_param()
    camera_delta_s = viz_data_for_market()
    for camera_delta in camera_delta_s:
        for delta_s in camera_delta:
            delta_s.sort()
    # for python
    pickle_save(fusion_param['distribution_pickle_path'], camera_delta_s)


def interval_scores():
    fusion_param = get_fusion_param()
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
    pickle_save(fusion_param['interval_pickle_path'], camera_pair_travel_probs)
    return camera_pair_travel_probs


if __name__ == '__main__':
    get_predict_tracks()
    store_sorted_deltas()

    # scores = interval_scores()
