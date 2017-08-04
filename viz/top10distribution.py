# coding=utf-8
#coding=utf-8
from profile.fusion_param import get_fusion_param
from util.file_helper import read_lines_and, safe_remove
from util.file_helper import write_line
from util.serialize import pickle_save
from viz.delta_track import viz_data_for_market


def get_tracks(fusion_param):
    # 这个函数是用来获取probe图片列表的，training的probe和gallery相同
    # answer_path: data/top-m2g-std0-train/test_track.txt
    test_path = fusion_param['answer_path']
    tracks = list()

    def add_track(line):
        tracks.append(line)

    read_lines_and(test_path, add_track)
    return tracks


predict_line_idx = 0


def get_predict_tracks(fusion_param, useful_predict_cnt=10):
    # 这个函数根据预测的pid，生成更多的图片名，从而构造时空模型，
    # 例如，左图1_c1_t2的匹配图片id是2_c2_t3,3_c3_t3,6_c4_t1,
    # 则生成1_c2_t3, 1_c3_t3, 1_c4_t1，用于构造时空概率模型
    # useful_predcit_cnt为10,实际上会对每张左图的名字产生10个右图的名字，加上原图就是11个新的名字
    # 为了加快后续检索速度，将生成的图片名中，摄像头相同的，写到同一个文件里，即predict_camera_path: predict_c%d.txt
    # 每次运行这个函数都会删除predict_c%d.txt和predict_tracks.txt，所以不会有缓存旧结果的情况
    # todo: 实际上可以在这一步直接生成cameras_deltas,之前是出于重用可视化代码考虑才使用了delta_track.py中的代码

    # renew_pid_path: data/top-m2g-std0-train/renew_pid.log'，包含左图预测的图片id, 250*249
    renew_pid_path = fusion_param['renew_pid_path']
    # predict_track_path：data/top-m2g-std0-train/predict_tracks.txt，存储get predict tracks结果
    predict_track_path = fusion_param['predict_track_path']
    # 获取左图列表
    origin_tracks = get_tracks(fusion_param)
    #
    safe_remove(predict_track_path)
    camera_cnt = 8
    global predict_line_idx
    predict_line_idx = 0
    for i in range(camera_cnt):
        safe_remove(fusion_param['predict_camera_path'] + str(i) + '.txt')

    def add_predict_track(line):
        global predict_line_idx
        # print predict_line_idx
        if line == '\n':
            predict_line_idx += 1
            return
        if predict_line_idx >= 248:
            print(predict_line_idx)
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
        # 这里写入的是predict_line_idx，而非原来的person id，保证了无监督无标签
        write_line(predict_track_path,
                   ('%04d_c%ds%d_%d_n.jpg' % (int(predict_line_idx) + 1, int(camera), s_num, int(track_time))))
        write_line(fusion_param['predict_camera_path'] + str(camera) + '.txt',
                   ('%04d_c%ds%d_%d_n.jpg' % (int(predict_line_idx) + 1, int(camera), s_num, int(track_time))))

        for i, mid in enumerate(mids):
            if i >= useful_predict_cnt:
                break
            write_line(predict_track_path,
                       ('%04d_c%ds%d_%d_n.jpg' % (int(mid), int(camera), s_num, int(track_time))))
            write_line(fusion_param['predict_camera_path'] + str(camera) + '.txt',
                       ('%04d_c%ds%d_%d_n.jpg' % (int(mid), int(camera), s_num, int(track_time))))
        predict_line_idx += 1
        # print('done')
    read_lines_and(renew_pid_path, add_predict_track)


def store_sorted_deltas(fusion_param):
    # 时空模型构建核心函数，
    # 存储每对摄像头的时间差分布，
    # 共6×6个数组，每个数组长度为该对摄像头统计到的时间差数目
    # 这个函数运行前会删除distribution_pickle_path： sorted_deltas.pickle，因此也不会有缓存问题
    camera_delta_s = viz_data_for_market(fusion_param)
    # 对时间差做排序，在预测的时候能快速定位时间差位置，得到时间差的概率
    for camera_delta in camera_delta_s:
        for delta_s in camera_delta:
            delta_s.sort()
    # for python
    safe_remove(fusion_param['distribution_pickle_path'])
    pickle_save(fusion_param['distribution_pickle_path'], camera_delta_s)


def interval_scores(fusion_param):
    # fusion_param = get_fusion_param()
    camera_delta_s = viz_data_for_market(fusion_param)
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
    fusion_param = get_fusion_param()
    get_predict_tracks(fusion_param)
    store_sorted_deltas(fusion_param)

    # scores = interval_scores()
