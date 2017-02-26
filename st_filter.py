from file_helper import read_lines, read_lines_and, write
from serialize import pickle_load
from track_prob import track_score

line_idx = 0
camera_delta_s = pickle_load('top10/sorted_deltas.pickle')


def predict_filter():
    predict_path = 'top10/predict_test.log'
    answer_path = 'top10/test_tracks.txt'
    filter_path = 'top10/filter_pid.log'
    sure_path = 'top10/sure_pid.log'
    answer_lines = read_lines(answer_path)
    real_tracks = list()
    for answer in answer_lines:
        info = answer.split('_')
        real_tracks.append([info[0], int(info[1][1]), int(info[2])])

    top_cnt = 10

    def predict_judge(line):
        global line_idx
        line_idx += 1
        predict_idx_es = line.split()

        for i, predict_idx in enumerate(predict_idx_es):
            if i >= top_cnt:
                break
            time1 = real_tracks[int(predict_idx) - 1][1]
            time2 = real_tracks[line_idx][1]
            c1 = real_tracks[int(predict_idx) - 1][1]
            c2 = real_tracks[line_idx][1]
            score = track_score(camera_delta_s, c1, time1, c2, time2)
            # score, [0, 0.48] shot rate up, [0.48, 0.7], shot rate keep the same
            if score > 0.48:
                write(filter_path, predict_idx + ' ')
            # score [0.7, 0.9], all shot up, line shot down
            # score [0.9, 1], show rate down
            # all shot max is 53%, line shot max is 83%
            if score > 0.9:
                write(sure_path, predict_idx + ' ')
        write(filter_path, '\n')
        write(sure_path, '\n')
    read_lines_and(predict_path, predict_judge)


if __name__ == '__main__':
    predict_filter()