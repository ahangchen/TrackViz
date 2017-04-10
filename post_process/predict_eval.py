import random

from profile.fusion_param import fusion_param
from util.file_helper import read_lines_and, write_line, read_lines
from util.str_helper import folder

line_idx = 0
shot_line_cnt = 0
predict_cnt = 0
predict_line_cnt = 0


shot_cnt = 0
top_cnt = 10


def percent_shot_eval(target_path, top_cnt):
    answer_path = folder(target_path) + '/test_tracks.txt'
    predict_path = target_path
    answer_lines = read_lines(answer_path)
    real_pids = [answer.split('_')[0] for answer in answer_lines]

    def is_shot(line):
        global line_idx
        global shot_line_cnt
        global shot_cnt
        global predict_cnt
        global predict_line_cnt

        predict_idx_es = line.split()
        has_shot = False
        if len(predict_idx_es) > top_cnt:
            predict_cnt += top_cnt
        else:
            predict_cnt += len(predict_idx_es)

        if len(predict_idx_es) > 0:
            predict_line_cnt += 1

        for i, predict_idx in enumerate(predict_idx_es):
            if i >= top_cnt:
                break
            # print(line_idx)
            if real_pids[int(predict_idx) - 1] == real_pids[line_idx]:
                if not has_shot:
                    shot_line_cnt += 1
                    has_shot = True
                shot_cnt += 1
        line_idx += 1

    read_lines_and(predict_path, is_shot)
    global line_idx
    global shot_line_cnt
    global shot_cnt
    global predict_cnt
    global predict_line_cnt
    # print('all predict shot(ac1): %f' % (float(shot_cnt) / predict_cnt))
    valid_line_cnt = 250
    print('top%d shot: %f' % (top_cnt, shot_line_cnt / float(valid_line_cnt)))
    line_idx = 0
    shot_cnt = 0
    shot_line_cnt = 0
    predict_cnt = 0
    predict_line_cnt = 0


def pos_neg_shot_eval(target_pid_path, target_score_path):
    answer_path = folder(target_pid_path) + '/test_tracks.txt'
    answer_lines = read_lines(answer_path)
    real_pids = [answer.split('_')[0] for answer in answer_lines]

    predict_score_lines = read_lines(target_score_path)
    predict_scores = [scores.split() for scores in predict_score_lines]

    predict_pid_lines = read_lines(target_pid_path)
    predict_pids = [pids.split() for pids in predict_pid_lines]

    neg_sample_cnt = 0
    neg_shot_cnt = 0
    pos_sample_cnt = 0
    pos_shot_cnt = 0

    for i in range(len(predict_pids)):
        for j in range(len(predict_pids[i])):
            predict_idx = int(predict_pids[i][j]) - 1
            if real_pids[predict_idx] == real_pids[line_idx]:
                # pos sample
                pos_sample_cnt += 1
                if float(predict_scores[i][j]) > 0.9:
                    # judge same
                    pos_shot_cnt += 1
            else:
                # neg sample
                neg_sample_cnt += 1
                if float(predict_scores[i][j]) < 0.7:
                    # judge diff
                    neg_shot_cnt += 1

    print('positive shot rate: %f' % (float(pos_shot_cnt) / pos_sample_cnt))
    print('negative shot rate: %f' % (float(neg_shot_cnt) / neg_sample_cnt))


def rand_predict():
    raw_path = 'grid_predict/predict_grid_rand.log'

    for i in range(500):
        rand_output = [random.randint(i + 1, 500) for _ in range(10)]
        rand_output_str = ''
        for rand_id in rand_output:
            rand_output_str += str(rand_id) + ' '
        write_line(raw_path, rand_output_str)


def eval_on_train_test():
    print('\nMarket to GRID:')
    percent_shot_eval(fusion_param['renew_pid_path'], 10)
    percent_shot_eval(fusion_param['renew_pid_path'], 5)
    percent_shot_eval(fusion_param['renew_pid_path'], 1)
    print('\nMarket to GRID with track score:')
    percent_shot_eval(fusion_param['eval_fusion_path'], 10)
    percent_shot_eval(fusion_param['eval_fusion_path'], 5)
    percent_shot_eval(fusion_param['eval_fusion_path'], 1)


if __name__ == '__main__':
    # eval_on_train_test()
    pos_neg_shot_eval(fusion_param['renew_pid_path'], fusion_param['renew_ac_path'])
