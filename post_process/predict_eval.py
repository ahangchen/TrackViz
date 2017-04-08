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


def predict_market_eval(target_path, top_cnt):
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

clean_line_idx = 0


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
    predict_market_eval(fusion_param['renew_pid_path'], 10)
    predict_market_eval(fusion_param['renew_pid_path'], 5)
    predict_market_eval(fusion_param['renew_pid_path'], 1)
    print('\nMarket to GRID with track score:')
    predict_market_eval(fusion_param['eval_fusion_path'], 10)
    predict_market_eval(fusion_param['eval_fusion_path'], 5)
    predict_market_eval(fusion_param['eval_fusion_path'], 1)


if __name__ == '__main__':
    eval_on_train_test()
