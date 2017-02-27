import random

from file_helper import read_lines_and, write, write_line, read_lines

line_idx = 0
shot_line_cnt = 0
predict_cnt = 0
predict_line_cnt = 0


def predict_eval():
    path = 'grid/predict_grid.log'

    def is_shot(line):
        global line_idx
        global shot_line_cnt
        line_idx += 1
        predict_idx = line.split()
        if str(line_idx + 2) in predict_idx[0:4]:
            shot_line_cnt += 1
            print(line_idx)

    read_lines_and(path, is_shot)
    print(shot_line_cnt * 2 / float(line_idx))


shot_cnt = 0


def predict_market_eval(target_path):
    answer_path = 'top10/test_tracks.txt'
    predict_path = target_path
    answer_lines = read_lines(answer_path)
    real_pids = [answer.split('_')[0] for answer in answer_lines]
    top_cnt = 1

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
    print('all predict shot(ac1): %f' % (float(shot_cnt) / predict_cnt))
    print('top10 shot(ac2): %f\n' % (shot_line_cnt / 3914.0))
    line_idx = 0
    shot_cnt = 0
    shot_line_cnt = 0
    predict_cnt = 0
    predict_line_cnt = 0

clean_line_idx = 0


def predict_clean():
    raw_path = 'grid_predict/predict_grid_raw.log'

    def idx_up(line):
        global clean_line_idx
        predict_idx = line.split()
        for idx in predict_idx:
            write('grid_predict/predict_grid.log', str(int(idx) + clean_line_idx + 1) + ' ')
        write('grid_predict/predict_grid.log', '\n')
        clean_line_idx += 1
    read_lines_and(raw_path, idx_up)


def rand_predict():
    raw_path = 'grid_predict/predict_grid_rand.log'

    for i in range(500):
        rand_output = [random.randint(i + 1, 500) for _ in range(10)]
        rand_output_str = ''
        for rand_id in rand_output:
            rand_output_str += str(rand_id) + ' '
        write_line(raw_path, rand_output_str)

if __name__ == '__main__':
    # predict_clean()
    print('origin:')
    predict_market_eval('top10/renew_pid.log')
    print('appearance and track filter:')
    predict_market_eval('top10/cross_filter_pid.log')
    # rand_predict()