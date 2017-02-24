import random

from file_helper import read_lines_and, write, write_line, read_lines

line_idx = 0
shot_line_cnt = 0


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


def predict_market_eval():
    answer_path = 'top10/test_tracks.txt'
    predict_path = 'top10/predict_test.log'
    answer_lines = read_lines(answer_path)
    real_pids = [answer.split('_')[0] for answer in answer_lines]
    top_cnt = 5

    def is_shot(line):
        global line_idx
        global shot_line_cnt
        global shot_cnt
        line_idx += 1
        predict_idx_es = line.split()
        has_shot = False

        for i, predict_idx in enumerate(predict_idx_es):
            if i >= top_cnt:
                break
            if real_pids[int(predict_idx) - 1] == real_pids[line_idx]:
                if not has_shot:
                    shot_line_cnt += 1
                    has_shot = True
                shot_cnt += 1

    read_lines_and(predict_path, is_shot)
    print('all shot: %f' % (shot_cnt / float(line_idx) / top_cnt))
    print('top10 shot: %f' % (shot_line_cnt / float(line_idx)))


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
    predict_market_eval()
    # rand_predict()