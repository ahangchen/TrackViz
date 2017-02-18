import random

from file_helper import read_lines_and, write, write_line

line_idx = 0
shot_cnt = 0


def predict_eval():
    path = 'grid_predict/predict_grid.log'

    def is_shot(line):
        global line_idx
        global shot_cnt
        line_idx += 1
        predict_idx = line.split()
        if str(line_idx + 2) in predict_idx:
            shot_cnt += 1
            print(line_idx)

    read_lines_and(path, is_shot)
    print(shot_cnt * 2 / float(line_idx))


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
    # predict_eval()
    rand_predict()