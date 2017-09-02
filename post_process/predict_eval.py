import random

from profile.fusion_param import get_fusion_param, ctrl_msg
from util.file_helper import read_lines_and, write_line, read_lines
from util.str_helper import folder

line_idx = 0
shot_line_cnt = 0
predict_cnt = 0
predict_line_cnt = 0
gallery_cnt = 0


shot_cnt = 0
top_cnt = 10


def percent_shot_eval(target_path, top_cnt, test_mode=False):
    global gallery_cnt
    gallery_cnt = 0
    answer_path = folder(target_path) + '/test_tracks.txt'
    predict_path = target_path
    answer_lines = read_lines(answer_path)
    real_pids = [int(answer.split('_')[0]) for answer in answer_lines]

    def is_shot(line):
        global line_idx
        global shot_line_cnt
        global shot_cnt
        global predict_cnt
        global predict_line_cnt
        global gallery_cnt

        predict_idx_es = line.split()
        has_shot = False
        if len(predict_idx_es) > top_cnt:
            predict_cnt += top_cnt
        else:
            predict_cnt += len(predict_idx_es)

        if len(predict_idx_es) > 0:
            predict_line_cnt += 1
        # line_idx > 774 means label img,
        # gallery_idxs[(line_idx - 775)/2] means iseven in gallery,
        # if iseven is equal, means gallery img
        if test_mode and line_idx > 774 and (line_idx - 774) % 2 == 1:
            gallery_cnt += 1
            line_idx += 1
            return
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
    if test_mode:
        valid_line_cnt = 125
    else:
        valid_line_cnt = 250
    shot_rate = shot_line_cnt / float(valid_line_cnt)
    print('top%d shot: %f' % (top_cnt, shot_rate))
    print('gallery cnt: %d' % gallery_cnt)
    line_idx = 0
    shot_cnt = 0
    shot_line_cnt = 0
    predict_cnt = 0
    predict_line_cnt = 0
    return shot_rate


def rand_predict():
    raw_path = 'grid_predict/predict_grid_rand.log'

    for i in range(500):
        rand_output = [random.randint(i + 1, 500) for _ in range(10)]
        rand_output_str = ''
        for rand_id in rand_output:
            rand_output_str += str(rand_id) + ' '
        write_line(raw_path, rand_output_str)


def eval_on_train_test(fusion_param, pst=True, test_mode=False):
    # fusion_param = get_fusion_param()
    print('\nMarket to GRID:')
    top10_shot_pure = percent_shot_eval(fusion_param['renew_pid_path'], 10, test_mode=test_mode)
    top5_shot_pure = percent_shot_eval(fusion_param['renew_pid_path'], 5, test_mode=test_mode)
    top1_shot_pure = percent_shot_eval(fusion_param['renew_pid_path'], 1, test_mode=test_mode)
    print('\nMarket to GRID with track score:')
    top10_shot_fusion = percent_shot_eval(fusion_param['eval_fusion_path'], 10, test_mode=test_mode)
    top5_shot_fusion = percent_shot_eval(fusion_param['eval_fusion_path'], 5, test_mode=test_mode)
    top1_shot_fusion = percent_shot_eval(fusion_param['eval_fusion_path'], 1, test_mode=test_mode)

    if pst:
        write_line('data/predict_result.txt', ctrl_msg['data_folder_path'])
        write_line(
            'data/predict_result.txt',
            '%f %f %f %f %f %f' % (top10_shot_pure, top5_shot_pure, top1_shot_pure,
                                   top10_shot_fusion, top5_shot_fusion, top1_shot_fusion)
        )


if __name__ == '__main__':
    # eval_on_train_test()
    fusion_param = get_fusion_param()
    # pos_neg_shot_eval(fusion_param['renew_pid_path'], fusion_param['renew_ac_path'])
    # target_pos_neg_shot_eval(fusion_param['fusion_normal_score_path'], fusion_param['renew_pid_path'], fusion_param['renew_ac_path'])
    eval_on_train_test(fusion_param)