import random

from profile.fusion_param import get_fusion_param, ctrl_msg
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
    shot_rate = shot_line_cnt / float(valid_line_cnt)
    print('top%d shot: %f' % (top_cnt, shot_rate))
    line_idx = 0
    shot_cnt = 0
    shot_line_cnt = 0
    predict_cnt = 0
    predict_line_cnt = 0
    return shot_rate


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
            if real_pids[predict_idx] == real_pids[i]:
                # pos sample
                pos_sample_cnt += 1
                if float(predict_scores[i][j]) > 0.5:
                    # judge same
                    pos_shot_cnt += 1
            else:
                # neg sample
                neg_sample_cnt += 1
                if float(predict_scores[i][j]) <= 0.5:
                    # judge diff
                    neg_shot_cnt += 1
    pos_shot_rate = 1 - float(pos_shot_cnt) / pos_sample_cnt
    neg_shot_rate = 1 - float(neg_shot_cnt) / neg_sample_cnt
    print('positive shot error rate: 1-%d/%d=%f' %
          (pos_shot_cnt, pos_sample_cnt, pos_shot_rate))
    print('negative shot error rate: 1-%d/%d=%f' %
          (neg_shot_cnt, neg_sample_cnt, neg_shot_rate))
    return pos_shot_rate, neg_shot_rate


def target_pos_neg_shot_eval(fusion_score_path, target_pid_path, target_score_path):
    """
    call after fusion done, get the positive shot rate and negative shot rate by:
    Ep = E((1-pij)*boolean(Si==Sj))
    En = E(pij*boolean(Si!=Sj))
    :param fusion_score_path: target dataSet fusion predict scores
    :param target_pid_path: target dataSet person ids
    :param target_score_path: target dataSet vision predict scores
    :return: 
    """
    predict_score_lines = read_lines(target_score_path)
    predict_scores = [scores.split() for scores in predict_score_lines]

    fusion_predict_score_lines = read_lines(fusion_score_path)
    fusion_predict_scores = [scores.split() for scores in fusion_predict_score_lines]

    predict_pid_lines = read_lines(target_pid_path)
    predict_pids = [pids.split() for pids in predict_pid_lines]

    ep_sum = 0.0
    ep_cnt = 0
    en_sum = 0.0
    en_cnt = 0
    for i in range(len(predict_pids)):
        for j in range(len(predict_pids[i])):
            if float(predict_scores[i][j]) > 0.5:
                ep_sum += 1 - float(fusion_predict_scores[i][j])
                ep_cnt += 1
                pass
            else:
                en_sum += float(fusion_predict_scores[i][j])
                en_cnt += 1
                pass

    pos_shot_rate = ep_sum / ep_cnt
    neg_shot_rate = en_sum / en_cnt
    print('positive shot error rate: %f/%f=%f' %
          (ep_sum, ep_cnt, pos_shot_rate))
    print('negative shot error rate: %f/%f=%f' %
          (en_sum, en_cnt, neg_shot_rate))
    return pos_shot_rate, neg_shot_rate


def rand_predict():
    raw_path = 'grid_predict/predict_grid_rand.log'

    for i in range(500):
        rand_output = [random.randint(i + 1, 500) for _ in range(10)]
        rand_output_str = ''
        for rand_id in rand_output:
            rand_output_str += str(rand_id) + ' '
        write_line(raw_path, rand_output_str)


def eval_on_train_test(fusion_param, pst=True):
    # fusion_param = get_fusion_param()
    print('\nMarket to GRID:')
    top10_shot_pure = percent_shot_eval(fusion_param['renew_pid_path'], 10)
    top5_shot_pure = percent_shot_eval(fusion_param['renew_pid_path'], 5)
    top1_shot_pure = percent_shot_eval(fusion_param['renew_pid_path'], 1)
    print('\nMarket to GRID with track score:')
    top10_shot_fusion = percent_shot_eval(fusion_param['eval_fusion_path'], 10)
    top5_shot_fusion = percent_shot_eval(fusion_param['eval_fusion_path'], 5)
    top1_shot_fusion = percent_shot_eval(fusion_param['eval_fusion_path'], 1)

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