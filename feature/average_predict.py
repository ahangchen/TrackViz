import random
import shutil

from feature.top10distribution import get_tracks, get_predict_tracks, store_sorted_deltas
from profile.fusion_param import get_fusion_param, ctrl_msg
from train.st_filter import predict_track_scores
from util.file_helper import write, safe_mkdir, safe_remove, read_lines
from util.str_helper import folder


def write_rand_pid(fusion_param):
    # fusion_param = get_fusion_param()
    rand_answer_path = fusion_param['answer_path'].replace(ctrl_msg['data_folder_path'], ctrl_msg['data_folder_path'] + '_rand')
    rand_folder_path = folder(rand_answer_path)
    safe_mkdir(rand_folder_path)
    # although copy all info including pid info, but not use in later training
    shutil.copy(fusion_param['answer_path'], rand_answer_path)
    rand_path = rand_folder_path + '/renew_pid.log'
    safe_remove(rand_path)

    origin_tracks = get_tracks(fusion_param)
    pid_cnt = len(origin_tracks)
    origin_pids = map(lambda x: x+1, range(pid_cnt))
    persons_rand_predict_idx_s = [random.sample(origin_pids, pid_cnt) for _ in range(pid_cnt)]
    write_content = ''
    for rand_predict_idx_s in persons_rand_predict_idx_s:
        for rand_predict_idx in rand_predict_idx_s:
            write_content += str(rand_predict_idx) + ' '
        write_content += '\n'
    write(rand_path, write_content)


def write_unequal_rand_st_model(fusion_param):
    # fusion_param = get_fusion_param()
    rand_answer_path = fusion_param['answer_path'].replace(ctrl_msg['data_folder_path'],
                                                           ctrl_msg['data_folder_path'] + '_uerand')
    rand_folder_path = folder(rand_answer_path)
    safe_mkdir(rand_folder_path)
    # although copy all info including pid info, but not use in later training
    shutil.copy(fusion_param['answer_path'], rand_answer_path)
    rand_path = rand_folder_path + '/renew_pid.log'
    safe_remove(rand_path)

    origin_tracks = get_tracks(fusion_param)
    pid_cnt = len(origin_tracks)
    origin_pids = map(lambda x: x + 1, range(pid_cnt))
    persons_rand_predict_idx_s = [random.sample(origin_pids, pid_cnt) for _ in range(pid_cnt)]

    viz_pid_path = fusion_param['renew_pid_path']
    viz_score_path = fusion_param['renew_ac_path']
    viz_pids = read_lines(viz_pid_path)
    viz_pids = [per_viz_pids.split() for per_viz_pids in viz_pids]
    viz_scores = read_lines(viz_score_path)
    viz_scores = [per_viz_scores.split() for per_viz_scores in viz_scores]
    viz_same_pids = [
        [
            int(viz_pid) for viz_pid, viz_score in zip(per_viz_pids, per_viz_scores) if float(viz_score) > 0.7
        ] for per_viz_scores, per_viz_pids in zip(viz_scores, viz_pids)
    ]

    persons_unequal_rand_predict_idx_s = list()
    for i in range(pid_cnt):
        diff_persons = list(set(persons_rand_predict_idx_s[i]) ^ set(viz_same_pids[i]))
        diff_cnt = len(diff_persons)
        persons_unequal_rand_predict_idx_s.append(
            random.sample(diff_persons, diff_cnt)
        )

    write_content = ''
    for rand_predict_idx_s in persons_unequal_rand_predict_idx_s:
        for rand_predict_idx in rand_predict_idx_s:
            write_content += str(rand_predict_idx) + ' '
        write_content += '\n'
    write(rand_path, write_content)


def gen_rand_st_model(fusion_param):
    get_predict_tracks(fusion_param, useful_predict_cnt=10)
    # get distribution sorted list for probability compute
    store_sorted_deltas(fusion_param)

if __name__ == '__main__':
    fusion_param = get_fusion_param()
    write_unequal_rand_st_model(fusion_param)

    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'] + '_uerand'
    fusion_param = get_fusion_param()
    gen_rand_st_model(fusion_param)
