import random
import shutil

from feature.top10distribution import get_tracks, get_predict_tracks, store_sorted_deltas
from profile.fusion_param import get_fusion_param, ctrl_msg
from train.st_filter import predict_track_scores
from util.file_helper import write, safe_mkdir, safe_remove
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
    origin_pids = map(lambda x: x+1, range(5))
    persons_rand_predict_idx_s = [random.sample(origin_pids, pid_cnt) for _ in range(pid_cnt)]
    write_content = ''
    for rand_predict_idx_s in persons_rand_predict_idx_s:
        for rand_predict_idx in rand_predict_idx_s:
            write_content += rand_predict_idx + ' '
        write_content += '\n'
    write(rand_path, write_content)


def gen_rand_st_model(fusion_param):
    get_predict_tracks(fusion_param)
    # get distribution sorted list for probability compute
    store_sorted_deltas(fusion_param)

if __name__ == '__main__':
    fusion_param = get_fusion_param()
    write_rand_pid(fusion_param)

    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'] + '_rand'
    fusion_param = get_fusion_param()
    gen_rand_st_model(fusion_param)
