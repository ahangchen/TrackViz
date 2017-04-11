import random

from feature.top10distribution import get_tracks
from profile.fusion_param import fusion_param, ctrl_msg
from util.file_helper import write, safe_mkdir
from util.str_helper import folder


def get_average_predict():
    rand_folder_path = folder(fusion_param['answer_path'].replace(ctrl_msg['data_folder_path'], ctrl_msg['data_folder_path'] + '_rand'))
    safe_mkdir(rand_folder_path)
    rand_path = rand_folder_path + '/renew_pid.log'

    origin_tracks = get_tracks()
    origin_pids = [track.split('_')[0] for track in origin_tracks]
    pid_cnt = len(origin_pids)
    persons_rand_predict_idx_s = [random.sample(origin_pids, pid_cnt) for _ in range(pid_cnt)]
    write_content = ''
    for rand_predict_idx_s in persons_rand_predict_idx_s:
        for rand_predict_idx in rand_predict_idx_s:
            write_content += rand_predict_idx + ' '
        write_content += '\n'
    write(rand_path, write_content)

if __name__ == '__main__':
    get_average_predict()