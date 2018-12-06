from profile.fusion_param import ctrl_msg, get_fusion_param
from train.st_filter import train_tracks
import numpy as np


def eval(fusion_param, path):
    real_tracks = train_tracks(fusion_param)
    persons_pids = np.genfromtxt(path, delimiter=' ')[:, :50]
    # persons_pids = np.genfromtxt(fusion_param['eval_fusion_path'], delimiter=' ')[:, :50]
    avg_score = 0.0
    for i, person_pids in enumerate(persons_pids):
        shot_cnt = 0
        for pid in person_pids:
            if real_tracks[int(pid)][0] == real_tracks[i][0]:
                shot_cnt += 1
        score = shot_cnt / 50.
        avg_score += score
    print(avg_score / len(real_tracks))

if __name__ == '__main__':
    ctrl_msg['data_folder_path'] = 'cuhk_duke-train'
    fusion_param = get_fusion_param()
    eval(fusion_param, fusion_param['eval_fusion_path']) # 0.0271298874228
    eval(fusion_param, fusion_param['renew_ac_path']) # 0.00127103256264