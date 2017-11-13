import numpy as np
import pandas as pd

from post_process.track_prob import track_score
from profile.fusion_param import get_fusion_param, ctrl_msg
from util.serialize import pickle_load
from viz.delta_track import viz_fusion_curve


def fusion_curve(fusion_param):
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    rand_camera_deltas = pickle_load(fusion_param['rand_distribution_pickle_path'])
    # for actual calculate
    delta_width = 2000.0
    delta_cnt = int(delta_width/100)
    # for distribution viz
    # delta_width = 3000.0
    # delta_cnt = 10
    interval_width = delta_width/ delta_cnt
    delta_stripe = delta_width / delta_cnt
    delta_range = map(lambda x: x * delta_stripe - delta_width/2, range(delta_cnt))
    # delta_range = map(lambda x: x*1.0 - 60.0, range(120))
    over_probs = [[list() for j in range(6)] for i in range(6)]
    probs = list()
    for i in range(6):
        for j in range(i+1, 6):
            if i == j:
                continue
            cur_prob = list()
            for k in range(len(delta_range)):
                match_track_score = track_score(camera_delta_s, i + 1, 0, j + 1, delta_range[k], interval=interval_width, filter_interval=delta_width/2)
                rand_track_score = track_score(rand_camera_deltas, i + 1, 0, j + 1, delta_range[k], interval=interval_width)
                if rand_track_score < 0.00002:
                    # print rand_track_score
                    rand_track_score = 0.00002
                if i == 3 and j == 4 and abs(delta_range[k]) <= 2000 :
                    print i
                # else:
                #     print match_track_score / rand_track_score
                cur_prob.append(rand_track_score)
                # raw_probs[i][j].append(match_track_score)
                # rand_probs[i][j].append(rand_track_score)
                # over_probs[i][j].append(rand_track_score)
                over_probs[i][j].append(match_track_score)
            probs.append(over_probs[i][j])
    np_probs = np.array(probs)
    np.savetxt('market-grid-cv0_scores.txt', np_probs, fmt='%.6f\t')

    return delta_range, over_probs


def fusion_heat(fusion_param):
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    rand_camera_deltas = pickle_load(fusion_param['rand_distribution_pickle_path'])
    delta_width = 3000.0
    delta_cnt = 10
    interval_width = delta_width/ delta_cnt
    delta_stripe = delta_width / delta_cnt
    delta_range = map(lambda x: x * delta_stripe - delta_width/2, range(delta_cnt))
    # delta_range = map(lambda x: x*1.0 - 60.0, range(120))
    viz_deltas = list()
    viz_camera_pairs = list()
    viz_values = list()
    for i in range(6):
        for j in range(i+1, 6):
            for k in range(len(delta_range)):
                match_track_score = track_score(camera_delta_s, i + 1, 0, j + 1, delta_range[k], interval=interval_width)
                rand_track_score = track_score(rand_camera_deltas, i + 1, 0, j + 1, delta_range[k], interval=interval_width)
                if rand_track_score < 0.00002:
                    # print rand_track_score
                    rand_track_score = 0.00002
                else:
                    print match_track_score / rand_track_score

                # raw_probs[i][j].append(match_track_score)
                # rand_probs[i][j].append(rand_track_score)
                viz_deltas.append(delta_range[k])
                viz_camera_pairs.append('c%d-c%d' % (i + 1, j + 1))
                viz_values.append(match_track_score)
        break
    df = pd.DataFrame({'transfer_time': viz_deltas,
                       'camera_pair': viz_camera_pairs,
                       'values': viz_values})
    pt = df.pivot_table(index='camera_pair', columns='transfer_time', values='values', aggfunc=np.sum)
    return pt


if __name__ == '__main__':
    ctrl_msg['data_folder_path'] = 'market_grid-cv0-test'
    fusion_param = get_fusion_param()
    # fusion_param['distribution_pickle_path'] = 'true_grid-cv0_test.pck'
    fusion_curve(fusion_param)
    delta_range, over_probs = fusion_curve(fusion_param)
    viz_fusion_curve(delta_range, [over_probs])