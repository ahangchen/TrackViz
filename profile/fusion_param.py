from util.file_helper import write

ctrl_msg = {
    'data_folder_path': 'top-m2g-std1-train'
}

update_msg = {}


def get_fusion_param():
    origin_dict = {
        'renew_pid_path': 'data/' + ctrl_msg['data_folder_path'] + '/renew_pid.log',
        'renew_ac_path': 'data/' + ctrl_msg['data_folder_path'] + '/renew_ac.log',
        'predict_pid_path': 'data/' + ctrl_msg['data_folder_path'] + '/predict_pid.log',
        'eval_fusion_path': 'data/' + ctrl_msg['data_folder_path'] + '/cross_filter_pid.log',
        'origin_answer_path': 'data/' + ctrl_msg['data_folder_path'] + '/test_track.txt',
        'answer_path': 'data/' + ctrl_msg['data_folder_path'] + '/test_tracks.txt',
        'predict_track_path': 'data/' + ctrl_msg['data_folder_path'] + '/predict_tracks.txt',
        'predict_camera_path': 'data/' + ctrl_msg['data_folder_path'] + '/predict_c',

        'distribution_pickle_path': 'data/' + ctrl_msg['data_folder_path'] + '/sorted_deltas.pickle',
        'src_distribution_pickle_path': 'data/' + ctrl_msg['data_folder_path'][:-3]+'rain' + '/sorted_deltas.pickle',
        'rand_distribution_pickle_path': 'data/' + ctrl_msg['data_folder_path'] + '_rand/sorted_deltas.pickle',
        'interval_pickle_path': 'data/' + ctrl_msg['data_folder_path'] + '/interval_scores.pickle',
        'persons_deltas_path': 'data/' + ctrl_msg['data_folder_path'] + '/persons_deltas_score.pickle',
        'persons_ap_path': 'data/' + ctrl_msg['data_folder_path'] + '/persons_ap_scores.pickle',
        'predict_person_path': 'data/' + ctrl_msg['data_folder_path'] + '/predict_persons.pickle',

        'fusion_pid_path': 'data/' + ctrl_msg['data_folder_path'] + '/renew_pid1.log',
        'fusion_score_path': 'data/' + ctrl_msg['data_folder_path'] + '/renew_ac1.log',
        'fusion_normal_score_path': 'data/' + ctrl_msg['data_folder_path'] + '/cross_filter_score.log',
        'fusion_raw_score_path': 'data/' + ctrl_msg['data_folder_path'] + '/raw_cross_filter_score.log',
        'pos_shot_rate': 0.003302,
        'neg_shot_rate': 0.001252,
    }

    for (k, v) in update_msg.items():
        origin_dict[k] = v
    return origin_dict


def update_fusion_param(key, value):
    update_msg[key] = value


def save_fusion_param():
    write('data/fusion_param.json', fusion_param.__str__)

fusion_param = get_fusion_param()