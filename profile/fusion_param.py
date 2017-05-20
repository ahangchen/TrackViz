from util.file_helper import write

ctrl_msg = {
    'cross_idx': 3,
    'data_folder_path': 'market_train'
}

update_msg = {}


def get_fusion_param():
    origin_dict = {
        'renew_pid_path': 'data/' + ctrl_msg['data_folder_path'] + '/renew_pid.log',
        'renew_ac_path': 'data/' + ctrl_msg['data_folder_path'] + '/renew_ac.log',
        'predict_pid_path': 'data/' + ctrl_msg['data_folder_path'] + '/predict_pid.log',
        'eval_fusion_path': 'data/' + ctrl_msg['data_folder_path'] + '/cross_filter_pid.log',
        'origin_answer_path': 'data/' + ctrl_msg['data_folder_path'] + '/test_tracks.txt',
        'answer_path': 'data/' + ctrl_msg['data_folder_path'] + '/test_tracks.txt',
        'gallery_path': 'data/' + ctrl_msg['data_folder_path'] + '/gallery_tracks.txt',
        'predict_track_path': 'data/' + ctrl_msg['data_folder_path'] + '/predict_tracks.txt',
        'predict_camera_path': 'data/' + ctrl_msg['data_folder_path'] + '/predict_c',

        'distribution_pickle_path': 'data/' + ctrl_msg['data_folder_path'] + '/sorted_deltas.pickle',
        'src_distribution_pickle_path': 'data/' + ctrl_msg['data_folder_path'].replace('test', 'train') + '/sorted_deltas.pickle',
        'interval_pickle_path': 'data/' + ctrl_msg['data_folder_path'] + '/interval_scores.pickle',
        'persons_deltas_path': 'data/' + ctrl_msg['data_folder_path'] + '/persons_deltas_score.pickle',
        'persons_ap_path': 'data/' + ctrl_msg['data_folder_path'] + '/persons_ap_scores.pickle',
        'predict_person_path': 'data/' + ctrl_msg['data_folder_path'] + '/predict_persons.pickle',

        'fusion_pid_path': 'data/' + ctrl_msg['data_folder_path'] + '/renew_pid1.log',
        'fusion_score_path': 'data/' + ctrl_msg['data_folder_path'] + '/renew_ac1.log',
        'fusion_normal_score_path': 'data/' + ctrl_msg['data_folder_path'] + '/cross_filter_score.log',
        'fusion_raw_score_path': 'data/' + ctrl_msg['data_folder_path'] + '/raw_cross_filter_score.log',
        'cross_gallery_path': 'data/grid_gallery_idx.txt',
        'pos_shot_rate': 0.5,
        # 'pos_shot_rate': 0.003302,
        'neg_shot_rate': 0.01,
        # 'neg_shot_rate': 0.001252,
    }
    if 'train' in origin_dict['src_distribution_pickle_path']:
        origin_dict['rand_distribution_pickle_path'] = origin_dict['src_distribution_pickle_path'].replace('train', 'train_rand')
    else:
        origin_dict['rand_distribution_pickle_path'] = origin_dict['src_distribution_pickle_path'].replace('.pickle','_rand.pickle')
    # if 'r-' in origin_dict['src_distribution_pickle_path']:
    #     origin_dict['rand_distribution_pickle_path'] = origin_dict['src_distribution_pickle_path'].replace('r-train',
    #                                                                                                        'train_rand')

    for (k, v) in update_msg.items():
        origin_dict[k] = v
    return origin_dict


def update_fusion_param(key, value):
    update_msg[key] = value


def save_fusion_param():
    write('data/fusion_param.json', fusion_param.__str__)

fusion_param = get_fusion_param()