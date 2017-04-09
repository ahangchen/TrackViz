data_folder_path = 'top-m2g-std0-r-test'
dist_src_path = 'top-m2g-std0-r-train'
fusion_param = {
    'renew_pid_path': 'data/' + data_folder_path + '/renew_pid.log',
    'renew_ac_path': 'data/' + data_folder_path + '/renew_ac.log',
    'predict_pid_path': 'data/' + data_folder_path + '/predict_pid.log',
    'eval_fusion_path': 'data/' + data_folder_path + '/cross_filter_pid.log',
    'origin_answer_path': 'data/' + data_folder_path + '/test_track.txt',
    'answer_path': 'data/' + data_folder_path + '/test_tracks.txt',
    'predict_track_path': 'data/' + data_folder_path + '/predict_tracks.txt',
    'predict_camera_path': 'data/' + data_folder_path + '/predict_c',

    'distribution_pickle_path': 'data/' + data_folder_path + '/sorted_deltas.pickle',
    'src_distribution_pickle_path': 'data/' + dist_src_path + '/sorted_deltas.pickle',
    'interval_pickle_path': 'data/' + data_folder_path + '/interval_scores.pickle',
    'persons_deltas_path': 'data/' + data_folder_path + '/persons_deltas_score.pickle',
    'persons_ap_path': 'data/' + data_folder_path + '/persons_ap_scores.pickle',
    'predict_person_path': 'data/' + data_folder_path + '/predict_persons.pickle',

    'fusion_pid_path': 'data/' + data_folder_path + '/renew_pid1.log',
    'fusion_score_path': 'data/' + data_folder_path + '/renew_ac1.log',
    'fusion_normal_score_path': 'data/' + data_folder_path + '/cross_filter_score.log',
    'fusion_raw_score_path': 'data/' + data_folder_path + '/raw_cross_filter_score.log',
}
