import os

from ctrl.img_st_fusion import init_strict_img_st_fusion
from profile.fusion_param import ctrl_msg, get_fusion_param
from util.file_helper import safe_mkdir


def fusion_dir_prepare(source, target):
    fusion_data_path = '/home/cwh/coding/TrackViz/data/'
    fusion_train_dir = fusion_data_path + '/' + source + '_' + target + '-train'
    fusion_test_dir = fusion_data_path + '/' + source + '_' + target + '-test'
    safe_mkdir(fusion_train_dir)
    safe_mkdir(fusion_test_dir)
    return fusion_train_dir, fusion_test_dir


def vision_rank(source, target):
    # vision classifier predict similarity rank table
    fusion_train_dir, fusion_test_dir = fusion_dir_prepare(source, target)
    vision_train_rank_pids_path = fusion_train_dir + '/renew_pid.log'
    vision_train_rank_scores_path = fusion_train_dir + '/renew_ac.log'
    vision_test_rank_pids_path = fusion_test_dir + '/renew_pid.log'
    vision_test_rank_scores_path = fusion_test_dir + '/renew_ac.log'
    os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    # os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 0 '
    #           + source + ' ' + target + ' '
    #           + vision_train_rank_pids_path + ' '
    #           + vision_train_rank_scores_path + ' '
    #           + vision_test_rank_pids_path + ' '
    #           + vision_test_rank_scores_path)
    return vision_train_rank_pids_path, vision_train_rank_scores_path, vision_test_rank_pids_path, vision_test_rank_scores_path


def dataset_eval(source, target, rank_pids_path):
    # if 'market' in target:
    #     return
    os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
              + target + ' ' + rank_pids_path)


def st_fusion(source, target):
    ctrl_msg['data_folder_path'] = source + '_' + target + '-train'
    init_strict_img_st_fusion()

    ctrl_msg['data_folder_path'] = source + '_' + target + '-train'
    fusion_data_path = '/home/cwh/coding/TrackViz/'
    fusion_param = get_fusion_param()
    fusion_train_rank_pids_path = fusion_data_path + fusion_param['eval_fusion_path']
    fusion_train_rank_scores_path = fusion_data_path + fusion_param['fusion_normal_score_path']
    ctrl_msg['data_folder_path'] = source + '_' + target + '-test'
    fusion_param = get_fusion_param()
    fusion_test_rank_pids_path = fusion_data_path + fusion_param['eval_fusion_path']
    fusion_test_rank_scores_path = fusion_data_path + fusion_param['fusion_normal_score_path']
    return fusion_train_rank_pids_path, fusion_train_rank_scores_path, fusion_test_rank_pids_path, fusion_test_rank_scores_path


def rank_transfer(source, target, fusion_train_rank_pids_path, fusion_train_rank_scores_path):
    fusion_train_dir, fusion_test_dir = fusion_dir_prepare(source, target + '-r')
    transfer_train_rank_pids_path = fusion_train_dir + '/renew_pid.log'
    transfer_train_rank_scores_path = fusion_train_dir + '/renew_ac.log'
    transfer_test_rank_pids_path = fusion_test_dir + '/renew_pid.log'
    transfer_test_rank_scores_path = fusion_test_dir + '/renew_ac.log'
    if 'grid' in target:
        target_train_list = '/home/cwh/coding/TrackViz/data/grid/' + target + '-train.txt'
    elif target == 'markets1':
        target_train_list = '/home/cwh/coding/TrackViz/data/markets1/train.txt'
    elif target == 'market':
        target_train_list = '/home/cwh/coding/TrackViz/data/market/train.txt'
    else:
        target_train_list = 'error_target_dataset'
    os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 1 '
              + source + ' ' + target + ' '
              + fusion_train_rank_pids_path + ' '
              + fusion_train_rank_scores_path + ' '
              + transfer_train_rank_pids_path + ' '
              + transfer_train_rank_scores_path + ' '
              + transfer_test_rank_pids_path + ' '
              + transfer_test_rank_scores_path + ' '
              + target_train_list)
    return transfer_train_rank_pids_path, transfer_train_rank_scores_path, transfer_test_rank_pids_path, transfer_test_rank_scores_path


def fusion_transfer(source, target):
    # vision rank and eval
    vision_train_rank_pids_path, vision_train_rank_scores_path, \
    vision_test_rank_pids_path, vision_test_rank_scores_path \
        = vision_rank(source, target)

    # fusion rank and eval
    fusion_train_rank_pids_path, fusion_train_rank_scores_path, \
    fusion_test_rank_pids_path, fusion_test_rank_scores_path = st_fusion(source, target)
    dataset_eval(source, target, fusion_test_rank_pids_path)

    for i in range(1):
    # rank transfer, rank and eval
        transfer_train_rank_pids_path, transfer_train_rank_scores_path, \
        transfer_test_rank_pids_path, transfer_test_rank_scores_path \
            = rank_transfer(source, target, fusion_train_rank_pids_path, fusion_train_rank_scores_path)
        transfer_target = target + '-r'
        # fusion rank and eval

        fusion_train_rank_pids_path, fusion_train_rank_scores_path, \
        fusion_test_rank_pids_path, fusion_test_rank_scores_path \
            = st_fusion(source, transfer_target)
        dataset_eval(source, transfer_target, fusion_test_rank_pids_path)


def dataset_fusion_transfer():
    # sources = ['market', 'cuhk', 'viper', 'grid']
    # sources = ['grid']
    # sources = ['market']
    # for source in sources:
    #     for i in range(0, 1):
    #         if 'grid' in source:
    #             fusion_transfer('grid-cv-%d' % i, 'grid-cv%d' % i)
    #         else:
    #             fusion_transfer(source, 'grid-cv%d' % i)
    # sources = ['market', 'grid', 'cuhk', 'viper']
    # sources = ['grid']
    # sources = ['market']
    # for source in sources:
    #     fusion_transfer(source, 'market')
    sources = ['grid', 'viper', 'cuhk']
    # sources = ['market']
    for source in sources:
        fusion_transfer(source, 'market')


if __name__ == '__main__':
    dataset_fusion_transfer()
    # vision_rank('grid', 'market')
    # dataset_eval('market', 'grid-cv1', '/home/cwh/coding/TrackViz/data/market_grid-cv1-r-test/renew_pid.log')
    # dataset_eval('market', 'grid-cv1', '/home/cwh/coding/TrackViz/data/market_grid-cv1-r-test/cross_filter_pid.log')