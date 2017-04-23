import shutil
from feature.average_predict import write_rand_pid, gen_rand_st_model
from feature.top10distribution import get_predict_tracks, store_sorted_deltas
from post_process.predict_eval import eval_on_train_test, pos_neg_shot_eval, target_pos_neg_shot_eval
from pre_process.shot_rate import get_shot_rate
from profile.fusion_param import get_fusion_param, ctrl_msg, update_fusion_param, save_fusion_param
from train.st_filter import cross_st_img_ranker, fusion_st_img_ranker


# need to run on src directory
from util.file_helper import write_line


def img_st_fusion():
    # pick predict tracks into different class file
    fusion_param = get_fusion_param()
    get_predict_tracks(fusion_param)
    # get distribution sorted list for probability compute
    store_sorted_deltas(fusion_param)

    # merge visual probability and track distribution probability
    cross_st_img_ranker(fusion_param)

    # evaluate
    eval_on_train_test(fusion_param)


def test_fusion(fusion_param, ep, en):
    # copy sort pickle
    shutil.copy(fusion_param['src_distribution_pickle_path'], fusion_param['distribution_pickle_path'])
    # merge visual probability and track distribution probability
    fusion_st_img_ranker(fusion_param, ep, en)
    # evaluate
    eval_on_train_test(fusion_param)


def train_fusion(fusion_param, ep, en):
    get_predict_tracks(fusion_param)
    # get distribution sorted list for probability compute
    store_sorted_deltas(fusion_param)
    fusion_st_img_ranker(fusion_param, ep, en)
    # evaluate
    eval_on_train_test(fusion_param)


def update_epen(fusion_param, pst=True):
    ep, en = target_pos_neg_shot_eval(fusion_param['fusion_normal_score_path'], fusion_param['renew_pid_path'],
                                      fusion_param['renew_ac_path'])
    if pst:
        write_line('data/ep_en.txt', ctrl_msg['data_folder_path'])
        write_line('data/ep_en.txt', '%f %f' % (ep, en))
    update_fusion_param('pos_shot_rate', ep)
    update_fusion_param('neg_shot_rate', en)


def init_strict_img_st_fusion():
    fusion_param = get_fusion_param()
    print('init predict tracks into different class files')
    # pick predict tracks into different class file
    get_predict_tracks(fusion_param)
    # get distribution sorted list for probability compute
    store_sorted_deltas(fusion_param)
    # only get rand model for train dataset

    print('generate random predict')
    write_rand_pid(fusion_param)
    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'] + '_rand'
    fusion_param = get_fusion_param()
    gen_rand_st_model(fusion_param)

    print('init fusion, try to get ep en')
    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-5]
    fusion_param = get_fusion_param()
    # need init ep, en to do first fusion and get first ep, en
    ep, en = pos_neg_shot_eval(fusion_param['renew_pid_path'], fusion_param['renew_ac_path'])
    fusion_st_img_ranker(fusion_param, ep, en)
    eval_on_train_test(fusion_param)

    update_epen(fusion_param)

    # has prepared more accurate ep, en
    print('fusion on training dataset')
    iter_strict_img_st_fusion(on_test=False)
    print('fusion on test dataset')
    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-4] + 'est'
    iter_strict_img_st_fusion(on_test=True)


def iter_strict_img_st_fusion(on_test=False):
    """
    call after img classifier update, train with new vision score and ep en
    :param on_test: 
    :return: 
    """
    fusion_param = get_fusion_param()
    ep, en = get_shot_rate()
    if on_test:
        test_fusion(fusion_param, ep, en)
    else:
        train_fusion(fusion_param, ep, en)
        update_epen(fusion_param, True)


if __name__ == '__main__':
    # img_st_fusion()
    # retrain_fusion()
    init_strict_img_st_fusion()
