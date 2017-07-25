#coding=utf-8
import shutil
from feature.average_predict import write_rand_pid, gen_rand_st_model, write_unequal_rand_st_model
from feature.top10distribution import get_predict_tracks, store_sorted_deltas
from post_process.predict_eval import eval_on_train_test, pos_neg_shot_eval, target_pos_neg_shot_eval
from pre_process.shot_rate import get_shot_rate
from profile.fusion_param import get_fusion_param, ctrl_msg, update_fusion_param, save_fusion_param
from train.delta_track import viz_fusion_curve, viz_market_distribution
from train.st_filter import cross_st_img_ranker, fusion_st_img_ranker, fusion_curve

# need to run on src directory
from util.file_helper import write_line, safe_remove


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


def test_fusion(fusion_param, ep=0.5, en=0.01):
    # copy sort pickle
    safe_remove(fusion_param['distribution_pickle_path'])
    try:
        # 直接使用训练集的时空模型
        shutil.copy(fusion_param['src_distribution_pickle_path'], fusion_param['distribution_pickle_path'])
        print 'copy train track distribute pickle done'
    except shutil.Error:
        print 'pickle ready'
    # merge visual probability and track distribution probability
    fusion_st_img_ranker(fusion_param, ep, en)
    # evaluate
    eval_on_train_test(fusion_param, test_mode=True)


def train_fusion(fusion_param, ep=0.5, en=0.01):
    # 这里不需要再做一次时空模型建立
    # get_predict_tracks(fusion_param)
    # get distribution sorted list for probability compute
    # store_sorted_deltas(fusion_param)
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
    # 全局调度入口，会同时做训练集和测试集上的融合与评分
    fusion_param = get_fusion_param()
    print('init predict tracks into different class files')
    # pick predict tracks into different class file
    get_predict_tracks(fusion_param)
    # get distribution sorted list for probability compute
    store_sorted_deltas(fusion_param)

    # # only get rand model for train dataset
    # todo 随机时空模型在增量后不需要反复建立
    print('generate random predict')
    write_rand_pid(fusion_param)
    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'] + '_rand'
    fusion_param = get_fusion_param()
    # 生成随机时空点的时空模型
    gen_rand_st_model(fusion_param)

    # 改回非随机的train目录
    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-5]

    # has prepared more accurate ep, en
    print('fusion on training dataset')
    iter_strict_img_st_fusion(on_test=False)
    # 改成测试目录
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
    # ep, en = get_shot_rate()
    if on_test:
        test_fusion(fusion_param)
    else:
        train_fusion(fusion_param)
        # update_epen(fusion_param, True)


if __name__ == '__main__':
    # img_st_fusion()
    # retrain_fusion()
    # init_strict_img_st_fusion()
    # for i in range(10):
    #     print('iteration %d' % i)
    #     ctrl_msg['cross_idx'] = i
    #     # ctrl_msg['data_folder_path'] = 'top-m2g-std%d-r-train' % i
    #     # fusion_param = get_fusion_param()
    #     # get_predict_tracks(fusion_param)
    #     # store_sorted_deltas(fusion_param)
    #     # ctrl_msg['data_folder_path'] = 'top-m2g-std%d-r-test' % i
    #     # iter_strict_img_st_fusion(on_test=True)
    #     ctrl_msg['data_folder_path'] = 'top-m2g-std%d-test' % i
    #     iter_strict_img_st_fusion(on_test=True)
    # # viz fusion curve
    # fusion_param = get_fusion_param()
    # get_predict_tracks(fusion_param)
    # store_sorted_deltas(fusion_param)
    #
    # print('generate random predict')
    # write_rand_pid(fusion_param)
    # ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'] + '_rand'
    # fusion_param = get_fusion_param()
    # gen_rand_st_model(fusion_param)
    #
    # ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-5]
    # fusion_param = get_fusion_param()
    ctrl_msg['data_folder_path'] = 'top-m2g-std0-test'
    fusion_param = get_fusion_param()
    delta_range, raw_probs, rand_probs, over_probs = fusion_curve(fusion_param)
    viz_fusion_curve(delta_range, [raw_probs, rand_probs, over_probs])

    # viz smooth dist
    # viz_market_distribution(fusion_param)
