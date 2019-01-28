import os

from ctrl.img_st_fusion import init_strict_img_st_fusion
from profile.fusion_param import ctrl_msg, get_fusion_param, update_msg


def real_fusion(src, dst):
    ctrl_msg['data_folder_path'] = src + '_' + dst + '-train'
    update_msg['gt_fusion'] = True
    init_strict_img_st_fusion()
    ctrl_msg['data_folder_path'] = src + '_' + dst + '-test'
    os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    # PYTHON eval_on_result.py --target_dataset_path $data_dir --pid_path $pid_path --result_path $log_path
    python_path = '/home/cwh/anaconda3/bin/python'
    eval_sh_path = '/home/cwh/coding/taudl_pyt/baseline/eval_on_result.py'
    target_dataset_path = '/home/cwh/coding/dataset/' + dst
    fusion_param = get_fusion_param()
    pid_path = '/home/cwh/coding/TrackViz/' + fusion_param['eval_fusion_path']
    log_path = src + '_' + dst + '_gt_eval.log'
    os.system(
        'export PYTHONPATH=/home/cwh/coding/taudl_pyt; %s %s --target_dataset_path %s --pid_path %s --result_path %s ' % (
            python_path, eval_sh_path, target_dataset_path, pid_path, log_path))


if __name__ == '__main__':
    srcs = ['grid', 'cuhk', 'viper', 'duke', 'market']
    dsts = ['market', 'duke']
    for src in srcs:
        for dst in dsts:
            real_fusion(src, dst)
        for i in range(10):
            dst = 'grid-cv-%d' % i
            real_fusion(src, dst)