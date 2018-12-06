import argparse
import os

from ctrl.img_st_fusion import init_strict_img_st_fusion
from profile import fusion_param
from profile.fusion_param import get_fusion_param
from util.file_helper import safe_link


def arg_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data_folder_path', default='cuhk_grid-cv-0-train', type=str, help='fusion data output dir')
    parser.add_argument('--cv_num', default=0, type=int, help='0...9, for grid cross validation')
    parser.add_argument('--ep', default=0.0, type=float, help='[0,1], error of position sample')
    parser.add_argument('--en', default=0.0, type=float, help='[0,1], error of negative sample')
    parser.add_argument('--window_interval', default=500, type=int, help='')
    parser.add_argument('--filter_interval', default=80000, type=int, help='')
    parser.add_argument('--vision_folder_path', default='/home/cwh/coding/taudl_pyt/baseline/eval/cuhk_grid-cv-0-train', type=str, help='')


    opt = parser.parse_args()
    return opt

def build_param(opt):
    fusion_param.ctrl_msg['data_folder_path'] = opt.data_folder_path
    fusion_param.ctrl_msg['cv_num'] = opt.cv_num
    fusion_param.ctrl_msg['ep'] = opt.ep
    fusion_param.ctrl_msg['en'] = opt.en
    fusion_param.ctrl_msg['window_interval'] = opt.window_interval
    fusion_param.ctrl_msg['filter_interval'] = opt.filter_interval
    param = get_fusion_param()
    safe_link(opt.vision_folder_path + '/pid.txt', param['renew_pid_path'])
    safe_link(opt.vision_folder_path + '/score.txt', param['renew_ac_path'])
    fusion_param.ctrl_msg['data_folder_path'] = opt.data_folder_path.replace('train', 'test')
    param = get_fusion_param()
    safe_link(opt.vision_folder_path.replace('train', 'test') + '/pid.txt', param['renew_pid_path'])
    safe_link(opt.vision_folder_path.replace('train', 'test') + '/score.txt', param['renew_ac_path'])
    fusion_param.ctrl_msg['data_folder_path'] = opt.data_folder_path

def main():
    opt = arg_parse()
    build_param(opt)
    init_strict_img_st_fusion()

if __name__ == '__main__':
    main()