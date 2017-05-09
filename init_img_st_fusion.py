import sys

from ctrl.img_st_fusion import init_strict_img_st_fusion
from profile.fusion_param import ctrl_msg

if __name__ == '__main__':
    if len(sys.argv) > 1:
        ctrl_msg['data_folder_path'] = sys.argv[1]
        ctrl_msg['cross_idx'] = int(sys.argv[1][11])
        print(ctrl_msg)
    init_strict_img_st_fusion()
