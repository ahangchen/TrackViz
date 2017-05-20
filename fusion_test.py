from post_process.predict_eval import eval_on_train_test
from profile.fusion_param import get_fusion_param, ctrl_msg
from train.st_filter import fusion_st_gallery_ranker 

if __name__ == '__main__':
    ctrl_msg['data_folder_path'] = 'market_test'
    fusion_param = get_fusion_param()
    # cross_st_img_ranker(fusion_param)
    fusion_st_gallery_ranker(fusion_param)
    # eval_on_train_test(fusion_param)
