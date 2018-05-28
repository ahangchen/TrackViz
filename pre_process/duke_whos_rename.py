import shutil

from util.file_helper import safe_mkdir, safe_remove, write
from viz.compute_gt_deltas import load_train_img_infos


def duke2whos_fmt(dir_path, data_type='train'):
    real_tracks, max_camera_cnt = load_train_img_infos(dir_path)
    duke_tracks_realloc(real_tracks, 'duke_whos', data_type)


def duke_tracks_realloc(tracks, track_name, data_type):
    target_list_path = '../data/%s/%s.list' % (track_name, data_type)
    target_track_info_dir_path = '/home/cwh/coding/TrackViz/data/%s' % track_name
    safe_mkdir(target_track_info_dir_path)
    target_track_dir_path = '/home/cwh/coding/%s' % track_name
    safe_mkdir(target_track_dir_path)
    target_track_type_dir_path = '/home/cwh/coding/%s/%s' % (track_name, data_type)
    safe_mkdir(target_track_type_dir_path)
    for track in tracks:
        source_img_name = '%04d_c%d_f%07d.jpg' % (int(track[0]), track[1], track[2])
        target_img_name = '%05d%07d%03d.jpg' % (int(track[0]), track[2], track[1])
        shutil.copyfile('/home/cwh/coding/DukeMTMC-reID/%s/%s' % (data_type, source_img_name), '%s/%s' % (target_track_type_dir_path, target_img_name))
    safe_remove(target_list_path)

if __name__ == '__main__':
    duke2whos_fmt('../data/duke/train.list', 'train')
    duke2whos_fmt('../data/duke/probe.list', 'probe')
    duke2whos_fmt('../data/duke/test.list', 'test')