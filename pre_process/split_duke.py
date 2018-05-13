from util.file_helper import write, safe_mkdir, safe_remove
from viz.compute_gt_deltas import load_train_img_infos
from viz.compute_local_gt_deltas import calculate_mid_frame
import shutil


def duke_tracks_realloc(tracks, track_name, data_type):
    target_list_path = '../data/%s/%s.list' % (track_name, data_type)
    target_track_info_dir_path = '/home/cwh/coding/TrackViz/data/%s' % track_name
    safe_mkdir(target_track_info_dir_path)
    target_track_dir_path = '/home/cwh/coding/%s' % track_name
    safe_mkdir(target_track_dir_path)
    target_track_type_dir_path = '/home/cwh/coding/%s/%s' % (track_name, data_type)
    safe_mkdir(target_track_type_dir_path)
    names = list()
    for track in tracks:
        img_name = '%04d_c%d_f%07d.jpg' % (int(track[0]), track[1], track[2])
        shutil.copyfile('/home/cwh/coding/DukeMTMC-reID/%s/%s' % (data_type, img_name), '%s/%s' % (target_track_type_dir_path, img_name))
        names.append(img_name)
        names.append('\n')
    list_stmt = ''.join(names)
    safe_remove(target_list_path)
    write(target_list_path, list_stmt)



def split_duke(img_list_path, data_type):
    tracks, camera_cnt = load_train_img_infos(img_list_path)
    mid = calculate_mid_frame(tracks)
    head_tracks = filter(lambda x: x[2] < mid, tracks)
    tail_tracks = filter(lambda x: x[2] > mid, tracks)
    track_name = 'dukehead'
    duke_tracks_realloc(head_tracks, track_name, data_type)
    track_name = 'duketail'
    safe_mkdir('../data/%s' % track_name)
    duke_tracks_realloc(tail_tracks, track_name, data_type)


if __name__ == '__main__':
    split_duke('../data/duke/train.list', data_type='train')
    split_duke('../data/duke/probe.list', data_type='probe')
    split_duke('../data/duke/test.list', data_type='test')

