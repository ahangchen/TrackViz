from util.file_helper import read_lines
from util.serialize import pickle_save
from viz.compute_gt_deltas import load_train_img_infos


def calculate_part_frame(tracks, slice_cnt):
    frames = zip(*tracks)[2]
    slice_bounds = list()
    sorted_frames = sorted(frames)
    for i in range(slice_cnt + 1):
        slice_bounds.append(sorted_frames[min(len(tracks) / slice_cnt * i, len(tracks) - 1)])
    for slice_bound in slice_bounds:
        print slice_bound
    return slice_bounds


def calculate_mid_frame(tracks):
    mid = calculate_part_frame(tracks, 2)[0]
    print mid
    return mid


def split_trainset_deltas(train_imgs_path, pickle_path, slice_cnt):
    train_tracks, camera_cnt = load_train_img_infos(train_imgs_path)
    slice_bounds =calculate_part_frame(train_tracks, slice_cnt)
    split_tracks = list()
    for i in range(slice_cnt/2 + 1):
        split_tracks.append(filter(lambda x: slice_bounds[i]<x[2] < slice_bounds[i+slice_cnt/2], train_tracks))

    for i, split_track in enumerate(split_tracks):
        deltas = [[list() for _ in range(camera_cnt)] for _ in range(camera_cnt)]
        for query_track in split_track:
            for gallery_track in split_track:
                if query_track[0] == gallery_track[0] and query_track[1] != gallery_track[1]:
                    delta = query_track[2] - gallery_track[2]
                    if abs(delta) < 60000:
                        deltas[query_track[1]-1][gallery_track[1] - 1].append(delta)
        pickle_save(('part%d_' % i) + pickle_path, deltas)


def compute_trainset_deltas(train_imgs_path, pickle_path):
    train_tracks, camera_cnt = load_train_img_infos(train_imgs_path)
    mid = calculate_mid_frame(train_tracks)
    head_train_tracks = filter(lambda x: x[2] < mid, train_tracks)
    deltas = [[list() for _ in range(camera_cnt)] for _ in range(camera_cnt)]
    for query_track in head_train_tracks:
        for gallery_track in head_train_tracks:
            if query_track[0] == gallery_track[0] and query_track[1] != gallery_track[1]:
                delta = query_track[2] - gallery_track[2]
                if abs(delta) < 60000:
                    deltas[query_track[1]-1][gallery_track[1] - 1].append(delta)
    pickle_save('head_' + pickle_path, deltas)

    tail_train_tracks = filter(lambda x: x[2] > mid, train_tracks)
    deltas = [[list() for _ in range(camera_cnt)] for _ in range(camera_cnt)]
    for query_track in tail_train_tracks:
        for gallery_track in tail_train_tracks:
            if query_track[0] == gallery_track[0] and query_track[1] != gallery_track[1]:
                delta = query_track[2] - gallery_track[2]
                if abs(delta) < 60000:
                    deltas[query_track[1]-1][gallery_track[1] - 1].append(delta)
    pickle_save('tail_' + pickle_path, deltas)


if __name__ == '__main__':
    train_tracks, camera_cnt = load_train_img_infos('../data/duke/train.list')
    # calculate_part_frame(train_tracks, 4)
    split_trainset_deltas('../data/duke/train.list', 'duke_real.pck', 10)
    # compute_trainset_deltas('../data/duke/train.list', 'duke_real.pck')

