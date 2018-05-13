from util.file_helper import read_lines
from util.serialize import pickle_save
from viz.compute_gt_deltas import load_train_img_infos


def calculate_mid_frame(tracks):
    frames = zip(*tracks)[2]
    mid = sorted(frames)[len(tracks)/2]
    print mid
    return mid

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
    compute_trainset_deltas('../data/duke/train.list', 'duke_real.pck')

