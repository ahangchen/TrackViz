from util.file_helper import read_lines
from util.serialize import pickle_save


def load_train_img_infos(train_imgs_path):
    train_imgs_lines = read_lines(train_imgs_path)
    real_tracks = list()
    max_camera_cnt = 0
    for answer in train_imgs_lines:
        info = answer.split('_')
        if 'bmp' in info[2]:
            # viper
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            # grid
            real_tracks.append([info[0], int(info[1][0]), int(info[2]), 1])
        elif 'f' in info[2]:
            # duke
            real_tracks.append([info[0], int(info[1][1]), int(info[2][1:-5]), 1])
        else:
            # market
            real_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])
        # real track: person_id, camera_num, time, sequence num
        max_camera_cnt = max(real_tracks[-1][1], max_camera_cnt)
    return real_tracks, max_camera_cnt


def compute_trainset_deltas(train_imgs_path, pickle_path):
    train_tracks, camera_cnt = load_train_img_infos(train_imgs_path)
    deltas = [[list() for _ in range(camera_cnt)] for _ in range(camera_cnt)]
    for query_track in train_tracks:
        for gallery_track in train_tracks:
            if query_track[0] == gallery_track[0]:
                delta = query_track[2] - gallery_track[2]
                if abs(delta) < 300000:
                    deltas[query_track[1]-1][gallery_track[1] - 1].append(delta)
    pickle_save(pickle_path, deltas)


if __name__ == '__main__':
    compute_trainset_deltas('../data/duke/train.list', 'duke_real.pck')

