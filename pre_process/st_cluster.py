import argparse


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.cluster import SpectralClustering, AffinityPropagation, DBSCAN, KMeans
from sklearn import metrics
from sklearn.metrics import euclidean_distances

from util.file_helper import read_lines
import numpy as np
import scipy.io
from util.serialize import pickle_save, pickle_load


def parse_tracks(track_path):
    answer_lines = read_lines(track_path)
    real_tracks = list()
    cameras = set()
    seqs = set()
    for i, answer in enumerate(answer_lines):
        info = answer.split('_')
        if len(info) < 1:
            continue
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            real_tracks.append([int(info[0]), int(info[1][0]), int(info[2])])
        elif 'f' in info[2]:
            real_tracks.append([int(info[0]), int(info[1][1]), int(info[2][1:-5]), 1])
        else:
            real_tracks.append([int(info[0]), int(info[1][1]), int(info[2]), int(info[1][3])])
        cameras.add(real_tracks[i][1])
        seqs.add(real_tracks[i][3])
    return real_tracks, cameras, seqs


def single_camera_time_cluster(track_path):
    # assign tracklet id to images in the same camera with
    real_tracks, cameras, seqs = parse_tracks(track_path)
    camera_cnt = len(cameras)
    seq_cnt = len(seqs)
    camera_tracks = [[ [] for j in range(seq_cnt)] for i in range(camera_cnt)]
    for i, (pid, camera, t, seq) in enumerate(real_tracks):
        camera_tracks[camera - 1][seq - 1].append([pid, t, i])

    pseudo_camera_tracks = [[ [] for j in range(seq_cnt)] for i in range(camera_cnt)]
    avg_flow_camera_seq = [[ [] for j in range(seq_cnt)] for i in range(camera_cnt)]
    for i, camera_seqs in enumerate(camera_tracks):
        for j, camera_seq in enumerate(camera_seqs):
            if len(camera_seq) == 0:
                continue
            sort_seq_idx = sorted(range(len(camera_seq)), key=lambda k: camera_seq[k][1])
            last_time = camera_seq[sort_seq_idx[0]][1]
            last_pid = 0
            camera_seq[sort_seq_idx[0]].append(last_pid)
            pseudo_camera_tracks[i][j].append(camera_seq[sort_seq_idx[0]])

            for idx in sort_seq_idx[1:]:
                cur_time_flow = camera_seq[idx][1] - last_time
                if abs(cur_time_flow) > 1000:
                    last_pid += 1
                last_time = camera_seq[idx][1]
                camera_seq[idx].append(last_pid)
                pseudo_camera_tracks[i][j].append(camera_seq[idx])

                if camera_seq[idx][0] != camera_seq[idx - 1]:
                    avg_flow_camera_seq[i][j].append(cur_time_flow)
            avg_flow_camera_seq[i][j] = sum(avg_flow_camera_seq[i][j]) / len(avg_flow_camera_seq[i][j])
    print(avg_flow_camera_seq)
    # [[215, 258, 277, 154, 185, 222],
    # [280, 240, 216, [], [], []],
    # [200, 133, 151, [], [], []],
    # [385, 484, 444, 396, 375, 883],
    # [152, 185, 242, [], [], []],
    # [89, 92, 123, 295, [], []]]
    return pseudo_camera_tracks


class TrackFeatureCluster:
    def __init__(self, pseudo_cameras_tracks, visual_feature_path):
        # pseudo_cameras_tracks: camera_cnt * seq_cnt lists
        print('loading affinity')
        self.feature = scipy.io.loadmat(visual_feature_path)['ft']
        self.cameras_tracks = pseudo_cameras_tracks
        self.ap_score = 0
        self.tracklet_cnt = 0
        self.spectral_score = 0
        self.kmeans_score = 0

    def cluster(self, tracklet):
        # tracklet: [[pid, time, img_id, pseudo_id],...], only img_id is used
        ids = map(lambda arr: arr[2], tracklet)
        # cluster = SpectralClustering(n_clusters=2, affinity='precomputed')
        track_features = []
        for i, img_i in enumerate(ids):
            track_features.append(self.feature[img_i])
        similarity = -euclidean_distances(track_features, squared=True)
        # cls = cluster.fit_predict(affinity)
        cluster = AffinityPropagation(preference=np.median(similarity))
        cls = cluster.fit_predict(track_features).reshape(-1)
        self.spectral_score += metrics.adjusted_rand_score([info[0] for info in tracklet], cls)

        cls_cnt = len(set(cls))
        #
        # cluster = SpectralClustering(n_clusters=cls_cnt)
        # cls = cluster.fit_predict(track_features).reshape(-1)
        # self.ap_score += metrics.adjusted_rand_score([info[0] for info in tracklet], cls)

        cluster = KMeans(n_clusters=cls_cnt)
        cls = cluster.fit_predict(track_features).reshape(-1)
        self.kmeans_score += metrics.adjusted_rand_score([info[0] for info in tracklet], cls)

        self.tracklet_cnt += 1
        return [(ids[i], cls[i]) for i in range(len(ids))]

    def fit(self):
        for i, pseudo_camera_tracks in enumerate(self.cameras_tracks):
            # same camera
            new_track_ids = []  # [[img_id, new_track_id], ...]
            for j, pseudo_camera_seq in enumerate(pseudo_camera_tracks):
                if len(pseudo_camera_seq) == 0:
                    continue
                # same seq
                seq_start_idx = len(new_track_ids)
                print('cluster on camera %d, seq %d' % (i, j))
                k = 0
                tmp_tracklet = []
                last_pse_track_id = 0
                while k < len(pseudo_camera_seq):
                    if pseudo_camera_seq[k][-1] != last_pse_track_id:
                        # do cluster; reassign id; clear tracklet
                        ids = self.cluster(tmp_tracklet)
                        if len(new_track_ids) > 0:
                            max_new_track_id = new_track_ids[-1][1]
                        else:
                            max_new_track_id = 0
                        print('pseudo before assgin:')
                        print(pseudo_camera_seq[k - len(ids):k])
                        print('assign ids:')
                        print(ids)
                        ids = map(lambda p: [p[0], p[1] + max_new_track_id + 1], ids)
                        new_track_ids.extend(ids)
                        del tmp_tracklet[:]
                    tmp_tracklet.append(pseudo_camera_seq[k])
                    last_pse_track_id = pseudo_camera_seq[k][3]
                    k += 1
                # do cluster; reassign id; clear tracklet
                ids = self.cluster(tmp_tracklet)
                if len(new_track_ids) > 0:
                    max_new_track_id = new_track_ids[-1][1]
                else:
                    max_new_track_id = 0
                print('pseudo before assgin:')
                print(pseudo_camera_seq[k - len(ids):k])
                print('assign ids:')
                print(ids)
                ids = map(lambda p: [p[0], p[1] + max_new_track_id + 1], ids)
                new_track_ids.extend(ids)
                del tmp_tracklet[:]

                print(pseudo_camera_seq)
                for l, (img_id, new_track_id) in enumerate(new_track_ids[seq_start_idx:]):
                    self.cameras_tracks[i][j][l].append(new_track_id)
            for j, pseudo_camera_seq in enumerate(pseudo_camera_tracks):
                if j == 0:
                    continue
                pseudo_camera_tracks[0].extend(pseudo_camera_seq)
                pseudo_camera_tracks[j] = []
        print('spectral:')
        print(self.ap_score / self.tracklet_cnt)
        print('kmeans:')
        print(self.kmeans_score / self.tracklet_cnt)
        print('ap:')
        print(self.spectral_score / self.tracklet_cnt)

def arg_parse():
    parser = argparse.ArgumentParser(description='eval on txt')
    parser.add_argument('--train_list', default='../data/duke/train.list', type=str, help='')
    parser.add_argument('--transfer_feature', default='/home/cwh/coding/taudl_pyt/baseline/eval/grid_duke-train/train_ft.mat', type=str, help='')
    parser.add_argument('--transfer', default='grid_duke', type=str, help='')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = arg_parse()
    # a = pickle_load(opt.transfer+'_cluster.pck')
    pseudo_camera_tracks = single_camera_time_cluster(opt.train_list)
    c = TrackFeatureCluster(pseudo_camera_tracks, opt.transfer_feature) #0.31
    c.fit()
    pickle_save(opt.transfer + '_cluster.pck', c.cameras_tracks)
