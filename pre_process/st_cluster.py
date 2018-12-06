from util.file_helper import read_lines


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
            real_tracks.append([info[0], int(info[1][0]), int(info[2])])
        elif 'f' in info[2]:
            real_tracks.append([info[0], int(info[1][1]), int(info[2][1:-5]), 1])
        else:
            real_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])
        cameras.add(real_tracks[i][1])
        seqs.add(real_tracks[i][3])
    return real_tracks, cameras, seqs


def single_camera_time_cluster(track_path):
    real_tracks, cameras, seqs = parse_tracks(track_path)
    camera_cnt = len(cameras)
    seq_cnt = len(seqs)
    camera_tracks = [[ [] for j in range(seq_cnt)] for i in range(camera_cnt)]
    for pid, camera, t, seq in real_tracks:
        camera_tracks[camera - 1][seq - 1].append([pid, t])

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
                if abs(cur_time_flow) > 400:
                    last_pid += 1
                last_time = camera_seq[idx][1]
                camera_seq[idx].append(last_pid)
                pseudo_camera_tracks[i][j].append(camera_seq[idx])

                if camera_seq[idx][0] != camera_seq[idx - 1]:
                    avg_flow_camera_seq[i][j].append(cur_time_flow)
            avg_flow_camera_seq[i][j] = sum(avg_flow_camera_seq[i][j]) / len(avg_flow_camera_seq[i][j])
    print avg_flow_camera_seq
    # [[215, 258, 277, 154, 185, 222],
    # [280, 240, 216, [], [], []],
    # [200, 133, 151, [], [], []],
    # [385, 484, 444, 396, 375, 883],
    # [152, 185, 242, [], [], []],
    # [89, 92, 123, 295, [], []]]
    return pseudo_camera_tracks


if __name__ == '__main__':
    single_camera_time_cluster('../data/duke/train.list')
