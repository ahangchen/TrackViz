from delta_track import viz_market_distribution, viz_data_for_market
from file_helper import read_lines_and
from file_helper import write
from raw_data import train_track_path
from serialize import pickle_save

predict_path = 'grid_predict/predict_grid.log'
test_path = 'grid_predict/grid_tracks.txt'


last_class = 0


def get_person_idx():
    ids = list()

    def count_id(line):
        global last_class
        cur_class = int(line.split('_')[0])
        if cur_class != last_class:
            ids.append(cur_class)
            last_class = cur_class

    read_lines_and(train_track_path, count_id)
    print(ids)
    return ids


def get_tracks():
    tracks = list()

    def add_track(line):
        tracks.append(line)

    read_lines_and(test_path, add_track)
    return tracks


predict_line_idx = 0
predict_track_path = 'grid_predict/predict_grid.txt'


def get_predict_tracks():
    origin_tracks = get_tracks()
    # person_ids = get_person_idx()

    def add_predict_track(line):
        global predict_line_idx
        if origin_tracks[predict_line_idx].startswith('-1'):
            tail = origin_tracks[predict_line_idx][2:-1]
        else:
            tail = origin_tracks[predict_line_idx][4: -1]
        camera = tail[1]
        mids = line.split()
        for mid in mids:
            write(predict_track_path, origin_tracks)
            write(predict_track_path, '%04d' % int(mid) + tail)
            write('grid_predict/grid_c%d.txt' % int(camera), ('%04d' % int(mid)) + tail)
            write('grid_predict/grid_c%d.txt' % int(camera), origin_tracks)
        predict_line_idx += 1

    read_lines_and(predict_path, add_predict_track)


def store_sorted_deltas():
    camera_delta_s = viz_data_for_market()
    for camera_delta in camera_delta_s:
        for delta_s in camera_delta:
            delta_s.sort()
    # for matlab
    for camera_delta in camera_delta_s:
        for delta_s in camera_delta:
            per_camera_deltas = ' '.join(map(str, delta_s))
            write('top10/sorted_deltas.txt', per_camera_deltas)
    # for python
    pickle_save('top10/sorted_deltas.pickle', camera_delta_s)

if __name__ == '__main__':
    get_predict_tracks()
    # store_sorted_deltas()
