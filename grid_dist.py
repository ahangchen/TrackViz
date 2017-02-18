from delta_track import viz_data_for_market
from file_helper import read_lines_and
from file_helper import write_line
from serialize import pickle_save
data_type = 1

predict_person_path = 'grid_predict/predict_grid.log'
rand_person_path = 'grid_predict/rand/predict_grid_rand.log'
origin_path = 'grid_predict/grid_tracks.txt'


last_class = 0


def get_tracks():
    tracks = list()

    def add_track(line):
        tracks.append(line)

    read_lines_and(origin_path, add_track)
    return tracks


predict_line_idx = 0
predict_track_path = 'grid_predict/predict_grid.txt'
rand_final_path = 'grid_predict/rand/predict_grid.txt'


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
        write_line(predict_track_path, '%04d' % (int(predict_line_idx) + 1) + tail)
        write_line('grid_predict/grid_c%d.txt' % int(camera), ('%04d' % (int(predict_line_idx) + 1) + tail))
        for i, mid in enumerate(mids):
            if i >= 5:
                break
            if data_type == 0:
                write_line(predict_track_path, '%04d' % int(mid) + tail)
                write_line('grid_predict/grid_c%d.txt' % int(camera), ('%04d' % int(mid)) + tail)
            else:
                write_line(rand_final_path, '%04d' % int(mid) + tail)
                write_line('grid_predict/rand/grid_c%d.txt' % int(camera), ('%04d' % int(mid)) + tail)

        predict_line_idx += 1
    if data_type == 0:
        read_lines_and(predict_person_path, add_predict_track)
    else:
        read_lines_and(rand_person_path, add_predict_track)


def store_sorted_deltas():
    camera_delta_s = viz_data_for_market()
    for camera_delta in camera_delta_s:
        for delta_s in camera_delta:
            delta_s.sort()
    # for matlab
    for camera_delta in camera_delta_s:
        for delta_s in camera_delta:
            per_camera_deltas = ' '.join(map(str, delta_s))
            write_line('top10/sorted_deltas.txt', per_camera_deltas)
    # for python
    pickle_save('top10/sorted_deltas.pickle', camera_delta_s)

if __name__ == '__main__':
    get_predict_tracks()
    # store_sorted_deltas()
