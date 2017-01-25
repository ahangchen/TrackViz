from delta_track import viz_market
from file_helper import read_lines_and
from file_helper import write
from raw_data import train_track_path

predict_path = 'top10/predict.log'

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
    read_lines_and(train_track_path, add_track)
    return tracks


predict_line_idx = 0
predict_track_path = 'top10/predict_tracks.txt'


def get_predict_tracks():
    train_tracks = get_tracks()
    person_ids = get_person_idx()

    def add_predict_track(line):
        global predict_line_idx
        tail = train_tracks[predict_line_idx][4:-1]
        mids = line.split()
        for mid in mids:
            write(predict_track_path, '%04d' % person_ids[int(mid)] + tail)
        predict_line_idx += 1
    read_lines_and(predict_path, add_predict_track)


if __name__ == '__main__':
    # get_predict_tracks()
    viz_market()