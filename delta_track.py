import random

import numpy as np
import seaborn
import matplotlib.pyplot as plt

from file_helper import read_lines_and
from raw_data import camera_cnt, viz_camera


def camera_intervals(camera_num):
    intervals = list()
    cur_values = { 'id': 0, 'start': 0, 'end': 0}

    def count_interval(img_name):
        if '.' not in img_name:
            return
        track_info = img_name.split('.')[0].split('_')
        person_id = track_info[0]
        track_time = int(track_info[2])
        if person_id != cur_values['id']:
            intervals.append([cur_values['id'], cur_values['start'], cur_values['end']])
            cur_values['id'] = person_id
            cur_values['start'] = track_time
            cur_values['end'] = track_time
        else:
            if track_time > cur_values['end']:
                cur_values['end'] = track_time

    read_lines_and('data_s1/track_c%ds1.txt' % (camera_num), count_interval)
    return intervals[1:]


def find_id_delta(intervals, id, frame):
    frame = int(frame)
    for interval in intervals:
        if interval[0] == id:
            delta0 = frame - interval[1]
            delta1 = frame - interval[2]
            if abs(delta0) < abs(delta1):
                return delta0
            else:
                return delta1
        else:
            continue
    return -0.1


def camera_distribute(camera_num):
    intervals = camera_intervals(camera_num)
    print('get intervals for c%d' % camera_num)
    deltas = list()
    cur_delta = {'id': 0, 'delta': 1000000, 'camera': -1}

    def shuffle_person(img_name):
        if '.' not in img_name:
            return
        track_info = img_name.split('.')[0].split('_')
        person_id = track_info[0]
        track_delta = find_id_delta(intervals, person_id, int(track_info[2]))
        camera_id = int(track_info[1][1])
        if cur_delta['id'] != person_id or cur_delta['camera'] != camera_id:
            # new person
            if cur_delta['id'] != 0:
                # exclude first zero record and not found id records
                # deltas.append([cur_delta['id'], cur_delta['camera'], cur_delta['delta']])
                # ignore large data
                if abs(cur_delta['delta']) < 10000:
                    deltas.append([cur_delta['camera'] + random.uniform(-0.4, 0.4), cur_delta['delta']])
            if track_delta == -0.1:
                # id not found
                cur_delta['id'] = 0
                return
            # new id, has appeared in camera -camera_num
            cur_delta['id'] = person_id
            cur_delta['delta'] = track_delta
            cur_delta['camera'] = camera_id
        elif abs(cur_delta['delta'] ) > abs(track_delta):
            # get the smallest delta
            cur_delta['delta'] = track_delta

    read_lines_and('data_s1/track_s1.txt', shuffle_person)
    return deltas


def viz_data_for_market():
    track_distribute = list()
    for i in range(camera_cnt):
        track_distribute.append(camera_distribute(i + 1))
    return track_distribute


def viz_market():
    viz_data = viz_data_for_market()
    fig = plt.figure()
    for i in range(camera_cnt):
        track_data = np.array(viz_data[i]).transpose()
        viz_camera(fig, track_data, i + 1, 10, m='x')
        print('viz camera %d' % (i + 1))
    plt.show()


if __name__ == '__main__':
    # print(camera_distribute(1))
    viz_market()