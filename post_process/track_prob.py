# coding=utf-8
from profile.fusion_param import ctrl_msg, get_fusion_param
from util.serialize import pickle_load
import numpy as np
from bisect import bisect_left

def binary_search(a, target):
    return bisect_left(a, target)
    # return np.searchsorted(a, target)
    # return c_bisect_lib.bisect_left(a, len(a) - 1, target)
    # return bisect_left(a, target)
    # 不同于普通的二分查找，目标是寻找target最适合的index
    # low = 0
    # high = len(a) - 1
    #
    # while low <= high:
    #     mid = (low + high) // 2
    #     mid_val = a[mid]
    #     if mid_val < target:
    #         low = mid + 1
    #     elif mid_val > target:
    #         high = mid - 1
    #     else:
    #         return mid
    # return low


def track_score(camera_delta_s, camera1, time1, camera2, time2, interval=100, moving_st=False, filter_interval=1000):
    if moving_st:
        return local_model_score(camera_delta_s, camera1, time1, camera2, time2, interval=interval,
                                 filter_interval=filter_interval)
    else:
        return global_track_score(camera_delta_s, camera1, time1, camera2, time2, interval=interval,
                                  filter_interval=filter_interval)


def global_track_score(camera_delta_s, camera1, time1, camera2, time2, interval=100, filter_interval=1000):
    if abs(time1 - time2) > filter_interval:
        return -1.
    camera1 -= 1
    camera2 -= 1
    cur_delta = time1 - time2
    delta_distribution = camera_delta_s[camera1][camera2]
    total_cnt = sum(map(len, camera_delta_s[camera1]))
    # 10 second
    left_bound = cur_delta - interval
    right_bound = cur_delta + interval
    # 二分查找位置，得到容错区间内时空点数量
    left_index = binary_search(delta_distribution, left_bound)
    right_index = binary_search(delta_distribution, right_bound)
    if total_cnt == 0 or len(camera_delta_s[camera1][camera2]) == 0:
        return 0.0
    # 这里除以total_cnt而非len(camera_delta_s[camera1][camera2])，体现空间概率
    score = (right_index - left_index) / float(total_cnt)
    # 训练集中同摄像头概率很高,但评估又不要同摄像头的,体现空间概率很不划算
    # score = (right_index - left_index + 1) / float(len(camera_delta_s[camera1][camera2]))
    if len(delta_distribution) == 0:
        return 0.0
    return score


def local_frame_infos(frames, time1, search_interval_length):
    delta_range = abs(frames[-1] - frames[0])

    left_frame = max(time1 - search_interval_length / 2, frames[0])
    if left_frame == frames[0]:
        right_frame = min(left_frame + search_interval_length, frames[-1])
    else:
        right_frame = min(time1 + search_interval_length / 2, frames[-1])
    if right_frame == frames[-1]:
        left_frame = max(right_frame - search_interval_length, frames[0])
    left_local_index = binary_search(frames, left_frame)
    right_local_index = binary_search(frames, right_frame)
    local_interval_length = max(right_local_index - left_local_index, 1)
    return local_interval_length, left_local_index, right_local_index


def other_frames_length(frames, left_frame, right_frame):
    left_frame = max(left_frame, frames[0])
    right_frame = min(right_frame, frames[-1])
    left_local_index = binary_search(frames, left_frame)
    right_local_index = binary_search(frames, right_frame)
    local_interval_length = max(right_local_index - left_local_index, 1)
    return local_interval_length


def local_model_score(distribution_dict, camera1, time1, camera2, time2, interval=100, filter_interval=50000):
    # if abs(time1 - time2) > filter_interval:
    #     return -1.
    cameras_delta_s = distribution_dict['deltas']
    cameras_frame_s = distribution_dict['frames']

    camera1 -= 1
    camera2 -= 1
    cur_delta = time1 - time2
    deltas = cameras_delta_s[camera1][camera2]
    frames = cameras_frame_s[camera1][camera2]

    total_cnt = len(deltas)
    if total_cnt == 0:
        return 0.0
    if abs(time1 - time2) > abs(frames[-1] - frames[0]):
        # print 'delta larger than frames width'
        return -1.
    # delta_range = abs(frames[-1] - frames[0])
    # local_prop = 2
    # filter_interval = delta_range / local_prop
    local_interval_length, left_local_index, right_local_index = local_frame_infos(frames, time1, filter_interval)

    left_time_min = frames[left_local_index]
    right_time_max = frames[right_local_index]
    for i in range(len(cameras_delta_s)):
        if i == camera1 or i == camera2:
            continue
        local_interval_length, left_local_index, right_local_index = local_frame_infos(frames, time1, filter_interval)
        left_time_min = min(frames[left_local_index], left_time_min)
        right_time_max = max(frames[right_local_index], right_time_max)


    left_bound = cur_delta - interval
    right_bound = cur_delta + interval
    local_deltas = deltas[left_local_index: right_local_index]
    target_interval_cnt = len(local_deltas[(local_deltas > left_bound) & (local_deltas < right_bound)])
    # target_interval_cnt = c_bisect_lib.in_count(local_deltas, len(local_deltas), left_bound, right_bound)

    all_frame_cnt_for_camera1 = local_interval_length
    for i in range(len(cameras_delta_s)):
        if i == camera1 or i == camera2:
            continue
        tmp_frames = cameras_frame_s[camera1][i]
        tmp_local_interval_length = other_frames_length(tmp_frames, left_time_min, right_time_max)
        all_frame_cnt_for_camera1 += tmp_local_interval_length
    # if we use uniform Denominator, we means spatial & temporal probability
    score = target_interval_cnt / float(all_frame_cnt_for_camera1)
    return score


if __name__ == '__main__':
    ctrl_msg['ep'] = 0
    ctrl_msg['en'] = 0
    ctrl_msg['data_folder_path'] = 'market_duke-test'
    fusion_param = get_fusion_param()
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    track_score(camera_delta_s, 1, 25, 2, 5250, filter_interval=40000)
