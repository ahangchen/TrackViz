# coding=utf-8
from util.serialize import pickle_load


def binary_search(a, target):
    # 不同于普通的二分查找，目标是寻找target最适合的index
    low = 0
    high = len(a) - 1

    while low <= high:
        # 在其它语言中,如果low + high的和大于Integer的最大值,比如2 ** 31 - 1,
        # 计算便会发生溢出,使它成为一个负数,然后被2除时结果仍为负数。在Java语言中,
        # 这个Bug导致一个ArrayIndexOutOfBoundsException异常被抛出,而在C语言中,
        # 你会得到一个无法预测的越界的数组下标。推荐的解决方法是修改中间值的计算过程,
        # 方法之一是用减法而不是加法——来实现：mid = low + ((high - low) / 2)；或者,
        # 如果你想炫耀一下自己掌握的移位运算的知识,可以使用更快的移位运算操作,
        # 在Python中是mid = (low + high) >> 1,Java中是int mid = (low + high) >>> 1。
        mid = (low + high) // 2
        midVal = a[mid]

        if midVal < target:
            low = mid + 1
        elif midVal > target:
            high = mid - 1
        else:
            return mid
    return low


def track_score(camera_delta_s, camera1, time1, camera2, time2, interval=100):
    camera1 -= 1
    camera2 -= 1
    cur_delta = time2 - time1
    delta_distribution = camera_delta_s[camera1][camera2]
    total_cnt = sum(map(len, camera_delta_s[camera1]))
    # 10 second
    left_bound = cur_delta - interval
    right_bound = cur_delta + interval
    left_index = binary_search(delta_distribution, left_bound)
    right_index = binary_search(delta_distribution, right_bound)
    if total_cnt == 0:
        return 0.0
    score = (right_index - left_index + 1) / float(total_cnt)
    if len(delta_distribution) == 0:
        return 0.0
    # score = (right_index - left_index + 1) / float(len(camera_delta_s[camera1][2]))
    # if score > 0:
    #     print(len(delta_distribution))
    #     print('delta range %d ~ %d' % (delta_distribution[0], delta_distribution[-1]))
    #     print(left_index)
    #     print(right_index)
    #     print('probablity: %f%%' % (score * 100))
    return score


def track_interval_score(interval_score_s, camera1, time1, camera2, time2):
    delta = time2 - time1
    for i, camera_pair_travel_prob in enumerate(interval_score_s[camera1 - 1][camera2 - 1]):
        if camera_pair_travel_prob['left'] < delta < camera_pair_travel_prob['right']:
            print('camera1: %d, camera2: %d, delta: %d, interval: %d, prob: %f' % (
                camera1, camera2, delta, i, camera_pair_travel_prob['prob']))
            return camera_pair_travel_prob['prob']
    return 0

if __name__ == '__main__':
    camera_delta_s = pickle_load('data/top10/sorted_deltas.pickle')
    track_score(camera_delta_s, 1, 25, 2, 250)
