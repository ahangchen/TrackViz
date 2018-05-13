from util.serialize import pickle_load
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_viz_deltas_distribution(delta_path):
    cameras_delta_s = pickle_load(delta_path)
    return cameras_delta_s


def distribute_in_cameras(data_s, subplot, camera_id):
    sns.set(color_codes=True)
    for i, data in enumerate(data_s):
        if len(data) < 1:
            continue
        else:
            print len(data)
        sns.distplot(np.array(data), label='camera %d' % (i + 1), hist=False, ax=subplot,
                     axlabel='Distribution for camera %d' % camera_id)


def viz_market_distribution(viz_data, camera_cnt):
    f, axes = plt.subplots(camera_cnt/2, 2, figsize=(15, 10))
    for ax_s in axes:
        for ax in ax_s:
            ax.set_xlabel('time')
            ax.set_ylabel('appear density')
            # ax.set_xlim([-2000, 2000])
            # ax.set_ylim([0, 0.025])
    sns.despine(left=True)
    for i in range(camera_cnt):
        # sns.plt.title('Appear distribution in cameras %d' % (i + 1))
        distribute_in_cameras(viz_data[i], axes[i / 2, i % 2], i + 1)
        print('viz camera %d' % (i + 1))
    sns.plt.show()


if __name__ == "__main__":
    # cameras_delta_s = load_viz_deltas_distribution('/home/cwh/coding/TrackViz/data/market_duke-train/sorted_deltas.pickle')
    cameras_delta_s = load_viz_deltas_distribution('head_duke_real.pck')
    viz_market_distribution(cameras_delta_s, 8)
