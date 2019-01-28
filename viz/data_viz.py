from util.serialize import pickle_load
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


def load_viz_deltas_distribution(delta_path):
    cameras_delta_s = pickle_load(delta_path)
    return cameras_delta_s


def distribute_in_cameras(data_s, subplot, camera_id):
    sns.set(color_codes=True)
    # markers = ['', '*', '', '.']
    markers = ['', '', '', '.']
    line_styles = ['-','-', '--', '-']
    for i, data in enumerate(data_s):
        # if len(data) < 1 or i > 3:
        #     continue
        # else:
        #     print len(data)
        # sns.distplot(np.array(data), label='camera %d' % (i + 1), hist=False, ax=subplot,
        #              kde_kws={'marker': markers[i % len(markers)], 'linestyle': line_styles[i % len(line_styles)]},
        #              axlabel='Distribution for camera 1 in window %d' % camera_id)
        sns.distplot(np.array(data), label='camera %d' % (i + 1), hist=False, ax=subplot,
                     kde_kws={'marker': markers[i % len(markers)], 'linestyle': line_styles[i % len(line_styles)]})
        # sns.distplot(np.array(data), label='camera %d' % (i + 1), hist=False, ax=subplot,
        #              axlabel='Distribution for camera %d' % camera_id)


def viz_market_distribution(transfer_name, viz_data, camera_cnt):
    f, axes = plt.subplots(camera_cnt/2, 2, figsize=(15, 10))
    for i, ax_s in enumerate(axes):
        for j, ax in enumerate(ax_s):
            ax.set_xlabel('Transfer time from camera %d' % (i * 2 + j + 1), fontsize=14)
            ax.set_ylabel('Frequency')
            ax.set_xlim([-25000, 50000])
            # ax.set_ylim([0, 0.025])
    # sns.despine(left=True)
    for i in range(camera_cnt):
        # sns.plt.title('Appear distribution in cameras %d' % (i + 1))
        distribute_in_cameras(viz_data[i], axes[i / 2, i % 2], i + 1)
        print('viz camera %d' % (i + 1))
    f.tight_layout()
    f.savefig(transfer_name + '_dist.pdf')


def viz_two_part_distribution(viz_data, camera_cnt):
    f, axes = plt.subplots(2, 1, figsize=(15, 10))
    for ax in axes:
        ax.set_xlabel('time')
        ax.set_ylabel('appear density')
        ax.set_xlim([-25000, 50000])
        # ax.set_ylim([0, 0.025])
    sns.despine(left=True)
    for i in range(camera_cnt):
        # sns.plt.title('Appear distribution in cameras %d' % (i + 1))
        distribute_in_cameras(viz_data[i], axes[i], i + 1)
        print('viz camera %d' % (i + 1))
    sns.plt.show()


def evolving_deltas_viz(slice_cnt):
    evolving_cameras_delta_s = load_viz_deltas_distribution('part0_duke_real.pck')
    for i in range(slice_cnt):
        cameras_delta_s = load_viz_deltas_distribution('part%d_duke_real.pck' % i)
        evolving_cameras_delta_s[i] = cameras_delta_s[0]
    viz_market_distribution(evolving_cameras_delta_s, slice_cnt)


def two_part_evolving_deltas_viz(slice_cnt):
    evolving_cameras_delta_s = load_viz_deltas_distribution('part0_duke_real.pck')
    cameras_delta_s = load_viz_deltas_distribution('part%d_duke_real.pck' % (slice_cnt - 1))
    evolving_cameras_delta_s[1] = cameras_delta_s[0]
    viz_two_part_distribution(evolving_cameras_delta_s, 2)


if __name__ == "__main__":
    # cameras_delta_s = load_viz_deltas_distribution('/home/cwh/coding/TrackViz/data/market_duke-train/sorted_deltas.pickle')
    # cameras_delta_s = load_viz_deltas_distribution('head_duke_real.pck')
    # viz_market_distribution(cameras_delta_s, 8)
    transfer_name = 'duke_market'
    cameras_deltas = load_viz_deltas_distribution('/home/cwh/coding/TrackViz/data/'+transfer_name+'-train/sorted_deltas.pickle')
    viz_market_distribution(transfer_name, cameras_deltas, 6)
    # slice_cnt = 6
    # for i in range(slice_cnt):
    #     cameras_delta_s = load_viz_deltas_distribution('part%d_duke_real.pck' % i)
    #     viz_market_distribution(cameras_delta_s, 8)
    # evolving_deltas_viz(slice_cnt)
    # two_part_evolving_deltas_viz(slice_cnt)
