# coding=utf-8
from profile.fusion_param  import ctrl_msg, get_fusion_param
from util.file_helper import read_lines
from util.viz import draw_line
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sys
import pandas as pd

def viz_heat_map(pt):
    # f, ax = plt.subplots(figsize=(15, 15))
    f, ax = plt.subplots()
    # cmap = sns.color_palette("coolwarm", 7)
    cmap = sns.cubehelix_palette(n_colors=8, start=3, rot=0.7, dark=0.4, light=0.92, gamma=1.0, hue=2.5, as_cmap=True)
    sns.heatmap(pt, cmap=cmap, linewidths=0.0, ax=ax, annot=True, fmt='.3f')
    ax.set_title('Transfer heat map on Market1501')
    ax.set_xlabel('transfer time')
    ax.set_ylabel('camera_pair')
    sns.plt.show()
    f.savefig('sns_heatmap_normal.jpg', bbox_inches='tight')


def viz_gray_map(pt):
    f, ax = plt.subplots()
    # cmap = sns.color_palette("coolwarm", 7)
    sns.set(font_scale=2.5)
    cmap = sns.cubehelix_palette(n_colors=8, start=3, rot=0.7, dark=0.4, light=0.92, gamma=1.0, hue=0, as_cmap=True)
    sns.heatmap(pt, cmap=cmap, linewidths=0.0, ax=ax, annot=True, fmt='.3f')
    ax.set_title('')
    print sys.getdefaultencoding()
    reload(sys)
    sys.setdefaultencoding('utf8')
    # ax.set_xlabel('α', fontsize=32)
    # ax.set_ylabel('β', fontsize=32)
    ax.set_xlabel('W/10³', fontsize=32)
    ax.set_ylabel('t', fontsize=32)
    ax.invert_yaxis()
    plt.yticks(fontsize=32)
    plt.xticks(fontsize=32)
    sns.plt.show()
    f.savefig('sns_gray.jpg', bbox_inches='tight')


def gray_data(values):
    # market
    df = pd.DataFrame({'a': [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.5, 0.5, 0.75],
                       'b': [0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.0, 0.25, 0.0],
                       'values': values})
    pt = df.pivot_table(index='a', columns='b', values='values', aggfunc=np.sum)
    return pt


def duke_sensity_data(values):
    df = pd.DataFrame({'a': [(j+1)*200 for j in range(5) for i in range(5)],
                       'b': [(i % 5 + 1) * 40 for i in range(25)],
                       'values': values})
    pt = df.pivot_table(index='a', columns='b', values='values')
    return pt


def iter_acc_data(data_path):
    lines = read_lines(data_path)
    cv_accs = list()
    acc_cnt = 0
    cv_cnt = -1
    for i, line in enumerate(lines):
        if i % 2 == 0:
            continue
        if (i - 1) % 44 == 0:
            cv_cnt += 1
            cv_accs.append([list(), list()])
        if acc_cnt % 2 == 0:
            cv_accs[cv_cnt][0].append(float(line.split()[0]))
        else:
            cv_accs[cv_cnt][1].append(float(line.split()[0]))
        acc_cnt += 1

    avg_accs = np.array(cv_accs).mean(axis=0)
    return avg_accs


def iter_vision_acc_data(data_path):
    lines = read_lines(data_path)
    cv_accs = list()
    acc_cnt = 0
    cv_cnt = -1
    for i, line in enumerate(lines):
        if i % 2 == 0:
            continue
        if (i - 1) % 22 == 0:
            cv_cnt += 1
            cv_accs.append(list())
        cv_accs[cv_cnt].append(float(line.split()[0]))
        acc_cnt += 1
    avg_accs = np.array([np.array(cv_accs[i]) for i in range(cv_cnt + 1)])
    avg_accs = avg_accs.mean(axis=0)
    return avg_accs

def sensitivity_eval(sense_file_path):
    lines = read_lines(sense_file_path)
    rank1_accs = list()
    cur_cv = -1
    for i, line in enumerate(lines):
        if i % 20 == 0:
            rank1_accs.append(list())
            cur_cv += 1
        if i % 2 == 0:
            continue
        rank1_accs[cur_cv].append(float(line.split()[0]))
    grid_avg_accs = np.array(rank1_accs).mean(axis=0)
    viz_gray_map(gray_data(grid_avg_accs))


def st_hp_sensitivity_eval(sense_file_path):
    lines = read_lines(sense_file_path)
    grid_avg_accs = list()
    for i, line in enumerate(lines):
        if i % 2 == 0:
            continue
        grid_avg_accs.append(float(line.split()[0]))
    viz_gray_map(duke_sensity_data(grid_avg_accs))


if __name__ == '__main__':
    # print(camera_distribute(1))
    # ctrl_msg['data_folder_path'] = 'grid_market-test'
    # fusion_param = get_fusion_param()
    # accs = iter_acc_data('grid_market_iter.txt')
    # accs = iter_acc_data('market_grid_iter.txt')
    # draw_line(accs, np.arange(0, len(accs[0])), 'Rank-1 precision', 'Number of iterations',
    #           ['Visual Classifier C', 'Fusion Model F'], title='')
    # accs = iter_vision_acc_data('vision_market_grid_iter.txt')
    # draw_line([accs], np.arange(0, len(accs)), 'Rank-1 precision', 'Number of iterations', ['Visual Classifier C'],
    #           title='')

    # draw_line(accs, np.arange(0, len(accs[0])), 'Rank-1 precision', 'Number of iterations', ['Visual Classifier C', 'Fusion Model F'], title='')
    # sensitivity_eval(sense_file_path='market_sense.txt')
    # sensitivity_eval(sense_file_path='grid_sense.txt')
    # sensitivity_eval(sense_file_path='duke2market_sense.txt')
    st_hp_sensitivity_eval(sense_file_path='dukesense.txt')
