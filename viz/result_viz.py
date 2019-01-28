# coding=utf-8
from profile.fusion_param  import ctrl_msg, get_fusion_param
from util.file_helper import read_lines
from util.viz import draw_line
import numpy as np
# import matplotlib as mpl
# import matplotlib.style
# mpl.style.use('classic')
# mpl.use('agg')
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
    f = plt.figure(figsize=(9.5,8))
    ax = plt.gca()
    # f, ax = plt.subplots()
    # cmap = sns.color_palette("coolwarm", 7)
    sns.set(font_scale=2., font='ubuntu')
    cmap = sns.cubehelix_palette(n_colors=8, start=3, rot=0.7, dark=0.4, light=0.92, gamma=1.0, hue=0, as_cmap=True)
    sns.heatmap(pt, cmap=cmap, linewidths=0.0, ax=ax, annot=True, fmt='.3f')
    ax.set_title('')
    print sys.getdefaultencoding()
    reload(sys)
    # sys.setdefaultencoding('utf8')
    ax.set_xlabel(r'$\alpha$', fontsize=25)
    ax.set_ylabel(r'$\beta$', fontsize=25)
    # ax.set_xlabel('W/10³', fontsize=20)
    # ax.set_ylabel('t', fontsize=20)
    ax.invert_yaxis()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    # ax.axis["top", 'right', 'left', 'bottom'].set_visible(False)
    plt.tight_layout()
    # sns.plt.show()
    f.savefig('grid2marketsense.pdf', bbox_inches='tight')


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
    # rank transfer market2grid
    # accs = [[0.1208, 0.2192, 0.222, 0.217, 0.224, 0.222], [0.4818, 0.672, 0.667, 0.668, 0.673, 0.674 ]]
    # rank transfer grid2duke
    g2d_rank_accs = [[0.077199, 0.098294, 0.096050, 0.098357, 0.096664 , 0.096215],
            [0.157092, 0.271221, 0.283214 , 0.293986, 0.288969 , 0.288151]]
    # rank transfer grid2market
    g2m_rank_accs = [[0.1366, 0.178147, 0.176960, 0.177397, 0.173397, 0.178147 ],
            [0.4819, 0.559085, 0.562055,0.560523 , 0.552553, 0.554038]]
    draw_line(g2d_rank_accs, np.arange(0, len(g2d_rank_accs[0])), 'Rank-1 precision', 'Number of iterations',
              ['Visual Classifier C', 'Fusion Model F'], 'grid2duke_rank.pdf')
    draw_line(g2d_rank_accs, np.arange(0, len(g2d_rank_accs[0])), 'Rank-1 precision', 'Number of iterations',
              ['Visual Classifier C', 'Fusion Model F'], 'grid2duke_rank.pdf')
    draw_line(g2m_rank_accs, np.arange(0, len(g2m_rank_accs[0])), 'Rank-1 precision', 'Number of iterations',
              ['Visual Classifier C', 'Fusion Model F'], 'grid2market_rank.pdf')
    # grid2duke vision and fusion
    g2d_multi_accs = [[0.077199, 0.154847, 0.208259, 0.241921, 0.249102, 0.254039], [0.157092, 0.422352, 0.536355, 0.549820, 0.552962, 0.559246]]
    # grid2market vision and fusion
    g2m_multi_accs = [[0.136580, 0.294537, 0.375297, 0.403504, 0.404691, 0.404691], [0.481888, 0.628266, 0.648159, 0.664489, 0.655582, 0.657957]]
    draw_line(g2d_multi_accs, np.arange(0, len(g2d_multi_accs[0])), 'Rank-1 precision', 'Number of iterations',
              ['Visual Classifier C', 'Fusion Model F'], 'grid2duke_multi_iter.pdf')
    draw_line(g2m_multi_accs, np.arange(0, len(g2m_multi_accs[0])), 'Rank-1 precision', 'Number of iterations',
              ['Visual Classifier C', 'Fusion Model F'], 'grid2market_multi_iter.pdf')
    # grid2duke vision and fusion
    g2d_vis_accs = [[0.077199, 0.099192, 0.095601, 0.091113, 0.088869, 0.085278]]
    # grid2market vision and fusion

    g2m_vis_accs = [[0.136580, 0.178741, 0.176366, 0.167755, 0.165083, 0.151425]]
    draw_line(g2d_vis_accs, np.arange(0, len(g2d_vis_accs[0])), 'Rank-1 precision', 'Number of iterations',
              ['Visual Classifier C'], 'grid2duke_vis_iter.pdf')
    draw_line(g2m_vis_accs, np.arange(0, len(g2m_vis_accs[0])), 'Rank-1 precision', 'Number of iterations',
              ['Visual Classifier C'], 'grid2market_vis_iter.pdf')
    g2d_all_accs =  [g2d_multi_accs[0], g2d_multi_accs[1],g2d_rank_accs[0], g2d_rank_accs[1],  g2d_vis_accs[0]]
    draw_line(g2d_all_accs, np.arange(0, len(g2d_all_accs[0])), 'Rank-1 precision', 'Number of iterations',
              ['Multi Visual', 'Multi Fusion','Rank Visual', 'Rank Fusion',  'Pure Visual'], 'grid2duke_incre_all.pdf')
    g2m_all_accs = [g2m_multi_accs[0], g2m_multi_accs[1], g2m_rank_accs[0], g2m_rank_accs[1],  g2m_vis_accs[0]]
    draw_line(g2m_all_accs, np.arange(0, len(g2m_all_accs[0])), 'Rank-1 precision', 'Number of iterations',
              ['Multi Visual', 'Multi Fusion', 'Rank Visual', 'Rank Fusion', 'Pure Visual'], 'grid2market_incre_all.pdf')
    # draw_line(accs, np.arange(0, len(accs[0])), 'Rank-1 precision', 'Number of iterations', ['Visual Classifier C', 'Fusion Model F'], title='')
    # sensitivity_eval(sense_file_path='market_sense.txt')
    # sensitivity_eval(sense_file_path='grid_sense.txt')
    # sensitivity_eval(sense_file_path='duke2market_sense.txt')
    # sensitivity_eval('grid2marketsense.log')
    # st_hp_sensitivity_eval(sense_file_path='dukesense.txt')
