import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def draw_line(y_s, x_s, y_label, x_label, y_titles, save_path, line_color=None):
    # plt.subplots()
    f, ax = plt.subplots(figsize=(12, 12))
    sns.set(font_scale=2.4)
    line_styles = ['-', '-', '-', '-', '-']
    marker_styles = ['o', '.', 'v', 's', '<']
    line_color = [ '#FF00FF','#FF0000',  '#0088FF','#0000FF', '#00FF00']
    for i in range(len(y_s)):
        plt.plot(x_s, y_s[i],  color=line_color[i], label=y_titles[i], linestyle=line_styles[i], marker=marker_styles[i], markersize=20., linewidth=5.)
    plt.xlabel(x_label, fontsize=32)
    plt.ylabel(y_label, fontsize=32)
    plt.ylim(min([min(y_si) for y_si in y_s])*0.8, max([max(y_si) for y_si in y_s])*1.5)
    plt.xlim(min(x_s), max(x_s))
    plt.yticks(fontsize=32)
    plt.xticks(fontsize=32)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.savefig('grid2duke_rank.pdf')
    # plt.savefig('market2grid_rank.pdf')


if __name__ == '__main__':
    accs = np.genfromtxt('../increment_acc.txt', delimiter='\t')
    draw_line(accs, np.arange(1, 11), 'Rank1_acc', 'iteration times',['vision', 'fusion'],  title='GRID->Market-1501')
    # plt.subplots()
    # plt.plot(np.arange(1, 11), accs[0], label='vision')
    # plt.legend()
    # plt.plot(np.arange(1, 11), accs[0], label='fusion')
    # plt.legend()
    # plt.xlabel('Rank1_acc')
    # plt.ylabel('iteration times')
    # plt.ylim(0.2, 0.4)
    # plt.title('')
    # plt.show()