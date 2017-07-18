from util.file_helper import read_lines
import seaborn as sns
import numpy as np

# lines = read_lines('data/viper-s1_train/cross_filter_score.log')
lines = read_lines('/home/cwh/coding/TrackViz/data/top-m2g-std2-train/cross_filter_score.log')
scores = [float(line.split(' ')[0]) for line in lines]
sns.distplot(np.array(scores), label='market top1 predict scores')
sns.plt.show()