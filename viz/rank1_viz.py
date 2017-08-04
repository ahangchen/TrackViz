from util.file_helper import read_lines, write
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    # for i in range(10):
    for i in range(1):

        person_lines = read_lines('data/top-m2g-std%d-train/cross_filter_pid.log' % i)
        predict_persons = [predict_line.split() for predict_line in person_lines]
        score_lines = read_lines('data/top-m2g-std%d-train/cross_filter_score.log' % i)
        predict_scores = [score_line.split() for score_line in score_lines]
        write('data/top1.txt', 'std %d top1\n' % i)
        scores = list()
        pos_cnt = 0
        in_cnt = 0
        up_b = 1.
        down_b = 0.16
        for k in range(10):
            for j in range(len(predict_persons)):
                score = float(predict_scores[j][0])
                r = int(predict_persons[j][0]) - j - 1
                if abs(r) == 1:
                    write('data/top1.txt', 'left %d, right %d, score %f\n' % (j + 1, int(predict_persons[j][0]), score))
                scores.append(score)
                if down_b < score < up_b:
                    in_cnt += 1
                    if abs(r) == 1:
                        pos_cnt += 1
            if in_cnt == 0:
                in_cnt = 1
            print('[%f, %f]: %f' % (down_b, up_b, float(pos_cnt) / in_cnt))
            sns.distplot(np.array(scores), label='market top1 predict scores')
            sns.plt.show()
            break
