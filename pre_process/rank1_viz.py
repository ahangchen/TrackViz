from util.file_helper import read_lines, write

if __name__ == '__main__':
    for i in range(10):
        person_lines = read_lines('data/top-m2g-std%d-train/cross_filter_pid.log' % i)
        predict_persons = [predict_line.split() for predict_line in person_lines]
        score_lines = read_lines('data/top-m2g-std%d-train/cross_filter_score.log' % i)
        predict_scores = [score_line.split() for score_line in score_lines]
        write('data/top1.txt', 'std %d top1\n' % i)
        for j in range(len(predict_persons)):
            r = int(predict_persons[j][0]) - j -1
            if abs(r) == 1:
                write('data/top1.txt', 'left %d, right %d, score %f\n' % (j+1, int(predict_persons[j][0]), float(predict_scores[j][0])))
