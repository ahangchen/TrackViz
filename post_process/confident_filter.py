from predict_eval import percent_shot_eval
from util.file_helper import read_lines, write, safe_remove


def write_with_confidence(confidence_bound):
    cross_score_path = 'data/top10-test/cross_filter_score.log'
    cross_pid_path = 'data/top10-test/cross_filter_pid.log'

    filter_pid_path = 'data/top10/conf_filter_pid.log'
    filter_score_path = 'data/top10/conf_filter_score.log'

    safe_remove(filter_pid_path)
    safe_remove(filter_score_path)
    score_lines = read_lines(cross_score_path)
    persons_scores = [[float(score) for score in score_line.split()] for score_line in score_lines]
    max_score = max([max(predict_scores) for predict_scores in persons_scores])

    pid_lines = read_lines(cross_pid_path)


    write_line_cnt = 0
    for i in range(len(persons_scores)):
        scores = persons_scores[i]
        pids = pid_lines[i].split()
        has_write = False
        for j in range(len(persons_scores[0])):
            confidence = max(scores[j]/max_score, 1 - scores[j]/max_score)
            if confidence < confidence_bound:
                # print(confidence)
                if not has_write:
                    write_line_cnt += 1
                has_write = True
                write(filter_pid_path, pids[j] + ' ')
                write(filter_score_path, '%f ' % scores[j])
        write(filter_pid_path, '\n')
        write(filter_score_path, '\n')
    return write_line_cnt


if __name__ == '__main__':
    for i in range(8):
        write_line_cnt = write_with_confidence(1-1.0/(i + 3))

        print('\nremain confidence < %f' % (1-1.0/(i+3)))
        print('write line cnt: %d' % write_line_cnt)
        percent_shot_eval('data/top10/conf_filter_pid.log', 10)
        percent_shot_eval('data/top10/conf_filter_pid.log', 5)
        percent_shot_eval('data/top10/conf_filter_pid.log', 1)
