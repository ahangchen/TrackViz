import argparse
import numpy as np

from sklearn import metrics
from sklearn.metrics import accuracy_score

from util.file_helper import write
from util.serialize import pickle_load


def arg_parse():
    parser = argparse.ArgumentParser(description='eval on cluster result')
    parser.add_argument('--track_path', default='/home/cwh/coding/TrackViz/pre_process/duke_market_cluster.pck', type=str,
                        help='')
    parser.add_argument('--result_path', default='/home/cwh/coding/TrackViz/pre_process/cluster_eval.txt', type=str,
                        help='')
    opt = parser.parse_args()
    return opt


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) * 1. / np.sum(contingency_matrix)


def purity_score_raw(y_true, y_pred):
    """Purity score

    To compute purity, each cluster is assigned to the class which is most frequent
    in the cluster [1], and then the accuracy of this assignment is measured by counting
    the number of correctly assigned documents and dividing by the number of documents.
    We suppose here that the ground truth labels are integers, the same with the predicted clusters i.e
    the clusters index.

    Args:
        y_true(np.ndarray): n*1 matrix Ground truth labels
        y_pred(np.ndarray): n*1 matrix Predicted clusters

    Returns:
        float: Purity score

    References:
        [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def purity_score_for_track(cluster_tracks):
    purity_sum = 0
    seq_cnt = 0
    for camera_track_infos in cluster_tracks:
        for seq in camera_track_infos:
            if len(seq) == 0:
                continue
            y_true = []
            y_pred = []
            for info in seq:
                y_true.append(info[0])
                y_pred.append(info[-1])
            purity_sum += purity_score(y_true, y_pred)
            # purity_sum += purity_score_raw(np.array(y_true), np.array(y_pred))
            seq_cnt += 1
    return purity_sum * 1.0 / seq_cnt


def main():
    opt = arg_parse()
    cluster_tracks = pickle_load(opt.track_path)
    score = purity_score_for_track(cluster_tracks)
    print('score %f' % score)
    write(opt.result_path, 'cluster task: %s \tscore: %f\n' % (opt.track_path, score))


if __name__ == '__main__':
    main()
