import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
"""
each row is an instance
each column is the prediction of a class
"""


def _score_to_rank(score_list):
    rank_array = np.zeros([len(score_list)])
    score_array = np.array(score_list)
    idx_sorted = (-score_array).argsort()
    rank_array[idx_sorted] = np.arange(len(score_list))+1

    rank_list = rank_array.tolist()
    return rank_list


# For clip evaluation
def auc(Y_target, Y_score):
    """
    Y_target: list of lists. {0, 1}
        real labels

    Y_score: list of lists. real values
        prediction values
    """
    Y_target = np.array(Y_target)
    Y_score = np.array(Y_score)
    auc_list = []
    for i in range(Y_score.shape[1]):
        try:
            auc = roc_auc_score(Y_target[:, i], Y_score[:, i])
        except:
            continue
        auc_list.append(auc)
    return auc_list


def mean_auc(Y_target, Y_score):
    auc_list = auc(Y_target, Y_score)
    mean_auc = np.mean(auc_list)
    return mean_auc


def mean_auc_y(Y_target, Y_score):
    '''
    along y-axis
    '''
    return mean_auc(Y_target, Y_score)


def mean_auc_x(Y_target, Y_score):
    '''
    along x-axis
    '''
    return mean_auc(np.array(Y_target).T, np.array(Y_score).T)


def mean_average_precision(Y_target, Y_score):
    """
    mean average precision
    raw-based operation

    Y_target: list of lists. {0, 1}
        real labels

    Y_score: list of lists. real values
        prediction values
    """
    p = float(len(Y_target))
    temp_sum = 0
    for y_target, y_score in zip(Y_target, Y_score):
        y_target = np.array(y_target)
        y_score = np.array(y_score)
        if (y_target == 0).all() or (y_target == 1).all():
            p -= 1
            continue
        idx_target = np.nonzero(y_target > 0)[0]
        n_target = float(len(idx_target))
        rank_list = np.array(_score_to_rank(y_score))
        target_rank_list = rank_list[idx_target]

        temp_sum_2 = 0
        for target_rank in target_rank_list:
            mm = sum([1 for ii in idx_target
                      if rank_list[ii] <= target_rank])/float(target_rank)
            temp_sum_2 += mm
        temp_sum += temp_sum_2/n_target

    measure = temp_sum/p
    return measure


def map(Y_target, Y_score):
    return mean_average_precision(Y_target, Y_score)


def map_x(Y_target, Y_score):
    return mean_average_precision(Y_target, Y_score)


def map_y(Y_target, Y_score):
    return mean_average_precision(np.array(Y_target).T,
                                  np.array(Y_score).T)


# For frame evaluation
def f1_micro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'micro'
    score = f1_score(y_target, y_predicted, average=average)
    return score


def f1_macro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'macro'
    score = f1_score(y_target, y_predicted, average=average)
    return score


def precision_micro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'micro'
    score = precision_score(y_target, y_predicted, average=average)
    return score


def precision_macro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'macro'
    score = precision_score(y_target, y_predicted, average=average)
    return score


def recall_micro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'micro'
    score = recall_score(y_target, y_predicted, average=average)
    return score


def recall_macro(y_target, y_predicted):
    """
    y_target: m x n 2D array. {0, 1}
        real labels

    y_predicted: m x n 2D array {0, 1}
        prediction labels

    m (y-axis): # of instances
    n (x-axis): # of classes
    """
    average = 'macro'
    score = recall_score(y_target, y_predicted, average=average)
    return score
