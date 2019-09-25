"""
File to hold methods that could be useful across the entire module
"""
import numpy as np

def threshold_matrix(A, threshold=0.0001, absolute=True, binary=False):
    """
    Thresholds a matrix at the set value
    Parameters
    ----------
    A : array_like
        p by p matrix
    threshold : (optional, default = 0.0001)
        value to set the threshold at
    absolute : (optional, default=True)
        if we set absolute values below the threshold to 0
    binary : (optional, default=False)
        if we set values above the threshold to 1

    Returns
    -------
    array_like
        p by p matrix
        Matrix that has been thresholded
    """
    A = A.copy()

    if absolute:
        A[np.abs(A) < threshold] = 0
    else:
        A[A < threshold] = 0

    if binary:
        A[np.abs(A) > 0] = 1

    return A 

def matrix_similarity(A, B):
    """
    Counts how similar the two matrices are
    Parameters
    ----------
    A : array_like
        p by p matrix
    B : array_like
        p by p matrix

    Returns
    -------
    (int, int)
        number of values that are the same, number that are different
    """
    X = A == B
    num_correct = np.count_nonzero(X)
    num_wrong = np.count_nonzero(~X)

    return (num_correct, num_wrong)

def precision_matrix_to_partial_corr(theta):
    """
    Turns a precision matrix into a partial correlation one

    Parameters
    ----------
    theta : array_like
        p by p precision matrix

    Returns
    -------
    p by p partial correlation matrix
    """
    p = theta.shape[0]
    partial_corr = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            partial_corr[i, j] = -theta[i, j] / np.sqrt(theta[i, i] * theta[j, j])
    np.fill_diagonal(partial_corr, 1)
    return partial_corr

def make_matrix_symmetric(M):
    """
    Makes matrix M symmetric by comparing i and j and taking the smaller absolute value

    Parameters
    ----------
    M : array_like
        p by p matrix

    Returns
    -------
    p by p matrix symmetric matrix
    """
    p = M.shape[0]
    for i in range(p):
        for j in range(p):
            if np.abs(M[i, j]) < np.abs(M[j, i]):
                M[j, i] = M[i, j]
            else:
                M[i, j] = M[j, i]

    return M

def calculate_matrix_accuracy(M_true, M_est, include_diag=False):
    """
    Compares the true known matrix (M_true) to one estimated from data (M_est)
    and returns the true positive rate (tpr), false positive rate (fpr)
    and precision 

    Parameters
    ----------
    M_true : array_like
        p by p correct matrix

    M_est : array_like
        p by p estimated matrix

    include_diag : (optional, default=False)
        whether to include the matrix diagonal in the accuracy measures

    Returns
    -------
    (float, float, float) - true positive rate, false positive rate, precision
    """
    M_true = threshold_matrix(M_true.copy(), binary=True)
    M_est = threshold_matrix(M_est.copy(), binary=True)

    p = M_true.shape[0]

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(p):
        for j in range(p):
            if not include_diag and i == j:
                continue
            
            if M_true[i, j] != 0:
                if M_est[i, j] != 0:
                    tp += 1
                else:
                    fn += 1
            else:
                if M_est[i, j] != 0:
                    fp += 1
                else:
                    tn += 1

    # Stop divide by 0 errors
    if tp == 0:
        tpr = 0
        precision = 0
    else:
        tpr = tp / (tp + fn)
        precision = tp / (tp + fp)


    if fp == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    return (tpr, fpr, precision)

def calculate_f1_score(recall, precision):
    """
    Calculates the F1 score from the recall (tpr) and precison
    
    Parameters
    ----------
    recall : float
    recall or true positive rate
    precision : float
    precision

    Returns
    -------
    float : f1_score
    """
    if recall == 0 and precision == 0:
        return 0
    return 2 * recall * precision / ( recall + precision )