"""
File to hold methods that could be useful across the entire module
"""
import numpy as np

def threshold_matrix(A, threshold, absolute=True, binary=False):
    """
    Thresholds a matrix at the set value
    Parameters
    ----------
    A : array_like
        p by p matrix
    threshold : float
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
        A[A > 0] = 1

    return A 
