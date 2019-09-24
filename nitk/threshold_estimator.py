import numpy as np
from . import methods
from sklearn.base import BaseEstimator

class ThresholdEstimator(BaseEstimator):
    """
    Estimates a covariance matrix and then thresholds the off-diagonal
    at the desired level

    Attributes
    ----------
    covariance_ : array_like
    Estimated covariance matrix
    alpha_ : float
    Threshold parameter
    """

    def __init__(self, alpha):
        self.covariance_ = None
        self.alpha_ = alpha

    def fit(self, X):
        """
        Estimates a covariance matrix from X and thresholds the off-diagonal
        at the set level
        Parameters
        ----------
        X : array_like
            n by p matrix - data matrix
        Returns
        ----------
        """
        n, p = X.shape
        cov = np.cov(X.T)
        offdiag_ind = ~np.eye(p, dtype=bool)
        cov_off_diag = cov[offdiag_ind]
        cov_off_diag = methods.threshold_matrix(cov_off_diag, self.alpha_)
        cov[offdiag_ind] = cov_off_diag
        self.covariance_ = cov
