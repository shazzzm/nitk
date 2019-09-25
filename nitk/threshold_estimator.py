import numpy as np
from . import methods
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
import collections

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
    
    def likelihood(self, estimated_cov, test_cov):
        """
        Calculates the likelihood of the estimated covariance
        matrix compared to the one from the test set

        Parameters
        ----------
        estimated_cov : array_like
        p by p sparse covariance matrix estimated from the training set
        test_cov : array_like
        p by p sample covariance matrix from the test set
        """
        p = estimated_cov.shape[0]
        theta = np.linalg.pinv(estimated_cov)
        sgn, logdet = np.linalg.slogdet(theta)
        return -logdet + np.trace(test_cov @ theta)

class ThresholdEstimatorCV(ThresholdEstimator):
    """
    Estimates a covariance matrix and then decides the 
    optimal threshold using cross validation

    Attributes
    ----------
    covariance_ : array_like
    Estimated covariance matrix
    alpha_ : float
    Threshold parameter
    n_splits : int
    Number of K-folds to use in cross validation
    """
    def __init__(self, n_splits=4):
        super().__init__(None)
        self.n_splits_ = n_splits

    def fit(self, X):
        """
        Estimates a covariance matrix from X and then uses cross
        validation to select the threshold
    
        Attributes
        ----------
        X : array_like
            n by p matrix - data matrix
        Returns
        """
        n, p = X.shape

        kf = KFold(n_splits = self.n_splits_)
        l_likelihood = collections.defaultdict(list)
        for train, test in kf.split(X):
            X_train = X[train, :]
            X_test = X[test, :]
            lambdas = np.logspace(-3, 1)

            likelihoods = []
            S_test = np.cov(X_test, rowvar=False)

            for l in lambdas:
                est = ThresholdEstimator(l)
                est.fit(X)
                cov = est.covariance_
                likelihood = self.likelihood(cov, S_test)
                l_likelihood[l].append(likelihood)

        likelihoods = []
        for l in l_likelihood:
            mean_likelihood = np.mean(l_likelihood[l])
            likelihoods.append(mean_likelihood)
        best_l_index = np.argmin(likelihoods)
        self.alpha_ = lambdas[best_l_index]
        super().fit(X)