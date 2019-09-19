"""
Implements an L2 regularised covariance estimator - based on Scout by Witten and Tibshirani
We solve -logdet Theta + tr(S Theta) + p ||Theta||_2^2
"""
import numpy as np
import sklearn
import scipy as sp
import sklearn.datasets
from sklearn.model_selection import KFold
import collections
from sklearn.utils.extmath import fast_logdet
from sklearn.base import BaseEstimator

class L2PenalizedEstimator(BaseEstimator):
    """
    Estimators an L2 penalized precision matrix

    See:
    https://www.sciencedirect.com/science/article/pii/S1532046413001019
    Attributes
    ----------
    alpha_ : float
        Regression parameter for the problem
    precision_ : array_like
        p by p array containing the estimated precision matrix
    """

    def __init__(self, alpha):
        self.alpha_ = alpha
        self.precision_ = None

    def fit(self, X):
        """
        Solves the problem using eigendecomposition
        
        Parameters
        ----------
        X : array_like
            n by p matrix - data matrix
        Returns
        -------
        """
        S = np.cov(X.T)
        p = S.shape[0]
        eigvals, eigvecs = np.linalg.eig(S)
        new_eigvals = np.zeros(p)

        for i in range(p):
            s = eigvals[i]
            e = -s/(4*self.alpha_) + np.sqrt(s**2 + 8*self.alpha_)/(4*self.alpha_)
            new_eigvals[i] = e

        D = np.diag(new_eigvals)
        theta = eigvecs @ D @ eigvecs.T
        self.precision_ = theta

    def _precision_likelihood_function(self, S, theta):
        """
        Estimates the likelihood of the precision matrix theta
        given the sample covariance matrix S
        
        Parameters
        ----------
        S : array_like
            p by p matrix - sample covariance matrix
        theta : array_like
            p by p matrix - estimated precision matrix
        Returns
        -------
        likelihood - float
        Likelihood of the problem
        """
        p = S.shape[0]
        log_likelihood_ = -fast_logdet(theta) + np.trace(S @ theta)    
        log_likelihood_ -= p * np.log(2 * np.pi)
        return log_likelihood_    

class L2PenalizedEstimatorCV(L2PenalizedEstimator):
    """
    Estimators an L2 penalized precision matrix and uses 
    cross-validation to estimate the regularization parameter
    
    Attributes
    ----------
    n_splits_ : int
        Number of K-folds to use for the cross-validation
    alpha_ : float
        Regression parameter for the problem
    precision_ : array_like
        p by p array containing the estimated precision matrix
    """
    def __init__(self, n_splits=4):
        self.n_splits_ = n_splits
        super().__init__(None)

    def fit(self, X):
        """
        Estimates an L2 penalized precision matrix using
        cross-validation
        
        Parameters
        ----------
        X : array_like
            n by p matrix - data matrix
        Returns
        -------
        """
        p = X.shape[1]
        prec = np.zeros((p, p))
        kf = KFold(n_splits = self.n_splits_)
        l_likelihood = collections.defaultdict(list)
        for train, test in kf.split(X):
            X_train = X[train, :]
            X_test = X[test, :]
            S_test = np.cov(X_test.T)
            lambdas = np.logspace(-1, 1)
            likelihoods = []

            for l in lambdas:
                est = L2PenalizedEstimator(l)
                est.fit(X_train)
                likelihood = self._precision_likelihood_function(S_test, est.precision_)
                l_likelihood[l].append(likelihood)

        likelihoods = []
        for l in l_likelihood:
            mean_likelihood = np.mean(l_likelihood[l])
            likelihoods.append(mean_likelihood)
        best_l_index = np.argmin(likelihoods)

        est = L2PenalizedEstimator(l)
        est.fit(X_train)
        self.alpha_ = lambdas[best_l_index]
        super().fit(X)
