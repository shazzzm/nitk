import numpy as np
import sklearn.linear_model as lm
from sklearn.base import BaseEstimator

class NeighbourhoodSelection(BaseEstimator):
    """
    Implementation of neighbourhood selection by Meinshausen and Bulhmann.
    Uses lasso regression to estimate the precision matrix.

    See
    https://projecteuclid.org/euclid.aos/1152540754

    for more information

    Attributes
    ----------
    precision_ : array_like
    Estimated precision matrix
    alpha_ : float
    Regularization parameter
    """
    def __init__(self, alpha):
        self.precision_ = None
        self.alpha_ = alpha

    def fit(self, X):
        """
        Runs p lasso problems to estimate the off-diagonal of the precision
        matrix 
        Parameters
        ----------
        X : array_like
            n by p matrix - data matrix
        Returns
        ----------
        """

        n, p = X.shape
        indices = np.arange(p)
        self.precision_ = np.zeros((p, p))
        for i in range(p):
            X_new = X[:, indices!=i]
            y = X[:, i]

            l = lm.Lasso(self.alpha_)
            l.fit(X_new, y)
            y_hat = X_new @ l.coef_
            res = ((y - y_hat)**2).mean()

            # Estimate the diagonal
            self.precision_[i, i] = 1/res
            # Then the off-diagonal
            self.precision_[indices!=i, i] = -l.coef_/self.precision_[i, i]


    def _likelihood(self, X_test, prec):
        """
        Estimates the likelihood of the neighbourhood selection
        model by a large regression problem
        Parameters
        ----------
        X_test : array_like
            n by p matrix - data matrix of test data
        prec : array_like
            p by p matrix - estimated precision matrix 
        """
        n, p = X.shape
        indices = np.arange(p)
        err = 0
        for i in range(p):
            pass                                                                                                                                                                               




 
        