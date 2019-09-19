import numpy as np
from . import methods
import math
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator

class ScaledLasso(BaseEstimator):
    """
    Solves a regression problem using the 'Scaled Lasso' - 
    a lasso based solver that estimates its own regularization
    parameter by estimating the noise in the problem
    
    See
    https://arxiv.org/abs/1202.2723
    http://www.jmlr.org/papers/volume14/sun13a/sun13a.pdf
    https://www.jstor.org/stable/41720740?seq=1#page_scan_tab_contents

    for more information

    Attributes
    ----------
    coefs_ : p by 1 array
        Estimated coefficients for the regression problem
    best_alpha_ : float
        Regression parameter estimated for the problem
    noise_ : float
        Estimated noise in the problem
    """
    def __init__(self):
        self.coefs_ = None
        self.best_alpha_ = None
        self.noise_ = None

    def fit(self, X, y):
        """
        Solves a scaled lasso problem for X and y
        Parameters
        ----------
        X : array_like
            n by p matrix - data matrix
        y : array_like
            n by 1 matrix - produced values

        Returns
        -------
        """
        n, p = X.shape

        # Our initial estimate of lambda
        lam0=np.sqrt(2*np.log(p)/n)
        sigma = np.inf
        new_sigma = 5

        for i in range(100):
            sigma = new_sigma
            lam = lam0 * sigma
            lm = Lasso(alpha=lam)
            lm.fit(X, y)
            curr_coefs = lm.coef_
            predicted_y = X @ curr_coefs 
            new_sigma = np.sqrt(((y - predicted_y) ** 2).mean())

            if abs(new_sigma - sigma) < 0.0001:
                break

        sigma = new_sigma
        lm = Lasso(alpha=lam)
        lm.fit(X, y)
        self.coefs_ = lm.coef_
        self.noise_ = sigma
        self.best_alpha_ = lam

class ScaledLassoInference(BaseEstimator):
    """
    Estimates a precision matrix using the 'scaled lasso' - this
    is a lasso based estimator that calculates its own regularization parameter
    
    See
    https://arxiv.org/abs/1202.2723
    http://www.jmlr.org/papers/volume14/sun13a/sun13a.pdf
    https://www.jstor.org/stable/41720740?seq=1#page_scan_tab_contents

    for more information.

    Attributes
    ----------
    precision_ : p by p matrix
        Estimated precision matrix 
    noise_ : p by 1 vector
        Estimation of the residual of each regression
    """

    def __init__(self):
        self.precision_ = None
        self.noise_ = None

    def fit(self, X):
        """
        Solves p scaled lassos problems to estimate the precision matrix
        of X
        Parameters
        ----------
        X : array_like
            n by p matrix - data matrix
        """
        n, p = X.shape
        indices = np.arange(p)
        noise = np.zeros(p)
        beta = np.zeros((p, p))
        cov = np.cov(X.T)
        scalefac = np.sqrt(np.var(X, axis=0))
        for i in range(p):
            X_i = np.divide(X[:, indices!=i], scalefac[indices!=i])
            sl = ScaledLasso()
            sl.fit(X_i, X[:, i])
            
            noise[i] = sl.noise_
            beta[indices!=i, i] = sl.coefs_/scalefac[indices!=i]
            beta[i, i] = -1
        self.noise_ = np.copy(noise)

        noise = np.power(noise, -2)
        tTheta = np.diag(noise)
        tTheta = -beta @ tTheta
        tTheta = methods.make_matrix_symmetric(tTheta)
        self.precision_ = tTheta    
