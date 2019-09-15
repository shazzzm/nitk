import numpy as np
import methods
import math
import numpy as np
from sklearn.linear_model import lars_path, Lasso
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

            if abs(new_sigma - sigma):
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
    """

    def __init__(self):
        self.precision_ = None

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
        hsigma = np.zeros(p)
        beta = np.zeros((p, p))
        residuals = np.zeros((n, p))
        for i in range(p):
            scalefac = np.sqrt((X[:, indices!=i]**2).sum(axis=0)/n)
            X_j = X[:, indices!=i]/scalefac
            sl = ScaledLasso()
            sl.fit(X_j, X[:, i])
            
            hsigma[i] = sl.noise_
            beta[indices!=i, i] = sl.coefs_/scalefac
            residuals[:,i] = np.power((X_i - X_j @ beta), 2)

        hsigma = np.reciprocal(hsigma**2)
        tTheta = np.diag(hsigma)
        tTheta = -beta @ tTheta
        hTheta = methods.make_matrix_symmetric(tTheta)
        ind = np.diag_indices(p)
        hTheta[ind] = hsigma
        self.precision_ = hTheta