import numpy as np
import sklearn.linear_model as lm
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import fast_logdet
from sklearn.model_selection import KFold
import collections

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


    def _gaussian_likelihood(self, S_test, prec):
        """
        Estimates the likelihood of the neighbourhood selection
        using the Gaussian log-likelihood model
        Parameters
        ----------
        S_test : array_like
            n by p matrix - data matrix of test data
        prec : array_like
            p by p matrix - estimated precision matrix 
        """
        p = S_test.shape[0]
        log_likelihood_ = -fast_logdet(prec) + np.trace(S_test @ prec)    
        log_likelihood_ -= p * np.log(2 * np.pi)
        return log_likelihood_                                                                                                                                                                                

class NeighbourhoodSelectionColumnwiseCV(NeighbourhoodSelection):
    """
    Implementation of neighbourhood selection by Meinshausen and Bulhmann
    where we do cross validation on each column to estimate a regularization
    parameter

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
    def __init__(self):
        super().__init__(None)


    def fit(self, X):
        """
        Runs p lasso problems to estimate the off-diagonal of the precision
        matrix with cross validation on each one
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
        self.alpha_ = np.zeros(p)
        for i in range(p):
            X_new = X[:, indices!=i]
            y = X[:, i]

            l = lm.LassoCV()
            l.fit(X_new, y)
            y_hat = X_new @ l.coef_
            res = ((y - y_hat)**2).mean()

            # Estimate the diagonal
            self.precision_[i, i] = 1/res
            # Then the off-diagonal
            self.precision_[indices!=i, i] = -l.coef_/self.precision_[i, i]

            self.alpha_[i] = l.alpha_

class NeighbourhoodSelectionCV(NeighbourhoodSelection):
    """
    Implementation of neighbourhood selection by Meinshausen and Bulhmann
    where we do cross validation on the overall matrix to select
    the regularization parameter

    See
    https://projecteuclid.org/euclid.aos/1152540754

    for more information

    Attributes
    ----------
    precision_ : array_like
    Estimated precision matrix
    alpha_ : float
    Regularization parameter
    n_splits : int
    Number of CV splits to use
    """
    def __init__(self, n_splits=3):
        self.n_splits_ = n_splits
        super().__init__(None)

    def fit(self, X):
        """
        Runs p lasso problems to estimate the off-diagonal of the precision
        matrix and selects an appropriate regularization parameter
        Parameters
        ----------
        X : array_like
            n by p matrix - data matrix
        Returns
        ----------
        """
        p = X.shape[1]
        prec = np.zeros((p, p))
        kf = KFold(n_splits = self.n_splits_)
        l_likelihood = collections.defaultdict(list)
        S = np.cov(X.T)
        offdiag = ~np.eye(p, dtype=bool)
        max_l = np.max(np.abs(S[offdiag]))
        min_l = 0.0001 * max_l
        lambdas = np.logspace(np.log10(min_l), np.log10(max_l))

        for train, test in kf.split(X):
            X_train = X[train, :]
            X_test = X[test, :]
            likelihoods = []
            S_test = np.cov(X_test, rowvar=False)

            for l in lambdas:
                ns = NeighbourhoodSelection(l)
                ns.fit(X_train)
                prec = ns.precision_
                likelihood = self._gaussian_likelihood(S_test, prec)
                l_likelihood[l].append(likelihood)

        likelihoods = []
        for l in l_likelihood:
            mean_likelihood = np.mean(l_likelihood[l])
            likelihoods.append(mean_likelihood)
        best_l_index = np.argmin(likelihoods)
        self.alpha_ = lambdas[best_l_index]
        super().fit(X)