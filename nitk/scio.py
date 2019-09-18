import numpy as np
import sklearn
from sklearn.linear_model import cd_fast
import math
from sklearn.utils.validation import check_random_state
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils.extmath import fast_logdet
import collections
from sklearn.base import BaseEstimator
from . import methods

class SCIO(BaseEstimator):
    """
    Estimates a sparse precision matrix using SCIO: "Sparse Columnwise Inverse Operator"
    by Liu and Luo. See https://www.sciencedirect.com/science/article/pii/S0047259X14002607for more details

    Parameters
    -----------
    alpha: regularization parameter to use - a larger value will give a sparser network
    """
    def __init__(self, alpha, penalize_diag = False):
        self.alpha_ = alpha
        self.precision_ = None
        self.penalize_diag = penalize_diag

    def fit(self, X):
        """
        Estimate a sparse precision matrix for dataset X

        Parameters
        ----------
        X : array_like
            n by p matrix - Dataset to estimate the precision matrix of

        Returns
        -------
        None
        """
        S = np.cov(X, rowvar=False)
        p = S.shape[0]
        n = X.shape[0]

        # Calculate the addition to the diagonal
        #S = self.calculate_scaled_covariance(X)
        self.precision_ = np.zeros((p, p))
        for i in range(p):
            self.precision_[:, i] = self._solve_column_problem(S, i, self.alpha_).flatten()
        
        self.precision_ = methods.make_matrix_symmetric(self.precision_)
            
    def _solve_column_problem(self, S, i, alpha):
        """
        Solve one of the columnwise problems of SCIO
        Parameters
        ----------
        S : array_like
            p by p matrix - Covariance matrix of problem
        i : int
            index of column to estimate
        alpha : float
            regularization parameter 

        Returns
        -------
        p by 1 vector - column i of the precision matrix
        """
        p = S.shape[0]
        beta = np.zeros(p)
        e = np.zeros(p)
        e[i] = 1

        # Removes the penalization of the diagonal value
        if not self.penalize_diag:
            e[i] += alpha

        beta, _, _, _ = cd_fast.enet_coordinate_descent_gram(beta, alpha, 0, S, e, e, 100, 1e-4, check_random_state(None), False)
        return beta

    def column_likelihood_function(self, S, beta, i):
        """
        Likelihood function for column i
        Parameters
        ----------
        S : array_like
            p by p matrix - Covariance matrix of problem
        beta : array_like
            p by 1 matrix - column i of the precision matrix
        i : int
            index of column to estimate
 
        Returns
        -------
        float - value of the objective function
        """
        p = S.shape[0]
        e = np.zeros(p)
        e[i] = 1
        return 0.5 * beta.T @ S @ beta - e @ beta        

    def precision_likelihood_function(self, S, theta):
        """
        Likelihood function for a Gaussian model
        Parameters
        ----------
        S : array_like
            p by p matrix - Covariance matrix of problem
        theta : array_like
            estimated precision matrix 
 
        Returns
        -------
        float - Gaussian loglikelihood of the estimated model
        """
        p = S.shape[0]
        log_likelihood_ = -fast_logdet(theta) + np.trace(S @ theta)    
        log_likelihood_ -= p * np.log(2 * np.pi)
        return log_likelihood_                   

    def calculate_scaled_covariance(self, X):
        """
        Calculates the covariance for X with an appropriate amount added to the diagonal
        Parameters
        ----------
        X : array_like
            n by p matrix - data of the problem

        Returns
        -------
        p by p matrix - Covariance matrix
        """
        p = X.shape[1]
        n = X.shape[0]
        S = np.cov(X, rowvar=False)
        diag_addition = np.power(n, -0.5) * math.log(p, 0.5)
        S = S + diag_addition*np.eye(p)
        return S

class SCIOColumnwiseCV(SCIO):
    """
    Uses columnwise cross validation with SCIO to estimate a sparse precision matrix
    and an appropriate parameter per column
    See https://www.sciencedirect.com/science/article/pii/S0047259X14002607for more details

    Parameters
    -----------
    """
    def __init__(self, penalize_diag=False):
        super().__init__(None, penalize_diag)

    def _solve_column_with_cv(self, X, i):
        """
        Solves column i using cross validation
        Parameters
        ----------
        X : array_like
            n by p matrix - data of the problem
        i : int
            index of column to solve

        Returns
        -------
        p by 1 vector, best_l - row of precision matrix and the chosen best lambda
        """
        X_train, X_test = train_test_split(X, test_size=0.4, random_state=0)
        S_train = self.calculate_scaled_covariance(X_train)
        S_test = np.cov(X_test, rowvar=False)
        # Calculate the lambdas to check
        lambdas = np.arange(0.005, 51)
        lambdas = lambdas/50
        test_errors = []
        for l in lambdas:
            beta = self._solve_column_problem(S_train, i, l)
            error = self.column_likelihood_function(S_test, beta, i)
            test_errors.append(error)

        min_err_i = np.argmin(test_errors)

        best_l = lambdas[min_err_i]
        S = np.cov(X, rowvar=False)
        return self._solve_column_problem(S, i, best_l), best_l

    def fit(self, X):
        """
        Runs the SCIO algorithm with cross validation on each column to decide lambda

        Parameters
        ----------
        X : array_like
            n by p matrix - Dataset to estimate the precision matrix of

        Returns
        -------
        None
        """
        p = X.shape[1]
        # Calculate the addition to the diagonal
        prec = np.zeros((p, p))

        for i in range(p):
            prec[:, i], l = self._solve_column_with_cv(X, i)
        self.precision_ = methods.make_matrix_symmetric(prec)

class SCIOOverallCV(SCIO):
    """
    Uses cross validation over the entire dataset with SCIO to estimate a sparse precision matrix
    and an appropriate parameter per column
    See https://www.sciencedirect.com/science/article/pii/S0047259X14002607 for more details

    Parameters
    -----------
    n_splits : int
        number of splits to use for the k-fold cross validation procedure
    """
    def __init__(self, n_splits=3):
        self.precision_ = None
        self.n_splits_ = n_splits

    def fit(self, X):
        """
        Runs the SCIO algorithm with cross validation on the overall precision matrix to decide lambda

        Parameters
        -----------
        X : array_like
            n by p matrix to estimate the precision matrix of
        """
        p = X.shape[1]
        prec = np.zeros((p, p))
        kf = KFold(n_splits = self.n_splits_)
        l_likelihood = collections.defaultdict(list)
        for train, test in kf.split(X):
            X_train = X[train, :]
            X_test = X[test, :]
            lambdas = np.arange(0.005, 51)
            lambdas = lambdas/50
            likelihoods = []
            S_test = np.cov(X_test, rowvar=False)

            for l in lambdas:
                sc = SCIO(l)
                sc.fit(X)
                prec = sc.precision_
                likelihood = self.precision_likelihood_function(S_test, prec)
                l_likelihood[l].append(likelihood)

        likelihoods = []
        for l in l_likelihood:
            mean_likelihood = np.mean(l_likelihood[l])
            likelihoods.append(mean_likelihood)
        best_l_index = np.argmin(likelihoods)
        sc = SCIO(lambdas[best_l_index])
        sc.fit(X)
        self.precision_ = sc.precision_
