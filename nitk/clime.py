import numpy as np
from scipy.optimize import linprog
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.base import BaseEstimator
from nitk import methods
from sklearn.model_selection import KFold
import collections
from sklearn.utils.extmath import fast_logdet

class CLIME(BaseEstimator):
    """
    Estimates a sparse precision matrix using the CLIME - "A Constrained ℓ1 Minimization Approach to Sparse Precision Matrix Estimation"
    by Cai et al. See https://amstat.tandfonline.com/doi/abs/10.1198/jasa.2011.tm10155#.XXkp6vfTVhE for more details

    Parameters
    -----------
    alpha : float
        regularization parameter to use - a smaller value will give a sparser network
    perturb : bool
        whether to add a bit to the diagonal to make the problem nicer
    """
    def __init__(self, alpha, perturb=True):
        self.precision_ = None
        self.alpha_ = alpha
        self.perturb_ = perturb
        
    def _solve_row(self, cov, i):
        """
        Solves the CLIME problem for row i using linear programming
        Parameters
        ----------
        cov : array_like
            p by p matrix - Covariance matrix of the problem
        i : int
            which row of the precision matrix to estimate

        Returns
        -------
        p-1 by 1 vector
        off diagonal of row i of the precision matrix
        """
        p = cov.shape[0]
        e_i = np.zeros(p)
        e_i[i] = 1
        p = cov.shape[0]
        c = np.ones(2*p)
        con1 = np.concatenate([-cov, cov], axis=1)
        b1 = np.ones(p) * self.alpha_
        b1[i] = b1[i] - 1
        b2 = np.ones(p) * self.alpha_
        b2[i] = 1 + self.alpha_
        con = np.concatenate([-np.eye(2*p), con1, -con1], axis=0)
        A_ub = con
        b_ub = np.concatenate([np.zeros(2*p), b1, b2])
        A_eq = np.zeros(A_ub.shape)
        b_eq = np.zeros(b_ub.shape)
        result = linprog(c, A_ub, b_ub, A_eq, b_eq)
        #if result['success']:
        solution = result['x']
        beta = solution[0:p] - solution[p:(2*p)]

        return beta
        #else:
        #    raise FloatingPointError("Optimizer could not find a solution")

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
        n, p = X.shape
        self.precision_ = np.zeros((p, p))
        cov = np.cov(X.T)
        indices = np.arange(p)

        if self.perturb_:
            eigs, _ = np.linalg.eig(cov)
            perturb_val = eigs.max() - p * eigs.min()
            if perturb_val > 0:
                perturb_val = perturb_val / p
            else:
                perturb_val = 0
            
            cov = cov + perturb_val * np.eye(p)

        for i in range(p):
            row = self._solve_row(cov, i)
            self.precision_[i, :] = row

        self.precision_ = methods.make_matrix_symmetric(self.precision_)

class CLIMECV(BaseEstimator):
    """
    Estimates a sparse precision matrix using the CLIME - "A Constrained ℓ1 Minimization Approach to Sparse Precision Matrix Estimation"
    by Cai et al and estimates the regularization parameter using cross validation.
    See https://amstat.tandfonline.com/doi/abs/10.1198/jasa.2011.tm10155#.XXkp6vfTVhE for more details

    Parameters
    -----------
    alpha : float
        regularization parameter decided by CV
    perturb : bool
        whether to add a bit to the diagonal to make the problem nicer
    n_splits : int
        number of CV splits to use
    loss : str
        which loss function to use (choices are likelihood or trace)
    """
    def __init__(self, perturb=False,  n_splits=3, loss="likelihood"):
        self.perturb_ = perturb
        self.precision_ = None
        self.n_splits_ = n_splits
        self.loss_ = loss

    def fit(self, X):
        """
        Runs the CLIME algorithm with cross validation to decide lambda

        Parameters
        ----------
        X : array_like
            n by p matrix - Dataset to estimate the precision matrix of

        Returns
        -------
        None
        """
        lambdas = np.logspace(-3, 0)
        p = X.shape[1]
        prec = np.zeros((p, p))
        kf = KFold(n_splits = self.n_splits_)
        l_likelihood = collections.defaultdict(list)
        for train, test in kf.split(X):
            X_train = X[train, :]
            X_test = X[test, :]
            likelihoods = []
            S_test = np.cov(X_test, rowvar=False)

            for l in lambdas:
                cl = CLIME(l)
                cl.fit(X)
                prec = cl.precision_
                if self.loss_ == "likelihood":
                    likelihood = self.likelihood(S_test, prec)
                elif self.loss_ == "trace":
                    likelihood = self.trace_l2_loss(S_test, prec)
                l_likelihood[l].append(likelihood)

        likelihoods = []
        for l in l_likelihood:
            mean_likelihood = np.mean(l_likelihood[l])
            likelihoods.append(mean_likelihood)
        best_l_index = np.argmin(likelihoods)
        cl = CLIME(lambdas[best_l_index])
        cl.fit(X)
        self.precision_ = cl.precision_

    def likelihood(self, S, theta):
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

    def trace_l2_loss(self, S, theta):
        """
        CLIME specific loss function - sum(trace(Sigma * Theta - I) ^ 2)
        Doesn't require the precision matrix to be positive definite
        Parameters
        ----------
        S : array_like
            p by p matrix - Covariance matrix of problem
        theta : array_like
            estimated precision matrix 
 
        Returns
        -------
        float - model loss
        """
        return 
