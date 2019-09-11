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

def solve(S, l):
    """
    Solves the problem with the specified penalty level. This is an eigenvalue problem
    """
    p = S.shape[0]
    eigvals, eigvecs = np.linalg.eig(S)
    new_eigvals = np.zeros(p)

    for i in range(p):
        s = eigvals[i]
        e = -s/(4*l) + np.sqrt(s**2 + 8*l)/(4*l)
        new_eigvals[i] = e
    #print(new_eigvals)
    theta = new_eigvals * eigvecs * eigvecs.T
    return theta

def precision_likelihood_function(S, theta):
    p = S.shape[0]
    log_likelihood_ = -fast_logdet(theta) + np.trace(S @ theta)    
    print(log_likelihood_)
    log_likelihood_ -= p * np.log(2 * np.pi)
    return log_likelihood_    

def run_cv(X, n_splits = 4):
    """
    Runs the L2 regularization algorithm with cross validation to decide lambda
    """
    p = X.shape[1]
    # Calculate the addition to the diagonal
    prec = np.zeros((p, p))
    kf = KFold(n_splits = n_splits)
    l_likelihood = collections.defaultdict(list)
    for train, test in kf.split(X):
        X_train = X[train, :]
        X_test = X[test, :]
        lambdas = np.logspace(0, 1)
        #print(lambdas)
        likelihoods = []
        S_train = np.cov(X_train, rowvar=True)
        S_test = np.cov(X_test, rowvar=False)

        for l in lambdas:
            prec = solve(S_train, l)
            likelihood = precision_likelihood_function(S_test, prec)
            l_likelihood[l].append(likelihood)

    likelihoods = []
    for l in l_likelihood:
        mean_likelihood = np.mean(l_likelihood[l])
        likelihoods.append(mean_likelihood)
    best_l_index = np.argmin(likelihoods)
    print(likelihoods)
    print(lambdas[best_l_index])
    return solve(X, lambdas[best_l_index])

if __name__=="__main__":
    p=50
    P = sklearn.datasets.make_sparse_spd_matrix(dim=p, alpha=0.8, smallest_coef=.4, largest_coef=.7,)
    C = np.linalg.inv(P)
    X = np.random.multivariate_normal(np.zeros(p), C, 50)
    S = np.cov(X, rowvar=False)

    print(run_cv(S))