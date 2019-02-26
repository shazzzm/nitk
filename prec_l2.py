"""
Implements an L2 regularised covariance estimator - based on Scout by Witten and Tibshirani
We solve -logdet Theta + tr(S Theta) + p ||Theta||_2^2
"""
import numpy as np
import sklearn
import scipy as sp
import sklearn.datasets

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

    theta = new_eigvals * eigvecs * eigvecs.T
    print(theta)
    return theta

p=50
P = sklearn.datasets.make_sparse_spd_matrix(dim=p, alpha=0.8, smallest_coef=.4, largest_coef=.7,)
C = np.linalg.inv(P)
X = np.random.multivariate_normal(np.zeros(p), C, 100)
S = np.cov(X, rowvar=False)

print(solve(S, 2))