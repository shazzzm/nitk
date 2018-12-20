"""
Implements the SPACE algorithm proposed by Peng et al
"""
import numpy as np
import cvxpy as cvx
import sklearn.datasets
import sklearn.linear_model as lm
import matrix_methods

def build_chi_matrix(X, sigma, w):
    n = X.shape[0]
    p = X.shape[1]
    chi = np.zeros((n*p, p*(p-1)))
    chi_Y = np.zeros(n*p)
    for i in range(p):
        for j in range(p):
            sigma_tilde_i = sigma[i]/w[i]
            sigma_tilde_j = sigma[j]/w[j]

            row = np.zeros(n*p)
            row[i:(i+n)] = np.sqrt(sigma_tilde_j/sigma_tilde_i) * X[:, j]
            row[j:(j+n)] = np.sqrt(sigma_tilde_i/sigma_tilde_j) * X[:, i]
            chi[:, i*j] = row

    for i in range(p):
        chi_Y[i*n:(i+1)*n] = np.sqrt(w[i]) * X[:, i]

    return chi, chi_Y

def reconstruct_precision_matrix(coefs, p):
    prec = np.zeros((p, p))
    indices = np.arange(p)
    for i in range(p):
        prec[i, indices!=i] = coefs[i*(p-1):(i+1)*(p-1)]

    return prec 

def run(X):
    """
    Runs the space algorithm on the dataset X and returns a sparse precision matrix
    """
    n = X.shape[0]
    p = X.shape[1]
    sigma = np.ones(p)
    w = np.ones(p)
    chi, chiY = build_chi_matrix(X, sigma, w)

    clf = lm.LassoCV()
    clf.fit(chi, chiY)

    space_prec = reconstruct_precision_matrix(clf.coef_, p)
    return space_prec