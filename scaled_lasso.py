import numpy as np
import glmnet_python
from glmnet import glmnet
from sklearn.datasets import make_sparse_spd_matrix
import methods
import math

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import numpy as np
import rpy2.rinterface as rinterface
from sklearn.linear_model import lars_path, Lasso

rpy2.robjects.numpy2ri.activate()
buf = []

def f(x):
    buf.append(x)
#rinterface.set_writeconsole_regular(f)

scalreg = importr('scalreg')

def run(X, l2=0):
    """
    """
    prec = scalreg.scalreg(X, lam="univ")
    prec = np.array(prec[0])
    return prec

def solve_scaled_lasso_r(X, y):
    outp = scalreg.scalreg(X, y, lam="univ")
    return (outp[0][0], outp[1], np.array(outp[2]))

def calculate_noise(beta, cov):
    p = cov.shape[0]
    return beta.T @ cov @ beta

def calculate_lambda(l_0, noise):
    return noise * l_0

def find_nearest_arg(array,value):
    idx = np.searchsorted(array[::-1], value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def solve_scaled_lasso(X, y):
    n, p = X.shape
    alphas, active_coefficients, coefs = lars_path(X, y, method='lasso')
    print(alphas)
    lam0=np.sqrt(2*np.log(p)/n)
    sigmaint=0.1
    sigmanew=5
    flag = 0
    while abs(sigmaint-sigmanew)>0.0001 and flag <= 100:
        flag += 1
        sigmaint = sigmanew
        lam = lam0 * sigmaint
        print(lam)
        lm = Lasso(alpha=lam)
        lm.fit(X, y)
        #idx = find_nearest_arg(alphas, lam)
        #curr_coefs = coefs[:, idx]
        curr_coefs = lm.coef_
        hy = X @ curr_coefs 
        sigmanew = np.sqrt(((y - hy) ** 2).mean())

    hsigma = sigmanew
    hlam = lam
    lm = Lasso(alpha=lam)
    lm.fit(X, y)
    best_coef = lm.coef_
    #best_alpha = alphas[idx]
    #print(idx)
    #best_coef = coefs[:, idx]
    hy = X @ best_coef
    return lam, best_coef, hsigma, y - hy, hy

def estimate_precision_matrix(X):
    n, p = X.shape
    indices = np.arange(p)
    hsigma = np.zeros(p)
    beta = np.zeros((p, p))
    residuals = np.zeros((n, p))
    for i in range(p):
        scalefac = np.sqrt((X[:, indices!=i]**2).sum(axis=0)/n)
        X_j = X[:, indices!=i]/scalefac
        best_alpha, best_coef, hsigma_i, res, hy = solve_scaled_lasso(X_j, X[:, i])
        #hsigma_i, best_coef, res = solve_scaled_lasso_r(X_j, X[:, i])
        hsigma[i] = hsigma_i
        beta[indices!=i, i] = best_coef/scalefac
        residuals[:,i] = res

    hsigma = np.reciprocal(hsigma**2)
    tTheta = np.diag(hsigma)
    tTheta = -beta @ tTheta
    hTheta = methods.make_matrix_symmetric(tTheta)
    ind = np.diag_indices(p)
    hTheta[ind] = hsigma
    return tTheta

def fit(X):
    n, p = X.shape
    l_0 = np.sqrt((2/n) * np.log(p))
    cov = np.cov(X.T)
    indices = np.arange(p)
    precision_ = np.linalg.pinv(cov)

    for i in range(p):
        for it in range(10):
            noise = precision_[i, i]
            coefs = solve_scaled_lasso(X[:, indices!=i], X[:, i], cov, noise, l_0).flatten()
            beta = np.zeros(p)
            beta[indices!=i] = coefs
            beta[i] = -1
            noise = calculate_noise(beta, cov)
            prec_diag = np.diag(precision_)
            precision_[i, indices!=i] = np.divide(coefs, prec_diag[indices!=i])
            l_0 = calculate_lambda(l_0, noise)
            precision_[i, i] = 1/noise


    print(precision_)

p = 5
n = 200
K = make_sparse_spd_matrix(p, 0.7)
C = np.linalg.inv(K)
X = np.random.multivariate_normal(np.zeros(p), C, n)
print(estimate_precision_matrix(X))
print(run(X))
print(K)