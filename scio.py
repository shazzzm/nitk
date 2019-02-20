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

def run_scio(X, l):
    """
    Runs the SCIO algorithm for estimating an inverse covariance matrix proposed by Liu et al
    """
    S = np.cov(X, rowvar=False)
    p = S.shape[0]
    n = X.shape[0]

    # Calculate the addition to the diagonal
    diag_addition = np.power(n, -0.5) * math.log(p, 0.5)
    S = S + diag_addition*np.eye(p)

    prec = np.zeros((p, p))

    for i in range(p):
        prec[:, i] = solve_column_problem(S, i, l).flatten()
    return make_matrix_symmetric(prec)

def run_scio_with_columnwise_cv(X):
    """
    Runs the SCIO algorithm with cross validation on each column to decide lambda
    """
    p = X.shape[1]
    # Calculate the addition to the diagonal
    prec = np.zeros((p, p))

    for i in range(p):
        prec[:, i], l = solve_column_with_cv(X, i)
    return make_matrix_symmetric(prec)

def run_scio_with_cv(X, n_splits=3):
    """
    Runs the SCIO algorithm with cross validation on the overall precision matrix to decide lambda
    """
    p = X.shape[1]
    # Calculate the addition to the diagonal
    prec = np.zeros((p, p))
    kf = KFold(n_splits = n_splits)
    l_likelihood = collections.defaultdict(list)
    for train, test in kf.split(X):
        X_train = X[train, :]
        X_test = X[test, :]
        lambdas = np.arange(0, 51)
        lambdas = lambdas/50
        likelihoods = []
        S_test = np.cov(X_test, rowvar=False)
        print(X_train.shape)
        print(X_test.shape)
        for l in lambdas:
            prec = run_scio(X_train, l)
            likelihood = precision_likelihood_function(S_test, prec)
            l_likelihood[l].append(likelihood)

    likelihoods = []
    for l in l_likelihood:
        mean_likelihood = np.mean(l_likelihood[l])
        likelihoods.append(mean_likelihood)
    best_l_index = np.argmin(likelihoods)
    return run_scio(X, lambdas[best_l_index])
        
def solve_column_problem(S, i, l):
    """
    Solve one of the columnwise problems of SCIO
    """
    p = S.shape[0]
    beta = np.zeros(p)
    e = np.zeros(p)
    e[i] = 1
    beta, _, _, _ = cd_fast.enet_coordinate_descent_gram(beta, l, 0, S, e, e, 100, 1e-4, check_random_state(None), False)
    return beta

def column_likelihood_function(S, beta, i):
    p = S.shape[0]
    e = np.zeros(p)
    e[i] = 1
    return 0.5 * beta.T @ S @ beta - e @ beta        

def precision_likelihood_function(S, theta):
    p = S.shape[0]
    log_likelihood_ = -fast_logdet(theta) + np.trace(S @ theta)    
    log_likelihood_ -= p * np.log(2 * np.pi)
    return log_likelihood_                   

def solve_column_with_cv(X, i):
    """
    Solves column i using cross validation
    """
    X_train, X_test = train_test_split(X, test_size=0.4, random_state=0)
    S_train = calculate_scaled_covariance(X_train)
    S_test = np.cov(X_test, rowvar=False)
    # Calculate the lambdas to check
    lambdas = np.arange(0, 51)
    lambdas = lambdas/50
    test_errors = []
    for l in lambdas:
        beta = solve_column_problem(S_train, i, l)
        error = column_likelihood_function(S_test, beta, i)
        test_errors.append(error)

    min_err_i = np.argmin(test_errors)

    best_l = lambdas[min_err_i]
    S = np.cov(X, rowvar=False)
    return solve_column_problem(S, i, best_l), best_l

def make_matrix_symmetric(M):
    p = M.shape[0]
    for i in range(p):
        for j in range(i, p):
            if np.abs(M[i, j]) < np.abs(M[j, i]):
                M[j, i] = M[i, j]
            else:
                M[i, j] = M[j, i]

    return M

def calculate_scaled_covariance(X):
    """
    Calculates the covariance for X with an appropriate amount added to the diagonal
    """
    p = X.shape[1]
    n = X.shape[0]
    S = np.cov(X, rowvar=False)
    diag_addition = np.power(n, -0.5) * math.log(p, 0.5)
    S = S + diag_addition*np.eye(p)
    return S

if __name__ == "__main__":
    # Run a little test
    p = 100
    number_samples = 2000
    P = sklearn.datasets.make_sparse_spd_matrix(dim=p, alpha=0.8, smallest_coef=.4, largest_coef=.7,)
    C = np.linalg.inv(P)
    X = np.random.multivariate_normal(np.zeros(p), C, number_samples)
    prec = run_scio_with_cv(X)
    print(prec)
