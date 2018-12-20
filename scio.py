import numpy as np
import sklearn
from sklearn.linear_model import cd_fast
import math
from sklearn.utils.validation import check_random_state
import sklearn.datasets

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

def make_matrix_symmetric(M):
    p = M.shape[0]
    for i in range(p):
        for j in range(i, p):
            if np.abs(M[i, j]) < np.abs(M[j, i]):
                M[j, i] = M[i, j]
            else:
                M[i, j] = M[j, i]

    return M

# Run a little test
p = 100
number_samples = 2000
P = sklearn.datasets.make_sparse_spd_matrix(dim=p, alpha=0.8, smallest_coef=.4, largest_coef=.7,)
C = np.linalg.inv(P)
X = np.random.multivariate_normal(np.zeros(p), C, number_samples)
prec = run_scio(X, 0.05)
print(prec)