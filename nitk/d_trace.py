import numpy as np
import sklearn.datasets

class DTRACE():
    def __init__(self, alpha):
        """
        """
        self.alpha_ = alpha
    def solve_g(self, A, B):
        return np.linalg.inv(A) @ B

    def solve_S(self, theta, l):
        p = theta.shape[0]
        output_theta = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                if np.abs(theta[i, j]) > self.alpha_:
                    if theta[i, j] < 0:
                        output_theta[i, j] = theta[i, j] + l
                    else:
                        output_theta[i, j] = theta[i, j] - l

        return output_theta

    def solve_delta(self, delta, theta, theta_0, p):
        return delta + p*(theta - theta_0)

    def fit(self, X):
        n,p = X.shape
        rho = 1
        S = np.cov(X.T)
        delta = np.zeros((p, p))
        theta_0 = np.zeros((p, p))
        theta_diag = np.reciprocal(np.diag(S))
        ind = np.diag_indices(p)
        theta_0[ind] = theta_diag
        prev_theta = theta_0.copy()
        diff = np.inf

        while diff > 0.001:
            theta_hat = self.solve_g(S + rho * np.eye(p), np.eye(p) + rho * theta_0 - delta)
            theta_0 = self.solve_S(theta_hat + (1/rho)*delta, (1/rho)*self.alpha_)
            delta = delta + rho * (theta_hat - theta_0)
            diff = np.linalg.norm(prev_theta - theta_hat)
            prev_theta = theta_hat.copy()

        return theta_hat
if __name__=='__main__':
    p = 100
    n = 200
    l = 0.1
    P = sklearn.datasets.make_sparse_spd_matrix(dim=p, alpha=0.8, smallest_coef=.4, largest_coef=.7,)
    C = np.linalg.inv(P)
    X = np.random.multivariate_normal(np.zeros(p), C, n)
    dt = DTRACE(alpha=l)
    prec = dt.fit(X)

    prec[np.abs(prec) < 0.0001] = 0
    
