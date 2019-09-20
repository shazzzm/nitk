import numpy as np
import sklearn.datasets
from . import methods

class DTrace():
    """
    Estimates a sparse precision matrix using the D-Trace loss,
    proposed by Zhang and Zou.

    See
    https://academic.oup.com/biomet/article/101/1/103/2366063 

    for more information

    Attributes
    ----------
    precision_ : p by p matrix
        Estimated precision matrix 
    alpha_ : float
        L1 regularization parameter
    iter_ : int
        Number of iterations to completion
    """
    def __init__(self, alpha):
        self.precision_ = None
        self.alpha_ = alpha
        self.iter_ = None

    def build_C(self, eigs):
        """
        Creates the C matrix mentioned in the paper where
        C_ij = 2/(eigs[i] + eigs[j])
        Parameters
        ----------
        eigs : array_like
            p by 1 matrix - sorted array of eigenvalues of the covariance matrix
        """
        p = eigs.shape[0]
        C = np.zeros((p, p))

        for i in range(p):
            for j in range(p):
                C[i, j] = 2 / (eigs[i] + eigs[j])

        return C

    def solve_g(self, eigs, eigv, C, B):
        """
        Returns the value of the G function in the paper (equation 19)
        Parameters
        ----------
        eigs : array_like
            p by 1 matrix - sorted array of eigenvalues of the covariance matrix
        eigv : array_like
            p by p matrix - matrix of eigenvectors corresponding to the eigenvalues
        C : array_like
            p by p matrix - C matrix mentioned above
        B : array_like
            p by p matrix - B matrix from equation 19 (usually I + rho * Theta_ - delta_K)
        """

        C = self.build_C(eigs)

        D = np.multiply(eigv.T @ B @ eigv, C)
        return eigv @ D @ eigv.T

    def solve_S(self, theta, l):
        """
        Soft thresholding function 
        Parameters
        ----------
        theta : array_like
            p by p matrix - precision matrix
        l : float
            regularization paramter to threshold at
        """
        p = theta.shape[0]
        output_theta = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                if i == j:
                    output_theta[i, j] = theta[i, j]
                if np.abs(theta[i, j]) > l:
                    if theta[i, j] < 0:
                        output_theta[i, j] = theta[i, j] + l
                    else:
                        output_theta[i, j] = theta[i, j] - l

        return output_theta

    def eigenvalue_threshold(self, theta, e=0.01):
        """
        Ensures all the eigenvalues are positive and 
        greater than e
        theta : array_like
            p by p matrix - precision matrix
        e : float
            minimum permitted eigenvalue
        """
        eigs, eigv = np.linalg.eig(theta)

        eigs[eigs < e] = e
        D = np.diag(eigs)

        return eigv.T @ D @ eigv


    def fit(self, X):
        """
        Estimates the precision matrix of X using the DTRACE method
        X : array_like
            n by p matrix - data matrix
        """
        n,p = X.shape
        rho = 1
        S = np.cov(X.T)
        delta_0 = np.zeros((p, p))
        delta_1 = np.zeros((p, p))
        theta_0 = np.zeros((p, p))
        theta_1 = np.zeros((p, p))
        theta_diag = np.reciprocal(np.diag(S))
        ind = np.diag_indices(p)
        theta_0[ind] = theta_diag
        theta_1 = theta_0
        prev_theta_hat = np.zeros((p, p))
        prev_theta_0 = np.zeros((p, p))
        diff_0 = np.inf
        diff_1 = np.inf

        i = 0

        # Cache this relatively expensive bit of solving G
        Z = S + rho * np.eye(p)
        eigs, eigv = np.linalg.eig(Z)
        ind = np.argsort(eigs)[::-1]
        eigs = eigs[ind]
        eigv = eigv[:, ind]
        C = self.build_C(eigs)

        while (diff_0 > 10e-7 or diff_1 > 10e-7) and i < 100: 
            theta_hat = self.solve_g(eigs, eigv, C, np.eye(p) + rho * theta_0 - delta_0)
            theta_0 = self.solve_S(theta_hat + (1/rho)*delta_0, (1/rho)*self.alpha_)
            #theta_1 = self.eigenvalue_threshold(theta_hat + delta_1/rho )
            delta_0 = delta_0 + rho * (theta_hat - theta_0)
            #delta_1 = delta_1 + rho * (theta_hat - theta_1)
            diff_0 = np.linalg.norm(prev_theta_hat - theta_hat) / max(1, np.linalg.norm(theta_hat), np.linalg.norm(prev_theta_hat))
            diff_1 = np.linalg.norm(prev_theta_0 - theta_0) / max(1, np.linalg.norm(theta_0), np.linalg.norm(prev_theta_0))
            prev_theta_hat = theta_hat.copy()
            prev_theta_0 = theta_0.copy()
            i += 1
            

        self.precision_ = methods.threshold_matrix(theta_hat, 0.0001)
        self.iter_ = i