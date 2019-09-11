import unittest
from sklearn.datasets import make_sparse_spd_matrix
import correlation_network
import methods
import numpy as np


class TestCorrelationPermutation(unittest.TestCase):
    def test_correlation_network_permutation(self):
        """
        Generates a distribution with a sparse covariance matrix and sees if the non-zero values are correctly picked up
        by the correlation permuter
        """
        p = 10
        n = 200
        C = make_sparse_spd_matrix(p, 0.7)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        corr_model = correlation_network.CorrelationPermutationNetwork()
        corr_model.fit(X)
        corr = corr_model.correlation_
        C = methods.threshold_matrix(C, 0.001, binary=True)
        corr = methods.threshold_matrix(corr, 0.001, binary=True)

        print(methods.matrix_similarity(C,corr))
        print(corr)
        print(C)

    def test_partial_correlation_network_permutation(self):
        """
        Generates a distribution with a sparse partial correlation matrix and sees if the non-zero values are correctly picked up
        """
        p = 10
        n = 200
        K = make_sparse_spd_matrix(p, 0.7)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        par_corr = correlation_network.significant_partial_correlation_matrix(X)
        K = methods.threshold_matrix(K, 0.001, binary=True)
        par_corr = methods.threshold_matrix(par_corr, 0.001, binary=True)

        print(methods.matrix_similarity(K,par_corr))
        print(par_corr)
        print(K)

if __name__ == '__main__':
    unittest.main()