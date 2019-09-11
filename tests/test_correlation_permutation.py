import unittest
from sklearn.datasets import make_sparse_spd_matrix
import correlation_network
import methods
import numpy as np


class TestCorrelationPermutation(unittest.TestCase):
    def test_correlation_network_permutation(self):
        """
        Generate a sparse matrix and see if the non-zero values are correctly picked up
        by the 
        """
        p = 10
        n = 200
        C = make_sparse_spd_matrix(p, 0.7)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        corr = correlation_network.significant_correlation_matrix(X)
        C = methods.threshold_matrix(C, 0.001, binary=True)
        corr = methods.threshold_matrix(corr, 0.001, binary=True)

        print(methods.matrix_similarity(C,corr))
        print(corr)
        print(C)

if __name__ == '__main__':
    unittest.main()