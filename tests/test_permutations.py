import unittest
from sklearn.datasets import make_sparse_spd_matrix
import correlation_permuter
import methods
import numpy as np

class TestCorrelationPermutation(unittest.TestCase):
    def test_correlation_permutation(self):
        """
        Generates a distribution with a sparse covariance matrix and sees if the non-zero values are correctly picked up
        by the correlation permuter
        """
        p = 10
        n = 200
        C = make_sparse_spd_matrix(p, 0.7)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        corr_model = correlation_permuter.CorrelationPermutationNetwork()
        corr_model.fit(X)
        corr = corr_model.correlation_
        C = methods.threshold_matrix(C, 0.001, binary=True)
        corr = methods.threshold_matrix(corr, 0.001, binary=True)

        print(methods.matrix_similarity(C,corr))
        print(corr)
        print(C)

if __name__ == '__main__':
    unittest.main()