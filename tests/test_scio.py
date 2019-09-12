import unittest
from sklearn.datasets import make_sparse_spd_matrix
import methods
import numpy as np
from scio import *

class TestSCIO(unittest.TestCase):
    def test_scio(self):
        """
        Generates a distribution with a sparse precision matrix and sees if the non-zero values are correctly picked up
        by SCIO
        """
        p = 10
        n = 200
        K = make_sparse_spd_matrix(p, 0.7)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        l = 0.5
        sc = SCIO(l)
        sc.fit(X)
        K = methods.threshold_matrix(K, 0.001, binary=True)
        prec_ = methods.threshold_matrix(sc.precision_, 0.001, binary=True)

        print(methods.matrix_similarity(K,prec_))

    def test_scio_cv_all(self):
        """
        Generates a distribution with a sparse precision matrix and sees if the non-zero values are correctly picked up
        by SCIO using CV over the entire matrix
        """
        p = 10
        n = 200
        K = make_sparse_spd_matrix(p, 0.7)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        sc = SCIOOverallCV()
        sc.fit(X)
        K = methods.threshold_matrix(K, 0.001, binary=True)
        prec_ = methods.threshold_matrix(sc.precision_, 0.001, binary=True)

        print(methods.matrix_similarity(K,prec_))

    def test_scio_cv_columnwise(self):
        """
        Generates a distribution with a sparse precision matrix and sees if the non-zero values are correctly picked up
        by SCIO using CV over each column
        """
        p = 10
        n = 200
        K = make_sparse_spd_matrix(p, 0.7)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        sc = SCIOColumnwiseCV()
        sc.fit(X)
        K = methods.threshold_matrix(K, 0.001, binary=True)
        prec_ = methods.threshold_matrix(sc.precision_, 0.001, binary=True)

        print(methods.matrix_similarity(K,prec_))
if __name__ == '__main__':
    unittest.main()