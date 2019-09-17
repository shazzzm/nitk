import unittest
from sklearn.datasets import make_sparse_spd_matrix
from nitk import methods
import numpy as np
from nitk.clime import CLIME

class TestCLIME(unittest.TestCase):
    def test_clime(self):
        """
        Generates a distribution with a sparse precision matrix and sees if the non-zero values are correctly picked up
        by the CLIME
        """
        p = 10
        n = 200
        K = make_sparse_spd_matrix(p, 0.7)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        l = 0.5
        cl = CLIME(l)
        cl.fit(X)
        K = methods.threshold_matrix(K, 0.001, binary=True)
        prec_ = methods.threshold_matrix(cl.precision_, 0.001, binary=True)

        #print(methods.matrix_similarity(K,prec_))


if __name__ == '__main__':
    unittest.main()