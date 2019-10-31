import unittest
from sklearn.datasets import make_sparse_spd_matrix
import numpy as np
from nitk.scio import *
from nitk import methods
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.rinterface as rinterface
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_less
class TestSCIO(unittest.TestCase):
    def _estimate_precision_matrix_using_r(self, X, l, penalize_diag):
        """
        Estimates a precision matrix using the R implementation
        of SCIO
        """
        rpy2.robjects.numpy2ri.activate()
        scio = importr('scio')
        ob = scio.scio(np.cov(X.T), l, 0.00001, 10000, penalize_diag, True)

        return np.array(ob[0])

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

        r_prec = self._estimate_precision_matrix_using_r(X, l, False)
        assert_array_almost_equal(r_prec, sc.precision_, decimal=4)

    def test_scio_with_diag_penalty(self):
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
        sc = SCIO(l, penalize_diag=True)
        sc.fit(X)

        r_prec = self._estimate_precision_matrix_using_r(X, l, True)
        assert_array_almost_equal(r_prec, sc.precision_, decimal=4)

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

        #print(methods.matrix_similarity(K,prec_))

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

        #print(methods.matrix_similarity(K,prec_))

    def test_scio_bic_columnwise(self):
        """
        Generates a distribution with a sparse precision matrix and sees if the non-zero values are correctly picked up
        by SCIO using BIC over each column
        """
        p = 10
        n = 200
        K = make_sparse_spd_matrix(p, 0.7)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        sc = SCIOColumnBIC()
        sc.fit(X)
        K = methods.threshold_matrix(K, 0.001, binary=True)
        prec_ = methods.threshold_matrix(sc.precision_, 0.001, binary=True)
if __name__ == '__main__':
    unittest.main()