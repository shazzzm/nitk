import unittest
from sklearn.datasets import make_sparse_spd_matrix
from nitk import methods
import numpy as np
from nitk.clime import CLIME
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.rinterface as rinterface
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_less

class TestCLIME(unittest.TestCase):
    def _estimate_precision_matrix_using_r(self, X, l):
        """
        Estimates the precision matrix using the R
        implementation provided by the authors
        """
        rpy2.robjects.numpy2ri.activate()
        clime = importr('clime')
        prec = clime.clime(X, np.array([l]), perturb=True, standardize=False, linsolver="simplex")
        prec = np.array(prec[0][0])

        return prec
    def test_clime(self):
        """
        Generates a distribution with a sparse precision matrix and sees if the non-zero values are correctly picked up
        by the CLIME
        """
        p = 50
        n = 10
        K = make_sparse_spd_matrix(p, 0.7)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        l = 0.5
        r_prec = self._estimate_precision_matrix_using_r(X, l)
        print(r_prec)
        cl = CLIME(l, True)
        cl.fit(X)

        assert_array_almost_equal(r_prec, cl.precision_, decimal=2)

if __name__ == '__main__':
    unittest.main()