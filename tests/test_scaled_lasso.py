"""
Simple wrapper around the scaled lasso from the scalereg package in R
"""

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import numpy as np
import rpy2.rinterface as rinterface
from nitk import scaled_lasso
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_less
import unittest
from sklearn.datasets import make_sparse_spd_matrix

class TestScaledLasso(unittest.TestCase):
    def _estimate_precision_matrix_using_r(self, X):
        """
        Estimates the precision matrix using the R
        implementation provided by the authors
        """
        rpy2.robjects.numpy2ri.activate()
        scalreg = importr('scalreg')
        prec = scalreg.scalreg(X, lam0="univ")
        noise = np.array(prec[1])
        print(noise)
        prec = np.array(prec[0])
        
        return prec

    def _scaled_lasso_using_r(self, X, y):
        """
        Solves the regression problem with the scaled
        lasso for the given data
        """
        rpy2.robjects.numpy2ri.activate()
        scalreg = importr('scalreg')
        outp = scalreg.scalreg(X, y, lam="univ")
        return outp[1]

    def test_scaled_lasso(self):
        """
        Generates a trivial regression problem and compares our 
        scaled lasso solution to that of the authors
        """
        p = 10
        n = 200
        X = np.random.multivariate_normal(np.zeros(p), np.eye(p), n)
        beta = np.random.normal(size=p)
        y = X @ beta

        r_estimate = self._scaled_lasso_using_r(X, y)
        sl = scaled_lasso.ScaledLasso()
        sl.fit(X, y)
        assert_array_almost_equal(r_estimate, sl.coefs_, decimal=4)
 
    def test_scaled_lasso_precision_network(self):
        """
        We test our implementation of the scaled lasso
        based precision matrix estimation against that of the authors
        """
        p = 10
        n = 200
        K = make_sparse_spd_matrix(p, 0.7)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        sli = scaled_lasso.ScaledLassoInference()
        sli.fit(X)
        prec_r = self._estimate_precision_matrix_using_r(X)

        assert_array_almost_equal(prec_r, sli.precision_, decimal=2)

if __name__ == '__main__':
    unittest.main()