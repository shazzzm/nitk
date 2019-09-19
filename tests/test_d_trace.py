import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import numpy as np
import rpy2.rinterface as rinterface
from nitk import d_trace
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_less
import unittest
from sklearn.datasets import make_sparse_spd_matrix

class TestDTrace(unittest.TestCase):
    def test_d_trace_estimator(self):
        """
        Creates a distribution and ensures the estimator returns something
        sensible
        """
        n = 100
        p = 10
        K = make_sparse_spd_matrix(p)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        l_max = np.max(np.abs(np.linalg.inv(np.cov(X.T)) @ np.ones(p)))
        est = d_trace.DTrace(l_max)
        est.fit(X)
        # Check there is no NaNs
        self.assertFalse(np.all(np.isnan(est.precision_)))

        # Check if the off-diagonals are all zero
        self.assertTrue(np.count_nonzero(est.precision_) == p)