import numpy as np
from nitk import l2_penalized_estimator
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_less
import unittest
from sklearn.datasets import make_spd_matrix

class TestL2Estimator(unittest.TestCase):

    def test_l2_estimator(self):
        """
        Creates a distribution and ensures the estimator returns something
        sensible
        """
        n = 100
        p = 20
        C = make_spd_matrix(p)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        est = l2_penalized_estimator.L2PenalizedEstimator(0.2)
        est.fit(X)
        # Check there is no NaNs
        self.assertFalse(np.all(np.isnan(est.precision_)))

        # Check the result is positive semi-definite
        eigs, _ = np.linalg.eig(est.precision_)
        self.assertTrue(np.all(eigs > 0))

    def test_l2_estimator_cv(self):
        """
        Tests the cross validation estimator to ensure it
        returns something sensible
        """
        n = 100
        p = 20
        C = make_spd_matrix(p)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        est = l2_penalized_estimator.L2PenalizedEstimatorCV()
        est.fit(X)
        # Check there is no NaNs
        self.assertFalse(np.all(np.isnan(est.precision_)))