import nitk
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_less
import unittest
from sklearn.datasets import make_sparse_spd_matrix
import numpy as np

class TestNeighbourhoodSelection(unittest.TestCase):
    def test_neighbourhood_selection(self):
        """
        Create a sparse matrix that we attempt to estimate
        """
        p = 10
        n = 200
        l = 0.5
        K = make_sparse_spd_matrix(p, 0.7)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        ns = nitk.NeighbourhoodSelection(l)
        ns.fit(X)

    def test_neighbourhood_selection_columnwise_cv(self):
        """
        Create a sparse matrix that we attempt to estimate using neighbourhood selection

        """
        p = 10
        n = 200
        l = 0.5
        K = make_sparse_spd_matrix(p, 0.7)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        ns = nitk.NeighbourhoodSelectionColumnwiseCV()
        ns.fit(X)
        
    def test_neighbourhood_selection_overall_cv(self):
        p = 10
        n = 200
        l = 0.5
        K = make_sparse_spd_matrix(p, 0.7)
        C = np.linalg.inv(K)
        X = np.random.multivariate_normal(np.zeros(p), C, n)
        ns = nitk.NeighbourhoodSelectionCV()
        ns.fit(X)