import numpy as np
from sklearn.covariance import LedoitWolf
from statsmodels.stats.multitest import multipletests
from . import methods
from sklearn.base import BaseEstimator

class CorrelationPermutationNetwork(BaseEstimator):
    """
    Estimates a correlation matrix and sets non-significant correlations to 0.

    P-values are calculated using a permutation procedure, where we see how likely 
    a correlation is to occur if the variables are randomly shifted

    Parameters
    -----------
    significance_threshold: cut off p-value to use
    no_permutes : number of times to permute the variables - higher will give a higher 
    reliability but will take longer
    """
    def __init__(self, significance_threshold=0.05, no_permutes=100):
        self.significance_threshold_ = significance_threshold
        self.no_permutes_ = no_permutes
        self.correlation_ = None
    
    def _correlation_p_value(self, x, y):
        """
        Calculates the correlation between two variables and uses permutation testing to get a p-value
        Parameters
        ----------
        x : array_like
            variable 1 - 1 by p array
        y : array_like
            variable 2 - 1 by p array
        no_permutes : (optional, int=100)
            Number of times to permute the variables to calculate the p-value

        Returns
        -------
        (correlation, p-value)
        Pearson correlation and the p-value calculated using permutations
        """
        normal_corr = np.corrcoef(x, y)[0, 1]

        correlation_values = np.zeros(self.no_permutes_)

        for i in range(self.no_permutes_):
            x_new = np.random.permutation(x)
            y_new = np.random.permutation(y)
            correlation_values[i] = np.abs(np.corrcoef(x_new, y_new)[0, 1])

        no_significant = np.sum(np.abs(correlation_values) > np.abs(normal_corr))
        return normal_corr, no_significant/self.no_permutes_

    def fit(self, X):
        """
        Creates a correlation matrix consisting only of significant values, all other values are set to 0
        Parameters
        ----------
        X : array_like
            n by p matrix containing the data

        Returns
        -------
        p by p correlation matrix
        Pearson correlation matrix with non-significant values set to 0
        """
        p = X.shape[1]
        output_corr = np.zeros((p, p))
        p_vals = np.zeros((p, p))
        for i in range(p):
            for j in range(i+1, p):
                corr, p_value = self._correlation_p_value(X[:, i], X[:, j])
                output_corr[i, j] = corr
                p_vals[i, j] = p_value

        # Run a test to correct for the multiple comparisons
        ind = np.triu_indices(p, k=1)
        p_vals_flat = p_vals[ind].flatten()
        _, corrected_vals, _, _ = multipletests(p_vals_flat)
        reject = corrected_vals > self.significance_threshold_
        corrs = output_corr[ind].flatten()
        corrs[reject] = 0
        output_corr[ind] = corrs
        np.fill_diagonal(output_corr, 1)
        self.correlation_ =  output_corr + output_corr.T

def significant_partial_correlation_matrix(X, significance_threshold = 0.05):
    """
    Creates a partial correlation matrix consisting only of significant values, all other values are set to 0

    Parameters
    ----------
    X : array_like
        n by p matrix containing the data
    significance_threshold : (optional, float=0.05)
        value at which we will consider a p-value significant

    Returns
    -------
    p by p correlation matrix
    Partial correlation matrix with non-signficant values set to 0
    """
    n, p = X.shape
    output_partial_corr = np.zeros((p, p))
    partial_corr, p_vals = partial_correlation_p_value(X)
    ind = np.triu_indices(p, k=1)
    p_vals_flat = p_vals[ind].flatten()
    par_corr_values = partial_corr[ind].flatten()
    _, corrected_vals, _, _ = multipletests(p_vals_flat)
    reject = corrected_vals > significance_threshold
    par_corr_values[reject] = 0
    output_partial_corr[ind] = par_corr_values
    np.fill_diagonal(output_partial_corr, 1)
    return output_partial_corr + output_partial_corr.T

def partial_correlation_p_value(X, no_permutes = 100):
    """
    Calculates a partial correlation matrix and appropriate p-values using permutations

    Parameters
    ----------
    X : array_like
        n by p matrix containing the data
    no_permutes : (optional, int=100)
        Number of times to permute the variables to calculate the p-value

    Returns
    -------
    (p by p correlation matrix, p-value matrix)
    """
    corr = np.corrcoef(X)
    prec = np.linalg.inv(corr)
    normal_partial_corr = methods.precision_matrix_to_partial_corr(prec)
    p = normal_partial_corr.shape[0]
    partial_correlation_values = np.zeros((no_permutes, p, p))

    for i in range(no_permutes):
        X_new = np.random.permutation(X)
        corr = np.corrcoef(X)
        prec = np.linalg.inv(corr)
        partial_correlation_values[i, :, :] = methods.precision_matrix_to_partial_corr(prec)

    # Let's go through each value and look at it's p-value
    p_value_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            vals = partial_correlation_values[:, i, j].flatten()
            fraction_significant = np.sum(np.abs(vals) > np.abs(normal_partial_corr[i, j]))
            p_value_matrix[i, j] = fraction_significant/no_permutes
    
    return normal_partial_corr, p_value_matrix