import numpy as np
from sklearn.covariance import LedoitWolf

def correlation_p_value(x, y, no_permutes = 100):
    normal_corr = np.corrcoef(x, y)[0, 1]

    correlation_values = np.zeros(no_permutes)

    for i in range(no_permutes):
         x_new = np.random.permutation(x)
         y_new = np.random.permutation(y)
         correlation_values[i] = np.abs(np.corrcoef(x_new, y_new)[0, 1])

    no_significant = np.sum(correlation_values > np.abs(normal_corr))
    return normal_corr, no_significant/no_permutes

def significant_correlation_matrix(X, significance_threshold = 0.05):
    """
    Creates a correlation matrix consisting only of significant values, all other values are set to 0
    """
    p = X.shape[1]
    output_corr = np.zeros((p, p))

    for i in range(p):
        for j in range(p):
            corr, p_value = correlation_p_value(X[:, i], X[:, j])
            if p_value < significance_threshold:
                output_corr[i, j] = corr

    return output_corr

def significant_partial_correlation_matrix(X, significance_threshold = 0.05):
    """
    Creates a partial correlation matrix consisting only of significant values, all other values are set to 0
    """
    partial_corr, p_values = partial_correlation_p_value(X)

    ind = p_values > significance_threshold
    partial_corr[ind] = 0

    return partial_corr

def precision_matrix_to_partial_corr(theta):
    """
    Turns a precision matrix into a partial correlation one
    """
    p = theta.shape[0]
    partial_corr = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            partial_corr[i, j] = -theta[i, j] / np.sqrt(theta[i, i] * theta[j, j])
    np.fill_diagonal(partial_corr, 1)
    return partial_corr

def partial_correlation_p_value(X, no_permutes = 100):
    lw = LedoitWolf()
    lw.fit(X)
    normal_partial_corr = precision_matrix_to_partial_corr(lw.precision_)
    p = normal_partial_corr.shape[0]
    partial_correlation_values = np.zeros((no_permutes, p, p))

    for i in range(no_permutes):
        X_new = np.random.permutation(X)
        lw = LedoitWolf()
        lw.fit(X_new)
        partial_correlation_values[i, :, :] = precision_matrix_to_partial_corr(lw.precision_)

    # Let's go through each value and look at it's p-value
    p_value_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            vals = partial_correlation_values[:, i, j].flatten()
            fraction_significant = np.sum(np.abs(vals) > np.abs(normal_partial_corr[i, j]))
            p_value_matrix[i, j] = fraction_significant/no_permutes
    
    return normal_partial_corr, p_value_matrix