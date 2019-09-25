"""
Use the various paramter selection methods to estimate a
specific matrix
"""
import nitk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLassoCV
from sklearn.datasets import make_sparse_spd_matrix

p = 100
n = 20
K = make_sparse_spd_matrix(p)
C = np.linalg.inv(K)
X = np.random.multivariate_normal(np.zeros(p), C, n)

ns = nitk.NeighbourhoodSelectionColumnwiseCV()
ns.fit(X)
tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, ns.precision_)
ns_f1 = nitk.methods.calculate_f1_score(tpr, prec)

te = nitk.ThresholdEstimatorCV()
te.fit(X)
tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, te.covariance_)
te_f1 = nitk.methods.calculate_f1_score(tpr, prec)

sc = nitk.SCIOColumnwiseCV()
sc.fit(X)
tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, sc.precision_)
sc_f1 = nitk.methods.calculate_f1_score(tpr, prec)

gl = GraphicalLassoCV()
gl.fit(X)
tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, gl.precision_)
gl_f1 = nitk.methods.calculate_f1_score(tpr, prec)
print("Graphical Lasso: %s" % gl_f1)
print("Neighbourhood Selection: %s" % ns_f1)
print("Threshold Estimator: %s" % te_f1)
print("SCIO %s" % sc_f1)
