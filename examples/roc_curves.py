"""
Compare the AUC for the various inference methods
"""
from nitk import neighbourhood_selection, scio, methods, space
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLasso
from sklearn.datasets import make_sparse_spd_matrix
    

p = 50
n = 20
K = make_sparse_spd_matrix(p)
C = np.linalg.inv(K)
X = np.random.multivariate_normal(np.zeros(p), C, n)
S = np.cov(X.T)
offdiag_ind = ~np.eye(p, dtype=bool)
lambda_max = np.abs(S[offdiag_ind]).max()
neighbourhood_selection_tpr = []
neighbourhood_selection_fpr = []
neighbourhood_selection_precision = []

ls = np.logspace(np.log10(0.001*lambda_max), np.log10(lambda_max))

for l in ls:
    ns = neighbourhood_selection.NeighbourhoodSelection(l)
    ns.fit(X)
    tpr, fpr, prec = methods.calculate_matrix_accuracy(K, ns.precision_)
    neighbourhood_selection_tpr.append(tpr)
    neighbourhood_selection_fpr.append(fpr)
    neighbourhood_selection_precision.append(prec)

glasso_tpr = []
glasso_fpr = []
glasso_precision = []

for l in ls:
    try:
        gl = GraphicalLasso(l)
        gl.fit(X)
        tpr, fpr, prec = methods.calculate_matrix_accuracy(K, gl.precision_)
        glasso_tpr.append(tpr)
        glasso_fpr.append(fpr)
        glasso_precision.append(prec)
    except FloatingPointError as e:
        print(e)

space_tpr = []
space_fpr = []
space_precision = []

for l in ls:
    s = space.SPACE(l)
    s.fit(X)
    tpr, fpr, prec = methods.calculate_matrix_accuracy(K, s.precision_)
    space_tpr.append(tpr)
    space_fpr.append(fpr)
    space_precision.append(prec)

scio_tpr = []
scio_fpr = []
scio_precision = []

max_l = 1

ls = np.logspace(np.log10(0.0001*lambda_max), np.log10(lambda_max))
for l in ls:
    sc = scio.SCIO(l)
    sc.fit(X)
    tpr, fpr, prec = methods.calculate_matrix_accuracy(K, sc.precision_)
    scio_tpr.append(tpr)
    scio_fpr.append(fpr)
    scio_precision.append(prec)


plt.figure()
plt.scatter(glasso_fpr, glasso_tpr)
plt.title("Glasso ROC")

print("Glasso AUC:")
print(np.trapz(glasso_tpr, glasso_fpr))

plt.figure()
plt.scatter(neighbourhood_selection_fpr, neighbourhood_selection_tpr)
plt.title("NS ROC")
print("neighbourhood selection AUC:")
print(np.trapz(neighbourhood_selection_tpr, neighbourhood_selection_fpr))

plt.figure()
plt.scatter(scio_fpr, scio_tpr)
plt.title("SCIO ROC")
print("SCIO ROC:")
print(np.trapz(scio_tpr, scio_fpr))

plt.figure()
plt.scatter(space_fpr, space_tpr)
plt.title("SPACE ROC")
print("SPACE ROC:")
print(np.trapz(space_tpr, space_fpr))

plt.show()