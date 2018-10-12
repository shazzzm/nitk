import pca_network_inference as pni
import sklearn.datasets
import matrix_methods
import network_analysis as nta
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

no_runs = 10
num_thetas = 100
p = 50
number_samples = 2000
thetas = np.linspace(0, 1, num_thetas)
P = sklearn.datasets.make_sparse_spd_matrix(dim=p, alpha=0.8, smallest_coef=.4, largest_coef=.7,)
C = np.linalg.inv(P)
X = np.random.multivariate_normal(np.zeros(p), C, number_samples)
corr = np.corrcoef(X, rowvar=0)
emp_prec = np.linalg.inv(corr)
emp_prec = nta.precision_matrix_to_partial_corr(emp_prec)
P_true = np.copy(P)
P = matrix_methods.threshold_matrix(P, 0.0001)
np.fill_diagonal(P, 0)
np.fill_diagonal(corr, 0)

dense_pca_network = pni.pca_network_inference(X, 4)
#sparse_pca_network = pni.sparse_pca_network_inference(X, 5, 1)
np.fill_diagonal(corr, 0)

dense_pca_network_tpr = []
dense_pca_network_fpr = []

corr_network_tpr = []
corr_network_fpr = []

prec_network_tpr = []
prec_network_fpr = []

dense_pca_network_vals = dense_pca_network.flatten()
plt.figure()
plt.hist(dense_pca_network_vals)
plt.title("Dense PCA Network Values")

for th in thetas:
    th_dense_pca_network = matrix_methods.threshold_matrix(np.copy(dense_pca_network), th)
    accuracy = nta.network_accuracy(th_dense_pca_network, P)
    tpr, fpr, precision = nta.calculate_roc_pr_curve(accuracy)
    dense_pca_network_tpr.append(tpr)
    dense_pca_network_fpr.append(fpr)

    th_corr_network = matrix_methods.threshold_matrix(np.copy(corr), th)
    accuracy = nta.network_accuracy(th_corr_network, P)
    tpr, fpr, precision = nta.calculate_roc_pr_curve(accuracy)
    corr_network_tpr.append(tpr)
    corr_network_fpr.append(fpr)

    th_prec_network = matrix_methods.threshold_matrix(np.copy(emp_prec), th)
    accuracy = nta.network_accuracy(th_prec_network, P)
    tpr, fpr, precision = nta.calculate_roc_pr_curve(accuracy)
    prec_network_tpr.append(tpr)
    prec_network_fpr.append(fpr)

plt.figure()
plt.plot(dense_pca_network_fpr, dense_pca_network_tpr, '--o', label="PCA Network")
plt.plot(corr_network_fpr, corr_network_tpr, '--o', label="Corr network")
plt.plot(prec_network_fpr, prec_network_tpr, '--o', label="Prec network")

plt.legend(loc='upper left')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()