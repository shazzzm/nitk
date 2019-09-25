"""
Compare the AUC for the various inference methods
"""
#from nitk import neighbourhood_selection, scio, methods, space
import nitk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLasso
from sklearn.datasets import make_sparse_spd_matrix
    

p = 100
n = 10
no_runs = 10

scio_auc = np.zeros(no_runs)
glasso_auc =  np.zeros(no_runs)
threshold_auc =  np.zeros(no_runs)
ns_auc =  np.zeros(no_runs)
space_auc = np.zeros(no_runs)
clime_auc = np.zeros(no_runs)

for i in range(no_runs):
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
        ns = nitk.NeighbourhoodSelection(l)
        ns.fit(X)
        tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, ns.precision_)
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
            tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, gl.precision_)
            glasso_tpr.append(tpr)
            glasso_fpr.append(fpr)
            glasso_precision.append(prec)
        except FloatingPointError as e:
            print(e)

    space_tpr = []
    space_fpr = []
    space_precision = []

    for l in ls:
        s = nitk.SPACE(l)
        s.fit(X)
        tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, s.precision_)
        space_tpr.append(tpr)
        space_fpr.append(fpr)
        space_precision.append(prec)

    scio_tpr = []
    scio_fpr = []
    scio_precision = []

    max_l = 1

    ls = np.logspace(np.log10(0.0001*lambda_max), np.log10(lambda_max))
    for l in ls:
        sc = nitk.SCIO(l)
        sc.fit(X)
        tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, sc.precision_)
        scio_tpr.append(tpr)
        scio_fpr.append(fpr)
        scio_precision.append(prec)

    clime_tpr = []
    clime_fpr = []
    clime_precision = []

    for l in ls:
        cl = nitk.CLIME(l)
        cl.fit(X)
        tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, cl.precision_)

        clime_tpr.append(tpr)
        clime_fpr.append(fpr)
        clime_precision.append(prec)

    threshold_tpr = []
    threshold_fpr = []
    threshold_precision = []

    for l in ls:
        te = nitk.ThresholdEstimator(l)
        te.fit(X)
        tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, te.covariance_)

        threshold_tpr.append(tpr)
        threshold_fpr.append(fpr)
        threshold_precision.append(prec)

    glasso_auc[i] = np.trapz(glasso_tpr, glasso_fpr)
    ns_auc[i] = np.trapz(neighbourhood_selection_tpr, neighbourhood_selection_fpr)
    scio_auc[i] = np.trapz(scio_tpr, scio_fpr)
    space_auc[i] = np.trapz(space_tpr, space_fpr)
    clime_auc[i] = np.trapz(clime_tpr, clime_fpr)
    threshold_auc[i] = np.trapz(threshold_tpr, threshold_fpr)

print("Glasso Mean AUC: %s ± %s" % (np.mean(glasso_auc), np.std(glasso_auc)))
print("Neighbourhood Selection Mean AUC: %s ± %s" % (np.mean(ns_auc), np.std(ns_auc)))
print("SCIO Mean AUC: %s ± %s" % (np.mean(scio_auc), np.std(scio_auc)))
print("SPACE Mean AUC: %s ± %s" % (np.mean(space_auc), np.std(space_auc)))
print("CLIME Mean AUC: %s ± %s" % (np.mean(clime_auc), np.std(clime_auc)))
print("Threshold Mean AUC: %s ± %s" % (np.mean(threshold_auc), np.std(threshold_auc)))

