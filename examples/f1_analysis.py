"""
Use the various paramter selection methods to estimate a
specific matrix
"""
import nitk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLassoCV
from sklearn.datasets import make_sparse_spd_matrix
import networkx as nx

p = 50
n = 10
no_runs = 5

network_structure = "uniform"
network_structure = "power law"

# Whether we add noise to the system
noise = True

# Make the system heavy tailed
lognormal = False

# Add some outliers to the system
num_outliers = 0

ns_f1 = np.zeros(no_runs)
ts_f1 = np.zeros(no_runs)
sc_f1 = np.zeros(no_runs)
gl_f1 = np.zeros(no_runs)
sli_f1 = np.zeros(no_runs)
cli_f1 = np.zeros(no_runs)
space_f1 = np.zeros(no_runs)

for i in range(no_runs):
    if network_structure == "uniform":
        K = make_sparse_spd_matrix(p, norm_diag=True)
    elif network_structure == "power law":
        L = nx.to_numpy_array(nx.barabasi_albert_graph(p, 5))
        np.fill_diagonal(L, 1)
        alpha = 0.8
        K = (1 - alpha) * np.eye(p) + alpha * L
    elif network_structure == "caveman":
        no_cliques = int(p/5)
        L = nx.to_numpy_array(nx.caveman_graph(no_cliques, 5))
        np.fill_diagonal(L, 1)
        alpha = 0.8
        K = (1 - alpha) * np.eye(p) + alpha * L

    C = np.linalg.inv(K)
    X = np.random.multivariate_normal(np.zeros(p), C, n)
    if noise:
        X += np.random.multivariate_normal(np.zeros(p), np.eye(p), n)

    if lognormal:
        X = np.exp(X)

    if num_outliers > 0:
        ind = np.random.choice(n, size=num_outliers)
        Y = np.random.multivariate_normal(np.ones(p), np.eye(p), num_outliers)
        X[ind] = Y

    ns = nitk.NeighbourhoodSelectionColumnwiseCV()
    ns.fit(X)
    tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, ns.precision_)
    ns_f1[i] = nitk.methods.calculate_f1_score(tpr, prec)

    te = nitk.ThresholdEstimatorCV()
    te.fit(X)
    tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, te.covariance_)
    ts_f1[i] = nitk.methods.calculate_f1_score(tpr, prec)

    sc = nitk.SCIOColumnwiseCV()
    sc.fit(X)
    tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, sc.precision_)
    sc_f1[i] = nitk.methods.calculate_f1_score(tpr, prec)

    gl = GraphicalLassoCV()
    gl.fit(X)
    tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, gl.precision_)
    gl_f1[i] = nitk.methods.calculate_f1_score(tpr, prec)

    sli = nitk.ScaledLassoInference()
    sli.fit(X)
    tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, sli.precision_)
    sli_f1[i] = nitk.methods.calculate_f1_score(tpr, prec)

    cli = nitk.CLIMECV()
    cli.fit(X)
    tpr, fpr, prec = nitk.methods.calculate_matrix_accuracy(K, cli.precision_)
    cli_f1[i] = nitk.methods.calculate_f1_score(tpr, prec)
print("Graphical Lasso & %s & %s & %6.3f $\pm$ %6.3f" % (p, n, np.mean(gl_f1), np.std(gl_f1)))
print("Neighbourhood Selection Columnwise & %s & %s & %6.3f $\pm$ %6.3f" % (p, n, np.mean(ns_f1), np.std(ns_f1)))
print("SCIO & %s & %s & %6.3f $\pm$ %6.3f" % (p, n, np.mean(sc_f1), np.std(sc_f1)))
print("Scaled Lasso & %s & %s & %6.3f $\pm$ %6.3f" % (p, n, np.mean(sli_f1), np.std(sli_f1)))
print("CLIME & %s & %s & %6.3f $\pm$ %6.3f" % (p, n, np.mean(cli_f1), np.std(cli_f1)))
print("Threshold Estimator & & %s & %s & %6.3f $\pm$ %6.3f" % (p, n, np.mean(ts_f1), np.std(ts_f1)))
