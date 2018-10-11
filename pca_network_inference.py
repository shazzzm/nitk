import numpy as np
from sklearn.decomposition import TruncatedSVD, SparsePCA

def pca_network_inference(X, no_components=None):
    """
    Infers a network using normal PCA from dataset X
    """
    p = X.shape[1]
    if no_components is None:
        no_components = p
    svd = TruncatedSVD(n_components=no_components)
    lsa_matrix = svd.fit_transform(X.T)
    # TODO: This is not efficient. Fix
    word_similarity = np.zeros((p, p))
    for i,row in enumerate(lsa_matrix):
        for j, row_2 in enumerate(lsa_matrix):
            # Self links are a bit meaningless
            if i == j:
                continue
            dot = np.dot(row, row_2)
            dot /= (np.linalg.norm(row) * np.linalg.norm(row_2))
            word_similarity[i, j] = dot

    return word_similarity

def sparse_pca_network_inference(X, no_components=None, l=1):
    """
    Infers a network using sparse PCA from dataset X
    """
    p = X.shape[1]
    if no_components is None:
        no_components = p

    sparse_pca = SparsePCA(n_components=no_components,  alpha=l)
    sparse_pca_matrix = sparse_pca.fit_transform(X.T)
    print(sparse_pca_matrix.shape)
    sparse_similarity = np.zeros((p, p))
    for i,row in enumerate(sparse_pca_matrix):
        for j, row_2 in enumerate(sparse_pca_matrix):
            # Self links are a bit meaningless
            if i == j:
                continue
            dot = np.dot(row, row_2)
            row_mag = np.linalg.norm(row) 
            row_2_mag = np.linalg.norm(row_2)
            if row_mag == 0 or row_2_mag == 0:
                dot = 0
            else:
                dot /= (np.linalg.norm(row) * np.linalg.norm(row_2))
            sparse_similarity[i, j] = dot

    return sparse_similarity