import numpy as np
from sklearn.decomposition import TruncatedSVD, SparsePCA

def cosine_similarity(M):
    """
    Calculcates the cosine similarity between the colunms of M
    """
    p = M.shape[1]
    word_similarity = np.zeros((p, p))
    for i,row in enumerate(M):
        for j, row_2 in enumerate(M):
            # Self links are a bit meaningless
            if i == j:
                continue
            dot = np.dot(row, row_2)
            dot /= (np.linalg.norm(row) * np.linalg.norm(row_2))
            word_similarity[i, j] = dot

    return word_similarity

def pca_network_inference(X, no_components=None):
    """
    Infers a network using normal PCA from dataset X
    """
    p = X.shape[1]
    if no_components is None:
        no_components = p
    svd = TruncatedSVD(n_components=no_components)
    lsa_matrix = svd.fit_transform(X.T)
    return cosine_similarity(lsa_matrix)

def pca_network_inference_with_p_values(X, no_components=None, no_permutes = 100):
    """
    Infers a network using PCA and calculates the p-value 
    """
    p = X.shape[1]
    if no_components is None:
        no_components = p
    svd = TruncatedSVD(n_components=no_components)
    lsa_matrix = svd.fit_transform(X.T)
    word_similarity = cosine_similarity(lsa_matrix.T)
    word_similarity_matrices = np.zeros((no_permutes, p, p))
    # Now checkout the p-vaues
    for i in range(no_permutes):
        lsa_matrix_new = np.random.permutation(lsa_matrix)
        new_word_similarity = cosine_similarity(lsa_matrix_new.T)
        word_similarity_matrices[i, :, :] = new_word_similarity
    # Let's go through each value and look at it's p-value
    p_value_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            vals = word_similarity_matrices[:, i, j]
            fraction_significant = np.sum(np.abs(vals) > np.abs(word_similarity[i, j]))
            p_value_matrix[i, j] = fraction_significant/no_permutes

    return word_similarity, p_value_matrix

def significant_pca_network_inference(X, no_components=None, significance_threshold=0.05):
    """
    Creates a pca word similarity matrix matrix consisting only of significant values, all other values are set to 0
    """
    word_similarity, p_values = pca_network_inference_with_p_values(X, no_components)

    ind = p_values > significance_threshold
    word_similarity[ind] = 0

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