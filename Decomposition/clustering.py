import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

def print_labels(method_data_list, labels):
    clustered_methods = {}

    for i, label in enumerate(labels):
        method_data = method_data_list[i]
        method_name = method_data['MethodName']
        
        if label not in clustered_methods:
            clustered_methods[label] = []
        
        clustered_methods[label].append(method_name)
    
    sorted_labels = sorted(clustered_methods.keys())
    for label in sorted_labels:
        methods = clustered_methods[label]
        print(f"Cluster {label}:", end = ' ')
        for method in methods:
            print(f"{method}", end = ' ')
        print() 
    print('==========================================')

def spectral_clustering(similarity_matrix, affinity_method='nearest_neighbors',
                         n_neighbors=20, gamma=0.1, n_init=100,
                         silhouette_threshold=0.2, use_elbow_method=False):
    # Convert the similarity matrix to a NumPy array
    similarity_matrix = np.array(similarity_matrix)

    # Normalize and reduce dimensionality using PCA
    normalized_similarity_matrix = StandardScaler().fit_transform(similarity_matrix)
    pca = PCA(n_components=0.95)
    normalized_similarity_matrix = pca.fit_transform(normalized_similarity_matrix)

    # Convert similarity matrix to an affinity matrix using the specified method and parameters
    if affinity_method == 'nearest_neighbors':
        # Use Nearest Neighbors to create an affinity matrix
        neighbors = NearestNeighbors(n_neighbors=min(n_neighbors, similarity_matrix.shape[0]))
        neighbors.fit(normalized_similarity_matrix)
        affinity_matrix = neighbors.kneighbors_graph(mode='distance').toarray()
        # Apply exponential kernel to the distance matrix
        affinity_matrix = np.exp(-affinity_matrix ** 2 / (2. * gamma ** 2))
    else:
        if affinity_method == 'rbf':
            # Use Gaussian kernel for RBF
            affinity_matrix = np.exp(-gamma * normalized_similarity_matrix ** 2)
        else:
            affinity_matrix = pairwise_kernels(normalized_similarity_matrix, metric=affinity_method)

    if use_elbow_method:
        return spectral_clustering_elbow_method(affinity_matrix, n_init=n_init)

    # Use Silhouette score method
    silhouette_scores = []
    for num_clusters in range(2, similarity_matrix.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=DataConversionWarning)

            spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_init=n_init)
            predicted_labels = spectral.fit_predict(affinity_matrix)

        silhouette = silhouette_score(affinity_matrix, predicted_labels)

        if silhouette < silhouette_threshold:
            break

        silhouette_scores.append((num_clusters, silhouette))

    if not silhouette_scores:
        # If no silhouette scores meet the threshold, choose a default number of clusters
        optimal_clusters = 2
    else:
        # Choose the number of clusters with the highest silhouette score
        optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]

    # Use the optimal number of clusters in the final SpectralClustering
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=DataConversionWarning)

        spectral = SpectralClustering(n_clusters=optimal_clusters, affinity='nearest_neighbors', n_init=n_init)
        predicted_labels = spectral.fit_predict(affinity_matrix)

    return predicted_labels

def spectral_clustering_elbow_method(affinity_matrix, n_init=100):
    explained_variance = []

    for num_clusters in range(2, affinity_matrix.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=DataConversionWarning)

            spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_init=n_init)
            predicted_labels = spectral.fit_predict(affinity_matrix)

        eigenvalues = np.linalg.eigvals(affinity_matrix)
        explained_variance.append((num_clusters, np.sum(eigenvalues.real)))

    # Choose the number of clusters based on the elbow point
    optimal_clusters = 2  # Default to 2 clusters if no clear elbow
    for i in range(1, len(explained_variance) - 1):
        slope_before = (explained_variance[i][1] - explained_variance[i-1][1])
        slope_after = (explained_variance[i+1][1] - explained_variance[i][1])
        if slope_before > slope_after:
            optimal_clusters = explained_variance[i][0]
            break

    # Use the optimal number of clusters in the final SpectralClustering
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=DataConversionWarning)

        spectral = SpectralClustering(n_clusters=optimal_clusters, affinity='nearest_neighbors', n_init=n_init)
        predicted_labels = spectral.fit_predict(affinity_matrix)

    return predicted_labels
