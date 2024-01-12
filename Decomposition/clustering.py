import numpy as np
from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift, SpectralClustering
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings

def spectral_clustering(similarity_matrix, affinity_method='nearest_neighbors',
                         n_neighbors=20, gamma=0.1, n_init=100,
                         silhouette_threshold=0.2):
    
    # Normalize
    normalized_similarity_matrix = StandardScaler().fit_transform(similarity_matrix)
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

    # Initialize variables for the combined method
    silhouette_scores = []
    explained_variance = []

    optimal_clusters_elbow = None  # Initialize optimal_clusters_elbow outside the loop

    for num_clusters in range(2, similarity_matrix.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=DataConversionWarning)

            spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_init=n_init)
            predicted_labels = spectral.fit_predict(affinity_matrix)

        silhouette = silhouette_score(affinity_matrix, predicted_labels)

        eigenvalues = np.linalg.eigvals(affinity_matrix)
        explained_variance.append((num_clusters, np.sum(eigenvalues.real)))
        silhouette_scores.append((num_clusters, silhouette))

        if silhouette < silhouette_threshold:
            break

    if not silhouette_scores:
        # If no silhouette scores meet the threshold, choose a default number of clusters
        optimal_clusters = 2
    else:
        # Choose the number of clusters with the highest silhouette score
        optimal_clusters_silhouette = max(silhouette_scores, key=lambda x: x[1])[0]

        # Use the optimal number of clusters from Silhouette Score as a starting point
        optimal_clusters = optimal_clusters_silhouette

        # Find the number of clusters based on the elbow point in explained variance
        for i in range(1, len(explained_variance) - 1):
            slope_before = (explained_variance[i][1] - explained_variance[i - 1][1])
            slope_after = (explained_variance[i + 1][1] - explained_variance[i][1])
            if slope_before > slope_after:
                optimal_clusters_elbow = explained_variance[i][0]
                break

        # Choose the number of clusters that maximizes the Silhouette Score within a range of the elbow
        optimal_clusters = max(optimal_clusters_silhouette, optimal_clusters_elbow) if optimal_clusters_elbow is not None else optimal_clusters_silhouette

    # Use the optimal number of clusters in the final SpectralClustering
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=DataConversionWarning)

        spectral = SpectralClustering(n_clusters=optimal_clusters, affinity='nearest_neighbors', n_init=n_init)
        predicted_labels = spectral.fit_predict(affinity_matrix)

    return predicted_labels

def dbscan_clustering(similarity_matrix, max_iterations=10):
    distance_matrix = np.maximum(0, 1 - np.array(similarity_matrix))
    np.fill_diagonal(distance_matrix, 0)

    best_clusters = None
    best_silhouette = -1

    for _ in range(max_iterations):
        silhouette_scores = []
        eps_values = np.arange(0.1, 1.0, 0.1)
        min_samples_values = range(2, 10)

        for eps_val in eps_values:
            for min_samples_val in min_samples_values:
                dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val, metric='precomputed')
                clusters = dbscan.fit_predict(distance_matrix)

                # Check if there is a reasonable number of labels assigned by DBSCAN
                unique_labels = np.unique(clusters)
                if len(unique_labels) > 1:
                    silhouette = silhouette_score(distance_matrix, clusters, metric='precomputed')
                    silhouette_scores.append((eps_val, min_samples_val, silhouette))

        if not silhouette_scores:
            raise ValueError("DBSCAN failed to find a suitable solution. Adjust parameters or use a different method.")

        # Find the optimal parameters based on Silhouette Score
        optimal_params_silhouette = max(silhouette_scores, key=lambda x: x[2])
        optimal_eps_silhouette, optimal_min_samples_silhouette, optimal_silhouette = optimal_params_silhouette

        # Use the optimal parameters as a starting point for Elbow method
        optimal_eps = optimal_eps_silhouette
        optimal_min_samples = optimal_min_samples_silhouette

        # Find the optimal parameters based on Elbow Method
        explained_variance = []
        for num_clusters in range(2, len(unique_labels)):
            dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples, metric='precomputed')
            clusters = dbscan.fit_predict(distance_matrix)

            eigenvalues = np.linalg.eigvals(distance_matrix)
            explained_variance.append((num_clusters, np.sum(eigenvalues.real)))

        # Choose the number of clusters based on the elbow point
        optimal_clusers_elbow = 2 # Default to 2 clusters if no clear elbow
        for i in range(1, len(explained_variance) - 1):
            slope_before = (explained_variance[i][1] - explained_variance[i-1][1])
            slope_after = (explained_variance[i+1][1] - explained_variance[i][1])
            if slope_before > slope_after:
                optimal_clusters_elbow = explained_variance[i][0]
                break

        # Choose the larger of the two optimal cluster numbers
        optimal_clusters = max(optimal_clusters_elbow, len(unique_labels))
        
        if optimal_silhouette > best_silhouette:
            best_silhouette = optimal_silhouette
            dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples, metric='precomputed')
            best_clusters = dbscan.fit_predict(distance_matrix)

    if best_clusters is None:
        raise ValueError(f"DBSCAN failed to find a suitable solution after {max_iterations} iterations. Adjust parameters or use a different method.")

    return best_clusters

def mean_shift_clustering(similarity_matrix):
    distance_matrix = np.maximum(0, 1 - np.array(similarity_matrix))
    np.fill_diagonal(distance_matrix, 0)

    clustering = MeanShift().fit(distance_matrix)
    labels = clustering.labels_
    return labels

def affinity_propagation_clustering(similarity_matrix, damping=0.9):
    affinity_propagation = AffinityPropagation(affinity='precomputed', damping=damping)
    labels = affinity_propagation.fit_predict(similarity_matrix)
    return labels
