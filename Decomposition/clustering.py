import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import DBSCAN
import processing
from sklearn.metrics import jaccard_score

def hierarchical_clustering(similarity_matrix):
    actual_petclinic_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6]

    distance_matrix = 1 - np.array(similarity_matrix)
    linkage_matrix = linkage(distance_matrix, method='complete', metric='euclidean')
    dendrogram(linkage_matrix)
    threshold = 1

    for threshold in range(10):
        labels = fcluster(linkage_matrix, threshold, criterion='distance')

        jaccard_similarity = jaccard_score(actual_petclinic_labels, labels, average = None)
        for label, similarity in enumerate(jaccard_similarity):
            print(f"Similarity for label {label}: {similarity}")
        print('=====================================================')
    return labels

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

def dbscan_clustering(similarity_matrix):
    actual_petclinic_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6]

    for eps in range(1, 10):
        for ms in range(1, 10):
            dbscan = DBSCAN(eps = eps / 10, min_samples = ms)
            labels = dbscan.fit_predict(similarity_matrix)
            
            jaccard_similarity = jaccard_score(actual_petclinic_labels, labels, average = None)
            sum = 0
            for label, similarity in enumerate(jaccard_similarity):
                print(f"Similarity for label {label}: {similarity}")
                sum = sum + similarity
            print(sum)
            print('=====================================================')

    return labels