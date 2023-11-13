import json
import processing
import similarity
import clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics import jaccard_score

# Read method data from json file
with open('./inputs/petclinic.json', 'r') as file: 
    data = json.load(file)

# Extract dataset from json
method_data_list = []
document = []
data_list = []

for item in data['Data']:
    method_name = item['MethodName']
    class_name = item['ClassName']
    method_calls = item['Methodscalled']
    variables = item['Variables']
    data = " ".join([method_name, class_name] + 
                    [call['MethodCalled'] for call in method_calls] + 
                    [var['VariableName'] for var in variables])
    document.append(data)
    data_list.append(data)
    method_data_list.append({'MethodName': method_name, 'ClassName': class_name, 
                             'MethodCalls': [call['MethodCalled'] for call in method_calls],
                             'Variables': [var['VariableName'] for var in variables]}) 

preprocessed_document = processing.preprocess_input(" ".join(document))

method_count = len(method_data_list)

# Compute structural similarity
class_dependency_matrix = similarity.get_class_dependency_matrix(method_count, method_data_list)
method_dependency_matrix = similarity.get_method_dependency_matrix(method_count, method_data_list)
structural_similarity_matrix = processing.add_matrices(method_count, class_dependency_matrix, 0.6, method_dependency_matrix, 0.4)

# Compute semantic similarity
semantic_similarity_matrix = similarity.get_semantic_similarity_matrix(preprocessed_document, method_count, method_data_list)

# Combine structural and semantic similarity matrices
similarity_matrix = processing.add_matrices(method_count, structural_similarity_matrix, 0.4, semantic_similarity_matrix, 0.6)

actual_petclinic_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6]

# print(similarity_matrix)
# Clustering algorithm
# print('Hierarchical Clustering')
# labels = clustering.hierarchical_clustering(processing.normalize_matrix(similarity_matrix))


# print('====================')
 
# print('DBSCAN Clustering')
# labels = clustering.dbscan_clustering(similarity_matrix)
# print('====================')

num_clusters = 6
# print('KMeans Clustering')
# clustering_algorithm = KMeans(n_clusters=num_clusters, init='k-means++')
# clustering_algorithm.fit(similarity_matrix)
# # clustering.print_labels(method_data_list, clustering_algorithm.labels_)
# jaccard_similarity = jaccard_score(actual_petclinic_labels, clustering_algorithm.labels_, average = None)
# for label, similarity in enumerate(jaccard_similarity):
#     print(f"Similarity for label {label}: {similarity}")

# print('====================')

print('Agglomerative Clustering')
for num_clusters in range(3, 10):
    clustering_algorithm = AgglomerativeClustering(n_clusters = num_clusters, metric = 'precomputed', linkage = 'complete')
    clustering_algorithm.fit(similarity_matrix)
    # clustering.print_labels(method_data_list, clustering_algorithm.labels_)
    jaccard_similarity = jaccard_score(actual_petclinic_labels, clustering_algorithm.labels_, average = None)
    sum = 0
    for label, similarity in enumerate(jaccard_similarity):
        print(f"Similarity for label {label}: {similarity}")
        sum = sum + similarity
    print(sum)

    print('====================')

# print('Spectral Clustering')
# spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', n_init=100)
# predicted_petclinic_labels = spectral.fit_predict(similarity_matrix)
# # clustering.print_labels(method_data_list, predicted_petclinic_labels)
# jaccard_similarity = jaccard_score(actual_petclinic_labels, predicted_petclinic_labels, average = None)
# for label, similarity in enumerate(jaccard_similarity): 
#     print(f"Similarity for label {label}: {similarity}")
# Apply PCA for dimensionality reduction
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(similarity_matrix)

# Visualize the reduced data
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('Data Visualization before Clustering')
# plt.show()

