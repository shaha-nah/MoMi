import clustering
import json
import numpy as np
import processing
import similarity
import dummies
import sys

if len(sys.argv) != 2:
    print("Usage: python momi.py <jsonFilePath>")
    sys.exit(1)

jsonFilePath = sys.argv[1]

# Read data from json
with open(jsonFilePath, 'r') as file:
    data = json.load(file)

# Extract dataset from json
method_data_list = []
label_set = set()
class_calls = {}

for item in data['Data']:
    method_name = item['MethodName']
    class_name = item['ClassName']
    method_calls = item['Methodscalled']
    variables = item['Variables']
    ast_features = item['ASTFeatures']
    method_source_code = item['MethodSourceCode']
    label = item['Folder']
    parameters = item['Parameters']
    comments = item['Comments']

    method_data_list.append({'MethodName': method_name, 'ClassName': class_name,
                             'MethodCalls': [call['MethodCalled'] for call in method_calls],
                             'Variables': [var['VariableName'] for var in variables],
                             'Label': label,
                             'Parameters': parameters,
                             'MethodSourceCode': method_source_code,
                             'Comments': comments
                             })
    label_set.add(label)

    # Add method calls to class_calls dictionary
    if class_name not in class_calls:
        class_calls[class_name] = {}
    for call in method_calls:
        called_method = call['MethodCalled']
        class_calls[class_name][called_method] = class_calls[class_name].get(called_method, 0) + 1

actual_labels = []
for method_data in method_data_list:
    actual_labels.append(list(label_set).index(method_data['Label']))

method_count = len(method_data_list)

# Compute structural similarity matrices
# directed_graph_similarity_matrix = similarity.get_directed_graph_similarity_matrix(method_data_list)
# weighted_method_similarity_matrix = similarity.weighted_directed_graph_similarity_matrix(method_count, method_data_list)
# Replace with precomputed values
directed_graph_similarity_matrix = dummies.get_directed_graph_similarity_matrix()
weighted_method_similarity_matrix = dummies.get_weighted_method_similarity_matrix()

structural_matrices = [
    directed_graph_similarity_matrix,
    weighted_method_similarity_matrix
]
structural_matrices = [np.array(matrix) for matrix in structural_matrices]

# Compute semantic similarity matrices
preprocessed_method_data_list = []

for item in method_data_list:
    preprocessed_method_data = {
        'MethodName': processing.preprocess_text(item['MethodName']),
        'ClassName': processing.preprocess_text(item['ClassName']),
        'MethodCalls': [processing.preprocess_text(call) for call in item['MethodCalls']],
        'Variables': [processing.preprocess_text(var) for var in item['Variables']],
        'Label': processing.preprocess_text(item['Label']),
        'Parameters': [processing.preprocess_text(param) for param in item['Parameters']],
        'MethodSourceCode': processing.preprocess_text(item['MethodSourceCode']),
        'Comments': processing.preprocess_text(item['Comments'])
    }
    preprocessed_method_data_list.append(preprocessed_method_data)

# word2vec_similarity_matrix = similarity.get_word2vec_similarity_matrix(preprocessed_method_data_list)
# bert_similarity_matrix = similarity.get_bert_similarity_matrix(preprocessed_method_data_list)
# fasttext_similarity_matrix = similarity.get_fasttext_similarity_matrix(preprocessed_method_data_list)

# Replace with precomputed values
word2vec_similarity_matrix = dummies.get_word2vec_similarity_matrix_euclidean()
bert_similarity_matrix = dummies.get_bert_similarity_matrix_petclinic_euclidean()
fasttext_similarity_matrix = dummies.get_fasttext_petclinic_euclidean()

semantic_matrices = [
    word2vec_similarity_matrix,
    bert_similarity_matrix,
    fasttext_similarity_matrix
]
semantic_matrices = [np.array(matrix) for matrix in semantic_matrices]

# Find optimal weights
structural_weights, semantic_weights, similarity_weight = processing.get_optimized_weights(structural_matrices, semantic_matrices, actual_labels, method_data_list, 'SpectralClustering')
similarity_weights = [similarity_weight[0], 1 - similarity_weight[0]]

# Compute weighted sums 
structural_matrix = sum(weight * matrix for weight, matrix in zip(structural_weights, structural_matrices))
semantic_matrix = sum(weight * matrix for weight, matrix in zip(semantic_weights, semantic_matrices))

similarity_matrices = [ 
    structural_matrix,
    semantic_matrix
]

# Compute similarity matrix
similarity_matrix = sum(weight * matrix for weight, matrix in zip(similarity_weights, similarity_matrices))
similarity_matrix = processing.normalize_matrix(similarity_matrix)

# Cluster the similarity matrix
predicted_labels = clustering.spectral_clustering(similarity_matrix)
processing.print_methods_by_labels(method_data_list, predicted_labels, jsonFilePath)

# Compute precision
precision = processing.calculate_precision(actual_labels, predicted_labels, method_data_list)
print(precision)

# Compute Structural Modularity
structural_modularity = processing.calculate_structural_modularity(predicted_labels, method_data_list)
print(structural_modularity)

# Compute Interface Number
interface_number = processing.calculate_interface_number(class_calls, method_data_list)
print(interface_number)

# Compute Non Extreme Distribution
non_extreme_distribution = processing.calculate_non_extreme_distribution(method_data_list)
print(non_extreme_distribution)

# Compute Inter Call Percentage
inter_call_percentage = processing.calculate_inter_call_percentage(class_calls, method_data_list)
print(inter_call_percentage)