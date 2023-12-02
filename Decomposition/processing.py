import clustering
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import numpy as np
from pyswarm import pso
from random import uniform
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

def normalize_matrix(matrix):
    min_value = np.nanmin(matrix)
    max_value = np.nanmax(matrix)

    # Handle division by zero or invalid values
    if min_value == max_value or not np.isfinite(min_value) or not np.isfinite(max_value):
        return matrix

    normalized_matrix = (matrix - min_value) / (max_value - min_value)

    return normalized_matrix  

def calculate_structural_modularity(predicted_labels, method_data_list):
    # Convert actual_labels and predicted_labels to NumPy arrays for easier indexing
    predicted_labels = np.array(predicted_labels)

    # Get the number of microservices (clusters)
    num_microservices = max(predicted_labels) + 1

    # Initialize arrays for microservices cohesion and coupling
    microservices_cohesion = np.zeros(num_microservices)
    microservices_coupling = np.zeros((num_microservices, num_microservices))

    # Loop through each method in the dataset
    for i, method_data_i in enumerate(method_data_list):
        label_i = predicted_labels[i]

        for j, method_data_j in enumerate(method_data_list):
            label_j = predicted_labels[j]

            if label_i == label_j:
                # Cohesion within the same microservice
                microservices_cohesion[label_i] += method_data_i['MethodCalls'].count(method_data_j['MethodName'])
            else:
                # Coupling between different microservices
                microservices_coupling[label_i, label_j] += method_data_i['MethodCalls'].count(method_data_j['MethodName'])

    total_methods = len(method_data_list)

    structural_modularity = 0

    for i in range(num_microservices):
        microservice_size = np.sum(predicted_labels == i)

        # Handling division by zero
        cohesion_term = np.nan_to_num(np.divide(microservices_cohesion[i], (microservice_size ** 2)))
        coupling_term = np.nan_to_num(np.divide(microservices_coupling[i, :].sum(), (2 * microservice_size * total_methods)))

        structural_modularity += cohesion_term - coupling_term

    structural_modularity /= num_microservices

    return structural_modularity

def calculate_interface_number(class_calls, method_data_list):
    num_microservices = len(class_calls)
    total_interfaces = 0

    # Calculate total number of interfaces
    for i, method_data_i in enumerate(method_data_list):
        interface_classes = set()
        for j, method_data_j in enumerate(method_data_list):
            if i != j:
                class_i = method_data_i['ClassName']
                class_j = method_data_j['ClassName']
                if class_calls.get(class_i, {}).get(class_j, 0) > 0:
                    interface_classes.add(class_i)

        total_interfaces += len(interface_classes)

    # Calculate IFN
    if num_microservices == 0:
        ifn = 0
    else:
        ifn = 1 / num_microservices * total_interfaces

    return ifn

def calculate_non_extreme_distribution(method_data_list):
    num_microservices = len(method_data_list)
    extreme_microservices = sum(1 for method_data in method_data_list if 5 < len(method_data['MethodCalls']) < 20)

    # Calculate NED
    if num_microservices == 0:
        ned = 0
    else:
        ned = 1 - (extreme_microservices / num_microservices)

    return ned

def calculate_inter_call_percentage(class_calls, method_data_list):
    num_microservices = len(class_calls)
    total_interactions = 0
    total_possible_interactions = 0

    # Calculate total interactions and possible interactions
    for i, method_data_i in enumerate(method_data_list):
        for j, method_data_j in enumerate(method_data_list):
            if i != j:
                class_i = method_data_i['ClassName']
                class_j = method_data_j['ClassName']
                interactions = class_calls.get(class_i, {}).get(class_j, 0)
                total_interactions += interactions
                total_possible_interactions += 1

    # Calculate ICP
    if total_possible_interactions == 0:
        icp = 0
    else:
        icp = total_interactions / total_possible_interactions

    return icp

def calculate_precision(actual_labels, predicted_labels, method_data_list):
    precision_sum = 0.0
    method_count = len(method_data_list)

    unique_actual_labels = set(actual_labels)

    for actual_label in unique_actual_labels:
        actual_indices = [i for i, x in enumerate(actual_labels) if x == actual_label]
        predicted_indices = [i for i, x in enumerate(predicted_labels) if x == actual_label]

        actual_microservice_methods = [
            method_data_list[i]['MethodName']
            for i in actual_indices
        ]

        predicted_microservice_methods = [
            method_data_list[i]['MethodName']
            for i in predicted_indices
        ]

        intersection_count = len(set(actual_microservice_methods) & set(predicted_microservice_methods))
        precision_sum += intersection_count / len(actual_microservice_methods)

    precision = precision_sum / len(unique_actual_labels)
    return precision

def print_methods_by_labels(method_data_list, labels):
    unique_labels = set(labels)

    for label in unique_labels:
        print(f"Methods for Label {label}:")
        label_indices = [i for i, x in enumerate(labels) if x == label]

        for index in label_indices:
            method_data = method_data_list[index]
            method_name = method_data['MethodName']
            class_name = method_data['ClassName']
            print(f"  Class: {class_name}, Method: {method_name}")

        print("\n")

def get_optimized_weights_pyswarm(structural_matrices, semantic_matrices, actual_labels, method_data_list):
    # Objective function to minimize
    def objective_function(weights, structural_matrices, semantic_matrices, actual_labels, method_data_list):
        structural_weights, semantic_weights, similarity_weight = np.split(weights, [len(structural_matrices), len(structural_matrices) + len(semantic_matrices)])
    
        structural_matrix = sum(weight * matrix for weight, matrix in zip(structural_weights, structural_matrices))
        semantic_matrix = sum(weight * matrix for weight, matrix in zip(semantic_weights, semantic_matrices))
    
        similarity_matrix = similarity_weight * structural_matrix + (1 - similarity_weight) * semantic_matrix
    
        # Impute NaN values
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        similarity_matrix = imputer.fit_transform(similarity_matrix) 
    
        # Check for NaN values after imputation
        if np.isnan(similarity_matrix).any():
            print("NaN values still present after imputation. Check the imputation strategy.")
            return 1  # Return a high objective value to indicate failure
    
        # Check for infinite values and replace them with a large number
        structural_matrix[~np.isfinite(structural_matrix)] = 1e6
        semantic_matrix[~np.isfinite(semantic_matrix)] = 1e6

        # Normalize matrices
        structural_matrix = normalize_matrix(structural_matrix)
        semantic_matrix = normalize_matrix(semantic_matrix)

        # Check for NaN values after normalization
        if np.isnan(structural_matrix).any() or np.isnan(semantic_matrix).any():
            print("NaN values still present after normalization. Check the matrices.")
            return 1  # Return a high objective value to indicate failure

        # Assuming everything went well, return the objective value
        precision = calculate_precision(actual_labels, clustering.spectral_clustering(similarity_matrix), method_data_list)
        return -precision  # Minimize the negative of precision

    # Initial guess for weights
    initial_weights = np.concatenate([0.5 * np.ones(len(structural_matrices)), 0.5 * np.ones(len(semantic_matrices)), [0.5]])

    # Length of the bounds
    n_weights = len(initial_weights)

    # Bounds for each weight
    lb = [0] * n_weights
    ub = [1] * n_weights

    # Run PSO
    best_weights, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=50, args=(structural_matrices, semantic_matrices, actual_labels, method_data_list))

    # Extract individual weights
    structural_weights, semantic_weights, similarity_weight = np.split(best_weights, [len(structural_matrices), len(structural_matrices) + len(semantic_matrices)])
    return structural_weights, semantic_weights, similarity_weight

def objective_function(weights, structural_matrices, semantic_matrices, actual_labels, method_data_list):
    structural_weights, semantic_weights, similarity_weight = np.split(weights, [len(structural_matrices), len(structural_matrices) + len(semantic_matrices)])
    
    structural_matrix = sum(weight * matrix for weight, matrix in zip(structural_weights, structural_matrices))
    semantic_matrix = sum(weight * matrix for weight, matrix in zip(semantic_weights, semantic_matrices))
    
    similarity_matrix = similarity_weight * structural_matrix + (1 - similarity_weight) * semantic_matrix
    
    # Impute NaN values
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    similarity_matrix = imputer.fit_transform(similarity_matrix) 

    # Check for NaN values after imputation
    if np.isnan(similarity_matrix).any():
        print("NaN values still present after imputation. Check the imputation strategy.")
        return 1  # Return a high objective value to indicate failure

    # Check for infinite values and replace them with a large number
    structural_matrix[~np.isfinite(structural_matrix)] = 1e6
    semantic_matrix[~np.isfinite(semantic_matrix)] = 1e6

    # Normalize matrices
    structural_matrix = normalize_matrix(structural_matrix)
    semantic_matrix = normalize_matrix(semantic_matrix)

    # Check for NaN values after normalization
    if np.isnan(structural_matrix).any() or np.isnan(semantic_matrix).any():
        print("NaN values still present after normalization. Check the matrices.")
        return 1  # Return a high objective value to indicate failure

    # Assuming everything went well, return the objective value
    precision = calculate_precision(actual_labels, clustering.spectral_clustering(similarity_matrix), method_data_list)
    return -precision  # Minimize the negative of precision

def get_optimized_weights(structural_matrices, semantic_matrices, actual_labels, method_data_list):
    # Objective function wrapper for Nelder-Mead
    def objective_function_nm(weights, structural_matrices, semantic_matrices, actual_labels, method_data_list):
        return -objective_function(weights, structural_matrices, semantic_matrices, actual_labels, method_data_list)

    # Initial guess for weights
    initial_weights = np.concatenate([0.5 * np.ones(len(structural_matrices)), 0.5 * np.ones(len(semantic_matrices)), [0.5]])

    # Run Nelder-Mead
    result = minimize(
        objective_function_nm,
        initial_weights,
        args=(structural_matrices, semantic_matrices, actual_labels, method_data_list),
        method='Nelder-Mead',
        options={'maxiter': 10}  # Adjust the maximum number of iterations as needed
    )

    # Extract individual weights
    optimized_weights = result.x
    structural_weights, semantic_weights, similarity_weight = np.split(optimized_weights, [len(structural_matrices), len(structural_matrices) + len(semantic_matrices)])
    return structural_weights, semantic_weights, similarity_weight