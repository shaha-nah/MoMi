import re
import math
import numpy as np

def preprocess_input(words):
    # Split words based on camel case or Pascal case
    words = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', words)
    return ' '.join(words)

def add_matrices(size, matrix1, weight1, matrix2, weight2):
    matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = (weight1 * matrix1[i][j]) + (weight2 * matrix2[i][j])
    return matrix

def calculate_cosine_similarity(text1, text2):
    tokens1 = text1.split()
    tokens2 = text2.split()

    unique_tokens = set(tokens1 + tokens2)
    
    tf1 = {token: tokens1.count(token) for token in unique_tokens}
    tf2 = {token: tokens2.count(token) for token in unique_tokens}

    vector1 = [tf1[token] for token in unique_tokens]
    vector2 = [tf2[token] for token in unique_tokens]

    dot_product = sum(vector1[i] * vector2[i] for i in range(len(vector1)))

    magnitude1 = math.sqrt(sum(vector1[i] ** 2 for i in range(len(vector1))))
    magnitude2 = math.sqrt(sum(vector2[i] ** 2 for i in range(len(vector2))))

    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    return cosine_similarity

def normalize_matrix(matrix):
    method_count = len(matrix)
    min_value = np.min(matrix)
    max_value = np.max(matrix)

    normalized_matrix = [[0] * method_count for _ in range(method_count)]

    for i in range(method_count):
        for j in range(method_count):
            normalized_matrix[i][j] = (matrix[i][j] - min_value) / (max_value - min_value)
    return normalized_matrix     