import fasttext
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import euclidean_distances
from transformers import BertTokenizer, BertModel

def get_directed_graph_similarity_matrix(method_data_list):
    G = nx.DiGraph()

    for method_data in method_data_list:
        G.add_node(method_data['MethodName'])

    for method_data in method_data_list:
        method_name = method_data['MethodName']
        method_calls = method_data['MethodCalls']
        for call in method_calls:
            if call in G.nodes:
                G.add_edge(method_name, call)

    method_names = [method_data['MethodName'] for method_data in method_data_list]
    num_methods = len(method_names)
    directed_graph_similarity_matrix = np.zeros((num_methods, num_methods))

    for i in range(num_methods):
        for j in range(i, num_methods):
            try:
                path_similarity = nx.shortest_path_length(G, source=method_names[i], target=method_names[j])
                directed_graph_similarity_matrix[i, j] = directed_graph_similarity_matrix[j, i] = 1 / (1 + path_similarity)
            except nx.NetworkXNoPath:
                directed_graph_similarity_matrix[i, j] = directed_graph_similarity_matrix[j, i] = 0

    np.set_printoptions(threshold=np.inf, precision=2, suppress=True)

    # Print the entire matrix in the desired format
    print("[", end="")
    for row in directed_graph_similarity_matrix:
        print("[", end="")
        for value in row:
            print(f"{value:.2f}, ", end="")
        print("],")
    print("]")

    # Reset printing options to default
    np.set_printoptions()

    return np.array(directed_graph_similarity_matrix)

def get_weighted_method_similarity_matrix(method_count, method_data_list, class_calls):
    weighted_method_similarity_matrix = np.zeros((method_count, method_count))

    # Extract unique classes from class_calls
    unique_classes = list(class_calls.keys())

    # Create a dictionary to map class names to their indices in unique_classes
    class_index_map = {class_name: index for index, class_name in enumerate(unique_classes)}

    for i in range(method_count):
        for j in range(method_count):
            class_i = method_data_list[i]['ClassName']
            class_j = method_data_list[j]['ClassName']

            # Get the indices of classes in unique_classes
            index_i = class_index_map[class_i]
            index_j = class_index_map[class_j]

            common_calls = set(class_calls[unique_classes[index_i]].keys()) & set(class_calls[unique_classes[index_j]].keys())
            total_calls_i = sum(class_calls[unique_classes[index_i]].values())
            total_calls_j = sum(class_calls[unique_classes[index_j]].values())

            if total_calls_i != 0 and total_calls_j != 0:
                weighted_method_similarity_matrix[i, j] = len(common_calls) / ((total_calls_i + total_calls_j) / 2)
            else:
                weighted_method_similarity_matrix[i, j] = 0

        print("[", end="")
        for row in weighted_method_similarity_matrix:
            print("[", end="")
            for value in row:
                print(f"{value:.2f}, ", end="")
            print("],")
        print("]")

    return np.array(weighted_method_similarity_matrix)

def get_word2vec_similarity_matrix(method_data_list, vector_size=100, window=5, min_count=1, epochs=10):
    method_info_list = [" ".join([
        item['MethodName'],
        item['ClassName'],
        " ".join(item['MethodCalls']),
        " ".join(item['Variables']),
        item['Label'],
        " ".join(item['Parameters']),
        item['MethodSourceCode'],
        item['Comments']
    ]).split() for item in method_data_list]

    model = Word2Vec(sentences=method_info_list, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs)
    
    word2vec_similarity_matrix = np.zeros((len(method_data_list), len(method_data_list)))

    for i in range(len(method_data_list)):
        for j in range(i, len(method_data_list)):
            vec1 = np.mean([model.wv[word] for word in method_info_list[i] if word in model.wv], axis=0)
            vec2 = np.mean([model.wv[word] for word in method_info_list[j] if word in model.wv], axis=0)
            
            intersection_size = len(set(vec1.nonzero()[0]).intersection(set(vec2.nonzero()[0])))
            union_size = len(set(vec1.nonzero()[0]).union(set(vec2.nonzero()[0])))
            similarity = intersection_size / union_size if union_size != 0 else 0
            
            word2vec_similarity_matrix[i, j] = word2vec_similarity_matrix[j, i] = similarity

    np.set_printoptions(threshold=np.inf, precision=2, suppress=True)

    # Print the entire matrix in the desired format
    print("[", end="")
    for row in word2vec_similarity_matrix:
        print("[", end="")
        for value in row:
            print(f"{value:.2f}, ", end="")
        print("],")
    print("]")

    # Reset printing options to default
    np.set_printoptions()

    return word2vec_similarity_matrix

def get_bert_similarity_matrix(method_data_list):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    method_info_list = [" ".join([
        item['MethodName'],
        item['ClassName']
        " ".join(item['MethodCalls']),
        " ".join(item['Variables']),
        item['Label'],
        " ".join(item['Parameters']),
        item['MethodSourceCode'],
        item['Comments']
    ]) for item in method_data_list]

    encoded_inputs = tokenizer(method_info_list, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**encoded_inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling for simplicity
    bert_similarity_matrix = euclidean_distances(embeddings.detach().numpy())

    np.set_printoptions(threshold=np.inf, precision=2, suppress=True)

    # Print the entire matrix in the desired format
    print("[", end="")
    for row in bert_similarity_matrix:
        print("[", end="")
        for value in row:
            print(f"{value:.2f}, ", end="")
        print("],")
    print("]")

    # Reset printing options to default
    np.set_printoptions()

    return bert_similarity_matrix

def get_fasttext_similarity_matrix(method_data_list, model_path='cc.en.300.bin'):
    model = fasttext.load_model(model_path)
    
    method_info = [" ".join([
        item['MethodName'],
        item['ClassName']
        " ".join(item['MethodCalls']),
        " ".join(item['Variables']),
        item['Label'],
        " ".join(item['Parameters']),
        item['MethodSourceCode'],
        item['Comments']
    ]) for item in method_data_list]

    method_vectors = np.array([model.get_sentence_vector(text) for text in method_info])

    # Calculate pairwise cosine similarity
    fasttext_similarity_matrix = euclidean_distances(method_vectors)

    print("[", end="")
    for row in fasttext_similarity_matrix:
        print("[", end="")
        for value in row:
            print(f"{value:.2f}, ", end="")
        print("],")
    print("]")

    return fasttext_similarity_matrix
