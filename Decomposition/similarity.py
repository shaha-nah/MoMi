import fasttext
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, pairwise_distances
from scipy.spatial.distance import cosine, , correlation, cityblock

def get_directed_graph_similarity_matrix(method_data_list):
    # Create a directed graph   
    G = nx.DiGraph()

    # Add nodes to the graph using method names
    for method_data in method_data_list:
        G.add_node(method_data['MethodName'])

    # Add edges to the graph based on method calls
    for method_data in method_data_list:
        method_name = method_data['MethodName']
        method_calls = method_data['MethodCalls']
        for call in method_calls:
            # Check if the called method is in the graph
            if call in G.nodes:
                G.add_edge(method_name, call)

    # Get the list of method names
    method_names = [method_data['MethodName'] for method_data in method_data_list]
    method_count = len(method_names)

    # Initialize the directed graph similarity matrix
    directed_graph_similarity_matrix = np.zeros((method_count, method_count))

    # Calculate the similarity based on shortest path length
    for i in range(method_count):
        for j in range(i, method_count):
            try:
                # Calculate shortest path length and update the similarity matrix
                path_similarity = nx.shortest_path_length(G, source=method_names[i], target=method_names[j])
                directed_graph_similarity_matrix[i, j] = directed_graph_similarity_matrix[j, i] = 1 / (1 + path_similarity)
            except nx.NetworkXNoPath:
                # If there is no path, set similarity to 0
                directed_graph_similarity_matrix[i, j] = directed_graph_similarity_matrix[j, i] = 0

    return np.array(directed_graph_similarity_matrix)

def get_weighted_method_similarity_matrix(method_count, method_data_list):
    # Initialize an empty matrix to store the weighted method similarity
    weighted_method_similarity_matrix = np.zeros((method_count, method_count))

    # Extract unique classes from method_data_list
    unique_classes = set(method_data['ClassName'] for method_data in method_data_list)

    # Create a dictionary to map class names to their indices in unique_classes
    class_index_map = {class_name: index for index, class_name in enumerate(unique_classes)}

    # Iterate over all pairs of methods
    for i in range(method_count):
        for j in range(method_count):
            class_i = method_data_list[i]['ClassName']
            class_j = method_data_list[j]['ClassName']

            # Get the indices of classes in unique_classes
            index_i = class_index_map[class_i]
            index_j = class_index_map[class_j]

            # Find the common calls betweent the two methods
            common_calls = set(method_data_list[i]['MethodCalls']) & set(method_data_list[j]['MethodCalls'])

            # Get the total number of calls for each methods
            total_calls_i = len(method_data_list[i]['MethodCalls'])
            total_calls_j = len(method_data_list[j]['MethodCalls'])

            # Calculate the weighted similarity and store it in the matrix
            if total_calls_i != 0 and total_calls_j != 0:
                weighted_method_similarity_matrix[i, j] = len(common_calls) / ((total_calls_i + total_calls_j) / 2)
            else:
                # If either method has no calls, set similarity to 0 to avoid division by zero
                weighted_method_similarity_matrix[i, j] = 0

    return np.array(weighted_method_similarity_matrix)

def get_word2vec_similarity_matrix(method_data_list, vector_size=100, window=5, min_count=1, epochs=10, similarity_metric = 'jaccard'):
    # Prepare a list of strings combining various information from method_data_list
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

    # Train Word2Vec model on the prepared data
    model = Word2Vec(sentences=method_info_list, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs)
    
    # Initialize a matrix to store Word2Vec similarity scores
    word2vec_similarity_matrix = np.zeros((len(method_data_list), len(method_data_list)))

    # Iterate over all pairs of methods
    for i in range(len(method_data_list)):
        for j in range(i, len(method_data_list)):
            # Get the vectors for the two methods
            vec1 = np.mean([model.wv[word] for word in method_info_list[i] if word in model.wv], axis=0)
            vec2 = np.mean([model.wv[word] for word in method_info_list[j] if word in model.wv], axis=0)
            
            # Calculate the similarity based on the specified metric
            if similarity_metric == 'cosine':
                similarity = 1 - cosine(vec1, vec2)
            elif similarity_metric == 'jaccard':
                intersection_size = len(set(vec1.nonzero()[0]).intersection(set(vec2.nonzero()[0])))
                union_size = len(set(vec1.nonzero()[0]).union(set(vec2.nonzero()[0])))
                similarity = intersection_size / union_size if union_size != 0 else 0
            elif similarity_metric == 'euclidean':
                similarity = 1 / (1 + euclidean(vec1, vec2))
            elif similarity_metric == 'correlation':
                similarity = correlation(vec1, vec2)
            elif similarity_metric == 'manhattan':
                similarity = 1 / (1 + cityblock(vec1, vec2))
            else:
                raise ValueError("Invalid similarity metric. Choose from 'cosine', 'jaccard', 'euclidean', 'correlation', or 'manhattan'.")
            
            word2vec_similarity_matrix[i, j] = word2vec_similarity_matrix[j, i] = similarity

    return word2vec_similarity_matrix

def get_bert_similarity_matrix(method_data_list, similarity_metric='jaccard'):
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Prepare a list of strings combining information from method_data_list
    method_info_list = [" ".join([
        item['MethodName'],
        item['ClassName'],
        " ".join(item['MethodCalls']),
        " ".join(item['Variables']),
        item['Label'],
        " ".join(item['Parameters']),
        item['MethodSourceCode'],
        item['Comments']
    ]) for item in method_data_list]

    # Tokenize and encode inputs using BERT tokenizer
    encoded_inputs = tokenizer(method_info_list, return_tensors='pt', padding=True, truncation=True)

    # Pass the inputs through the BERT model
    outputs = model(**encoded_inputs)

    # Use mean pooling to obtain embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1) 

    if similarity_metric == 'cosine':
        bert_similarity_matrix = cosine_similarity(embeddings.detach().numpy())
    elif similarity_metric == 'jaccard':
        bert_distance_matrix = pairwise_distances(embeddings.detach().numpy(), metric='jaccard')
        bert_similarity_matrix = 1 - bert_distance_matrix   
    elif similarity_metric == 'euclidean':
        bert_similarity_matrix = euclidean_distances(embeddings.detach().numpy())
    elif similarity_metric == 'manhattan':
        bert_distance_matrix = pairwise_distances(embeddings.detach().numpy(),metric='manhattan')
        bert_similarity_matrix = 1 / (1 + bert_distance_matrix)
    else:
        raise ValueError("Invalid similarity metric. Choose from 'cosine', 'jaccard', 'euclidean', 'correlation', or 'manhattan'.")

    return bert_similarity_matrix

def get_fasttext_similarity_matrix(method_data_list, model_path='cc.en.300.bin', similarity_metric = 'jaccard'):
    # Load FastText model
    model = fasttext.load_model(model_path)
    
    # Prepare a list of strings combining information from method_data_list
    method_info = [" ".join([
        item['MethodName'],
        item['ClassName'],
        " ".join(item['MethodCalls']),
        " ".join(item['Variables']),
        item['Label'],
        " ".join(item['Parameters']),
        item['MethodSourceCode'],
        item['Comments']
    ]) for item in method_data_list]

    # Get vectors for each method using FastText model
    method_vectors = np.array([model.get_sentence_vector(text) for text in method_info])

    # Calculate pairwise cosine similarity
    if similarity_metric == 'cosine':
        fasttext_similarity_matrix = cosine_similarity(method_vectors)
    elif similarity_metric == 'jaccard':
        fasttext_distance_matrix = pairwise_distances(method_vectors, metric = 'jaccard')
        fasttext_similarity_matrix = 1 - fasttext_distance_matrix
    elif similarity_metric == 'euclidean':
        fasttext_similarity_matrix = euclidean_distances(method_vectors)
    elif similarity_metric == 'correlation':
        fasttext_similarity_matrix = np.corrcoef(method_vectors)
    elif similarity_metric == 'manhattan':
        fasttext_distance_matrix = pairwise_distances(method_vectors, metric = 'manhattan')
        fasttext_similarity_matrix = 1 / (1 + fasttext_similarity_matrix)
    else:
        raise ValueError("Invalid similarity metric. Choose from 'cosine', 'jaccard', 'euclidean', 'correlation', or 'manhattan'.")

    return fasttext_similarity_matrix
