from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import processing
import Levenshtein
from difflib import SequenceMatcher
from collections import defaultdict
import gensim
import spacy

def get_class_dependency_matrix(method_count, method_data_list):
    class_names = list(set(entry['ClassName'] for entry in method_data_list))
    num_classes = len(class_names)
    
    method_indices = {method_data['MethodName']: idx for idx, method_data in enumerate(method_data_list)}
    
    interaction_matrix = defaultdict(lambda: defaultdict(int))
    
    for method_data in method_data_list:
        method_class = method_data['ClassName']
        method_calls = method_data['MethodCalls']
        
        class_idx = class_names.index(method_class)
        
        for method_call in method_calls:
            if method_call in method_indices:
                called_method_idx = method_indices[method_call]
                called_class = method_data_list[called_method_idx]['ClassName']
                called_class_idx = class_names.index(called_class)
                interaction_matrix[class_idx][called_class_idx] += 1
    
    similarity_matrix = np.zeros((method_count, method_count))
    
    for i in range(method_count):
        class_i_idx = class_names.index(method_data_list[i]['ClassName'])
        
        for j in range(method_count):
            class_j_idx = class_names.index(method_data_list[j]['ClassName'])
            
            if interaction_matrix[class_i_idx][class_j_idx] != 0:
                similarity_matrix[i][j] = interaction_matrix[class_i_idx][class_j_idx]
    
    return similarity_matrix.tolist()
    
def get_method_dependency_matrix(method_count, method_data_list):
    similarity_matrix = [[0] * method_count for _ in range(method_count)]

    for method_idx, method_data in enumerate(method_data_list):
        method_calls = method_data['MethodCalls']
        variables = method_data['Variables']

        for call in method_calls:
            called_method_idx = method_data_list.index(next(item for item in method_data_list if item['MethodName'] == call))
            similarity_matrix[method_idx][called_method_idx] += 1
        
        for variable in variables:
            for index, data in enumerate(method_data_list):
                if variable in data['Variables']:
                    variable_method_idx = index
                    similarity_matrix[method_idx][variable_method_idx] += 1
    
    for i in range(method_count):
        similarity_matrix[i][i] = 1
    
    return similarity_matrix

def get_domain_list(document, method_data_list):
    tfidf_vectorizer = TfidfVectorizer()
    lemmatizer = WordNetLemmatizer()

    lemmatized_document = ' '.join(lemmatizer.lemmatize(word) for word in document.split())

    tfidf_matrix = tfidf_vectorizer.fit_transform([lemmatized_document])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    word_scores = zip(feature_names, tfidf_matrix.toarray()[0])
    sorted_word_scores = sorted(word_scores, key=lambda x:x[1], reverse = True)

    frequent_words = []
    seen_words = set()
    for word, score in sorted_word_scores:
        word_singular = lemmatizer.lemmatize(word)
        if word_singular not in seen_words:
            frequent_words.append(word_singular)
            seen_words.add(word_singular)
    
    
    domain_list = filter(lambda keyword: any(keyword.lower() in data['MethodName'].lower() and 
                                        keyword.lower() in data['ClassName'].lower() and keyword.lower() in 
                                        ' '.join(data['Variables']).lower() for data in method_data_list), frequent_words)
    return list(domain_list)

def get_semantic_similarity(method_count, method_data_list):
    semantic_similarity_matrix = [[0] * method_count for _ in range(method_count)]
    for i, method_data_i in enumerate(method_data_list):
        for j, method_data_j in enumerate(method_data_list):
            # # Cosine similarity for method names
            # vectorizer = CountVectorizer().fit_transform([method_data_i['MethodName'], method_data_j['MethodName']])
            # similarity_method_vector = cosine_similarity(vectorizer[0], vectorizer[1]).flatten()
            # similarity_method = similarity_method_vector[0]

            # # Cosine similarity for class names
            # vectorizer = CountVectorizer().fit_transform([method_data_i['ClassName'], method_data_j['ClassName']])
            # similarity_class_vector = cosine_similarity(vectorizer[0], vectorizer[1]).flatten()
            # similarity_class = similarity_class_vector[0]

            # # Cosine similarity for method calls
            # if method_data_i['MethodCalls'] and method_data_j['MethodCalls']:
            #     vectorizer = CountVectorizer().fit_transform([', '.join(method_data_i['MethodCalls']), ', '.join(method_data_j['MethodCalls'])])
            #     similarity_methodcall_vector = cosine_similarity(vectorizer[0], vectorizer[1]).flatten()
            #     similarity_methodcall = similarity_methodcall_vector[0]
            # else:
            #     similarity_methodcall = 0

            # # Cosine similarity for variables
            # if method_data_i['Variables'] and method_data_j['Variables']:
            #     vectorizer = CountVectorizer().fit_transform([', '.join(method_data_i['Variables']), ', '.join(method_data_j['Variables'])])
            #     similarity_variable_vector = cosine_similarity(vectorizer[0], vectorizer[1]).flatten()
            #     similarity_variable = similarity_variable_vector[0]
            # else:
            #     similarity_variable = 0

            # # Calculate the average similarity
            # avg_similarity = (similarity_method + similarity_class + similarity_methodcall + similarity_variable) / 4

            # semantic_similarity_matrix[i][j] = avg_similarity
            # semantic_similarity_matrix[i][j] = similarity
            
            # # Levenshtein distance
            # levenshtein_distance = Levenshtein.distance(method_data_i['MethodName'], method_data_j['MethodName'])
            # semantic_similarity_matrix[i][j] = levenshtein_distance
            
            # SequenceMatcher
            similarity_method = SequenceMatcher(None, method_data_i['MethodName'], method_data_j['MethodName']).ratio()
            # similarity_class = SequenceMatcher(None, method_data_i['ClassName'], method_data_j['ClassName'])
            # similarity_methodcall = SequenceMatcher(None, ' '.join(method_data_i['MethodCalls']), ' '.join(method_data_j['MethodCalls']))
            # similarity_variables = SequenceMatcher(None, ' '.join(method_data_i['Variables']), ' '.join(method_data_j['Variables']))
            # print(similarity_method)
            # semantic_similarity_matrix[i][j] = (similarity_method + similarity_class + similarity_methodcall + similarity_variables) / 4
            semantic_similarity_matrix[i][j] = similarity_method

    return semantic_similarity_matrix

def get_domain_similarity(method_count, method_data_list, domain_list):
    domain_matrix = np.zeros((method_count, len(domain_list)))
    for i, method_data in enumerate(method_data_list):
        for j in range(len(domain_list)):
            if domain_list[j].lower() in method_data['MethodName'].lower():
                domain_matrix[i][j] = 1

    domain_similarity_matrix = [[0] * method_count for _ in range(method_count)]
    for i in range(method_count):
        for j in range(method_count):
            for k in range(len(domain_list)):
                if domain_matrix[i][k] == 1 and domain_matrix[j][k] == 1:
                    domain_similarity_matrix[i][j] += 1
    
    return domain_similarity_matrix                                                                                                                                                                                                           

def compute_nlp_similarity(method_count, data_list):
    semantic_similarity_matrix = [[0 for _ in range(method_count)] for _ in range(method_count)]
    nlp = spacy.load('en_core_web_md')
    for i in range(method_count):
        for j in range(method_count):
            data_i = nlp(data_list[i])
            data_j = nlp(data_list[j])
            similarity_score = data_i.similarity(data_j)
            semantic_similarity_matrix[i][j] = similarity_score
            semantic_similarity_matrix[j][i] = similarity_score
    return semantic_similarity_matrix

def get_semantic_similarity_matrix(document, method_count, method_data_list):

    domain_list = get_domain_list(document, method_data_list)
    domain_similarity_matrix = get_domain_similarity(method_count, method_data_list, domain_list)
    
    semantic_similarity_matrix = get_semantic_similarity(method_count, method_data_list)
    
    similarity_matrix = processing.add_matrices(method_count, domain_similarity_matrix, 0.6, semantic_similarity_matrix, 0.4)
    # similarity_matrix = processing.add_matrices(method_count, compute_nlp_similarity(method_count, document), 0.6, similarity_matrix, 0.4)
    return similarity_matrix