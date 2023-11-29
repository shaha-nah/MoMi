import json
import processing
import similarity

# Read data from json
with open('./inputs/petclinic.json', 'r') as file:
    data = json.load(file)

# Extract dataset from json
method_data_list = []
label_set = set()
class_info = []
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

    class_info.append(data)

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
directed_graph_similarity_matrix = similarity.get_directed_graph_similarity_matrix(method_data_list)
weighted_method_similarity_matrix = similarity.get_weighted_method_similarity_matrix(method_data_list, class_calls, class_info)

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

tfidf_similarity_matrix = similarity.get_tfidf_similarity_matrix(preprocessed_method_data_list)
word2vec_similarity_matrix = similarity.get_word2vec_similarity_matrix(preprocessed_method_data_list)
bert_similarity_matrix = similarity.get_bert_similarity_matrix(preprocessed_method_data_list)
fasttext_similarity_matrix = similarity.get_fasttext_similarity_matrix(preprocessed_method_data_list)
spacy_similarity_matrix = similarity.get_spacy_similarity_matrix(preprocessed_method_data_list)
