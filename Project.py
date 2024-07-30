import os
import nltk
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
nltk.download('stopwords')

#---------------------------------------------------------------------------------------------------------------

def display_info():
    """Displays project and author information."""
    print("=================== SPAM CLASSIFICATION USING MULTINOMIAL NAIVE BAYES TEXT CLASSIFICATION ===============")
    print("First Name: Uma Maheshwari ")
    print("Last Name: Banna ")
    
display_info()

#---------------------------------------------------------------------------------------------------------------
def read_csv_file(file_path):
    """
    Reads a CSV file and returns its contents as X (messages) and y (labels) with labels converted to numeric.

    Args:
    - file_path (str): The path to the CSV file.

    Returns:
    - tuple: Tuple containing X (list of preprocessed messages) and y (list of numeric labels).
    """
    X = []
    y = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            next(file)  # Skip the header if the CSV has one
            for line in file:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    label = parts[0].strip()
                    message = parts[1].strip()
                    # Convert labels to numeric
                    numeric_label = 1 if label.lower() == 'spam' else 0
                    X.append(message)
                    y.append(numeric_label)
                else:
                    print(f"Error: Invalid line format - {line}")
    else:
        print(f"Error: File '{file_path}' not found.")
    return X, y

#---------------------------------------------------------------------------------------------------------------
def preprocess_text(text):
    """
    Processes the input text by tokenizing, converting to lowercase, removing punctuation and stopwords, and applying stemming.

    Args:
    - text (str): The input text to preprocess.

    Returns:
    - list: A list of stemmed and filtered words from the input text.
    """
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    # Tokenize the text
    words = word_tokenize(text)
    # convert to lowercase and remove punctuation
    words = [word.lower() for word in words if word.isalnum()]
    # Remove stopwords and apply stemming
    filtered_words = [porter.stem(word.lower()) for word in words if word.lower() not in stop_words]
    return filtered_words

#---------------------------------------------------------------------------------------------------------------
def extract_vocabulary(documents):
    """
    Extracts a set of unique words from a list of documents.

    Args:
    - documents (list): A list of documents from which to extract the vocabulary.

    Returns:
    - set: A set containing all unique words across the provided documents.
    """
    vocabulary = set()
    for doc in documents:
        if isinstance(doc, list): 
            doc = ' '.join(doc)  
        words = doc.split()  
        vocabulary.update(words)
    return vocabulary

#---------------------------------------------------------------------------------------------------------------
""" THE BELOW FUNCTION ARE SOME FUNCTIONS USED IN THE TRANING AND TESTING ALOGORITHM"""
def count_docs_in_class(documents, y_train, class_label):
    """
    Count the number of documents that belong to a specific class.

    Args:
    documents (list): A list of documents.
    y_train (list): A list of labels corresponding to the documents.
    class_label (int or str): The class label to count documents for.

    Returns:
    int: The number of documents that belong to the specified class label.
    """
    count = 0
    for doc, label in zip(documents, y_train):
        if label == class_label:
            count += 1
    return count

def concatenate_text_of_all_docs_in_class(documents, class_label):
    """
    Concatenate the text of all documents in a specific class into a single list of tokens.

    Args:
    documents (list of lists): A list where each element is a list of tokens from a document.
    y_train (list): A list of labels corresponding to the documents.
    class_label (int or str): The class label of the documents to concatenate.

    Returns:
    list: A list of tokens from all documents that belong to the specified class label.
    """
    text = []
    for doc, label in zip(documents, y_train):
        if label == class_label:
            text.extend(doc)
    return text

def count_tokens_of_term(document, term):
    """
    Count the occurrences of a specific term in a document.

    Args:
    document (list of str): The document to search within.
    term (str): The term to count in the document.

    Returns:
    int: The count of the specified term in the document.
    """
    return document.count(term)

def extract_tokens_from_doc(vocabulary, document):
    """
    Extract tokens from a document that are present in the given vocabulary.

    Args:
    vocabulary (set or list): A collection of unique tokens that constitutes the vocabulary.
    document (list of str): The document from which to extract tokens.

    Returns:
    list: A list of tokens that are both in the document and the vocabulary.
    """
    return [token for token in document if token in vocabulary]

def count_words_in_class(documents, y_train, class_label):
    """
    Count the total number of words in all documents of a specified class.

    Args:
    documents (list of lists): A list where each element is a list of tokens from a document.
    y_train (list): A list of labels corresponding to the documents.
    class_label (int or str): The class label whose documents' words are to be counted.

    Returns:
    int: The total number of words in all documents of the specified class.
    """
    total_words = 0
    for doc_list, label in zip(documents, y_train):
        if label == class_label:
            for doc in doc_list:  
                total_words += len(doc.split())
    return total_words

#---------------------------------------------------------------------------------------------------------------

def train_multinomial_nb(classes, X_train, y_train, vocabulary):
    """
    Train a Multinomial Naive Bayes classifier.

    Args:
    classes (list): A list of unique class labels.
    X_train (list of lists): The training data, where each sublist represents a document.
    y_train (list): The labels corresponding to each document in X_train.
    vocabulary (set): A set of all unique words across the training dataset.

    Returns:
    tuple: Returns a tuple containing:
           - vocabulary (set): The set of all vocabulary words.
           - prior (dict): A dictionary of prior probabilities for each class.
           - cond_prob (dict): A nested dictionary where cond_prob[c][t] is the conditional probability
                               of term t given class c.
    """
    N = len(X_train)
    prior = {}
    cond_prob = {}

    for c in classes:
        Nc = count_docs_in_class(X_train, y_train, c)
        prior[c] = Nc / N
        text_c = concatenate_text_of_all_docs_in_class(X_train, y_train)  
        cond_prob[c] = {}

        total_count = count_words_in_class(X_train, y_train, c) + len(vocabulary)
        for t in vocabulary:
            Tct = count_tokens_of_term(text_c, t)
            cond_prob[c][t] = (Tct + 1) / total_count
    
    return vocabulary, prior, cond_prob


def apply_multinomial_nb(classes, vocabulary, prior, cond_prob, document):
    """
    Apply a trained Multinomial Naive Bayes classifier to classify a document.

    Args:
    classes (list): A list of unique class labels.
    vocabulary (set): A set of all unique words used in the classifier.
    prior (dict): A dictionary of prior probabilities for each class.
    cond_prob (dict): A nested dictionary where cond_prob[c][t] is the conditional probability
                      of term t given class c.
    document (list of str): The document to classify, represented as a list of tokens.

    Returns:
    str: The predicted class label for the given document.
    """
    tokens = extract_tokens_from_doc(vocabulary, document)
    scores = {}

    for c in classes:
        score_c = math.log(prior[c])
        for t in tokens:
            if t in vocabulary:
                score_c += math.log(cond_prob[c][t])
        scores[c] = score_c

    return max(scores, key=scores.get)

def train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, selected_vocabulary):
    """
    Train and evaluate a Multinomial Naive Bayes classifier using a selected vocabulary.

    Args:
    X_train (list of lists): The training data, where each sublist represents a document.
    y_train (list): The labels corresponding to each document in X_train.
    X_test (list of lists): The test data, where each sublist represents a document for evaluation.
    y_test (list): The labels corresponding to each document in X_test.
    selected_vocabulary (set): A set of selected terms used as the vocabulary for the classifier.

    Returns:
    tuple: Returns a tuple containing:
           - accuracy (float): The accuracy of the classifier on the test data.
           - f1 (float): The F1 score of the classifier on the test data, computed for the positive class.
           - precision (float): The precision of the classifier on the test data.
           - recall (float): The recall of the classifier on the test data.

    Description:
    This function first trains a Multinomial Naive Bayes classifier using the provided training data and
    selected vocabulary. Then, it evaluates the classifier on the test data using accuracy, F1 score, precision, and recall as metrics.
    The `pos_label` parameter in evaluation metrics is specifically set to handle binary classification cases where the 
    positive class label needs to be explicitly defined.
    """
    # Train the model using the selected vocabulary
    trained_vocabulary, prior, cond_prob = train_multinomial_nb(set(y_train), X_train, y_train, selected_vocabulary)

    # Function to predict and evaluate the model
    def predict_and_evaluate(X_test, y_test, classes, trained_vocabulary, prior, cond_prob):
        y_pred = [apply_multinomial_nb(classes, trained_vocabulary, prior, cond_prob, doc) for doc in X_test]
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=0)  
        precision = precision_score(y_test, y_pred, pos_label=0)
        recall = recall_score(y_test, y_pred, pos_label=0)
        return accuracy, f1, precision, recall

    # Evaluate the model using the test data
    return predict_and_evaluate(X_test, y_test, set(y_train), trained_vocabulary, prior, cond_prob)

#---------------------------------------------------------------------------------------------------------------

""" MUTUAL INFORMATION FEATURE SELECTION """
def calculate_mi(X, y, vocabulary):
    # Vectorize the text data using the existing vocabulary
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    X_vec = vectorizer.fit_transform([' '.join(doc) for doc in X])

    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X_vec, y, discrete_features=True)
    return dict(zip(vectorizer.get_feature_names_out(), mi_scores))

def select_top_features_by_mi(vocabulary, mi_scores, top_n=100):
    # Sort features by MI score in descending order
    sorted_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
    selected_features = {feature for feature, score in sorted_features[:top_n]}
    return selected_features

#---------------------------------------------------------------------------------------------------------------

""" CHI2 FEATURE SELECTION """
def calculate_chi2(X, y, vocabulary, top_n=100):
    # Convert the list of documents into a single string per document and vectorize using the full vocabulary
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    X_vec = vectorizer.fit_transform([' '.join(doc) for doc in X])

    # Apply chi-squared test
    chi2_selector = SelectKBest(chi2, k=top_n)
    chi2_selector.fit(X_vec, y)
    chi2_scores = chi2_selector.scores_

    # Select top N features based on chi-squared scores
    features = vectorizer.get_feature_names_out()
    top_features = [features[i] for i in chi2_selector.get_support(indices=True)]
    return top_features

#---------------------------------------------------------------------------------------------------------------

""" FREQUENCY FEATURE SELECTION """
def frequency_feature_selection(documents, min_df=0.01, max_df=0.95, max_features=None):
    """
    Selects features based on frequency within documents.

    Args:
    - documents (list of list of str): The preprocessed documents.
    - min_df (float or int): The minimum frequency (as a proportion) or as an absolute count.
    - max_df (float or int): The maximum frequency (as a proportion) or as an absolute count.
    - max_features (int): The maximum number of features to keep.

    Returns:
    - list: The vocabulary list of selected features.
    """
    # Join the documents for the vectorizer
    joined_documents = [' '.join(doc) for doc in documents]
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features)
    X_vec = vectorizer.fit_transform(joined_documents)

    # Get the feature names that have been selected
    selected_features = vectorizer.get_feature_names_out()
    return selected_features

#---------------------------------------------------------------------------------------------------------------

#Loading the dataset
X, y = read_csv_file('SPAM text message 20170820 - Data.csv')

# Text preprocessing
preprocessed_X = []
for i in X:
    processed_message = preprocess_text(i)
    preprocessed_X.append(processed_message)

# Spliting dataset into 8:2 ratio
X_train, X_test, y_train, y_test = train_test_split(preprocessed_X, y, test_size=0.2, random_state=42)

# extracting unique words in the vocabulary
vocabulary = extract_vocabulary(X_train)

# extracting unique classes (labels) from the training set y_train using set data structure
classes = set(y_train)

#extracting mutual information features
mi_scores = calculate_mi(X_train, y_train, vocabulary)
mutual_info_selected_vocabulary_100 = select_top_features_by_mi(vocabulary, mi_scores, top_n=100)
mutual_info_selected_vocabulary_500 = select_top_features_by_mi(vocabulary, mi_scores, top_n=500)
mutual_info_selected_vocabulary_250 = select_top_features_by_mi(vocabulary, mi_scores, top_n=250)
#extracting chi2 features
top_features_by_chi2_100 = calculate_chi2(X_train, y_train, vocabulary, top_n=100)
top_features_by_chi2_250 = calculate_chi2(X_train, y_train, vocabulary, top_n=250)
top_features_by_chi2_500 = calculate_chi2(X_train, y_train, vocabulary, top_n=250)

#extracting frequency features
frequency_selected_vocabulary_500 = frequency_feature_selection(X_train, min_df=5, max_df=0.8, max_features=500)
frequency_selected_vocabulary_100 = frequency_feature_selection(X_train, min_df=5, max_df=0.8, max_features=100)
frequency_selected_vocabulary_250 = frequency_feature_selection(X_train, min_df=5, max_df=0.8, max_features=250)
#---------------------------------------------------------------------------------------------------------------

# Training and Evaluation on Original Vocabulary
accuracy, f1, precision, recall = train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, vocabulary)
print("Model trained on original vocabulary - Accuracy:", accuracy, "F1 Score:", f1, "Precision:", precision, "Recall:", recall)
print("\n")
#---------------------------------------------------------------------------------------------------------------

# Training and Evaluation on Mutual Information Vocabulary with different k values
accuracy_mi_100, f1_mi_100, precision_mi_100 , recall_mi_100 = train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, mutual_info_selected_vocabulary_100)
print("Model trained on Mutual Information selected vocabulary and k = 100 - Accuracy", accuracy_mi_100, "F1 Score:", f1_mi_100, "Precision:", precision_mi_100, "Recall:", recall_mi_100 )

accuracy_mi_250, f1_mi_250, precision_mi_250 , recall_mi_250 = train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, mutual_info_selected_vocabulary_250)
print("Model trained on Mutual Information selected vocabulary and k = 250- Accuracy:", accuracy_mi_250, "F1 Score:", f1_mi_250, "Precision:", precision_mi_250, "Recall:", recall_mi_250 )

accuracy_mi_500, f1_mi_500, precision_mi_500 , recall_mi_500 = train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, mutual_info_selected_vocabulary_500)
print("Model trained on Mutual Information selected vocabulary  and k = 500 - Accuracy:", accuracy_mi_500, "F1 Score:", f1_mi_500, "Precision:", precision_mi_500, "Recall:", recall_mi_500)
print("\n")
#---------------------------------------------------------------------------------------------------------------

# Training and Evaluation on Chi2 Vocabulary with different k values

accuracy_chi2_100, f1_chi2_100, precision_chi2_100, recall_chi2_100 = train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, top_features_by_chi2_100)
print("Model trained on Chi-squared selected vocabulary and k = 100 - Accuracy:", accuracy_chi2_100, "F1 Score:", f1_chi2_100, "Precision:", precision_chi2_100, "Recall:", recall_chi2_100)

accuracy_chi2_250, f1_chi2_250, precision_chi2_250, recall_chi2_250 = train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, top_features_by_chi2_250)
print("Model trained on Chi-squared selected vocabulary and k = 250 - Accuracy:", accuracy_chi2_250, "F1 Score:", f1_chi2_250, "Precision:", precision_chi2_250, "Recall:", recall_chi2_250)

accuracy_chi2_500, f1_chi2_500, precision_chi2_500, recall_chi2_500 = train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, top_features_by_chi2_500)
print("Model trained on Chi-squared selected vocabulary and k = 500 - Accuracy:", accuracy_chi2_500, "F1 Score:", f1_chi2_500, "Precision:", precision_chi2_500, "Recall:", recall_chi2_500)
print("\n")
#---------------------------------------------------------------------------------------------------------------

# Training and Evaluation on Frequency Vocabulary with different k values
accuracy_freq_100, f1_freq_100, precision_freq_100, recall_freq_100 = train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, frequency_selected_vocabulary_100)
print("Model trained on Frequency selected vocabulary and k=100 - Accuracy:", accuracy_freq_100, "F1 Score:", f1_freq_100, "Precision:", precision_freq_100, "Recall:", recall_freq_100)

accuracy_freq_250, f1_freq_250, precision_freq_250, recall_freq_250 = train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, frequency_selected_vocabulary_250)
print("Model trained on Frequency selected vocabulary and k=250 - Accuracy:", accuracy_freq_250, "F1 Score:", f1_freq_250, "Precision:", precision_freq_250, "Recall:", recall_freq_250)

accuracy_freq_500, f1_freq_500, precision_freq_500, recall_freq_500 = train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, frequency_selected_vocabulary_500)
print("Model trained on Frequency selected vocabulary and k = 500 - Accuracy:", accuracy_freq_500, "F1 Score:", f1_freq_500, "Precision:", precision_freq_500, "Recall:", recall_freq_500)
print("\n")
#---------------------------------------------------------------------------------------------------------------

"""Plotting the graph for number of features vs f1 score for Original Vocabulary, Mutual Information Vocabulary,
Chi2 Vocabulary and Frequency Vocabulary"""

import matplotlib.pyplot as plt

# Collect data for plotting
k_values = [100, 250, 500]

# F1 scores for original vocabulary
f1_original = [f1, f1, f1]

# F1 scores for Mutual Information vocabulary with different k values
f1_mi_values = [f1_mi_100, f1_mi_250, f1_mi_500]

# F1 scores for Chi2 vocabulary with different k values
f1_chi2_values = [f1_chi2_100, f1_chi2_250, f1_chi2_500]

# F1 scores for Frequency vocabulary with different k values
f1_freq_values = [f1_freq_100, f1_freq_250, f1_freq_500]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(k_values, f1_original, marker='o', color='blue', label='Original Vocabulary')
plt.plot(k_values, f1_mi_values, marker='o', color='green', label='Mutual Information Vocabulary')
plt.plot(k_values, f1_chi2_values, marker='o', color='red', label='Chi-squared Vocabulary')
plt.plot(k_values, f1_freq_values, marker='o', color='orange', label='Frequency Vocabulary')

plt.title('Number of Features vs. F1 Score')
plt.xlabel('Number of Features (k)')
plt.ylabel('F1 Score')
plt.xticks(k_values)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#---------------------------------------------------------------------------------------------------------------
