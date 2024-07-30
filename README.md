# Spam Classification using Naive Bayes Project

This project involves building a text classification model with various feature selection techniques and evaluating its performance. The main tasks include preprocessing text data, extracting vocabulary, and training and testing the model.

## Project Structure

- `Project.py`: Main script containing the implementation of the text classification model, text preprocessing functions, and evaluation methods.

## Requirements

To run this project, you need the following Python libraries:

- `nltk`
- `sklearn`
- `matplotlib`

You can install the required libraries using the following command:

```bash
pip install nltk sklearn matplotlib

```
## Usage

### 1. Preprocess Text

Tokenizes the text, converts it to lowercase, removes punctuation and stopwords, and applies stemming.

```python
def preprocess_text(text):
    ...
```

### 2. Extract Vocabulary
Extracts a set of unique words from a list of documents.

```
python

def extract_vocabulary(documents):
    ...
```

### 3. Train and Evaluate Model
Trains the text classification model using different feature selection methods and evaluates its performance.
```
python

def train_and_evaluate_with_selected_vocabulary(X_train, y_train, X_test, y_test, selected_vocabulary):
    ...
```

### 4. Plot Results
Plots the number of features versus F1 score for different feature selection methods.
```
python

import matplotlib.pyplot as plt
...
plt.show()

```

## Author

This project is developed by Uma Maheshwari Banna.