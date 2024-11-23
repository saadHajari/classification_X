#Author : Saad HAJARI 
#E-mail : saadhajari10@gmail.com

## Import required libraries

import pandas as pd  # data processing
import numpy as np  # mathemetical operations
import json 
import emoji # to read json files
import unicodedata

import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # data visualization

import re  # regular expression
import string  # to perform string operations
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords  # to remove stopwords
from nltk.stem import PorterStemmer  # for stemming
from sklearn.feature_extraction.text import CountVectorizer  # to create bag of words
from sklearn.feature_extraction.text import TfidfVectorizer  # to create tf-idf
from gensim.models import KeyedVectors  # to create word vectors

from sklearn.model_selection import train_test_split  # to split the data
from sklearn.preprocessing import LabelEncoder  # to label encode the categorical values

## Import required libraries for model building
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

## to evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score

## stop warnings
import warnings
warnings.filterwarnings('ignore')
import logging


## Load the data from the json file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the data
with open('data_x_posts.json') as f:
    data = json.load(f)

# Convert the JSON data into a pandas dataframe
df = pd.DataFrame(data)
logging.info(f"Data shape: {df.shape}")
logging.info(f"First few rows: {df.head()}")

# Label and their corresponding names in a dictionary
label_dict = {d['label']: d['label_name'] for d in data}
logging.info(f"Label dictionary: {label_dict}")

# Functions
def clean_text(text):
    """
    Cleans text by removing URLs, mentions, special characters, and emojis.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+", '', text)  # Remove mentions
    text = re.sub(r"[{}<>]", '', text)  # Remove special characters
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    return text.strip()

def contains_dangerous_keywords(text):
    """
    Checks if the text contains dangerous keywords.
    """
    dangerous_keywords = [
        'attack', 'kill', 'hate', 'bomb', 'threat', 'violence', 'weapon',
        'terror', 'murder', 'suicide', 'hacking', 'extortion', 'poison', 
        'danger', 'explosion', 'nuclear', 'hazardous', 'anarchy'
    ]
    normalized_text = unicodedata.normalize('NFKD', text.lower())
    pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in dangerous_keywords) + r')\b'
    return bool(re.search(pattern, normalized_text))

def contains_suspicious_links(text):
    """
    Detects if the text contains links from suspicious domains.
    """
    suspicious_domains = ['suspicious.com', 'phishy.net', 'malware.org']
    pattern = r"http[s]?://(?:www\.)?(" + '|'.join(re.escape(domain) for domain in suspicious_domains) + r")"
    return bool(re.search(pattern, text))

def excessive_special_characters(text):
    """
    Checks if the text contains an excessive amount of special characters.
    """
    special_chars = re.findall(r'[!@#$%^&*(),.?":{}|<>]', text)
    return len(special_chars) > 5

def contains_emojis(text):
    """
    Detects if the text contains emojis.
    """
    return bool(emoji.emoji_count(text))

# Apply preprocessing and checks
df['cleaned_text'] = df['text'].apply(clean_text)
df['contains_dangerous_keywords'] = df['cleaned_text'].apply(contains_dangerous_keywords)
df['contains_suspicious_links'] = df['text'].apply(contains_suspicious_links)
df['contains_emojis'] = df['text'].apply(contains_emojis)
df['excessive_special_characters'] = df['cleaned_text'].apply(excessive_special_characters)

# Combine checks into a single 'suspicious' column
df['suspicious'] = (
    df['contains_dangerous_keywords'] | 
    df['contains_suspicious_links'] | 
    df['contains_emojis'] | 
    df['excessive_special_characters']
)

# Logging suspicious counts
logging.info("Suspicious Posts Count:")
logging.info(df['suspicious'].value_counts())

# Feature and target selection
X = df['cleaned_text']
y = df['suspicious'].astype(int)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a model (RandomForest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
logging.info("Classification Report for Suspicious Detection:")
logging.info(classification_report(y_test, y_pred))

## Dropping date, id, and label_name columns

df.drop(['date', 'id', 'label_name'], axis=1, inplace=True)
print(df.shape)
df.head()

## class distribution in pie chart

df['label'].value_counts().plot(kind='pie', autopct='%1.0f%%', figsize=(6, 6),
                                title='Class Distribution', ylabel='',
                                labels=[label_dict[i] for i in df['label'].value_counts().index])
plt.show()


## Preprocessing the text to remove punctuations, convert text to lowercase, and remove stopwords

process_text = lambda text: ' '.join([re.sub(r'[^\w\s]', '', word.lower()) for word in text.split() if word not in stopwords.words('english')])


## Apply the lambda function to the 'text' column
df['cleaned_text'] = df['text'].apply(process_text)
df.head()


def stem_text(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

df['stemmed_text'] = df['text'].apply(stem_text)
df.head()

## Creating bag of words

# Creating bag of words
cv = CountVectorizer(max_features=5000, ngram_range=(1, 2))
bow = cv.fit_transform(df['stemmed_text'])
print(bow.shape)

# Creating tf-idf
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['stemmed_text'])
print(tfidf_matrix.shape)

# Splitting the data
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow, df['label'], test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_matrix, df['label'], test_size=0.2, random_state=42)

# Dataframe to store results
results = pd.DataFrame(columns=['Model', 'VectorType', 'CrossValidationScore', 'TestAccuracy', 'Precision', 'Recall', 'F1Score'])


# Logistic Regression - Bag of Words
cross_val = cross_val_score(LogisticRegression(max_iter=10000), X_train_bow, y_train_bow, cv=3, scoring='accuracy')
lr_bow = LogisticRegression(max_iter=10000).fit(X_train_bow, y_train_bow)

new_result_bow = pd.DataFrame([{
    'Model': 'Logistic Regression',
    'VectorType': 'Bag of Words',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_bow, lr_bow.predict(X_test_bow)),
    'Precision': precision_score(y_test_bow, lr_bow.predict(X_test_bow), average='weighted'),
    'Recall': recall_score(y_test_bow, lr_bow.predict(X_test_bow), average='weighted'),
    'F1Score': f1_score(y_test_bow, lr_bow.predict(X_test_bow), average='weighted')
}])

results = pd.concat([results, new_result_bow], ignore_index=True)
print('Bag of Words - Completed')

# Logistic Regression - TF-IDF
cross_val = cross_val_score(LogisticRegression(max_iter=10000), X_train_tfidf, y_train_tfidf, cv=3, scoring='accuracy')
lr_tfidf = LogisticRegression(max_iter=10000).fit(X_train_tfidf, y_train_tfidf)

new_result_tfidf = pd.DataFrame([{
    'Model': 'Logistic Regression',
    'VectorType': 'TF-IDF',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_tfidf, lr_tfidf.predict(X_test_tfidf)),
    'Precision': precision_score(y_test_tfidf, lr_tfidf.predict(X_test_tfidf), average='weighted'),
    'Recall': recall_score(y_test_tfidf, lr_tfidf.predict(X_test_tfidf), average='weighted'),
    'F1Score': f1_score(y_test_tfidf, lr_tfidf.predict(X_test_tfidf), average='weighted')
}])

results = pd.concat([results, new_result_tfidf], ignore_index=True)
print('TF-IDF - Completed')

# Display results
print(results)

## knn - bag of words
cross_val = cross_val_score(KNeighborsClassifier(), X_train_bow, y_train_bow, cv=3, scoring='accuracy')
knn_bow = KNeighborsClassifier().fit(X_train_bow, y_train_bow)

new_result_bow = pd.DataFrame([{
    'Model': 'KNN',
    'VectorType': 'Bag of Words',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_bow, knn_bow.predict(X_test_bow)),
    'Precision': precision_score(y_test_bow, knn_bow.predict(X_test_bow), average='weighted'),
    'Recall': recall_score(y_test_bow, knn_bow.predict(X_test_bow), average='weighted'),
    'F1Score': f1_score(y_test_bow, knn_bow.predict(X_test_bow), average='weighted')
}])

results = pd.concat([results, new_result_bow], ignore_index=True)
print('Bag of Words - Completed')

## knn - tfidf
cross_val = cross_val_score(KNeighborsClassifier(), X_train_tfidf, y_train_tfidf, cv=3, scoring='accuracy')
knn_tfidf = KNeighborsClassifier().fit(X_train_tfidf, y_train_tfidf)

new_result_tfidf = pd.DataFrame([{
    'Model': 'KNN',
    'VectorType': 'TF-IDF',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_tfidf, knn_tfidf.predict(X_test_tfidf)),
    'Precision': precision_score(y_test_tfidf, knn_tfidf.predict(X_test_tfidf), average='weighted'),
    'Recall': recall_score(y_test_tfidf, knn_tfidf.predict(X_test_tfidf), average='weighted'),
    'F1Score': f1_score(y_test_tfidf, knn_tfidf.predict(X_test_tfidf), average='weighted')
}])

results = pd.concat([results, new_result_tfidf], ignore_index=True)
print('TF-IDF - Completed')


# Display results filtered for 'KNN' model
print(results[results['Model'] == 'KNN'])


## svm - bag of words
cross_val = cross_val_score(SVC(), X_train_bow, y_train_bow, cv=3, scoring='accuracy')
svm_bow = SVC().fit(X_train_bow, y_train_bow)

new_result_bow = pd.DataFrame([{
    'Model': 'SVM',
    'VectorType': 'Bag of Words',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_bow, svm_bow.predict(X_test_bow)),
    'Precision': precision_score(y_test_bow, svm_bow.predict(X_test_bow), average='weighted'),
    'Recall': recall_score(y_test_bow, svm_bow.predict(X_test_bow), average='weighted'),
    'F1Score': f1_score(y_test_bow, svm_bow.predict(X_test_bow), average='weighted')
}])

results = pd.concat([results, new_result_bow], ignore_index=True)
print('Bag of Words - Completed')

## svm - tfidf
cross_val = cross_val_score(SVC(), X_train_tfidf, y_train_tfidf, cv=3, scoring='accuracy')
svm_tfidf = SVC().fit(X_train_tfidf, y_train_tfidf)

new_result_tfidf = pd.DataFrame([{
    'Model': 'SVM',
    'VectorType': 'TF-IDF',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_tfidf, svm_tfidf.predict(X_test_tfidf)),
    'Precision': precision_score(y_test_tfidf, svm_tfidf.predict(X_test_tfidf), average='weighted'),
    'Recall': recall_score(y_test_tfidf, svm_tfidf.predict(X_test_tfidf), average='weighted'),
    'F1Score': f1_score(y_test_tfidf, svm_tfidf.predict(X_test_tfidf), average='weighted')
}])

results = pd.concat([results, new_result_tfidf], ignore_index=True)
print('TF-IDF - Completed')

print(results[results['Model'] == 'SVM'])


## naive bayes - bag of words

cross_val = cross_val_score(MultinomialNB(), X_train_bow, y_train_bow, cv=3, scoring='accuracy')
nb_bow = MultinomialNB().fit(X_train_bow, y_train_bow)

new_result_bow = pd.DataFrame([{
    'Model': 'Naive Bayes',
    'VectorType': 'Bag of Words',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_bow, nb_bow.predict(X_test_bow)),
    'Precision': precision_score(y_test_bow, nb_bow.predict(X_test_bow), average='weighted'),
    'Recall': recall_score(y_test_bow, nb_bow.predict(X_test_bow), average='weighted'),
    'F1Score': f1_score(y_test_bow, nb_bow.predict(X_test_bow), average='weighted')
}])

results = pd.concat([results, new_result_bow], ignore_index=True)
print('Bag of Words - Completed')

## naive bayes - tfidf
cross_val = cross_val_score(MultinomialNB(), X_train_tfidf, y_train_tfidf, cv=3, scoring='accuracy')
nb_tfidf = MultinomialNB().fit(X_train_tfidf, y_train_tfidf)

new_result_tfidf = pd.DataFrame([{
    'Model': 'Naive Bayes',
    'VectorType': 'TF-IDF',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_tfidf, nb_tfidf.predict(X_test_tfidf)),
    'Precision': precision_score(y_test_tfidf, nb_tfidf.predict(X_test_tfidf), average='weighted'),
    'Recall': recall_score(y_test_tfidf, nb_tfidf.predict(X_test_tfidf), average='weighted'),
    'F1Score': f1_score(y_test_tfidf, nb_tfidf.predict(X_test_tfidf), average='weighted')
}])

results = pd.concat([results, new_result_tfidf], ignore_index=True)
print('TF-IDF - Completed')

# Afficher les résultats pour Naive Bayes
print(results[results['Model'] == 'Naive Bayes'])


## MLP - bag of words
cross_val = cross_val_score(MLPClassifier(), X_train_bow, y_train_bow, cv=3, scoring='accuracy')
mlp_bow = MLPClassifier().fit(X_train_bow, y_train_bow)

new_result_mlp_bow = pd.DataFrame([{
    'Model': 'MLP',
    'VectorType': 'Bag of Words',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_bow, mlp_bow.predict(X_test_bow)),
    'Precision': precision_score(y_test_bow, mlp_bow.predict(X_test_bow), average='weighted'),
    'Recall': recall_score(y_test_bow, mlp_bow.predict(X_test_bow), average='weighted'),
    'F1Score': f1_score(y_test_bow, mlp_bow.predict(X_test_bow), average='weighted')
}])

results = pd.concat([results, new_result_mlp_bow], ignore_index=True)
print('Bag of Words - Completed')

## MLP - tfidf
cross_val = cross_val_score(MLPClassifier(), X_train_tfidf, y_train_tfidf, cv=3, scoring='accuracy')
mlp_tfidf = MLPClassifier().fit(X_train_tfidf, y_train_tfidf)

new_result_mlp_tfidf = pd.DataFrame([{
    'Model': 'MLP',
    'VectorType': 'TF-IDF',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_tfidf, mlp_tfidf.predict(X_test_tfidf)),
    'Precision': precision_score(y_test_tfidf, mlp_tfidf.predict(X_test_tfidf), average='weighted'),
    'Recall': recall_score(y_test_tfidf, mlp_tfidf.predict(X_test_tfidf), average='weighted'),
    'F1Score': f1_score(y_test_tfidf, mlp_tfidf.predict(X_test_tfidf), average='weighted')
}])

results = pd.concat([results, new_result_mlp_tfidf], ignore_index=True)
print('TF-IDF - Completed')

print(results[results['Model'] == 'MLP'])

# XGBoost - Bag of Words
le = LabelEncoder()
y_train_bow_le = le.fit_transform(y_train_bow)
y_test_bow_le = le.transform(y_test_bow)

cross_val = cross_val_score(XGBClassifier(), X_train_bow, y_train_bow_le, cv=3, scoring='accuracy')
xgb_bow = XGBClassifier().fit(X_train_bow, y_train_bow_le)

result = pd.DataFrame([{
    'Model': 'XGBoost',
    'VectorType': 'Bag of Words',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_bow_le, xgb_bow.predict(X_test_bow)),
    'Precision': precision_score(y_test_bow_le, xgb_bow.predict(X_test_bow), average='weighted'),
    'Recall': recall_score(y_test_bow_le, xgb_bow.predict(X_test_bow), average='weighted'),
    'F1Score': f1_score(y_test_bow_le, xgb_bow.predict(X_test_bow), average='weighted')
}])

results = pd.concat([results, result], ignore_index=True)
print('Bag of Words - Completed')


# XGBoost - TF-IDF
le = LabelEncoder()
y_train_tfidf_le = le.fit_transform(y_train_tfidf)
y_test_tfidf_le = le.transform(y_test_tfidf)

cross_val = cross_val_score(XGBClassifier(), X_train_tfidf, y_train_tfidf_le, cv=3, scoring='accuracy')
xgb_tfidf = XGBClassifier().fit(X_train_tfidf, y_train_tfidf_le)

result = pd.DataFrame([{
    'Model': 'XGBoost',
    'VectorType': 'TF-IDF',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_tfidf_le, xgb_tfidf.predict(X_test_tfidf)),
    'Precision': precision_score(y_test_tfidf_le, xgb_tfidf.predict(X_test_tfidf), average='weighted'),
    'Recall': recall_score(y_test_tfidf_le, xgb_tfidf.predict(X_test_tfidf), average='weighted'),
    'F1Score': f1_score(y_test_tfidf_le, xgb_tfidf.predict(X_test_tfidf), average='weighted')
}])

results = pd.concat([results, result], ignore_index=True)
print('TF-IDF - Completed')

print(results[results['Model'] == 'XGBoost'])


# Gradient Boosting - Bag of Words
cross_val = cross_val_score(GradientBoostingClassifier(), X_train_bow, y_train_bow, cv=3, scoring='accuracy')
gb_bow = GradientBoostingClassifier().fit(X_train_bow, y_train_bow)

result = pd.DataFrame([{
    'Model': 'Gradient Boosting',
    'VectorType': 'Bag of Words',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_bow, gb_bow.predict(X_test_bow)),
    'Precision': precision_score(y_test_bow, gb_bow.predict(X_test_bow), average='weighted'),
    'Recall': recall_score(y_test_bow, gb_bow.predict(X_test_bow), average='weighted'),
    'F1Score': f1_score(y_test_bow, gb_bow.predict(X_test_bow), average='weighted')
}])

results = pd.concat([results, result], ignore_index=True)
print('Bag of Words - Completed')


# Gradient Boosting - TF-IDF
cross_val = cross_val_score(GradientBoostingClassifier(), X_train_tfidf, y_train_tfidf, cv=3, scoring='accuracy')
gb_tfidf = GradientBoostingClassifier().fit(X_train_tfidf, y_train_tfidf)

result = pd.DataFrame([{
    'Model': 'Gradient Boosting',
    'VectorType': 'TF-IDF',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_tfidf, gb_tfidf.predict(X_test_tfidf)),
    'Precision': precision_score(y_test_tfidf, gb_tfidf.predict(X_test_tfidf), average='weighted'),
    'Recall': recall_score(y_test_tfidf, gb_tfidf.predict(X_test_tfidf), average='weighted'),
    'F1Score': f1_score(y_test_tfidf, gb_tfidf.predict(X_test_tfidf), average='weighted')
}])

results = pd.concat([results, result], ignore_index=True)
print('TF-IDF - Completed')


print(results[results['Model'] == 'Gradient Boosting'])


# Random Forest - Bag of Words
cross_val = cross_val_score(RandomForestClassifier(), X_train_bow, y_train_bow, cv=3, scoring='accuracy')
rf_bow = RandomForestClassifier().fit(X_train_bow, y_train_bow)

result = pd.DataFrame([{
    'Model': 'Random Forest',
    'VectorType': 'Bag of Words',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_bow, rf_bow.predict(X_test_bow)),
    'Precision': precision_score(y_test_bow, rf_bow.predict(X_test_bow), average='weighted'),
    'Recall': recall_score(y_test_bow, rf_bow.predict(X_test_bow), average='weighted'),
    'F1Score': f1_score(y_test_bow, rf_bow.predict(X_test_bow), average='weighted')
}])

results = pd.concat([results, result], ignore_index=True)
print('Bag of Words - Completed')


# Random Forest - TF-IDF
cross_val = cross_val_score(RandomForestClassifier(), X_train_tfidf, y_train_tfidf, cv=3, scoring='accuracy')
rf_tfidf = RandomForestClassifier().fit(X_train_tfidf, y_train_tfidf)

result = pd.DataFrame([{
    'Model': 'Random Forest',
    'VectorType': 'TF-IDF',
    'CrossValidationScore': cross_val.mean(),
    'TestAccuracy': accuracy_score(y_test_tfidf, rf_tfidf.predict(X_test_tfidf)),
    'Precision': precision_score(y_test_tfidf, rf_tfidf.predict(X_test_tfidf), average='weighted'),
    'Recall': recall_score(y_test_tfidf, rf_tfidf.predict(X_test_tfidf), average='weighted'),
    'F1Score': f1_score(y_test_tfidf, rf_tfidf.predict(X_test_tfidf), average='weighted')
}])

results = pd.concat([results, result], ignore_index=True)
print('TF-IDF - Completed')


print(results[results['Model'] == 'Gradient Boosting'])


print(results)

print(results.sort_values(by='TestAccuracy', ascending=False).head(5))

## Performance by Model

# Group results by model and calculate mean metrics
grouped_results = results.groupby('Model').mean()

# Create bar plot with increased bar width and legend outside the plot
ax = grouped_results.plot(kind='bar', figsize=(10, 5), width=0.8,)
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1, borderaxespad=0, prop={'size': 8})

# Remove y-axis grid lines and keep x-axis grid lines
ax.grid(axis='y', linestyle='-')

# Set plot title and labels
plt.title('Average Metrics by Model')
plt.xlabel('Model')
plt.ylabel('Score')
plt.show()