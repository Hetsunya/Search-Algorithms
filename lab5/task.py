import os
import re
import json
import nltk
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка необходимых ресурсов
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Папка с текстовыми файлами
DATA_PATH = "News Articles"

def load_documents(data_path):
    documents = {}
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            with open(os.path.join(data_path, filename), "r", encoding="utf-8") as file:
                documents[filename] = file.read()
    return documents

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zа-яё ]', '', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(processed_tokens)

# Загрузка и предобработка документов
documents = load_documents(DATA_PATH)
preprocessed_docs = {doc_id: preprocess_text(text) for doc_id, text in documents.items()}

# Вычисление TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs.values())

# Функция поиска

def search(query, vectorizer, tfidf_matrix, documents):
    query = preprocess_text(query)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_docs = sorted(zip(documents.keys(), similarities), key=lambda x: x[1], reverse=True)
    return ranked_docs

# Пример поиска
query = "technology market"
search_results = search(query, vectorizer, tfidf_matrix, documents)
print("Результаты поиска:")
for doc_id, score in search_results[:10]:  # Выводим топ-10 результатов
    print(f"Документ: {doc_id}, Сходство: {score:.4f}")
