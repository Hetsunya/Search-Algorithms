import os
import re
import json
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Загрузка необходимых ресурсов
nltk.download('punkt')
nltk.download('punkt_tab')

nltk.download('wordnet')
nltk.download('stopwords')

# Папка с текстовыми файлами (указать путь к датасету)
DATA_PATH = "BBC News Summary/News Articles/tech"


# Функция для чтения текстов из файлов
def load_documents(data_path):
    documents = {}
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            with open(os.path.join(data_path, filename), "r", encoding="utf-8") as file:
                documents[filename] = file.read()
    return documents

# Функция предобработки текста
def preprocess_text(text):
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r'[^a-zа-яё ]', '', text)  # Удаление пунктуации
    tokens = word_tokenize(text)  # Токенизация
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return processed_tokens

# Функция для построения инвертированного индекса
def build_inverted_index(documents):
    inverted_index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        for position, token in enumerate(tokens):
            inverted_index[token][doc_id].append(position)
    return inverted_index

# Функция поиска по индексу
def search(query, index):
    query_tokens = preprocess_text(query)
    results = defaultdict(set)
    for token in query_tokens:
        if token in index:
            for doc_id in index[token]:
                results[doc_id].add(token)
    return sorted(results.items(), key=lambda x: len(x[1]), reverse=True)

# Запуск индексирования
documents = load_documents(DATA_PATH)
inverted_index = build_inverted_index(documents)

# Сохранение индекса в файл
with open("inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, ensure_ascii=False, indent=4)

# Пример поиска
query = "technology market"
search_results = search(query, inverted_index)
print("Результаты поиска:")
for doc_id, words in search_results:
    print(f"Документ: {doc_id}, Найденные слова: {words}")
