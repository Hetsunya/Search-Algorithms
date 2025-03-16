import os
import re
import json
import nltk
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Папка с текстовыми файлами
DATA_PATH = "kniga"

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

# Функция обычного поиска (поиск запроса в документах)
def simple_search(query, documents):
    results = {}
    query_lower = query.lower()
    for doc_id, text in documents.items():
        if query_lower in text.lower():
            results[doc_id] = text.lower().count(query_lower)
    return sorted(results.items(), key=lambda x: x[1], reverse=True)

# Функция поиска по индексу
def indexed_search(query, index):
    query_tokens = preprocess_text(query)
    results = defaultdict(set)  # Объединение
    intersection_results = None  # Пересечение
    doc_positions = defaultdict(lambda: defaultdict(list))
    
    for token in query_tokens:
        if token in index:
            for doc_id, positions in index[token].items():
                results[doc_id].add(token)
                doc_positions[doc_id][token] = sorted(positions)
    
    if query_tokens:
        found_docs = list(results.keys())
        intersection_results = set(found_docs) if found_docs else set()
        for token in query_tokens:
            intersection_results &= set(index.get(token, {}))
    
    return {
        "union": sorted(results.items(), key=lambda x: len(x[1]), reverse=True),
        "intersection": sorted(list(intersection_results))
    }

# Загрузка документов
documents = load_documents(DATA_PATH)
# Построение инвертированного индекса
inverted_index = build_inverted_index(documents)

# Тестовые запросы
queries = ["затрепетали на ветру знамена", "Гондора Мин-Риммон", "рохиррим", "войске"]

# Словари для хранения времени выполнения
execution_times = {}

for query in queries:
    start_time = time.perf_counter()
    simple_results = simple_search(query, documents)
    simple_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    index_results = indexed_search(query, inverted_index)
    index_time = time.perf_counter() - start_time
    
    execution_times[query] = {"simple": simple_time, "indexed": index_time}
    
    print(f"\nРезультаты поиска для: \"{query}\"")
    
    print("\nОбычный поиск:")
    if simple_results:
        for doc_id, count in simple_results:
            print(f"  Документ: {doc_id}, вхождений: {count}")
    else:
        print("  Совпадений не найдено.")
    print(f"  Время выполнения: {simple_time:.6f} секунд")
    
    print("\nПоиск по индексу (объединение):")
    if index_results["union"]:
        for doc_id, words in index_results["union"]:
            print(f"  Документ: {doc_id}, найденные слова: {', '.join(words)}")
    else:
        print("  Совпадений не найдено.")
    print(f"  Время выполнения: {index_time:.6f} секунд")
    
    print("\nПоиск по индексу (пересечение):")
    if index_results["intersection"]:
        print("  Документы: ", ", ".join(index_results["intersection"]))
    else:
        print("  Нет документов, содержащих все слова запроса.")
    print(f"  Время выполнения: {index_time:.6f} секунд")

# Построение графика
queries_labels = [f"Запрос {i+1}" for i in range(len(queries))]
simple_times = [execution_times[q]["simple"] for q in queries]
indexed_times = [execution_times[q]["indexed"] for q in queries]

plt.figure(figsize=(10, 5))
plt.bar(queries_labels, simple_times, label="Обычный поиск", alpha=0.7)
plt.bar(queries_labels, indexed_times, label="Индексированный поиск", alpha=0.7)
plt.xlabel("Запросы")
plt.ylabel("Время (сек)")
plt.title("Сравнение времени выполнения поиска")
plt.xticks(rotation=45)
plt.legend()
plt.show()
