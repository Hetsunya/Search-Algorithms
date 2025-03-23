import os
import re
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Папка с текстовыми файлами
DATA_PATH = "kniga"

# Загрузка необходимых ресурсов NLTK
import nltk
nltk.download('punkt')
nltk.download('stopwords')

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
    stemmer = SnowballStemmer("russian")
    stop_words = set(stopwords.words('russian'))
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return processed_tokens

# Функция построения инвертированного индекса
def build_inverted_index(documents):
    inverted_index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        for position, token in enumerate(tokens):
            inverted_index[token][doc_id].append(position)
    return inverted_index

# Поиск без индекса
def search_without_index(query, documents):
    query_tokens = preprocess_text(query)
    intersection_results = set(documents.keys()) if query_tokens else set()
    individual_results = defaultdict(lambda: defaultdict(int))
    phrase_results = {}

    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        # 1. Пересечение
        all_present = all(token in tokens for token in query_tokens)  # Исправлено
        if not all_present:
            intersection_results.discard(doc_id)
        
        # 2. Слова по отдельности
        for token in query_tokens:
            count = tokens.count(token)
            if count > 0:
                individual_results[doc_id][token] = count
        
        # 3. Поиск фразы
        for i in range(len(tokens) - len(query_tokens) + 1):
            if tokens[i:i + len(query_tokens)] == query_tokens:
                phrase_results[doc_id] = list(range(i, i + len(query_tokens)))
                break  # Берем первое совпадение

    return {
        "intersection": sorted(list(intersection_results)),
        "individual": individual_results,
        "phrase": phrase_results
    }

# Поиск с использованием индекса
def search_with_index(query, index):
    query_tokens = preprocess_text(query)
    intersection_results = set(index.get(query_tokens[0], {}).keys()) if query_tokens else set()
    individual_results = defaultdict(lambda: defaultdict(int))
    phrase_results = {}

    # 1. Пересечение
    for token in query_tokens[1:]:
        intersection_results &= set(index.get(token, {}).keys())

    # 2. Слова по отдельности
    for token in query_tokens:
        if token in index:
            for doc_id, positions in index[token].items():
                individual_results[doc_id][token] = len(positions)

    # 3. Поиск фразы
    if query_tokens:
        for doc_id in intersection_results:
            positions = [index[token][doc_id] for token in query_tokens if doc_id in index[token]]
            if len(positions) == len(query_tokens):  # Все слова есть в документе
                for start_pos in positions[0]:
                    match = True
                    phrase_positions = [start_pos]
                    for i, token in enumerate(query_tokens[1:], 1):
                        expected_pos = start_pos + i
                        if expected_pos not in positions[i]:
                            match = False
                            break
                        phrase_positions.append(expected_pos)
                    if match:
                        phrase_results[doc_id] = phrase_positions
                        break  # Берем первое совпадение

    return {
        "intersection": sorted(list(intersection_results)),
        "individual": individual_results,
        "phrase": phrase_results
    }

# Загрузка документов и построение индекса
documents = load_documents(DATA_PATH)
inverted_index = build_inverted_index(documents)

# Тестовые запросы
queries = ["затрепетали на ветру знамена", "Гондора Мин-Риммон",]

# Словари для хранения времени выполнения
execution_times = {}

for query in queries:
    print(f"\nРезультаты поиска для: \"{query}\"")
    
    # Поиск без индекса
    start_time = time.perf_counter()
    no_index_results = search_without_index(query, documents)
    no_index_time = time.perf_counter() - start_time
    
    print("\nПоиск без индекса:")
    # 1. Пересечение
    print("1. Документы, содержащие все слова (пересечение):")
    if no_index_results["intersection"]:
        print("  Документы: ", ", ".join(no_index_results["intersection"]))
    else:
        print("  Нет документов, содержащих все слова.")
    
    # 2. Слова по отдельности
    print("2. Поиск слов по отдельности:")
    if no_index_results["individual"]:
        for doc_id, words in no_index_results["individual"].items():
            print(f"  Документ: {doc_id}")
            for word, count in words.items():
                print(f"    Слово: {word}, вхождений: {count}")
    else:
        print("  Совпадений не найдено.")
    
    # 3. Поиск фразы
    print("3. Поиск фразы с позициями:")
    if no_index_results["phrase"]:
        for doc_id, positions in no_index_results["phrase"].items():
            print(f"  Документ: {doc_id}, позиции слов: {positions}")
    else:
        print("  Фраза не найдена в документах.")
    print(f"  Время выполнения: {no_index_time:.6f} секунд")
    
    # Поиск с индексом
    start_time = time.perf_counter()
    index_results = search_with_index(query, inverted_index)
    index_time = time.perf_counter() - start_time
    
    print("\nПоиск с индексом:")
    # 1. Пересечение
    print("1. Документы, содержащие все слова (пересечение):")
    if index_results["intersection"]:
        print("  Документы: ", ", ".join(index_results["intersection"]))
    else:
        print("  Нет документов, содержащих все слова.")
    
    # 2. Слова по отдельности
    print("2. Поиск слов по отдельности:")
    if index_results["individual"]:
        for doc_id, words in index_results["individual"].items():
            print(f"  Документ: {doc_id}")
            for word, count in words.items():
                print(f"    Слово: {word}, вхождений: {count}")
    else:
        print("  Совпадений не найдено.")
    
    # 3. Поиск фразы
    print("3. Поиск фразы с позициями:")
    if index_results["phrase"]:
        for doc_id, positions in index_results["phrase"].items():
            print(f"  Документ: {doc_id}, позиции слов: {positions}")
    else:
        print("  Фраза не найдена в документах.")
    print(f"  Время выполнения: {index_time:.6f} секунд")
    
    execution_times[query] = {"no_index": no_index_time, "index": index_time}

# Построение графика
# Данные
queries_labels = [f"Запрос {i+1}" for i in range(len(queries))]
no_index_times = np.array([execution_times[q]["no_index"] for q in queries])
index_times = np.array([execution_times[q]["index"] for q in queries])

# Логарифм времени (добавляет масштабируемость)
log_no_index_times = np.log10(no_index_times)
log_index_times = np.log10(index_times)

# Относительное ускорение (во сколько раз быстрее)
speedup = no_index_times / index_times  

# Создаем графики
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 1. Логарифмический график времени выполнения
axs[0].bar(queries_labels, log_no_index_times, label="Без индекса (log10)", alpha=0.7)
axs[0].bar(queries_labels, log_index_times, label="С индексом (log10)", alpha=0.7)
axs[0].set_ylabel("log10(Время), сек")
axs[0].set_title("Сравнение времени выполнения (логарифмическая шкала)")
axs[0].legend()
axs[0].set_xticklabels(queries_labels, rotation=45)

# 2. График ускорения
axs[1].bar(queries_labels, speedup, color="green", alpha=0.7)
axs[1].set_ylabel("Ускорение (X раз)")
axs[1].set_xlabel("Запросы")
axs[1].set_title("Во сколько раз быстрее поиск с индексом")
axs[1].set_xticklabels(queries_labels, rotation=45)

plt.tight_layout()
plt.show()
# queries_labels = [f"Запрос {i+1}" for i in range(len(queries))]
# no_index_times = [execution_times[q]["no_index"] for q in queries]
# index_times = [execution_times[q]["index"] for q in queries]

# plt.figure(figsize=(10, 5))
# plt.bar(queries_labels, no_index_times, label="Без индекса", alpha=0.7)
# plt.bar(queries_labels, index_times, label="С индексом", alpha=0.7)
# plt.xlabel("Запросы")
# plt.ylabel("Время (сек)")
# plt.title("Сравнение времени выполнения поиска")
# plt.xticks(rotation=45)
# plt.legend()
# plt.show()