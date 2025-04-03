import os
import re
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

DATA_PATH = "kniga"

import nltk
nltk.download('punkt')
nltk.download('stopwords')

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
    stemmer = SnowballStemmer("russian")
    stop_words = set(stopwords.words('russian'))
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return processed_tokens

def build_inverted_index(documents):
    inverted_index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        for position, token in enumerate(tokens):
            inverted_index[token][doc_id].append(position)
    return inverted_index

def search_without_index(query, documents):
    query_tokens = preprocess_text(query)
    intersection_results = set(documents.keys()) if query_tokens else set()
    individual_results = defaultdict(lambda: defaultdict(list))
    phrase_results = {}

    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        all_present = all(token in tokens for token in query_tokens)
        if not all_present:
            intersection_results.discard(doc_id)
        
        for token in query_tokens:
            positions = [i for i, t in enumerate(tokens) if t == token]
            if positions:
                individual_results[doc_id][token] = positions
        
        for i in range(len(tokens) - len(query_tokens) + 1):
            if tokens[i:i + len(query_tokens)] == query_tokens:
                phrase_results[doc_id] = list(range(i, i + len(query_tokens)))
                break

    return {
        "intersection": sorted(list(intersection_results)),
        "individual": individual_results,
        "phrase": phrase_results
    }

def search_with_index(query, index):
    query_tokens = preprocess_text(query)
    intersection_results = set(index.get(query_tokens[0], {}).keys()) if query_tokens else set()
    individual_results = defaultdict(lambda: defaultdict(list))
    phrase_results = {}

    for token in query_tokens[1:]:
        intersection_results &= set(index.get(token, {}).keys())

    for token in query_tokens:
        if token in index:
            for doc_id, positions in index[token].items():
                individual_results[doc_id][token] = positions

    if query_tokens:
        for doc_id in intersection_results:
            positions = [index[token][doc_id] for token in query_tokens if doc_id in index[token]]
            if len(positions) == len(query_tokens):
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
                        break

    return {
        "intersection": sorted(list(intersection_results)),
        "individual": individual_results,
        "phrase": phrase_results
    }

documents = load_documents(DATA_PATH)
inverted_index = build_inverted_index(documents)

queries = [ "должна быть выбрана"]

execution_times = {}

for query in queries:
    print(f"\nРезультаты поиска для: \"{query}\"")
    
    start_time = time.perf_counter()
    no_index_results = search_without_index(query, documents)
    no_index_time = time.perf_counter() - start_time
    
    print("\nПоиск без индекса:")
    print("1. Документы, содержащие все слова (пересечение):")
    if no_index_results["intersection"]:
        print("  Документы: ", ", ".join(no_index_results["intersection"]))
        for doc_id in no_index_results["intersection"]:
            print(f"    Документ: {doc_id}")
            for word, positions in no_index_results["individual"][doc_id].items():
                print(f"      Слово: {word}, вхождений: {len(positions)}, позиции: {positions}")
    else:
        print("  Нет документов, содержащих все слова.")
    
    print("2. Поиск слов по отдельности:")
    if no_index_results["individual"]:
        for doc_id, words in no_index_results["individual"].items():
            if doc_id not in no_index_results["intersection"]:  # Пропускаем документы из пересечения
                print(f"  Документ: {doc_id}")
                for word, positions in words.items():
                    print(f"    Слово: {word}, вхождений: {len(positions)}, позиции: {positions}")
    else:
        print("  Совпадений не найдено.")
    
    print("3. Поиск фразы с позициями:")
    if no_index_results["phrase"]:
        for doc_id, positions in no_index_results["phrase"].items():
            print(f"  Документ: {doc_id}, позиции слов: {positions}")
    else:
        print("  Фраза не найдена в документах.")
    print(f"  Время выполнения: {no_index_time:.6f} секунд")
    
    start_time = time.perf_counter()
    index_results = search_with_index(query, inverted_index)
    index_time = time.perf_counter() - start_time
    
    print("\nПоиск с индексом:")
    print("1. Документы, содержащие все слова (пересечение):")
    if index_results["intersection"]:
        print("  Документы: ", ", ".join(index_results["intersection"]))
        for doc_id in index_results["intersection"]:
            print(f"    Документ: {doc_id}")
            for word, positions in index_results["individual"][doc_id].items():
                print(f"      Слово: {word}, вхождений: {len(positions)}, позиции: {positions}")
    else:
        print("  Нет документов, содержащих все слова.")
    
    print("2. Поиск слов по отдельности:")
    if index_results["individual"]:
        for doc_id, words in index_results["individual"].items():
            if doc_id not in index_results["intersection"]:  # Пропускаем документы из пересечения
                print(f"  Документ: {doc_id}")
                for word, positions in words.items():
                    print(f"    Слово: {word}, вхождений: {len(positions)}, позиции: {positions}")
    else:
        print("  Совпадений не найдено.")
    
    print("3. Поиск фразы с позициями:")
    if index_results["phrase"]:
        for doc_id, positions in index_results["phrase"].items():
            print(f"  Документ: {doc_id}, позиции слов: {positions}")
    else:
        print("  Фраза не найдена в документах.")
    print(f"  Время выполнения: {index_time:.6f} секунд")
    
    execution_times[query] = {"no_index": no_index_time, "index": index_time}

plt.show()