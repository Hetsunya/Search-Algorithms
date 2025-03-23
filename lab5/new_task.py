import os
import math
import time
import matplotlib.pyplot as plt

# Папка с текстовыми файлами
DATA_PATH = "kniga"

# 1. Функция загрузки документов
def load_documents(data_path):
    documents = {}
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            with open(os.path.join(data_path, filename), "r", encoding="utf-8") as file:
                documents[filename] = file.read()
    return documents

# 2. Токенизация и предобработка текста
def preprocess_text(text):
    text = text.lower()
    text = ''.join(c if c.isalpha() or c.isspace() else ' ' for c in text)
    tokens = [word for word in text.split() if word]
    stemmed_tokens = [word[:-2] if len(word) > 4 else word for word in tokens]
    return stemmed_tokens

# 3. Построение матрицы частот (TF) и подсчёт документов с терминами
def build_tf_matrix_and_doc_freq(documents):
    tf_matrix = {}
    vocab = set()
    doc_freq = {}  # Подсчёт документов, содержащих слово
    preprocessed_docs = {}  # Кэшируем токены
    
    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        preprocessed_docs[doc_id] = tokens
        total_words = len(tokens)
        word_count = {}
        unique_words = set(tokens)  # Уникальные слова в документе
        
        for word in tokens:
            word_count[word] = word_count.get(word, 0) + 1
            vocab.add(word)
        
        for word in unique_words:
            doc_freq[word] = doc_freq.get(word, 0) + 1
        
        tf_matrix[doc_id] = {word: count / total_words for word, count in word_count.items()}
    
    return tf_matrix, vocab, doc_freq, preprocessed_docs

# 4. Вычисление IDF (оптимизировано)
def compute_idf(total_docs, doc_freq, vocab):
    idf = {}
    for word in vocab:
        idf[word] = math.log(total_docs / (doc_freq.get(word, 0) + 1))
    return idf

# 5. Построение матрицы TF-IDF
def build_tfidf_matrix(tf_matrix, idf):
    tfidf_matrix = {}
    for doc_id, tf_scores in tf_matrix.items():
        tfidf_matrix[doc_id] = {word: tf * idf.get(word, 0) for word, tf in tf_scores.items()}
    return tfidf_matrix

# 6. Вектор запроса в TF-IDF
def query_to_tfidf_vector(query, idf, vocab):
    query_tokens = preprocess_text(query)
    total_words = len(query_tokens)
    tf_query = {}
    for word in query_tokens:
        tf_query[word] = tf_query.get(word, 0) + 1 / total_words
    return {word: tf * idf.get(word, 0) for word, tf in tf_query.items() if word in vocab}

# 7. Косинусное сходство
def cosine_similarity(vec1, vec2):
    dot_product = 0
    norm1 = 0
    norm2 = 0
    all_keys = set(vec1.keys()) | set(vec2.keys())
    
    for key in all_keys:
        v1 = vec1.get(key, 0)
        v2 = vec2.get(key, 0)
        dot_product += v1 * v2
        norm1 += v1 ** 2
        norm2 += v2 ** 2
    
    norm1 = math.sqrt(norm1)
    norm2 = math.sqrt(norm2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

# 8. Ранжирование документов
def rank_documents(query, tfidf_matrix, idf, vocab):
    query_vector = query_to_tfidf_vector(query, idf, vocab)
    rankings = []
    for doc_id, doc_vector in tfidf_matrix.items():
        similarity = cosine_similarity(query_vector, doc_vector)
        rankings.append((doc_id, similarity))
    return sorted(rankings, key=lambda x: x[1], reverse=True)

# 9. Простой подсчёт встречаемости для сравнения
def simple_count_search(query, preprocessed_docs):
    query_tokens = preprocess_text(query)
    rankings = []
    for doc_id, tokens in preprocessed_docs.items():
        count = sum(tokens.count(token) for token in query_tokens)
        rankings.append((doc_id, count))
    return sorted(rankings, key=lambda x: x[1], reverse=True)

# 10. Генерация релевантных документов
def generate_relevant_docs(queries, preprocessed_docs):
    relevant_docs = {}
    for query in queries:
        query_tokens = set(preprocess_text(query))
        relevant = []
        for doc_id, tokens in preprocessed_docs.items():
            if query_tokens.issubset(set(tokens)):
                relevant.append(doc_id)
        relevant_docs[query] = relevant if relevant else [list(preprocessed_docs.keys())[0]]
    return relevant_docs

# 11. Оценка качества
def evaluate_search(ranked_docs, relevant_docs):
    retrieved = [doc_id for doc_id, _ in ranked_docs]
    relevant = set(relevant_docs)
    retrieved_relevant = set(retrieved) & relevant
    
    precision = len(retrieved_relevant) / len(retrieved) if retrieved else 0
    recall = len(retrieved_relevant) / len(relevant) if relevant else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Основной код
documents = load_documents(DATA_PATH)
start_time = time.perf_counter()
tf_matrix, vocab, doc_freq, preprocessed_docs = build_tf_matrix_and_doc_freq(documents)
idf = compute_idf(len(documents), doc_freq, vocab)
tfidf_matrix = build_tfidf_matrix(tf_matrix, idf)
prep_time = time.perf_counter() - start_time

print(f"Подготовка данных завершена за {prep_time:.4f} секунд")
print(f"Размер словаря: {len(vocab)} слов")
print(f"Количество документов: {len(documents)}")

# Тестовые запросы
queries = ["затрепетали на ветру знамена", "Гондора Мин-Риммон"]

# Генерация релевантных документов
relevant_docs = generate_relevant_docs(queries, preprocessed_docs)
print("\nСгенерированные релевантные документы:")
for query, docs in relevant_docs.items():
    print(f"  Запрос: \"{query}\", релевантные документы: {docs}")

# Выполнение и анализ
for query in queries:
    print(f"\n\n=== Анализ запроса: \"{query}\" ===")
    
    # Поиск с TF-IDF
    start_time = time.perf_counter()
    tfidf_ranked = rank_documents(query, tfidf_matrix, idf, vocab)
    tfidf_time = time.perf_counter() - start_time
    
    print("\nTF-IDF ранжирование:")
    print(f"  Всего документов: {len(tfidf_ranked)}")
    for doc_id, score in tfidf_ranked[:5]:
        print(f"  Документ: {doc_id}, косинусное сходство: {score:.4f}")
    print(f"  Время выполнения: {tfidf_time:.6f} секунд")
    
    # Простой подсчёт
    start_time = time.perf_counter()
    simple_ranked = simple_count_search(query, preprocessed_docs)
    simple_time = time.perf_counter() - start_time
    
    print("\nПростой подсчёт встречаемости:")
    print(f"  Всего документов: {len(simple_ranked)}")
    for doc_id, count in simple_ranked[:5]:
        print(f"  Документ: {doc_id}, вхождений: {count}")
    print(f"  Время выполнения: {simple_time:.6f} секунд")
    
    # Оценка качества для TF-IDF
    precision, recall, f1 = evaluate_search(tfidf_ranked, relevant_docs[query])
    print("\nОценка качества (TF-IDF):")
    print(f"  Релевантных документов в корпусе: {len(relevant_docs[query])}")
    print(f"  Найдено релевантных: {len(set([doc_id for doc_id, _ in tfidf_ranked]) & set(relevant_docs[query]))}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    
    # Сравнительный анализ
    print("\nСравнительный анализ:")
    tfidf_top = set(doc_id for doc_id, _ in tfidf_ranked[:5])
    simple_top = set(doc_id for doc_id, _ in simple_ranked[:5])
    print(f"  Пересечение топ-5 (TF-IDF и простой): {len(tfidf_top & simple_top)} документов")
    print(f"  Уникальные для TF-IDF: {tfidf_top - simple_top}")
    print(f"  Уникальные для простого подсчёта: {simple_top - tfidf_top}")

# График времени выполнения
queries_labels = [f"Запрос {i+1}" for i in range(len(queries))]
tfidf_times = []
simple_times = []

for query in queries:
    start = time.perf_counter()
    rank_documents(query, tfidf_matrix, idf, vocab)
    tfidf_times.append(time.perf_counter() - start)
    
    start = time.perf_counter()
    simple_count_search(query, preprocessed_docs)
    simple_times.append(time.perf_counter() - start)

plt.figure(figsize=(10, 5))
plt.bar(queries_labels, tfidf_times, label="TF-IDF", alpha=0.7)
plt.bar(queries_labels, simple_times, label="Простой подсчёт", alpha=0.7)
plt.xlabel("Запросы")
plt.ylabel("Время (сек)")
plt.title("Сравнение времени выполнения")
plt.xticks(rotation=45)
plt.legend()
plt.show()