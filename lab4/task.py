import os
import re
import json
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Загрузка необходимых ресурсов
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Папка с текстовыми файлами
DATA_PATH = "News Articles"

print("Стоп слова:")
print(set(stopwords.words('english')))

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
# def search(query, index):
#     query_tokens = preprocess_text(query)
#     results = defaultdict(set)
#     for token in query_tokens:
#         if token in index:
#             for doc_id in index[token]:
#                 results[doc_id].add(token)
#     return sorted(results.items(), key=lambda x: len(x[1]), reverse=True)


# Запуск индексирования
documents = load_documents(DATA_PATH)
inverted_index = build_inverted_index(documents)

# Сохранение индекса в файл
with open("inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, ensure_ascii=False, indent=4)

def search(query, index):
    query_tokens = preprocess_text(query)
    results = defaultdict(set)  # Обычный поиск
    phrase_results = defaultdict(list)  # Фразовый поиск с позициями

    doc_positions = defaultdict(lambda: defaultdict(list))  # {doc_id: {token: [pos1, pos2]}}

    for token in query_tokens:
        if token in index:
            for doc_id, positions in index[token].items():
                results[doc_id].add(token)
                doc_positions[doc_id][token] = sorted(positions)

    # Фразовый поиск возможен только если > 1 слова
    if len(query_tokens) > 1:
        for doc_id, token_pos in doc_positions.items():
            if len(token_pos) < len(query_tokens):
                continue

            # Список позиций каждого слова
            positions = [token_pos[token] for token in query_tokens]

            # Проверка фразы в прямом порядке
            for start_pos in positions[0]:
                if all(start_pos + i in positions[i] for i in range(1, len(query_tokens))):
                    phrase_results[doc_id].append(start_pos)

            reversed_positions = positions[::-1]
            for start_pos in reversed_positions[0]:
                if all(start_pos + i in reversed_positions[i] for i in range(1, len(query_tokens))):
                    phrase_results[doc_id].append(start_pos)

    return {
        "normal_search": sorted(results.items(), key=lambda x: len(x[1]), reverse=True),
        "phrase_search": phrase_results
    }


querys = ["technology market", "virus attack", "Windows virus disguising", "Virus poses", "Poses virus"]

for query in querys:
    search_results = search(query, inverted_index)

    print(f"\nРезультаты поиска для: \"{query}\"")

    # Обычный поиск (пересечение слов)
    print("\nОбычный поиск:")
    if search_results["normal_search"]:
        for doc_id, words in search_results["normal_search"]:
            print(f"  Документ: {doc_id}")
            for word in words:
                positions = inverted_index[word][doc_id]
                print(f"    - Слово: \"{word}\", позиции: {positions}")
    else:
        print("  Совпадений не найдено.")

    # Фразовый поиск (слова подряд)
    print("\nФразовый поиск:")
    if search_results["phrase_search"]:
        for doc_id, positions in search_results["phrase_search"].items():
            positions_str = ", ".join(str(pos) for pos in positions)
            print(f"  Документ: {doc_id}, позиции начала фразы: {positions_str}")
    else:
        print("  Фраза не найдена в текстах.")


