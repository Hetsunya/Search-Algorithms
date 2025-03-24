import os
import re
import math
from collections import Counter

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яёa-z0-9 ]', '', text)
    return text.split()

def compute_tf(doc_tokens):
    tf = Counter(doc_tokens)
    doc_length = len(doc_tokens)
    return {term: freq / doc_length for term, freq in tf.items()}

def compute_idf(documents):
    num_docs = len(documents)
    idf = {}
    all_tokens = set(token for doc in documents for token in doc)
    for token in all_tokens:
        containing_docs = sum(1 for doc in documents if token in doc)
        idf[token] = math.log((num_docs + 1) / (containing_docs + 1)) + 1
    return idf

def compute_tfidf(tf, idf):
    return {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items()}

def cosine_similarity(vec1, vec2):
    common_terms = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

def search(query, documents):
    query_tokens = preprocess_text(query)
    query_tf = compute_tf(query_tokens)
    idf = compute_idf(documents)
    query_tfidf = compute_tfidf(query_tf, idf)
    
    scores = []
    for i, doc in enumerate(documents):
        doc_tf = compute_tf(doc)
        doc_tfidf = compute_tfidf(doc_tf, idf)
        similarity = cosine_similarity(query_tfidf, doc_tfidf)
        scores.append((i, similarity))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def load_documents(folder):
    documents = []
    filenames = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            documents.append(preprocess_text(text))
            filenames.append(filename)
    return documents, filenames

def main():
    folder = "kniga"  # Папка с текстовыми файлами
    query = "Гондора Мин-Риммон"
    
    documents, filenames = load_documents(folder)
    search_results = search(query, documents)
    
    print("Результаты поиска:")
    for doc_idx, score in search_results:
        print(f"Документ: {filenames[doc_idx]}, Сходство: {score:.4f}")

if __name__ == "__main__":
    main()
