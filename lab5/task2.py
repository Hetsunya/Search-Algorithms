import json

def load_relevance_data():
    """Загружает разметку релевантных документов."""
    return {
        "Гондора Мин-Риммон": {"часть_1.txt", "часть_4.txt"},
        "рохиррим": {"часть_5.txt", "часть_8.txt"},
        "войске": {"часть_3.txt", "часть_6.txt", "часть_9.txt"}
    }

def load_search_results(filename="search_results.json"):
    """Загружает результаты поиска."""
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

def precision_recall_f1(retrieved, relevant):
    """Вычисляет Precision, Recall и F1-score."""
    retrieved = set(retrieved)
    relevant = set(relevant)
    
    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate():
    relevance_data = load_relevance_data()
    search_results = load_search_results()
    
    for query, retrieved_docs in search_results.items():
        relevant_docs = relevance_data.get(query, set())
        precision, recall, f1 = precision_recall_f1(retrieved_docs, relevant_docs)
        
        print(f"Запрос: {query}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}\n")

if __name__ == "__main__":
    evaluate()
