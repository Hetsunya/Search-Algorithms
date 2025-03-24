import sqlite3
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")

DB_PATH = "search_engine.db"

def create_index_table():
    """Создаёт таблицу для хранения инвертированного индекса"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS index_table (
            word TEXT,
            doc_id INTEGER,
            frequency INTEGER,
            PRIMARY KEY (word, doc_id),
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        )
    ''')
    conn.commit()
    conn.close()

def tokenize(text, use_stemming=False):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    if use_stemming:
        stemmer = PorterStemmer()
        return [stemmer.stem(w) for w in words if w.isalnum() and w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in words if w.isalnum() and w not in stop_words]

def build_index():
    """Создаёт инвертированный индекс и сохраняет его в SQLite"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, content FROM documents")
    docs = c.fetchall()
    conn.close()

    index = defaultdict(lambda: defaultdict(int))

    for doc_id, content in docs:
        words = tokenize(content)
        for word in words:
            index[word][doc_id] += 1  # Частота слова в документе

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for word, doc_dict in index.items():
        for doc_id, freq in doc_dict.items():
            c.execute("INSERT OR IGNORE INTO index_table (word, doc_id, frequency) VALUES (?, ?, ?)", 
                      (word, doc_id, freq))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_index_table()
    build_index()
    print("Индексирование завершено.")
