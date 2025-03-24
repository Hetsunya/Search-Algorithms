import requests
import sqlite3
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

DB_PATH = "search_engine.db"
MAX_DEPTH = 30  # Глубина обхода ссылок

def create_db():
    """Создаёт таблицу для хранения страниц"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            content TEXT,
            links TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_page(url, title, content, links):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO documents (url, title, content, links) VALUES (?, ?, ?, ?)", 
              (url, title, content, ','.join(links)))
    conn.commit()
    conn.close()

def clean_text(html):
    """Удаляет теги и возвращает чистый текст"""
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator=" ", strip=True)

def crawl(url, depth=0, visited=set()):
    """Рекурсивный обход страниц"""
    if depth > MAX_DEPTH or url in visited:
        return
    visited.add(url)

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        html = response.text
    except requests.RequestException:
        return

    text = clean_text(html)
    title = BeautifulSoup(html, "html.parser").title.string if BeautifulSoup(html, "html.parser").title else url

    # Поиск новых ссылок
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("a", href=True):
        next_url = urljoin(url, link["href"])
        if urlparse(next_url).netloc == urlparse(url).netloc:  # Остаёмся на том же домене
            crawl(next_url, depth + 1, visited)
    
    
    links = [urljoin(url, link["href"]) for link in soup.find_all("a", href=True) 
             if urlparse(urljoin(url, link["href"])).netloc == urlparse(url).netloc]
    save_page(url, title, text, links)

if __name__ == "__main__":
    create_db()
    start_url = "https://books.toscrape.com/"  # Задай свой сайт
    crawl(start_url)
