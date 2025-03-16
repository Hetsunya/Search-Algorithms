package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	_ "github.com/mattn/go-sqlite3"

	"mysearchengine/bm25"
	"mysearchengine/models"
	"mysearchengine/pagerank"
)

var db *sql.DB

func init() {
	var err error
	// Открытие базы данных
	db, err = sql.Open("sqlite3", "./search_engine.db")
	if err != nil {
		log.Fatal(err)
	}
}

func setHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

func searchHandler(w http.ResponseWriter, r *http.Request) {
	setHeaders(w)

	query := r.URL.Query().Get("query")
	if query == "" {
		http.Error(w, "Query parameter is missing", http.StatusBadRequest)
		return
	}

	queryWords := strings.Fields(strings.ToLower(query))
	var results []models.Document

	// Логируем запрос
	log.Printf("Search query: %s", query)

	// Получаем документы, соответствующие запросу
	for _, word := range queryWords {
		log.Printf("Searching for word: %s", word)
		rows, err := db.Query(`
            SELECT documents.id, documents.title, documents.url
            FROM documents
            JOIN index_table ON documents.id = index_table.doc_id
            WHERE index_table.word = ?
        `, word)

		if err != nil {
			http.Error(w, fmt.Sprintf("Database query error: %v", err), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		for rows.Next() {
			var doc models.Document
			if err := rows.Scan(&doc.ID, &doc.Title, &doc.URL); err != nil {
				http.Error(w, fmt.Sprintf("Error scanning row: %v", err), http.StatusInternalServerError)
				return
			}
			results = append(results, doc)
		}
	}

	// Логируем найденные документы до BM25
	log.Printf("Found documents before BM25 calculation: %d", len(results))

	// Расчет BM25
	results = bm25.CalculateBM25(queryWords, results, 1.5, 0.75, 150)

	// Логируем результаты после BM25
	log.Printf("Documents after BM25 calculation:")
	for _, doc := range results {
		log.Printf("Document: %s, Score: %f", doc.Title, doc.Score)
	}

	// Дополнительно можно добавить расчет PageRank
	pageRanks := pagerank.CalculatePageRank(results, 0.85, 10)

	// Логируем значения PageRank
	log.Printf("PageRank values:")
	for _, doc := range results {
		log.Printf("Document: %s, PageRank: %f", doc.Title, pageRanks[doc.URL])
	}

	// Отображение результатов с учетом BM25 и PageRank
	for i := range results {
		results[i].Score += pageRanks[results[i].URL] // Добавляем влияние PageRank в итоговый рейтинг
	}

	// Логируем итоговые результаты
	log.Printf("Final results after combining BM25 and PageRank:")
	for _, doc := range results {
		log.Printf("Document: %s, Final Score: %f", doc.Title, doc.Score)
	}

	// Отправляем результат в формате JSON
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

func main() {
	// Обработчик для поиска
	http.HandleFunc("/search", searchHandler)

	// Запуск HTTP-сервера
	fmt.Println("Server started on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
