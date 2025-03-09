package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	_ "github.com/mattn/go-sqlite3"
)

var db *sql.DB

// Структура для хранения документа
type Document struct {
	ID    int
	Title string
	URL   string
}

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
	var results []Document

	for _, word := range queryWords {
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
			var doc Document
			if err := rows.Scan(&doc.ID, &doc.Title, &doc.URL); err != nil {
				http.Error(w, fmt.Sprintf("Error scanning row: %v", err), http.StatusInternalServerError)
				return
			}
			results = append(results, doc)
		}
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
