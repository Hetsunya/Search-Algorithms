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
	log.Printf("Search query: %s", query)

	// Получаем все документы из базы для BM25 и PageRank
	rows, err := db.Query("SELECT id, title, url, content, links FROM documents")
	if err != nil {
		http.Error(w, fmt.Sprintf("Database query error: %v", err), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	allDocs := make(map[int]models.Document)
	for rows.Next() {
		var doc models.Document
		var linksStr string
		if err := rows.Scan(&doc.ID, &doc.Title, &doc.URL, &doc.Content, &linksStr); err != nil {
			http.Error(w, fmt.Sprintf("Error scanning row: %v", err), http.StatusInternalServerError)
			return
		}
		doc.URLLinks = strings.Split(linksStr, ",")
		allDocs[doc.ID] = doc
	}

	// Поиск релевантных документов через индекс
	results := make(map[int]models.Document)
	for _, word := range queryWords {
		log.Printf("Searching for word: %s", word)
		rows, err := db.Query(`
            SELECT documents.id, documents.title, documents.url, documents.content, documents.links
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
			var linksStr string
			if err := rows.Scan(&doc.ID, &doc.Title, &doc.URL, &doc.Content, &linksStr); err != nil {
				http.Error(w, fmt.Sprintf("Error scanning row: %v", err), http.StatusInternalServerError)
				return
			}
			doc.URLLinks = strings.Split(linksStr, ",")
			results[doc.ID] = doc
		}
	}

	// Преобразуем результаты в список
	var resultList []models.Document
	for _, doc := range results {
		resultList = append(resultList, doc)
	}
	log.Printf("Found documents before ranking: %d", len(resultList))

	// Расчёт BM25
	resultList = bm25.CalculateBM25(queryWords, resultList, 1.5, 0.75, allDocs)

	// Расчёт PageRank для всех документов
	var allDocList []models.Document
	for _, doc := range allDocs {
		allDocList = append(allDocList, doc)
	}
	pageRanks := pagerank.CalculatePageRank(allDocList, 0.85, 10)

	// Комбинируем BM25 и PageRank только для результатов поиска
	for i := range resultList {
		resultList[i].Score += pageRanks[resultList[i].URL]
	}

	// Отправляем результат
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resultList)
}

func main() {
	http.HandleFunc("/search", searchHandler)
	fmt.Println("Server started on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
