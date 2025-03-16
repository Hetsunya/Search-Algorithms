package bm25

import (
	"log"
	"math"
	"strings"

	"mysearchengine/models"
)

// Функция для вычисления BM25 для документа
func CalculateBM25(queryWords []string, documents []models.Document, k1 float64, b float64, avgDocLength float64) []models.Document {
	var scoredResults []models.Document

	for _, doc := range documents {
		score := 0.0
		docLength := float64(len(strings.Join(strings.Fields(doc.Title), " ") + " " + doc.URL)) // Длина документа

		// Логируем длину документа
		log.Printf("Document: %s, Length: %f", doc.Title, docLength)

		for _, word := range queryWords {
			tf := calculateTermFrequency(word, doc)                   // Функция для подсчета TF
			idf := calculateInverseDocumentFrequency(word, documents) // IDF для слова

			// Логируем TF и IDF
			log.Printf("Word: %s, TF: %f, IDF: %f", word, tf, idf)

			score += idf * (tf * (k1 + 1)) / (tf + k1*(1-b+(b*(docLength/avgDocLength))))
		}

		// Логируем итоговый score для документа
		log.Printf("Final score for Document: %s, Score: %f", doc.Title, score)

		doc.Score = score
		scoredResults = append(scoredResults, doc)
	}

	// Сортируем результаты по убыванию оценки релевантности
	log.Printf("Sorting documents by score...")
	return sortByScore(scoredResults)
}

// Функция для подсчета TF (Term Frequency)
func calculateTermFrequency(word string, doc models.Document) float64 {
	count := 0
	words := strings.Fields(doc.Title + " " + doc.URL)
	for _, w := range words {
		if w == word {
			count++
		}
	}
	return float64(count) / float64(len(words))
}

// Функция для подсчета IDF (Inverse Document Frequency)
func calculateInverseDocumentFrequency(word string, documents []models.Document) float64 {
	docCount := len(documents)
	docWithWord := 0
	for _, doc := range documents {
		if strings.Contains(doc.Title+" "+doc.URL, word) {
			docWithWord++
		}
	}
	return math.Log(float64(docCount) / float64(docWithWord+1)) // Добавляем 1 для предотвращения деления на ноль
}

// Функция для сортировки документов по релевантности
func sortByScore(docs []models.Document) []models.Document {
	for i := 0; i < len(docs)-1; i++ {
		for j := i + 1; j < len(docs); j++ {
			if docs[i].Score < docs[j].Score {
				docs[i], docs[j] = docs[j], docs[i]
			}
		}
	}
	return docs
}
