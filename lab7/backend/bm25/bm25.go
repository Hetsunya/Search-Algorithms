package bm25

import (
	"log"
	"math"
	"strings"

	"mysearchengine/models"
)

// CalculateBM25 вычисляет BM25 для списка документов
func CalculateBM25(queryWords []string, documents []models.Document, k1 float64, b float64, allDocs map[int]models.Document) []models.Document {
	// Вычисляем среднюю длину документа
	totalLength := 0.0
	for _, doc := range allDocs {
		totalLength += float64(len(strings.Fields(doc.Content)))
	}
	avgDocLength := totalLength / float64(len(allDocs))
	log.Printf("Average document length: %f", avgDocLength)

	var scoredResults []models.Document
	for _, doc := range documents {
		score := 0.0
		docLength := float64(len(strings.Fields(doc.Content)))
		log.Printf("Document: %s, Length: %f", doc.Title, docLength)

		for _, word := range queryWords {
			tf := calculateTermFrequency(word, doc)
			idf := calculateInverseDocumentFrequency(word, documents)
			log.Printf("Word: %s, TF: %f, IDF: %f", word, tf, idf)
			score += idf * (tf * (k1 + 1)) / (tf + k1*(1-b+b*(docLength/avgDocLength)))
		}
		log.Printf("Final BM25 score for Document: %s, Score: %f", doc.Title, score)
		doc.Score = score
		scoredResults = append(scoredResults, doc)
	}

	return sortByScore(scoredResults)
}

// calculateTermFrequency считает TF с учётом Content
func calculateTermFrequency(word string, doc models.Document) float64 {
	count := 0
	words := strings.Fields(doc.Content)
	for _, w := range words {
		if strings.ToLower(w) == word {
			count++
		}
	}
	return float64(count) / float64(len(words))
}

// calculateInverseDocumentFrequency считает IDF
func calculateInverseDocumentFrequency(word string, documents []models.Document) float64 {
	docCount := len(documents)
	docWithWord := 0
	for _, doc := range documents {
		if strings.Contains(strings.ToLower(doc.Content), word) {
			docWithWord++
		}
	}
	return math.Log(float64(docCount) / float64(docWithWord+1))
}

// sortByScore сортирует документы по убыванию Score
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
