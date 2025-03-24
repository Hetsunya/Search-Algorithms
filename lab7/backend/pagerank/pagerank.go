package pagerank

import (
	"log"

	"mysearchengine/models"

)

// CalculatePageRank вычисляет PageRank для списка документов
func CalculatePageRank(documents []models.Document, d float64, iterations int) map[string]float64 {
	n := len(documents)
	ranks := make(map[string]float64)

	// Начальные значения PR
	for _, doc := range documents {
		ranks[doc.URL] = 1.0 / float64(n)
	}
	log.Printf("Initial PageRank values:")
	for _, doc := range documents {
		log.Printf("Document: %s, Initial PageRank: %f", doc.Title, ranks[doc.URL])
	}

	// Итерации PageRank
	for i := 0; i < iterations; i++ {
		newRanks := make(map[string]float64)
		for _, doc := range documents {
			rankSum := 0.0
			// Ищем документы, ссылающиеся на текущий
			for _, otherDoc := range documents {
				for _, link := range otherDoc.URLLinks {
					if link == doc.URL {
						numLinks := len(otherDoc.URLLinks)
						if numLinks > 0 {
							rankSum += ranks[otherDoc.URL] / float64(numLinks)
						}
					}
				}
			}
			newRanks[doc.URL] = (1.0-d)/float64(n) + d*rankSum
		}
		ranks = newRanks
		log.Printf("PageRank values after iteration %d:", i+1)
		for _, doc := range documents {
			log.Printf("Document: %s, PageRank: %f", doc.Title, ranks[doc.URL])
		}
	}

	return ranks
}
