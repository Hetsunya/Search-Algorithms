package pagerank

import (
	"log"

	"mysearchengine/models" // Здесь предполагается, что у вас есть этот пакет для структуры Document
)

// Функция для вычисления PageRank
func CalculatePageRank(documents []models.Document, d float64, iterations int) map[string]float64 {
	n := len(documents)
	ranks := make(map[string]float64)

	// Начальная оценка: все документы имеют одинаковую оценку
	for _, doc := range documents {
		ranks[doc.URL] = 1.0 / float64(n) // Начальная оценка
	}

	// Логируем начальные оценки
	log.Printf("Initial PageRank values:")
	for _, doc := range documents {
		log.Printf("Document: %s, Initial PageRank: %f", doc.Title, ranks[doc.URL])
	}

	// Итерации PageRank
	for i := 0; i < iterations; i++ {
		newRanks := make(map[string]float64)
		for _, doc := range documents {
			rankSum := 0.0
			// Получаем ссылки на текущую страницу (doc.URLLinks)
			for _, link := range doc.URLLinks {
				// Проверяем, что ссылка существует в ranks
				if rank, exists := ranks[link]; exists {
					rankSum += rank / float64(len(doc.URLLinks)) // Равномерное распределение
				}
			}
			// Обновляем рейтинг документа
			newRanks[doc.URL] = (1.0-d)/float64(n) + d*rankSum
		}
		ranks = newRanks

		// Логируем промежуточные результаты после каждой итерации
		log.Printf("PageRank values after iteration %d:", i+1)
		for _, doc := range documents {
			log.Printf("Document: %s, PageRank: %f", doc.Title, ranks[doc.URL])
		}
	}

	// Логируем итоговые оценки PageRank
	log.Printf("Final PageRank values after %d iterations:", iterations)
	for _, doc := range documents {
		log.Printf("Document: %s, PageRank: %f", doc.Title, ranks[doc.URL])
	}

	return ranks
}
