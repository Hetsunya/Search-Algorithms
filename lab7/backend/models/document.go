package models

type Document struct {
	ID       int
	Title    string
	URL      string
	Content  string
	URLLinks []string // Ссылки на другие страницы, загруженные из базы
	Score    float64  // Оценка релевантности
}
