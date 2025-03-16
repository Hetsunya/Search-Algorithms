package models

type Document struct {
	ID       int
	Title    string
	URL      string
	Content  string
	URLLinks []string // Если страницы ссылаются друг на друга
	Score    float64  // Оценка релевантности
}
