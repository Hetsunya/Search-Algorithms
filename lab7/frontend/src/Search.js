import React, { useState } from 'react';
import axios from 'axios';

const Search = () => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSearch = async () => {
        if (!query.trim()) return;

        setLoading(true);
        setError('');
        
        try {
            const response = await axios.get(`http://localhost:8080/search?query=${query}`);
            setResults(response.data);
        } catch (err) {
            setError('Ошибка при получении результатов поиска.');
        }
        setLoading(false);
    };

    // Проверяем, что results является массивом перед сортировкой
    const sortedResults = Array.isArray(results) ? [...results].sort(function(a, b) {
        return b.Score - a.Score;
    }) : [];

    return (
        <div>
            <h1>Поиск на сайте</h1>
            <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Введите запрос"
            />
            <button onClick={handleSearch} disabled={loading}>
                {loading ? 'Поиск...' : 'Искать'}
            </button>

            {error && <p>{error}</p>}

            {sortedResults.length > 0 ? (
                <ul>
                    {sortedResults.map((result) => (
                        <li key={result.ID}>
                            <a href={result.URL} target="_blank" rel="noopener noreferrer">
                                {result.Title.replace(/\n/g, '').trim()}
                            </a>
                            <br />
                            <small>Рейтинг: {result.Score.toFixed(4)}</small>
                        </li>
                    ))}
                </ul>
            ) : (
                <p>Нет результатов для вашего запроса.</p>
            )}
        </div>
    );
};

export default Search;
