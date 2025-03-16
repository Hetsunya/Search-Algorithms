import numpy as np
import matplotlib.pyplot as plt

# Расширенный граф в виде матрицы смежности
G = np.array([
    [0, 1, 1, 1, 0],  # Страница A
    [1, 0, 1, 1, 0],  # Страница B
    [1, 1, 0, 0, 0],  # Страница C
    [0, 1, 1, 0, 0],  # Страница D
    [1, 0, 0, 1, 0],  # Страница E
])

# Инициализация
N = G.shape[0]  # количество страниц
d_values = [0.6, 0.7, 0.85, 0.95]  # разные значения damping factor
PR_init = np.ones(N) / N  # начальные значения PageRank

# Функция для расчета нового значения PR
def pagerank(G, PR, d, max_iter=100, tol=1e-6):
    for i in range(max_iter):
        new_PR = (1 - d) / N + d * np.dot(G.T, PR / np.sum(G, axis=1))
        if np.linalg.norm(new_PR - PR, 1) < tol:
            break
        PR = new_PR
    return PR

# Эксперименты с разными значениями damping factor
results = {}
for d in d_values:
    PR = pagerank(G, PR_init, d)
    results[d] = PR

    # Выводим результаты на график
    plt.plot(range(1, N + 1), PR, label=f'd = {d:.2f}')

# Отображение результатов на графике
plt.title('PageRank для разных значений damping factor')
plt.xlabel('Страница')
plt.ylabel('PageRank')
plt.xticks(range(1, N + 1), [f'Page {i+1}' for i in range(N)])
plt.legend()
plt.grid(True)
plt.show()

# Вывод результатов
print("PageRank для разных значений damping factor:")
for d, PR in results.items():
    print(f"\ndamping factor = {d}")
    for i, pr in enumerate(PR):
        print(f"Страница {i+1}: {pr:.4f}")
