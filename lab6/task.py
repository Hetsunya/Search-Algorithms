import numpy as np

def pagerank(graph, damping=0.85, max_iter=100, tol=1.0e-6):
    n = len(graph)
    pr = np.ones(n) / n
    M = np.zeros((n, n))

    for i in range(n):
        if len(graph[i]) > 0:
            M[i, graph[i]] = 1 / len(graph[i])
        else:
            M[i, :] = 1 / n  # Dangling node correction

    for _ in range(max_iter):
        new_pr = (1 - damping) / n + damping * M.T @ pr
        if np.linalg.norm(new_pr - pr, 1) < tol:
            break
        pr = new_pr

    return pr

# Пример графа: индексы соответствуют страницам, а значения — спискам ссылок
graph = {
    0: [1, 2],
    1: [2],
    2: [0],
    3: [2, 4],
    4: [3]
}

graph_list = [graph[i] if i in graph else [] for i in range(len(graph))]

damping_factors = [0.6, 0.7, 0.85, 0.95]

for d in damping_factors:
    pr_values = pagerank(graph_list, damping=d)
    print(f"Damping factor: {d}\nPageRank: {pr_values}\n")
