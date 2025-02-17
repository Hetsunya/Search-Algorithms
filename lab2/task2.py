import heapq
import time
import random
import matplotlib.pyplot as plt
import networkx as nx

# DFS (оставляем без изменений)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for next_vertex in sorted(graph[start].keys() - visited):  # Добавил sorted()
        order.extend(dfs(graph, next_vertex, visited))
    return order

# BFS с приоритетной очередью (чтобы совпадал порядок с Дейкстрой)
def bfs(graph, start, target=None):
    visited = set()
    pq = [(0, start)]  # Используем кортеж (0, вершина), чтобы эмулировать min-heap
    order = []

    while pq:
        _, vertex = heapq.heappop(pq)  # Берем вершину с минимальным "приоритетом"
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            # Если нашли целевую вершину, выходим
            if target is not None and vertex == target:
                break
            # Добавляем соседей в том же порядке, что и Дейкстра
            for neighbor in sorted(graph[vertex].keys()):
                if neighbor not in visited:
                    heapq.heappush(pq, (0, neighbor))  # "Приоритет" не влияет

    return order

# Дейкстра без учета весов (чтобы совпадал с BFS)
def dijkstra(graph, start, target=None):
    visited = set()
    pq = [(0, start)]  # Вес пути всегда 0, как в BFS
    order = []

    while pq:
        _, vertex = heapq.heappop(pq)
        if vertex in visited:
            continue
        visited.add(vertex)
        order.append(vertex)
        # Если нашли целевую вершину, выходим
        if target is not None and vertex == target:
            break
        # Добавляем соседей так же, как в BFS
        for neighbor in sorted(graph[vertex].keys()):
            if neighbor not in visited:
                heapq.heappush(pq, (0, neighbor))

    return order

# Генерация графа
def generate_graph(num_vertices, density=0.4):
    graph = {i: {} for i in range(num_vertices)}
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < density:
                graph[i][j] = graph[j][i] = 1  # Все веса = 1
    return graph

# Тест
graph = generate_graph(10)
start_vertex = 0

print("Граф:", graph)
print("BFS порядок:", bfs(graph, start_vertex))
print("Dijkstra порядок:", dijkstra(graph, start_vertex))
