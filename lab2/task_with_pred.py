import heapq
import time
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

# Заданный граф (список смежности)
graph = {
    0: {1: 1, 2: 1},
    1: {0: 1, 3: 1, 4: 1},
    2: {0: 1, 5: 1, 6: 1},
    3: {1: 1, 2: 1},
    4: {1: 1, 7: 1, 8: 1},
    5: {2: 1},
    6: {2: 1, 8: 1},
    7: {4: 1, 9: 1},
    8: {6: 1, 9: 1},
    9: {8: 1}
}

# DFS (поиск в глубину)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    if start in visited:  # Если вершина уже посещена, не обрабатываем её повторно
        return []
    visited.add(start)
    order = [start]
    for next_vertex in sorted(graph[start].keys()):  # Просто проходим по смежным вершинам
        order.extend(dfs(graph, next_vertex, visited))
    return order


# BFS (поиск в ширину)
def bfs(graph, start, target=None):
    visited = set()
    queue = deque([start])
    order = []

    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            if target is not None and vertex == target:
                break
            queue.extend(neighbor for neighbor in graph[vertex] if neighbor not in visited)

    return order

# Алгоритм Дейкстры
def dijkstra(graph, start, target=None):
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    order = []

    while pq:
        current_distance, vertex = heapq.heappop(pq)

        if current_distance > distances[vertex]:
            continue

        order.append(vertex)
        if target is not None and vertex == target:
            break

        for neighbor, weight in graph[vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return order, distances

# Визуализация графа
def visualize_graph(graph, title="Граф"):
    G = nx.Graph()
    for vertex in graph:
        for neighbor in graph[vertex]:
            weight = graph[vertex][neighbor]
            G.add_edge(vertex, neighbor, weight=weight)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(title)
    plt.show()

# Замер времени выполнения алгоритма
def measure_time(algorithm, graph, start, target):
    start_time = time.perf_counter()
    result = algorithm(graph, start, target)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, result

# Основная функция
def main():
    start_vertex = 0

    bfs_time, bfs_order = measure_time(bfs, graph, start_vertex, 7)
    print(f"BFS: {bfs_time:.6f} сек\nПорядок обхода: {bfs_order}")

    dijkstra_time, dijkstra_result = measure_time(dijkstra, graph, start_vertex, 7)
    dijkstra_order, _ = dijkstra_result
    print(f"Дейкстра: {dijkstra_time:.6f} сек\nПорядок обхода: {dijkstra_order}")

    if bfs_order == dijkstra_order:
        print("Обходы равны")
    else:
        print("Обходы не равны")
    visualize_graph(graph, title="Заданный граф")

if __name__ == "__main__":
    main()
