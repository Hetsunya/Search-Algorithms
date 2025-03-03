import heapq
import time
import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

# DFS (поиск в глубину)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for next_vertex in sorted(graph[start].keys() - visited):
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
        queue = deque([start])
        order = []

        while queue:
            vertex = queue.popleft()
            order.append(vertex)
            if target is not None and vertex == target:
                break

            for neighbor in graph[vertex]:
                if distances[neighbor] == float('inf'):  # Если еще не посещали
                    distances[neighbor] = distances[vertex] + 1
                    queue.append(neighbor)

        return order, distances


# Генерация графа (список смежности)
def generate_graph(num_vertices, density=0.5, equal_weight=False):
    graph = {i: {} for i in range(num_vertices)}
    weight = 1 if equal_weight else None
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < density:
                edge_weight = weight if equal_weight else random.randint(1, 10)
                graph[i][j] = edge_weight
                graph[j][i] = edge_weight
    return graph

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
def measure_time(algorithm, graph, start):
    start_time = time.perf_counter()
    result = algorithm(graph, start)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, result

# Основная функция для экспериментов
def main():
    sizes = [10, 20, 50, 100]
    bfs_times = []
    dfs_times = []
    dijkstra_times = []

    for size in sizes:
        print(f"\nГраф из {size} вершин:")
        graph = generate_graph(size)

        bfs_time, bfs_order = measure_time(bfs, graph, 0)
        bfs_times.append(bfs_time)
        print(f"BFS: {bfs_time:.6f} сек")

        dfs_time, dfs_order = measure_time(dfs, graph, 0)
        dfs_times.append(dfs_time)
        print(f"DFS: {dfs_time:.6f} сек")

        dijkstra_time, dijkstra_order = measure_time(dijkstra, graph, 0)
        dijkstra_times.append(dijkstra_time)
        print(f"Дейкстра: {dijkstra_time:.6f} сек")

        if size == 10:
            print(graph)
            print("Порядок обхода BFS:", bfs_order)
            print("Порядок обхода DFS:", dfs_order)
            print("Порядок обхода Дейкстры:", dijkstra_order)
            visualize_graph(graph, title=f"Граф ({size} вершин)")

    equal_graph = generate_graph(10, equal_weight=True)
    target_vertex = 7
    bfs_target_order = bfs(equal_graph, 0, target=target_vertex)
    dijkstra_target_order = dijkstra(equal_graph, 0, target=target_vertex)

    print("Порядок обхода BFS:", bfs_target_order, "Длина обхода: ", len(bfs_target_order))
    print("Порядок обхода Дейкстры:", dijkstra_target_order, "Длина обхода: ", len(dijkstra_target_order))
    if bfs_target_order == dijkstra_target_order[0]:
        print("Обходы равны")
    else:
        print("Обходы не равны")
    visualize_graph(equal_graph, title="Равновзвешенный граф (все веса = 1)")

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, bfs_times, marker='o', label='BFS')
    plt.plot(sizes, dfs_times, marker='o', label='DFS')
    plt.plot(sizes, dijkstra_times, marker='o', label='Дейкстра')
    plt.xlabel('Размер графа (количество вершин)')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Сравнение времени выполнения BFS, DFS и Дейкстры')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
