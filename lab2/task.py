import heapq
import time
import random
import matplotlib.pyplot as plt
import networkx as nx

# Реализация BFS (поиск в ширину)
def bfs(graph, start):
    visited = set()
    queue = [start]
    order = []
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            queue.extend(graph[vertex] - visited)
    return order

# Реализация DFS (поиск в глубину)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for next_vertex in graph[start] - visited:
        order.extend(dfs(graph, next_vertex, visited))
    return order

# Реализация алгоритма Дейкстры
def dijkstra(graph, start):
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

# Генерация графа (список смежности)
def generate_graph(num_vertices, density=0.5, weighted=False):
    if weighted:
        # Для взвешенного графа используем словарь для хранения смежных вершин с весами
        graph = {i: {} for i in range(num_vertices)}
    else:
        # Для невзвешенного графа используем множество для хранения соседей
        graph = {i: set() for i in range(num_vertices)}

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < density:
                if weighted:
                    weight = random.randint(1, 10)
                    graph[i][j] = weight
                    graph[j][i] = weight
                else:
                    graph[i].add(j)
                    graph[j].add(i)
    return graph

# Визуализация графа
def visualize_graph(graph, title="Граф"):
    G = nx.Graph()
    for vertex in graph:
        for neighbor in graph[vertex]:
            if isinstance(graph[vertex], dict):  # Взвешенный граф
                weight = graph[vertex][neighbor]
                G.add_edge(vertex, neighbor, weight=weight)
            else:  # Невзвешенный граф
                G.add_edge(vertex, neighbor)

    pos = nx.spring_layout(G)  # Позиционирование вершин
    plt.figure(figsize=(8, 6))
    if isinstance(graph[list(graph.keys())[0]], dict):  # Взвешенный граф
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    else:  # Невзвешенный граф
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

# Замер времени выполнения алгоритма
def measure_time(algorithm, graph, start):
    start_time = time.perf_counter()
    algorithm(graph, start)
    return time.perf_counter() - start_time

# Основная функция для экспериментов
def main():
    sizes = [10, 20, 50, 100, 200, 300, 500]  # Размеры графов
    bfs_times = []
    dfs_times = []
    dijkstra_times = []

    for size in sizes:
        print(f"\nГраф из {size} вершин:")
        graph = generate_graph(size, weighted=True)  # Взвешенный граф для Дейкстры
        unweighted_graph = generate_graph(size)      # Невзвешенный граф для BFS и DFS
        #

        # Замер времени для BFS
        bfs_time = measure_time(bfs, unweighted_graph, 0)
        bfs_times.append(bfs_time)
        print(f"BFS: {bfs_time:.6f} сек")

        # Замер времени для DFS
        dfs_time = measure_time(dfs, unweighted_graph, 0)
        dfs_times.append(dfs_time)
        print(f"DFS: {dfs_time:.6f} сек")

        # Замер времени для Дейкстры
        dijkstra_time = measure_time(dijkstra, graph, 0)
        dijkstra_times.append(dijkstra_time)
        print(f"Дейкстра: {dijkstra_time:.6f} сек")

    # Визуализация результатов
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
    # graph = generate_graph(10, weighted=True)  # Взвешенный граф для Дейкстры
    # unweighted_graph = generate_graph(10)
    # visualize_graph(graph, title=f"Взвешенный граф ({10} вершин)")
    # visualize_graph(unweighted_graph, title=f"Невзвешенный граф ({10} вершин)")
    main()
