import heapq
import time
import random
import matplotlib.pyplot as plt
import networkx as nx


def depth_first_search(graph, start, target=None):
    explored, stack, traversal = set(), [start], []
    while stack:
        node = stack.pop()
        if node not in explored:
            explored.add(node)
            traversal.append(node)
            if target is not None and node == target:
                break
            stack.extend(sorted(graph.get(node, {}).keys() - explored, reverse=True))
    return traversal


def breadth_first_search(graph, start, target=None):
    explored, queue, traversal = set(), [start], []
    while queue:
        node = queue.pop(0)
        if node not in explored:
            explored.add(node)
            traversal.append(node)
            if target is not None and node == target:
                break
            queue.extend(sorted(graph.get(node, {}).keys() - explored))
    return traversal


def shortest_path_dijkstra(graph, start, target=None):
    priority_queue, visited = [(0, start)], set()
    distances, predecessors = {start: 0}, {start: None}
    path_traversal = []

    while priority_queue:
        cost, node = heapq.heappop(priority_queue)
        if node in visited:
            continue
        visited.add(node)
        path_traversal.append(node)
        if target is not None and node == target:
            break
        for neighbor, weight in graph.get(node, {}).items():
            if neighbor not in visited:
                new_cost = cost + weight
                if new_cost < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_cost
                    predecessors[neighbor] = node
                    heapq.heappush(priority_queue, (new_cost, neighbor))

    path, total_cost = [], float('inf')
    if target in distances:
        while target is not None:
            path.append(target)
            target = predecessors[target]
        path.reverse()
        total_cost = distances[path[-1]]
    return path_traversal, total_cost, path


def draw_graph(graph, title="Граф"):
    G = nx.Graph()
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=600, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.title(title)
    plt.show()


def execution_time(algorithm, graph, start, target=None):
    start_time = time.perf_counter()
    result = algorithm(graph, start, target)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, result


def analyze_graph(graph, name):
    print(f"\n{name}:")
    algorithms = [(breadth_first_search, "BFS"), (depth_first_search, "DFS"), (shortest_path_dijkstra, "Дейкстра")]
    for algo, label in algorithms:
        if label == "Дейкстра":
            time_taken, (order, distance, path) = execution_time(algo, graph, 0, 4)
            print(f"{label}: {time_taken:.6f} сек, Обход: {order}, Длина пути: {distance}, Путь: {path}")
        else:
            time_taken, order = execution_time(algo, graph, 0)
            print(f"{label}: {time_taken:.6f} сек, Обход: {order}")
    draw_graph(graph, title=name)


def generate_graph(size):
    return {i: {random.randint(0, size - 1): random.randint(1, 10) for _ in range(random.randint(1, min(4, size - 1)))}
            for i in range(size)}


def compare_algorithms(sizes, bfs_times, dfs_times, dijkstra_times):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, bfs_times, marker='o', label='BFS', color='r')
    plt.plot(sizes, dfs_times, marker='o', label='DFS', color='g')
    plt.plot(sizes, dijkstra_times, marker='o', label='Дейкстра', color='b')
    plt.xlabel('Размер графа')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Сравнение алгоритмов обхода графов')
    plt.legend()
    plt.grid(True)
    plt.show()


# Заданные графы
sample_graphs = [
    ({0: {1: 1, 2: 1}, 1: {0: 1, 3: 1, 4: 1}, 2: {0: 1, 4: 1}, 3: {1: 1, 4: 1}, 4: {1: 1, 2: 1, 3: 1}}, "Малый граф"),
    ({0: {1: 4, 2: 1}, 1: {0: 4, 3: 1}, 2: {0: 1, 3: 2}, 3: {1: 1, 2: 2, 4: 3}, 4: {3: 3}}, "Взвешенный граф")
]

for graph, name in sample_graphs:
    analyze_graph(graph, name)

# Случайные графы
sizes = [10, 20, 40, 60, 80, 100]
bfs_times, dfs_times, dijkstra_times = [], [], []
for size in sizes:
    random_graph = generate_graph(size)
    print(f"\nСлучайный граф из {size} вершин")
    for algo, times, label in [(breadth_first_search, bfs_times, "BFS"), (depth_first_search, dfs_times, "DFS"),
                               (shortest_path_dijkstra, dijkstra_times, "Дейкстра")]:
        time_taken, _ = execution_time(algo, random_graph, 0)
        times.append(time_taken)
        print(f"{label}: {time_taken:.6f} сек")

compare_algorithms(sizes, bfs_times, dfs_times, dijkstra_times)
