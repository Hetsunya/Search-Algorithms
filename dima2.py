import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import heapq
import time

class Graph:
    def __init__(self, directed=False):
        self.graph = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, weight=1):
        self.graph[u].append((v, weight))
        if not self.directed:
            self.graph[v].append((u, weight))

    def bfs(self, start_vertex):
        if start_vertex not in self.graph:
            raise ValueError(f"Вершина {start_vertex} отсутствует в графе.")

        visited = set([start_vertex])
        queue = deque([start_vertex])
        order = []
        edges = []

        print("\n[Шаги выполнения BFS]")
        start_time = time.time()

        while queue:
            vertex = queue.popleft()
            order.append(vertex)
            print(f"Посещаем вершину: {vertex}")
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    edges.append((vertex, neighbor))
                    print(f"Добавляем в очередь: {neighbor}")

        end_time = time.time()
        print(f"BFS завершён за {end_time - start_time:.6f} секунд.")

        self.visualize(edges, title="BFS Traversal")
        return order, end_time - start_time

    def dfs(self, start_vertex):
        if start_vertex not in self.graph:
            raise ValueError(f"Вершина {start_vertex} отсутствует в графе.")

        visited = set()
        order = []
        edges = []

        def dfs_recursive(vertex):
            visited.add(vertex)
            order.append(vertex)
            print(f"Посещаем вершину: {vertex}")
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    edges.append((vertex, neighbor))
                    print(f"Переходим к вершине: {neighbor}")
                    dfs_recursive(neighbor)

        print("\n[Шаги выполнения DFS]")
        start_time = time.time()
        dfs_recursive(start_vertex)
        end_time = time.time()
        print(f"DFS завершён за {end_time - start_time:.6f} секунд.")

        self.visualize(edges, title="DFS Traversal")
        return order, end_time - start_time

    def dijkstra(self, start_vertex, weighted=True):
        if start_vertex not in self.graph:
            raise ValueError(f"Вершина {start_vertex} отсутствует в графе.")

        distances = {vertex: float('inf') for vertex in self.graph}
        distances[start_vertex] = 0
        priority_queue = [(0, start_vertex)]
        previous = {vertex: None for vertex in self.graph}
        edges = []

        print(f"\n[Шаги выполнения Dijkstra {'(взвешенный граф)' if weighted else '(невзвешенный граф)'}]")
        start_time = time.time()

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_distance > distances[current_vertex]:
                continue

            print(f"Обрабатываем вершину {current_vertex} с текущим расстоянием {current_distance}")

            for neighbor, weight in self.graph[current_vertex]:
                if not weighted:
                    weight = 1

                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))
                    print(f"Обновляем расстояние до {neighbor}: {distance} (через {current_vertex})")

        end_time = time.time()
        print(f"Dijkstra завершён за {end_time - start_time:.6f} секунд.")

        for node, prev in previous.items():
            if prev is not None:
                edges.append((prev, node))

        self.visualize(edges, title="Dijkstra's Shortest Path")
        return distances, end_time - start_time

    def visualize(self, edges, title="Graph"):
        G = nx.Graph() if not self.directed else nx.DiGraph()
        edge_set = set()

        for node in self.graph:
            for neighbor, weight in self.graph[node]:
                if (neighbor, node) not in edge_set:
                    G.add_edge(node, neighbor, weight=weight)
                    edge_set.add((node, neighbor))

        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))

        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1500, font_size=14)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="red", width=2)

        plt.title(title)
        plt.show()

g = Graph(directed=False)
edges = [
    (1, 2, 1), (1, 3, 4), (2, 4, 2),
    (3, 4, 1), (4, 5, 3), (2, 5, 5)
]
for u, v, w in edges:
    g.add_edge(u, v, w)

bfs_result, bfs_time = g.bfs(1)
dfs_result, dfs_time = g.dfs(1)
dijkstra_weighted, dijkstra_weighted_time = g.dijkstra(1, weighted=True)
dijkstra_unweighted, dijkstra_unweighted_time = g.dijkstra(1, weighted=False)

print("\n[Сравнение времени выполнения]")
print(f"BFS: {bfs_time:.6f} секунд")
print(f"DFS: {dfs_time:.6f} секунд")
print(f"Dijkstra (взвешенный граф): {dijkstra_weighted_time:.6f} секунд")
print(f"Dijkstra (невзвешенный граф): {dijkstra_unweighted_time:.6f} секунд")
