import heapq
import math
import time
import matplotlib.pyplot as plt

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def diagonal_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    D = 1
    D2 = math.sqrt(2)
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def a_star_search(grid, start, goal, heuristic):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited_nodes = 0

    while open_list:
        _, current = heapq.heappop(open_list)
        visited_nodes += 1

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, visited_nodes

        # for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Только 4 направления

            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + (math.sqrt(2) if dx != 0 and dy != 0 else 1)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None, visited_nodes

def plot_grids(grids, paths, starts, goals, heuristics, titles):
    fig, axes = plt.subplots(len(grids), len(heuristics), figsize=(5 * len(heuristics), 5 * len(grids)))

    for i, grid in enumerate(grids):
        for j, heuristic in enumerate(heuristics):
            ax = axes[i, j] if len(grids) > 1 else axes[j]
            ax.set_title(f"{titles[i]} - {heuristic.__name__}")

            for x in range(len(grid)):
                for y in range(len(grid[0])):
                    if grid[x][y] == 1:
                        ax.add_patch(plt.Rectangle((y, len(grid) - 1 - x), 1, 1, color='black'))

            path = paths[i][j]
            if path:
                for x, y in path:
                    ax.add_patch(plt.Rectangle((y, len(grid) - 1 - x), 1, 1, color='blue', alpha=0.5))

            ax.add_patch(plt.Rectangle((starts[i][1], len(grid) - 1 - starts[i][0]), 1, 1, color='green'))
            ax.add_patch(plt.Rectangle((goals[i][1], len(grid) - 1 - goals[i][0]), 1, 1, color='red'))

            ax.set_xticks(range(len(grid[0]) + 1))
            ax.set_yticks(range(len(grid) + 1))
            ax.grid(True)

    plt.show()

grids = [
    [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ]
]

start_positions = [(0, 0), (0, 0)]
goal_positions = [(4, 4), (9, 9)]
heuristics = [manhattan_distance, euclidean_distance, diagonal_distance]

titles = ["5x5", "10x10"]
paths = [[] for _ in grids]

for i, grid in enumerate(grids):
    for heuristic in heuristics:
        start_time = time.time()
        path, visited = a_star_search(grid, start_positions[i], goal_positions[i], heuristic)
        end_time = time.time()

        print(f"{titles[i]} - {heuristic.__name__}")
        print(f"Найденный путь: {path}")
        print(f"Посещённые вершины: {visited}")
        print(f"Время работы: {end_time - start_time:.6f} секунд\n")

        paths[i].append(path)

plot_grids(grids, paths, start_positions, goal_positions, heuristics, titles)
