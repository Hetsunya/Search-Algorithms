import heapq
import math
import time
import matplotlib.pyplot as plt


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


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

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None, visited_nodes


def plot_grid(grid, path, start, goal):
    fig, ax = plt.subplots()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                ax.add_patch(plt.Rectangle((j, len(grid) - 1 - i), 1, 1, color='black'))

    if path:
        for x, y in path:
            ax.add_patch(plt.Rectangle((y, len(grid) - 1 - x), 1, 1, color='blue', alpha=0.5))

    ax.add_patch(plt.Rectangle((start[1], len(grid) - 1 - start[0]), 1, 1, color='green'))
    ax.add_patch(plt.Rectangle((goal[1], len(grid) - 1 - goal[0]), 1, 1, color='red'))

    ax.set_xticks(range(len(grid[0]) + 1))
    ax.set_yticks(range(len(grid) + 1))
    ax.grid(True)
    plt.show()


# Усложнённый лабиринт (0 - проходимая клетка, 1 - стена)
grid = [
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
]

start = (0, 0)
goal = (9, 9)

# Сравнение эвристик
for heuristic in [manhattan_distance, euclidean_distance]:
    start_time = time.perf_counter()
    path, visited = a_star_search(grid, start, goal, heuristic)
    end_time = time.perf_counter()

    print(f"Эвристика: {heuristic.__name__}")
    print(f"Найденный путь: {path}")
    print(f"Посещённые вершины: {visited}")
    print(f"Время работы: {end_time - start_time:.9f} секунд\n")

    plot_grid(grid, path, start, goal)
