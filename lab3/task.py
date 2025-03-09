import heapq
import json
import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(grid, start, goal, heuristic):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited_nodes = []

    while open_list:
        _, current = heapq.heappop(open_list)
        visited_nodes.append(current)

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

    return None, visited_nodes  # Путь не найден

def visualize(grid, path, visited, start, goal):
    fig, ax = plt.subplots()
    ax.set_xticks(range(len(grid[0]) + 1))
    ax.set_yticks(range(len(grid) + 1))
    ax.grid(True)

    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if grid[x][y] == 1:
                ax.add_patch(plt.Rectangle((y, len(grid) - 1 - x), 1, 1, color='black'))

    ax.add_patch(plt.Rectangle((start[1], len(grid) - 1 - start[0]), 1, 1, color='green'))
    ax.add_patch(plt.Rectangle((goal[1], len(grid) - 1 - goal[0]), 1, 1, color='red'))

    frames = []
    for i in range(len(visited)):
        frame = []
        for x, y in visited[:i]:
            frame.append(ax.add_patch(plt.Rectangle((y, len(grid) - 1 - x), 1, 1, color='orange', alpha=0.3)))
        if path and i < len(path):
            x, y = path[i]
            frame.append(ax.add_patch(plt.Rectangle((y, len(grid) - 1 - x), 1, 1, color='blue', alpha=0.7)))
        frames.append(frame)

    def update(frame):
        for patch in frame:
            ax.add_patch(patch)

    ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)
    plt.show()

def load_mazes_from_folder(folder):
    """Загружает все лабиринты из папки."""
    mazes = {}
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), "r") as f:
                maze_data = json.load(f)
                size = (maze_data['width'], maze_data['height'])
                mazes[size] = maze_data
    return mazes

if __name__ == "__main__":
    maze_folder = "mazes"  # Папка с сохранёнными лабиринтами
    mazes = load_mazes_from_folder(maze_folder)

    for size, data in mazes.items():
        print(f"Лабиринт {size[0]}x{size[1]}")
        start, goal, grid = tuple(data['start']), tuple(data['goal']), data['maze']
        start_time = time.time()
        path, visited = a_star_search(grid, start, goal, manhattan_distance)
        end_time = time.time()

        if path:
            print(f"  Найденный путь: {path}")
        else:
            print(f"  Путь не найден!")

        print(f"  Посещённые вершины: {len(visited)}")
        print(f"  Время работы: {end_time - start_time:.6f} секунд")

        maze_filename = f"mazes/{size[0]}x{size[1]}_result.json"
        with open(maze_filename, "w") as maze_file:
            json.dump({"start": start, "goal": goal, "grid": grid, "path": path, "visited": visited}, maze_file, indent=4)

        visualize(grid, path, visited, start, goal)
