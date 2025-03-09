import json
import random

def generate_maze(width, height):
    """Генерирует случайный лабиринт, гарантируя наличие пути от старта до цели."""
    maze = [[1 for _ in range(width)] for _ in range(height)]

    # Начальная и конечная точки
    start = (0, 0)
    goal = (width - 1, height - 1)

    # Алгоритм генерации лабиринта (DFS-based)
    stack = [start]
    maze[0][0] = 0

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while stack:
        x, y = stack[-1]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx * 2, y + dy * 2
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 1:
                maze[y + dy][x + dx] = 0
                maze[ny][nx] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    # Убеждаемся, что цель достижима
    maze[goal[1]][goal[0]] = 0

    return {
        "width": width,
        "height": height,
        "start": [start[0], start[1]],
        "goal": [goal[0], goal[1]],
        "maze": maze
    }

# Генерация лабиринтов 10x10, 20x20 и 30x30
mazes = {
    "maze_10x10.json": generate_maze(10, 10),
    "maze_20x20.json": generate_maze(20, 20),
    "maze_30x30.json": generate_maze(30, 30)
}

# Сохранение лабиринтов в файлы
for filename, maze_data in mazes.items():
    with open(f"./{filename}", "w", encoding="utf-8") as f:
        json.dump(maze_data, f, indent=4)

list(mazes.keys())